"""Joint statistical analysis for the same checkpoint across audit levels."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def _correlation_summary(rows, x_key, y_key, seed=71, trials=2000):
    pairs = [
        (_finite(row.get(x_key)), _finite(row.get(y_key)))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return {"status": "insufficient_data", "n": len(pairs)}
    x = np.asarray([pair[0] for pair in pairs])
    y = np.asarray([pair[1] for pair in pairs])
    observed = _spearman(x, y)
    if not np.isfinite(observed):
        return {"status": "degenerate", "n": len(pairs), "spearman_rho": observed}

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(trials):
        idx = rng.integers(0, len(pairs), size=len(pairs))
        rho = _spearman(x[idx], y[idx])
        if np.isfinite(rho):
            boot.append(rho)
    permuted = np.asarray([
        _spearman(x, rng.permutation(y)) for _ in range(trials)
    ])
    p_value = (1 + int(np.sum(np.abs(permuted) >= abs(observed)))) / (trials + 1)
    return {
        "status": "ok",
        "n": len(pairs),
        "spearman_rho": observed,
        "paired_bootstrap_95ci": (
            [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if boot else [float("nan"), float("nan")]
        ),
        "permutation_p_value": float(p_value),
    }


def _ols(rows, outcome, exposure, numeric_controls, categorical_controls=()):
    complete = []
    for row in rows:
        numeric = [
            _finite(row.get(key))
            for key in (outcome, exposure, *numeric_controls)
        ]
        categories = [row.get(key) for key in categorical_controls]
        if all(value is not None for value in numeric + categories):
            complete.append((numeric, categories))
    category_levels = {
        key: sorted({str(item[1][index]) for item in complete})
        for index, key in enumerate(categorical_controls)
    }
    dummy_count = sum(max(0, len(levels) - 1) for levels in category_levels.values())
    parameter_count = 2 + len(numeric_controls) + dummy_count
    if len(complete) < max(8, parameter_count + 3):
        return {
            "status": "insufficient_data",
            "n": len(complete),
            "minimum_n": max(8, parameter_count + 3),
        }
    arr = np.asarray([item[0] for item in complete], dtype=np.float64)
    y = arr[:, 0]
    x_raw = arr[:, 1:]
    means = x_raw.mean(axis=0)
    scales = x_raw.std(axis=0)
    if bool((scales < 1e-12).any()):
        return {"status": "degenerate_controls", "n": len(complete)}
    x_parts = [np.ones((len(complete), 1)), (x_raw - means) / scales]
    coefficient_names = ["intercept", exposure, *numeric_controls]
    for category_index, key in enumerate(categorical_controls):
        levels = category_levels[key]
        for level in levels[1:]:
            x_parts.append(np.asarray([
                [float(str(item[1][category_index]) == level)] for item in complete
            ]))
            coefficient_names.append(f"{key}={level}")
    x = np.column_stack(x_parts)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    return {
        "status": "ok",
        "n": len(complete),
        "rank": int(rank),
        "r_squared": float(1 - residual.dot(residual) / max(((y - y.mean()) ** 2).sum(), 1e-12)),
        "coefficients": {
            key: float(beta[index]) for index, key in enumerate(coefficient_names)
        },
        "coefficient_scale": "numeric predictors are standardized; training seed uses categorical fixed effects",
    }


def _mediation(rows, x_key, mediator_key, y_key, seed=113, trials=2000):
    complete = []
    for row in rows:
        values = [_finite(row.get(key)) for key in (x_key, mediator_key, y_key)]
        if all(value is not None for value in values):
            complete.append(values)
    if len(complete) < 10:
        return {"status": "insufficient_data", "n": len(complete), "minimum_n": 10}

    arr = np.asarray(complete, dtype=np.float64)

    def indirect(sample):
        x, mediator, y = sample.T
        design_a = np.column_stack([np.ones(x.size), x])
        a = np.linalg.lstsq(design_a, mediator, rcond=None)[0][1]
        design_b = np.column_stack([np.ones(x.size), x, mediator])
        b = np.linalg.lstsq(design_b, y, rcond=None)[0][2]
        return float(a * b)

    observed = indirect(arr)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(trials):
        sampled = arr[rng.integers(0, arr.shape[0], size=arr.shape[0])]
        try:
            value = indirect(sampled)
            if np.isfinite(value):
                boot.append(value)
        except np.linalg.LinAlgError:
            continue
    return {
        "status": "exploratory_noncausal",
        "n": len(complete),
        "indirect_effect": observed,
        "bootstrap_95ci": [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
        "warning": "Statistical mediation does not identify a causal mechanism without additional assumptions.",
    }


class EvidenceChainAnalyzer:
    """Join metrics by checkpoint and run confirmatory/exploratory statistics."""

    def __init__(self, performance_k=50, diagnostic_k=5):
        self.performance_k = int(performance_k)
        self.diagnostic_k = int(diagnostic_k)

    @staticmethod
    def _training_dataset(load_report, model_name):
        report = load_report.get(model_name, {})
        status = report.get("checkpoint_status") or {}
        metadata = status.get("metadata") or {}
        value = str(metadata.get("training_dataset", "")).lower()
        for key, aliases in {
            "vg": ("vg", "visual genome"),
            "oi": ("openimages", "open images", "oi"),
            "psg": ("psg", "panoptic scene graph"),
            "gqa": ("gqa",),
            "vrd": ("vrd", "visual relationship detection"),
        }.items():
            if any(alias in value for alias in aliases):
                return key
        return None

    def _rows(self, all_results, load_report):
        rows = []
        for dataset_name, dataset_results in all_results.items():
            model_names = set()
            for audit in dataset_results.values():
                if isinstance(audit, dict):
                    model_names.update(audit.keys())
            for model_name in sorted(model_names):
                feature = dataset_results.get("feature_audit", {}).get(model_name, {})
                pair = dataset_results.get("pair_audit", {}).get(model_name, {})
                graph = dataset_results.get("graph_audit", {}).get(model_name, {})
                physical = dataset_results.get("physical_consistency", {}).get(model_name, {})
                standard = dataset_results.get("standard_sgg", {}).get(model_name, {})
                sweep = dataset_results.get("perturbation_sweep", {}).get(model_name, {})
                sgdet = standard.get("tasks", {}).get("sgdet", {}).get("metrics", {})
                report = load_report.get(model_name, {})
                status = report.get("checkpoint_status") or {}
                metadata = status.get("metadata") or {}
                level = "1.000"
                wrong_stable = (
                    sweep.get("strategies", {}).get("union_attenuation", {})
                    .get("curve", {}).get(level, {}).get("transitions", {})
                    .get("wrong_stable")
                )
                training_dataset = self._training_dataset(load_report, model_name)
                parameter_count = report.get("parameter_count", metadata.get("parameter_count"))
                if parameter_count:
                    parameter_count = np.log10(float(parameter_count))
                rows.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "checkpoint_sha256": status.get("sha256"),
                    "implementation_kind": report.get("implementation_kind"),
                    "training_dataset": training_dataset,
                    "is_ood": training_dataset is not None and dataset_name != training_dataset,
                    "effective_rank": feature.get("effective_rank"),
                    "normalized_effective_rank": feature.get("normalized_effective_rank"),
                    "collapse_score": (
                        1 - feature["normalized_effective_rank"]
                        if _finite(feature.get("normalized_effective_rank")) is not None else None
                    ),
                    "dirichlet_energy": feature.get("dirichlet_energy"),
                    "BRR": pair.get("brr_at_k", {}).get(str(self.diagnostic_k), pair.get("BRR")),
                    "MAR": graph.get("MAR"),
                    "PIR": graph.get("PIR"),
                    "WSR": graph.get("WSR"),
                    "PVR": physical.get("PVR"),
                    "PVR_coverage": physical.get("coverage"),
                    "wrong_stable_rate": wrong_stable,
                    "SGDet_mR": sgdet.get(f"mR@{self.performance_k}"),
                    "SGDet_R": sgdet.get(f"R@{self.performance_k}"),
                    "log10_parameter_count": parameter_count,
                    "baseline_mR": metadata.get("baseline_mR"),
                    "training_seed": metadata.get("training_seed"),
                })
        return rows

    def analyze(self, all_results, load_report):
        rows = self._rows(all_results, load_report)
        standard_confirmatory = [
            row for row in rows
            if row.get("implementation_kind") in {
                "official_adapter", "official_prediction_cache",
            }
            and row.get("checkpoint_sha256")
        ]
        diagnostic_confirmatory = [
            row for row in standard_confirmatory
            if row.get("implementation_kind") == "official_adapter"
        ]
        correlations = {}
        for x_key, y_key in (
            ("collapse_score", "BRR"),
            ("dirichlet_energy", "BRR"),
            ("BRR", "MAR"),
            ("BRR", "SGDet_mR"),
            ("MAR", "SGDet_mR"),
            ("wrong_stable_rate", "SGDet_mR"),
        ):
            correlations[f"{x_key}__vs__{y_key}"] = _correlation_summary(
                diagnostic_confirmatory, x_key, y_key
            )

        per_dataset = defaultdict(dict)
        for dataset_name in sorted({row["dataset"] for row in diagnostic_confirmatory}):
            subset = [
                row for row in diagnostic_confirmatory
                if row["dataset"] == dataset_name
            ]
            per_dataset[dataset_name]["BRR__vs__SGDet_mR"] = _correlation_summary(
                subset, "BRR", "SGDet_mR"
            )
            per_dataset[dataset_name]["MAR__vs__SGDet_mR"] = _correlation_summary(
                subset, "MAR", "SGDet_mR"
            )

        ood_rows = [row for row in diagnostic_confirmatory if row["is_ood"]]
        regression = _ols(
            ood_rows,
            outcome="SGDet_mR",
            exposure="BRR",
            numeric_controls=("log10_parameter_count", "baseline_mR"),
            categorical_controls=("training_seed",),
        )
        mediation = _mediation(
            ood_rows, "collapse_score", "BRR", "SGDet_mR"
        )
        return {
            "status": (
                "ok" if standard_confirmatory else "no_official_checkpoint_rows"
            ),
            "analysis_scope": (
                "Prediction caches count only for standard reproduction; "
                "cross-level statistics require a live adapter for the same checkpoint SHA."
            ),
            "causal_claims": "not_identified",
            "performance_k": self.performance_k,
            "diagnostic_k": self.diagnostic_k,
            "num_rows": len(rows),
            "num_standard_confirmatory_rows": len(standard_confirmatory),
            "num_live_diagnostic_rows": len(diagnostic_confirmatory),
            "rows": rows,
            "confirmatory_spearman": correlations,
            "per_dataset_spearman": dict(per_dataset),
            "ood_controlled_regression": regression,
            "exploratory_mediation": mediation,
        }
