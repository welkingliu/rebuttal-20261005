"""Manifest-driven bridge to official SGG repositories.

The local architecture-inspired models are useful smoke-test surrogates, but
their outputs must not be attributed to published methods. This adapter loads a
factory implemented inside a checked-out official repository and validates the
checkpoint and provenance metadata before exposing the audit interfaces.

Manifest schema:
{
  "name": "motifs_official",
  "paradigm": "RNN-based",
  "factory": "my_official_adapters.motifs:create_adapter",
  "checkpoints": {
    "predcls": {"path": "/abs/path/predcls.pth", "sha256": "..."},
    "sgcls": {"path": "/abs/path/sgcls.pth", "sha256": "..."},
    "sgdet": {"path": "/abs/path/sgdet.pth", "sha256": "..."}
  },
  "supported_tasks": ["predcls", "sgcls", "sgdet"],
  "source_url": "https://github.com/...",
  "source_commit": "full git commit",
  "training_dataset": "VG-150",
  "supported_datasets": ["vg"],
  "ontology_ids": {"vg": "vg150:<digest>"},
  "perturbation_contract": {"visual_noise": true, ...},
  "config": {...},
  "reference_metrics": {"SGDet/R@50": 0.0}
}

The factory receives ``checkpoint``, ``checkpoints``, ``device``, ``config``
and ``diagnostic_task`` keyword arguments. ``checkpoint`` is the selected
diagnostic checkpoint retained for simple single-task factories; ``checkpoints``
contains every task-specific path. The returned object must implement
``predict_scene_graph``, ``predict``, ``extract_node_features`` and
``diagnostic_input_fingerprint``. The final method hashes the exact visual
tensors consumed by diagnostic inference so unused-key perturbations fail.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import torch


REQUIRED_MANIFEST_FIELDS = (
    "name", "architecture_family", "paradigm", "factory",
    "source_url", "source_root", "source_commit", "training_dataset",
    "reference_dataset", "reference_metrics", "metric_scale", "input_source",
    "supported_datasets", "ontology_ids", "supported_tasks",
    "training_seed", "parameter_count", "baseline_mR", "perturbation_contract",
)

VALID_SGG_TASKS = ("predcls", "sgcls", "sgdet")

PERTURBATION_STRATEGIES = {
    "full", "noise", "swap", "union_zero", "boxes_only",
    "visual_noise", "color_jitter", "union_attenuation",
    "on_manifold_replacement", "random_node_mask", "key_node_mask",
    "unrelated_node_mask",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_source_url(value: str) -> str:
    normalised = str(value).lower()
    if normalised.endswith(".git"):
        normalised = normalised[:-4]
    return normalised.rstrip("/")


def verify_official_source(source_root: Path, expected_url: str,
                           expected_commit: str) -> dict:
    """Verify either a git checkout or a checksum-pinned commit archive."""
    expected_commit = str(expected_commit).lower()
    if (source_root / ".git").is_dir():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            remote_url = subprocess.check_output(
                ["git", "-C", str(source_root), "remote", "get-url", "origin"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Cannot verify official git checkout: {source_root}") from exc
        if commit.lower() != expected_commit:
            raise RuntimeError(
                f"Source commit mismatch for {source_root}: expected {expected_commit}, got {commit}"
            )
        if _normalise_source_url(remote_url) != _normalise_source_url(expected_url):
            raise RuntimeError(
                f"Source URL mismatch: manifest {expected_url}, git {remote_url}"
            )
        return {"commit": commit, "source_type": "git", "source_url": remote_url}

    marker_path = source_root / ".official_source.json"
    if not marker_path.is_file():
        raise RuntimeError(
            f"Official source requires .git or .official_source.json: {source_root}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid official archive marker: {marker_path}") from exc
    archive_path = Path(str(marker.get("archive_path", ""))).expanduser()
    valid = (
        str(marker.get("commit", "")).lower() == expected_commit
        and _normalise_source_url(marker.get("repository_url", ""))
        == _normalise_source_url(expected_url)
        and marker.get("archive_sha256")
        and archive_path.is_file()
        and _sha256_file(archive_path) == marker["archive_sha256"]
    )
    if not valid:
        raise RuntimeError(
            f"Official commit archive provenance mismatch: {marker_path}"
        )
    return {
        "commit": expected_commit,
        "source_type": "official_commit_archive",
        "source_url": marker.get("repository_url"),
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": marker["archive_sha256"],
    }


def fingerprint_tensors(**named_tensors) -> str:
    """Hash named tensors after every adapter-side preprocessing decision."""
    digest = hashlib.sha256()
    for name in sorted(named_tensors):
        tensor = named_tensors[name]
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Fingerprint value {name!r} is not a tensor")
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class OfficialSGGAdapter:
    implementation_kind = "official_adapter"
    supports_standard_sgg = True
    requires_checkpoint = False

    def __init__(self, manifest_path: str, device="cpu"):
        self.manifest_path = str(Path(manifest_path).resolve())
        self.device = torch.device(device)
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        missing = [
            key for key in REQUIRED_MANIFEST_FIELDS
            if key not in self.manifest
            or self.manifest[key] is None
            or self.manifest[key] == ""
            or self.manifest[key] == []
            or self.manifest[key] == {}
        ]
        if missing:
            raise ValueError(f"Official adapter manifest is missing fields: {missing}")

        supported_tasks = tuple(
            str(task).lower() for task in self.manifest["supported_tasks"]
        )
        invalid_tasks = sorted(set(supported_tasks) - set(VALID_SGG_TASKS))
        if invalid_tasks or not supported_tasks:
            raise ValueError(
                f"Invalid supported_tasks: invalid={invalid_tasks}, values={supported_tasks}"
            )
        if len(set(supported_tasks)) != len(supported_tasks):
            raise ValueError("manifest.supported_tasks contains duplicates")
        self.supported_tasks = supported_tasks
        self.diagnostic_task = str(
            self.manifest.get(
                "diagnostic_task",
                "sgdet" if "sgdet" in supported_tasks else supported_tasks[0],
            )
        ).lower()
        if self.diagnostic_task not in self.supported_tasks:
            raise ValueError("manifest.diagnostic_task must be listed in supported_tasks")

        self.name = str(self.manifest["name"])
        self.architecture_family = str(self.manifest["architecture_family"])
        self.paradigm = str(self.manifest["paradigm"])
        source_root = Path(self.manifest["source_root"]).expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Official source checkout not found: {source_root}")
        source_provenance = verify_official_source(
            source_root,
            self.manifest["source_url"],
            self.manifest["source_commit"],
        )
        source_commit = source_provenance["commit"]
        self.execution_mode = str(
            self.manifest.get("execution_mode", "live_adapter")
        ).lower()
        if self.execution_mode not in {"live_adapter", "prediction_cache"}:
            raise ValueError("execution_mode must be live_adapter or prediction_cache")
        self.implementation_kind = (
            "official_prediction_cache"
            if self.execution_mode == "prediction_cache" else "official_adapter"
        )
        allowed_inputs = {
            "raw_images", "official_precomputed_features", "detector_roi_features",
            "official_prediction_cache",
        }
        if self.manifest["input_source"] not in allowed_inputs:
            raise ValueError(
                f"Official input_source must be one of {sorted(allowed_inputs)}"
            )
        if self.manifest["reference_dataset"] not in self.manifest["supported_datasets"]:
            raise ValueError("reference_dataset must be listed in supported_datasets")
        contract = self.manifest["perturbation_contract"]
        if not isinstance(contract, dict):
            raise TypeError("manifest.perturbation_contract must be an object")
        missing_contract = sorted(PERTURBATION_STRATEGIES - set(contract))
        invalid_contract = sorted(
            key for key, value in contract.items()
            if key in PERTURBATION_STRATEGIES and not isinstance(value, bool)
        )
        if missing_contract or invalid_contract:
            raise ValueError(
                "Invalid perturbation_contract: "
                f"missing={missing_contract}, non_boolean={invalid_contract}"
            )
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        checkpoint_specs = self.manifest.get("checkpoints")
        if checkpoint_specs is None:
            # Backward-compatible single checkpoint manifests are valid only
            # when that checkpoint is explicitly scoped to one task.
            if "checkpoint" not in self.manifest or "sha256" not in self.manifest:
                raise ValueError(
                    "Official manifest requires checkpoints or checkpoint+sha256"
                )
            if len(self.supported_tasks) != 1:
                raise ValueError(
                    "A multi-task official adapter must declare task-specific checkpoints"
                )
            checkpoint_specs = {
                self.supported_tasks[0]: {
                    "path": self.manifest["checkpoint"],
                    "sha256": self.manifest["sha256"],
                }
            }
        if not isinstance(checkpoint_specs, dict):
            raise TypeError("manifest.checkpoints must be an object keyed by SGG task")
        if set(checkpoint_specs) != set(self.supported_tasks):
            raise ValueError(
                "manifest.checkpoints keys must exactly match supported_tasks: "
                f"checkpoints={sorted(checkpoint_specs)}, tasks={sorted(self.supported_tasks)}"
            )

        checkpoints = {}
        checkpoint_digests = {}
        for task, spec in checkpoint_specs.items():
            if not isinstance(spec, dict) or not spec.get("path") or not spec.get("sha256"):
                raise ValueError(
                    f"Checkpoint spec for {task} requires non-empty path and sha256"
                )
            checkpoint = Path(spec["path"]).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Official checkpoint not found: {checkpoint}")
            digest = _sha256_file(checkpoint)
            if digest.lower() != str(spec["sha256"]).lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {checkpoint}: expected {spec['sha256']}, got {digest}"
                )
            checkpoints[str(task)] = str(checkpoint)
            checkpoint_digests[str(task)] = digest

        for metric_name in self.manifest["reference_metrics"]:
            task_name, separator, _ = str(metric_name).partition("/")
            if separator and task_name.lower() not in self.supported_tasks:
                raise ValueError(
                    f"Reference metric {metric_name} is outside supported_tasks"
                )

        module_name, separator, factory_name = str(self.manifest["factory"]).partition(":")
        if not separator:
            raise ValueError("manifest.factory must use 'python.module:function' syntax")
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        diagnostic_checkpoint = checkpoints[self.diagnostic_task]
        factory_kwargs = {
            "checkpoint": diagnostic_checkpoint,
            "checkpoints": dict(checkpoints),
            "device": self.device,
            "config": self.manifest.get("config", {}),
            "diagnostic_task": self.diagnostic_task,
        }
        if self.execution_mode == "prediction_cache":
            factory_kwargs["manifest"] = self.manifest
        self.model = factory(**factory_kwargs)
        for method in (
            "predict_scene_graph", "predict", "extract_node_features",
            "diagnostic_input_fingerprint",
        ):
            if not callable(getattr(self.model, method, None)):
                raise TypeError(f"Official adapter '{self.name}' lacks required method: {method}")
        self.supports_joint_task_inference = callable(
            getattr(self.model, "predict_scene_graph_tasks", None)
        )
        declared_parameter_count = int(self.manifest["parameter_count"])
        if hasattr(self.model, "reported_parameter_count"):
            actual_parameter_count = int(
                getattr(self.model, "reported_parameter_count", -1)
            )
        else:
            actual_parameter_count = sum(
                parameter.numel() for parameter in self.model.parameters()
            )
        if actual_parameter_count != declared_parameter_count:
            raise RuntimeError(
                f"Parameter-count mismatch for {self.name}: manifest "
                f"{declared_parameter_count}, adapter {actual_parameter_count}"
            )

        self.checkpoint_status = {
            "path": diagnostic_checkpoint,
            "paths_by_task": checkpoints,
            "loaded": True,
            "sha256": checkpoint_digests[self.diagnostic_task],
            "sha256_by_task": checkpoint_digests,
            "parameter_coverage": 1.0,
            "metadata": self.manifest,
            "source_commit_verified": source_commit,
            "source_provenance": source_provenance,
            "parameter_count": actual_parameter_count,
            "execution_mode": self.execution_mode,
        }
        self.eval()

    def eval(self):
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def train(self, mode: bool = True):
        if hasattr(self.model, "train"):
            self.model.train(mode)
        return self

    def to(self, device):
        self.device = torch.device(device)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        return self

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def predict(self, batch: dict) -> dict:
        return self.model.predict(batch)

    def predict_scene_graph(self, batch: dict, task: str) -> dict:
        if str(task).lower() not in self.supported_tasks:
            raise NotImplementedError(
                f"Official adapter '{self.name}' does not support task: {task}"
            )
        return self.model.predict_scene_graph(batch, task=task)

    def predict_scene_graph_tasks(self, batch: dict, tasks) -> dict:
        """Optionally share detector inference across PredCls/SGCls/SGDet."""
        unsupported = sorted(set(tasks) - set(self.supported_tasks))
        if unsupported:
            raise NotImplementedError(
                f"Official adapter '{self.name}' does not support tasks: {unsupported}"
            )
        method = getattr(self.model, "predict_scene_graph_tasks", None)
        if not callable(method):
            raise NotImplementedError("official factory has no joint-task inference method")
        return method(batch, tasks=tuple(tasks))

    def extract_node_features(self, batch: dict) -> torch.Tensor:
        return self.model.extract_node_features(batch)

    def diagnostic_input_fingerprint(self, batch: dict) -> str:
        """Fingerprint the exact visual inputs consumed by ``predict``."""
        value = self.model.diagnostic_input_fingerprint(batch)
        if not isinstance(value, str) or not value:
            raise TypeError(
                "diagnostic_input_fingerprint must return a non-empty string"
            )
        return value

    def supports_dataset(self, dataset: str, ontology_id: str | None = None) -> bool:
        supported = {str(name).lower() for name in self.manifest["supported_datasets"]}
        if dataset.lower() not in supported:
            return False
        expected = self.manifest.get("ontology_ids", {}).get(dataset)
        return expected in (None, "*") or ontology_id == expected

    def supports_perturbation(self, strategy: str) -> bool:
        """Return the factory author's declared input-intervention support."""
        return bool(self.manifest["perturbation_contract"].get(strategy, False))

    def validate_reproduction(self, standard_result: dict, dataset: str) -> dict:
        """Compare local standard metrics with the manifest's paper values."""
        if dataset != str(self.manifest["reference_dataset"]).lower():
            return {"status": "not_applicable", "dataset": dataset}
        scale = str(self.manifest["metric_scale"]).lower()
        if scale not in ("fraction", "percent"):
            raise ValueError("manifest.metric_scale must be 'fraction' or 'percent'")
        tolerance = float(self.manifest.get("reproduction_tolerance", 0.02))
        metric_policies = self.manifest.get("reference_metric_policies", {})
        allowed_policies = {"required", "report_only_protocol_mismatch"}
        invalid_policies = {
            name: policy for name, policy in metric_policies.items()
            if policy not in allowed_policies
        }
        if invalid_policies:
            raise ValueError(
                f"Invalid reference_metric_policies: {invalid_policies}"
            )
        comparisons = {}
        observed_image_counts = set()
        for metric_name, reference in self.manifest["reference_metrics"].items():
            if reference is None:
                continue
            task_name, separator, key = str(metric_name).partition("/")
            if not separator:
                raise ValueError(f"Invalid reference metric name: {metric_name}")
            task_result = standard_result.get("tasks", {}).get(task_name.lower(), {})
            observed = task_result.get("metrics", {}).get(key)
            if task_result.get("num_images") is not None:
                observed_image_counts.add(int(task_result["num_images"]))
            reference_fraction = float(reference) / (100.0 if scale == "percent" else 1.0)
            delta = None if observed is None else float(observed) - reference_fraction
            policy = metric_policies.get(metric_name, "required")
            comparisons[metric_name] = {
                "reference": reference_fraction,
                "observed": observed,
                "absolute_delta": None if delta is None else abs(delta),
                "within_tolerance": delta is not None and abs(delta) <= tolerance,
                "policy": policy,
            }
        if not comparisons:
            return {"status": "missing_reference_values", "dataset": dataset}
        expected_images = self.manifest.get("reference_eval_images")
        if expected_images is not None:
            expected_images = int(expected_images)
            if observed_image_counts != {expected_images}:
                return {
                    "status": "subset_diagnostic",
                    "dataset": dataset,
                    "expected_eval_images": expected_images,
                    "observed_eval_images": sorted(observed_image_counts),
                    "absolute_tolerance": tolerance,
                    "comparisons": comparisons,
                }
        required = [
            item for item in comparisons.values() if item["policy"] == "required"
        ]
        if not required:
            return {
                "status": "missing_required_reference_values",
                "dataset": dataset,
                "absolute_tolerance": tolerance,
                "comparisons": comparisons,
            }
        passed = all(item["within_tolerance"] for item in required)
        protocol_qualified = any(
            item["policy"] == "report_only_protocol_mismatch"
            and not item["within_tolerance"]
            for item in comparisons.values()
        )
        return {
            "status": (
                "pass_with_protocol_qualification"
                if passed and protocol_qualified
                else "pass" if passed else "fail"
            ),
            "dataset": dataset,
            "absolute_tolerance": tolerance,
            "protocol_qualified": protocol_qualified,
            "comparisons": comparisons,
        }

    def grounding_parameters(self):
        """Parameters eligible for mitigation fine-tuning."""
        method = getattr(self.model, "grounding_parameters", None)
        return method() if callable(method) else self.model.parameters()

    def grounding_parameter_groups(self):
        """Named mitigation groups used to verify object-head optimization."""
        method = getattr(self.model, "grounding_parameter_groups", None)
        if callable(method):
            groups = method()
            if not isinstance(groups, dict):
                raise TypeError("grounding_parameter_groups must return a dict")
            return groups
        return {"all": list(self.grounding_parameters())}

    def grounding_state_dict(self):
        method = getattr(self.model, "grounding_state_dict", None)
        return method() if callable(method) else self.model.state_dict()

    def load_grounding_state_dict(self, state):
        method = getattr(self.model, "load_grounding_state_dict", None)
        if callable(method):
            method(state)
        else:
            self.model.load_state_dict(state)

    @property
    def supports_mitigation(self) -> bool:
        contract = self.manifest.get("mitigation_contract", {})
        required = (
            "forward_grounding", "trainable_grounding_parameters",
            "object_logits", "trainable_object_parameters",
        )
        return (
            self.execution_mode == "live_adapter"
            and all(contract.get(key) is True for key in required)
            and contract.get("relation_logit_alignment") == "gt_relations"
            and contract.get("object_logit_alignment") == "gt_entities"
            and callable(getattr(self.model, "forward_grounding", None))
        )

    def forward_grounding(self, batch: dict) -> dict:
        """Differentiable object/relation forward used by mitigation training.

        Required keys are ``pred_rel_scores`` and ``pred_entity_scores``.
        ``mask_entity_scores`` is optional and activates box/mask consistency.
        """
        method = getattr(self.model, "forward_grounding", None)
        if not callable(method):
            raise NotImplementedError(
                f"Official adapter '{self.name}' must implement forward_grounding"
            )
        output = method(batch)
        if not isinstance(output, dict):
            raise TypeError("forward_grounding must return a dict")
        missing = [
            key for key in ("pred_rel_scores", "pred_entity_scores")
            if key not in output
        ]
        if missing:
            raise KeyError(f"forward_grounding output is missing: {missing}")
        return output
