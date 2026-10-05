import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

import torch
import torch.nn as nn

from sgg_core.backbones.perception import PerceptionModule

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from prepare_foundation_models import (
    CRADIO_CODE_FILES, DINOV2_COMMIT, RADIO_HF_REVISION,
    download_model, status_for,
)


class FoundationCatalogTest(unittest.TestCase):
    def test_runnable_backbones_exist_in_registry(self):
        path = Path(__file__).resolve().parents[1] / "sgg_core" / "backbones" / "foundation_backbone_catalog.json"
        catalog = json.loads(path.read_text(encoding="utf-8"))
        runnable = {
            item["id"] for item in catalog["backbones"]
            if item["experiment_1_runnable"]
        }
        self.assertTrue(set(catalog["main_comparable_panel"]).issubset(runnable))
        self.assertTrue(runnable.issubset(PerceptionModule.BACKBONE_REGISTRY))
        self.assertEqual(len(catalog["main_comparable_panel"]), 6)
        self.assertNotIn("dinov3_b", catalog["main_comparable_panel"])
        self.assertIn("cradio_v4_so400m", catalog["main_comparable_panel"])
        self.assertIn("sam_vit_b", catalog["main_comparable_panel"])

    def test_dinov2_prefers_canonical_offline_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "foundation_repos" / "dinov2"
            weights = (
                root / "checkpoints" / "foundation" / "dinov2"
                / "dinov2_vitb14_pretrain.pth"
            )
            repo.mkdir(parents=True)
            weights.parent.mkdir(parents=True)
            (repo / "hubconf.py").write_text("# test\n", encoding="utf-8")
            weights.write_bytes(b"weights")
            module = PerceptionModule.__new__(PerceptionModule)
            nn.Module.__init__(module)
            with patch.dict(
                os.environ,
                {"SGG_PROJECT_ROOT": str(root), "TORCH_HOME": str(root / "torch")},
                clear=False,
            ), patch("torch.hub.load", return_value=nn.Identity()) as mocked, patch(
                "torch.load", return_value={}
            ):
                module._init_dinov2("dinov2_b")
            mocked.assert_called_once_with(
                str(repo), "dinov2_vitb14", source="local",
                pretrained=False,
            )

    def test_foundation_status_rejects_wrong_weight_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "foundation_repos" / "dinov2"
            weights = (
                root / "checkpoints" / "foundation" / "dinov2"
                / "dinov2_vitb14_pretrain.pth"
            )
            repo.mkdir(parents=True)
            weights.parent.mkdir(parents=True)
            (repo / "hubconf.py").write_text("# test\n", encoding="utf-8")
            (repo / ".sgg_source.json").write_text(
                json.dumps({"commit": DINOV2_COMMIT}), encoding="utf-8"
            )
            weights.write_bytes(b"not official weights")
            status = status_for("dinov2_b", root)
            self.assertFalse(status["ok"])
            self.assertTrue(status["mismatches"])

    def test_radio_weight_download_uses_hf_mirror(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "prepare_foundation_models._download_repo"
        ), patch("prepare_foundation_models._download_file") as download_file:
            download_model(
                "radio_v25_b", Path(tmp), token=None,
                hf_endpoint="https://hf-mirror.com",
            )
            url = download_file.call_args.args[0]
            self.assertEqual(
                url,
                "https://hf-mirror.com/nvidia/RADIO/resolve/"
                f"{RADIO_HF_REVISION}/radio-v2.5-b_half.pth.tar?download=true",
            )

    def test_hf_status_ignores_transport_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = (
                root / "checkpoints" / "foundation" / "hf_models" / "dinov3_b"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            (model_dir / ".sgg_source.json").write_text(
                json.dumps(
                    {
                        "repo_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                        "revision": (
                            "5931719e67bbdb9737e363e781fb0c67687896bc"
                        ),
                        "download_endpoint": "https://hf-mirror.com",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(status_for("dinov3_b", root)["ok"])

    def test_snapshot_download_receives_hf_mirror_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download = Mock()
            huggingface_hub = types.ModuleType("huggingface_hub")
            huggingface_hub.snapshot_download = snapshot_download
            with patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}):
                download_model(
                    "siglip2_b", Path(tmp), token="test-token",
                    hf_endpoint="https://hf-mirror.com/",
                )
            self.assertEqual(
                snapshot_download.call_args.kwargs["endpoint"],
                "https://hf-mirror.com",
            )
            self.assertEqual(snapshot_download.call_args.kwargs["token"], "test-token")

    def test_cradio_snapshot_includes_pinned_custom_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download = Mock()
            huggingface_hub = types.ModuleType("huggingface_hub")
            huggingface_hub.snapshot_download = snapshot_download
            with patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}):
                download_model(
                    "cradio_v4_so400m", Path(tmp), token=None,
                    hf_endpoint="https://hf-mirror.com",
                )
            kwargs = snapshot_download.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], "nvidia/C-RADIOv4-SO400M")
            self.assertEqual(
                kwargs["revision"],
                "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
            )
            self.assertIn("*.py", kwargs["allow_patterns"])

    def test_cradio_status_requires_nested_custom_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = (
                root / "checkpoints" / "foundation" / "hf_models"
                / "cradio_v4_so400m"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            for filename in CRADIO_CODE_FILES:
                if filename != "utils.py":
                    (model_dir / filename).write_text("# test\n", encoding="utf-8")
            status = status_for("cradio_v4_so400m", root)
            self.assertFalse(status["ok"])
            self.assertIn(str(model_dir / "utils.py"), status["missing"])

    def test_hf_backbone_prefers_canonical_local_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = (
                root / "checkpoints" / "foundation" / "hf_models" / "dinov3_b"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            module = PerceptionModule.__new__(PerceptionModule)
            nn.Module.__init__(module)
            mocked = Mock(return_value=nn.Identity())
            transformers = types.ModuleType("transformers")
            transformers.AutoModel = types.SimpleNamespace(from_pretrained=mocked)
            transformers.Siglip2VisionModel = types.SimpleNamespace(
                from_pretrained=Mock(return_value=nn.Identity())
            )
            with patch.dict(
                os.environ, {"SGG_PROJECT_ROOT": str(root)}, clear=False
            ), patch.dict(sys.modules, {"transformers": transformers}):
                module._init_hf_vision("dinov3_b")
            mocked.assert_called_once_with(
                str(model_dir.resolve()), local_files_only=True
            )

    def test_siglip2_fixres_uses_auto_model_vision_tower(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = (
                root / "checkpoints" / "foundation" / "hf_models"
                / "siglip2_b"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "siglip"}), encoding="utf-8"
            )
            vision_model = nn.Identity()
            full_model = types.SimpleNamespace(vision_model=vision_model)
            mocked = Mock(return_value=full_model)
            transformers = types.ModuleType("transformers")
            transformers.AutoModel = types.SimpleNamespace(
                from_pretrained=mocked
            )
            module = PerceptionModule.__new__(PerceptionModule)
            nn.Module.__init__(module)
            with patch.dict(
                os.environ,
                {
                    "SGG_PROJECT_ROOT": str(root),
                    "SGG_SIGLIP2_B_DIR": str(model_dir),
                },
                clear=False,
            ), patch.dict(sys.modules, {"transformers": transformers}):
                result = module._init_hf_vision("siglip2_b")
            self.assertIs(result, vision_model)
            mocked.assert_called_once_with(
                str(model_dir.resolve()), local_files_only=True
            )

    def test_radio_prefers_canonical_offline_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "foundation_repos" / "radio"
            weights = (
                root / "checkpoints" / "foundation" / "radio"
                / "radio-v2.5-b_half.pth.tar"
            )
            repo.mkdir(parents=True)
            weights.parent.mkdir(parents=True)
            (repo / "hubconf.py").write_text("# test\n", encoding="utf-8")
            weights.write_bytes(b"weights")
            module = PerceptionModule.__new__(PerceptionModule)
            nn.Module.__init__(module)
            with patch.dict(
                os.environ, {"SGG_PROJECT_ROOT": str(root)}, clear=False
            ), patch("torch.hub.load", return_value=nn.Identity()) as mocked:
                module._init_radio("radio_v25_b")
            mocked.assert_called_once_with(
                str(repo), "radio_model", source="local",
                version=str(weights.resolve()), progress=True,
                skip_validation=True,
            )

    def test_cradio_prefers_pinned_local_hf_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = (
                root / "checkpoints" / "foundation" / "hf_models"
                / "cradio_v4_so400m"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            module = PerceptionModule.__new__(PerceptionModule)
            nn.Module.__init__(module)
            expected = nn.Identity()
            with patch.dict(
                os.environ,
                {
                    "SGG_PROJECT_ROOT": str(root),
                    "SGG_CRADIO_V4_SO400M_DIR": str(model_dir),
                },
                clear=False,
            ), patch.object(
                module, "_load_local_cradio", return_value=expected
            ) as mocked:
                result = module._init_radio("cradio_v4_so400m")
            self.assertIs(result, expected)
            mocked.assert_called_once_with(model_dir.resolve())

    def test_local_cradio_loader_resolves_second_order_relative_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "cradio"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            for filename in CRADIO_CODE_FILES:
                (model_dir / filename).write_text("# test\n", encoding="utf-8")
            (model_dir / "utils.py").write_text(
                "VALUE = 'relative-import-ok'\n", encoding="utf-8"
            )
            (model_dir / "hf_model.py").write_text(
                "from .utils import VALUE\n"
                "class RADIOConfig:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *args, **kwargs):\n"
                "        return VALUE\n"
                "class RADIOModel:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, *args, **kwargs):\n"
                "        return kwargs['config']\n",
                encoding="utf-8",
            )
            module = PerceptionModule.__new__(PerceptionModule)
            nn.Module.__init__(module)
            result = module._load_local_cradio(model_dir.resolve())
            self.assertEqual(result, "relative-import-ok")

    def test_sam_prefers_local_vision_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = (
                root / "checkpoints" / "foundation" / "hf_models"
                / "sam_vit_b"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            vision_encoder = nn.Identity()
            full_model = types.SimpleNamespace(vision_encoder=vision_encoder)
            mocked = Mock(return_value=full_model)
            transformers = types.ModuleType("transformers")
            transformers.SamModel = types.SimpleNamespace(from_pretrained=mocked)
            module = PerceptionModule.__new__(PerceptionModule)
            nn.Module.__init__(module)
            with patch.dict(
                os.environ,
                {
                    "SGG_PROJECT_ROOT": str(root),
                    "SGG_SAM_VIT_B_DIR": str(model_dir),
                },
                clear=False,
            ), patch.dict(sys.modules, {"transformers": transformers}):
                result = module._init_sam_vision()
            self.assertIs(result, vision_encoder)
            mocked.assert_called_once_with(
                str(model_dir.resolve()), local_files_only=True
            )

    def test_hf_cradio_flat_tokens_become_spatial_map(self):
        class FakeRadio(nn.Module):
            def forward(self, x):
                features = torch.arange(24, dtype=x.dtype).reshape(1, 4, 6)
                return types.SimpleNamespace(features=features)

        module = PerceptionModule.__new__(PerceptionModule)
        nn.Module.__init__(module)
        module.backbone_type = "cradio_v4_so400m"
        module.native_dim = 6
        module.backbone = FakeRadio()
        module._radio_hf_wrapper = True
        output = module._extract_feature_map(torch.zeros(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 6, 2, 2))


if __name__ == "__main__":
    unittest.main()
