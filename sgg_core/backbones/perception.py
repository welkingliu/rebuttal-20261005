"""
perception_module.py

SGG 感知底座模块
支持的 Backbone（按论文实验分类）:
  CNN 系列:
    - resnet50      [2048-D 空间图]  torchvision，轻量级基线
    - resnext101    [2048-D 空间图]  torchvision，标准 SGG 重基线
  Transformer 系列:
    - swin_b        [1024-D]         timm Swin-Base
    - swin_l        [1536-D]         timm Swin-Large，更强的层级表征
    - dinov2_b      [768-D  空间图]  facebookresearch/dinov2 ViT-B/14
    - dinov2_l      [1024-D 空间图]  facebookresearch/dinov2 ViT-L/14
    - dinov3_b/l    [768/1024-D]     Hugging Face DINOv3 ViT-B/L/16
  VLM 系列:
    - clip_vitb32   [768-D]          OpenAI CLIP ViT-B/32 patch tokens
    - clip_vitl14   [1024-D]         OpenAI CLIP ViT-L/14 patch tokens
    - siglip2_b     [768-D]          Google SigLIP 2 ViT-B/16
    - sam_vit_b     [256-D]          Meta SAM ViT-B image encoder
  多教师底座:
    - radio_v25_b   [768-D]          NVIDIA RADIOv2.5-B/16
    - cradio_v4_so400m [1152-D]      NVIDIA C-RADIOv4-SO400M

所有 backbone 输出均统一为 [B, output_dim, H_feat, W_feat] 的空间特征图，
供下游 _extract_roi_features() 做 ROI 聚合使用。

输出规范变化说明：
  旧版本：各 backbone 输出格式不一致（部分输出 [B, D]，部分输出 [B, D, H, W]）
  新版本：统一输出 [B, D, H_feat, W_feat]
    - CNN: 保留空间维度，直接输出最后卷积层特征图
    - DINOv2: patch token reshape 为空间图
    - Swin: 取最后 stage 的特征图（timm feature_map 模式）
    - CLIP: 取 patch token reshape 为空间图（类似 DINOv2）
"""

import torch
import torch.nn as nn
import hashlib
import importlib
import os
import math
from pathlib import Path
import sys
import types
from typing import Optional


CRADIO_LOCAL_CODE_FILES = (
    "adaptor_attn.py", "adaptor_base.py", "adaptor_generic.py",
    "adaptor_mlp.py", "adaptor_module_factory.py", "adaptor_registry.py",
    "cls_token.py", "common.py", "dinov2_arch.py", "dual_hybrid_vit.py",
    "enable_cpe_support.py", "enable_damp.py", "enable_spectral_reparam.py",
    "eradio_model.py", "extra_models.py", "extra_timm_models.py",
    "feature_normalizer.py", "forward_intermediates.py", "hf_model.py",
    "input_conditioner.py", "open_clip_adaptor.py", "radio_model.py",
    "siglip2_adaptor.py", "utils.py", "vit_patch_generator.py", "vitdet.py",
)


class PerceptionModule(nn.Module):

    # backbone名称 → (原生特征维度, 是否为空间图输出, 推荐输入尺寸)
    BACKBONE_REGISTRY = {
        #  name            dim    spatial  input_size
        'resnet50':     (2048,  True,   224),
        'resnext101':   (2048,  True,   224),
        'convnext_l':   (1536,  True,   224),
        'swin_b':       (1024,  True,   224),
        'swin_l':       (1536,  True,   224),
        'vit_l16':      (1024,  True,   224),
        'dinov2_b':     (768,   True,   518),   # 518 = 37×14，DINOv2 推荐
        'dinov2_l':     (1024,  True,   518),
        'dinov3_b':     (768,   True,   256),
        'dinov3_l':     (1024,  True,   256),
        'clip_vitb32':  (768,   True,   224),
        'clip_vitl14':  (1024,   True,   224),
        'openclip_vith14': (1280, True, 224),
        'siglip_so400m':   (1152, True, 384),
        'siglip2_b':       (768,  True, 224),
        'sam_vit_b':       (256,  True, 1024),
        'radio_v25_b':     (768,  True, 512),
        'cradio_v4_so400m': (1152, True, 512),
    }

    def __init__(
        self,
        backbone_type: str = 'dinov2_b',
        cache_dir: str = './feat_cache',
        output_dim: int = 768,
        signal_threshold: float = 0.05,
    ):
        super().__init__()
        backbone_type = backbone_type.lower()

        if backbone_type not in self.BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_type}'. "
                f"Available: {list(self.BACKBONE_REGISTRY.keys())}"
            )
        
        
        self.backbone_type = backbone_type
        self.cache_dir = cache_dir
        
        self.signal_threshold = signal_threshold
        os.makedirs(cache_dir, exist_ok=True)

        native_dim, _, _ = self.BACKBONE_REGISTRY[backbone_type]
        
        self.native_dim = native_dim
        self.output_dim = output_dim
        self.pretrained_source = "library_pretrained_weights"

        # 1. 初始化底座（各类型有独立的初始化路径）
        self.backbone = self._init_backbone(backbone_type)

        # 2. 统一投影头：native_dim → output_dim
        #    注意：投影在空间维度的每个位置独立作用 (1×1 卷积等价于 Linear)
        if self.native_dim != self.output_dim:
            # 用 Conv2d(1×1) 而非 Linear，以兼容空间特征图
            self.proj = nn.Conv2d(self.native_dim, self.output_dim, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

        self.native_dim = native_dim

    # ------------------------------------------------------------------
    # Backbone 初始化（每类独立，清晰可扩展）
    # ------------------------------------------------------------------

    def _init_backbone(self, b_type: str) -> nn.Module:
        """
        分派不同 backbone 的初始化，统一返回 nn.Module。
        每类 backbone 的 forward 处理在 _extract_feature_map() 中实现。
        """
        if b_type == 'resnet50':
            return self._init_resnet50()
        elif b_type == 'resnext101':
            return self._init_resnext101()
        elif b_type == 'convnext_l':
            return self._init_timm_features('convnext_l')
        elif b_type in ('swin_b', 'swin_l'):
            return self._init_swin(b_type)
        elif b_type in ('vit_l16', 'openclip_vith14', 'siglip_so400m'):
            return self._init_timm_vit(b_type)
        elif b_type in ('dinov2_b', 'dinov2_l'):
            return self._init_dinov2(b_type)
        elif b_type in ('dinov3_b', 'dinov3_l', 'siglip2_b'):
            return self._init_hf_vision(b_type)
        elif b_type == 'sam_vit_b':
            return self._init_sam_vision()
        elif b_type in ('radio_v25_b', 'cradio_v4_so400m'):
            return self._init_radio(b_type)
        elif b_type in ('clip_vitb32', 'clip_vitl14'):
            return self._init_clip(b_type)
        else:
            raise ValueError(f"No initializer for backbone: {b_type}")

    def _init_resnet50(self) -> nn.Module:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 去掉 avgpool 和 fc，保留 layer4 输出 [B, 2048, H/32, W/32]
        return nn.Sequential(*list(model.children())[:-2])

    def _init_resnext101(self) -> nn.Module:
        from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
        model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
        return nn.Sequential(*list(model.children())[:-2])

    def _init_swin(self, b_type: str) -> nn.Module:
        try:
            import timm
        except ImportError:
            raise ImportError("请安装 timm: pip install timm")
        model_name = {
            'swin_b': 'swin_base_patch4_window7_224',
            'swin_l': 'swin_large_patch4_window7_224',
        }[b_type]
        # features_only=True 模式：返回各 stage 的特征图列表
        # out_indices=[-1] 只取最后一个 stage 输出
        model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=[-1],
        )
        return model

    def _create_timm_model(self, candidates, **kwargs) -> nn.Module:
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        last_error = None
        for model_name in candidates:
            try:
                print(f"[Backbone] Loading timm model: {model_name}")
                return timm.create_model(model_name, pretrained=True, **kwargs)
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Could not load any timm model from {candidates}: {last_error}")

    def _init_timm_features(self, b_type: str) -> nn.Module:
        candidates = {
            'convnext_l': [
                'convnext_large.fb_in22k_ft_in1k',
                'convnext_large_in22k',
                'convnext_large',
            ],
        }[b_type]
        return self._create_timm_model(
            candidates,
            features_only=True,
            out_indices=[-1],
        )

    def _init_timm_vit(self, b_type: str) -> nn.Module:
        candidates = {
            'vit_l16': [
                'vit_large_patch16_224.augreg_in21k_ft_in1k',
                'vit_large_patch16_224',
            ],
            'openclip_vith14': [
                'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
                'vit_huge_patch14_clip_224.laion2b',
                'vit_huge_patch14_224',
            ],
            'siglip_so400m': [
                'vit_so400m_patch14_siglip_384.webli',
                'vit_so400m_patch14_siglip_384',
            ],
        }[b_type]
        return self._create_timm_model(candidates, num_classes=0)

    def _init_dinov2(self, b_type: str) -> nn.Module:
        repo = 'facebookresearch/dinov2'
        model_name = {
            'dinov2_b': 'dinov2_vitb14',
            'dinov2_l': 'dinov2_vitl14',
        }[b_type]
        project_root = Path(
            os.environ.get(
                "SGG_PROJECT_ROOT", Path(__file__).resolve().parents[2]
            )
        ).expanduser()
        torch_home = Path(
            os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")
        ).expanduser()
        repo_candidates = [
            Path(os.environ["SGG_DINOV2_REPO"]).expanduser()
            if os.environ.get("SGG_DINOV2_REPO") else None,
            project_root / "external" / "foundation_repos" / "dinov2",
            torch_home / "hub" / "facebookresearch_dinov2_main",
        ]
        weights_env = {
            "dinov2_b": "SGG_DINOV2_B_WEIGHTS",
            "dinov2_l": "SGG_DINOV2_L_WEIGHTS",
        }[b_type]
        weights_name = {
            "dinov2_b": "dinov2_vitb14_pretrain.pth",
            "dinov2_l": "dinov2_vitl14_pretrain.pth",
        }[b_type]
        weights_candidates = [
            Path(os.environ[weights_env]).expanduser()
            if os.environ.get(weights_env) else None,
            project_root / "checkpoints" / "foundation" / "dinov2" / weights_name,
            torch_home / "hub" / "checkpoints" / weights_name,
        ]
        local_repo = next(
            (
                path for path in repo_candidates
                if path is not None and (path / "hubconf.py").is_file()
            ),
            None,
        )
        local_weights = next(
            (
                path for path in weights_candidates
                if path is not None and path.is_file()
            ),
            None,
        )

        if local_repo is not None:
            kwargs = {"source": "local"}
            if local_weights is None and os.environ.get("SGG_OFFLINE", "0") == "1":
                expected = project_root / "checkpoints" / "foundation" / "dinov2" / weights_name
                raise FileNotFoundError(
                    f"Offline DINOv2 weights are missing: {expected}"
                )
            print(
                f"[Backbone] Loading {model_name} from local repo={local_repo} "
                f"weights={local_weights or 'remote official URL'}"
            )
            self.pretrained_source = (
                f"local_torch_hub:{local_repo.resolve()};"
                f"weights={local_weights.resolve() if local_weights else 'remote'}"
            )
            if local_weights is None:
                return torch.hub.load(str(local_repo), model_name, **kwargs)
            model = torch.hub.load(
                str(local_repo), model_name, pretrained=False, **kwargs
            )
            try:
                state_dict = torch.load(
                    local_weights, map_location="cpu", weights_only=True
                )
            except TypeError:
                state_dict = torch.load(local_weights, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            return model

        if local_weights is not None:
            raise FileNotFoundError(
                "DINOv2 weights exist but the local source repository is missing. "
                f"Extract it to {project_root / 'external' / 'foundation_repos' / 'dinov2'}"
            )
        return torch.hub.load(repo, model_name, skip_validation=True)

    def _init_hf_vision(self, b_type: str) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv3/SigLIP2 require transformers>=4.56"
            ) from exc
        model_name = {
            'dinov3_b': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dinov3_l': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'siglip2_b': 'google/siglip2-base-patch16-224',
        }[b_type]
        project_root = Path(
            os.environ.get(
                "SGG_PROJECT_ROOT", Path(__file__).resolve().parents[2]
            )
        ).expanduser()
        local_name = {
            "dinov3_b": "dinov3_b",
            "dinov3_l": "dinov3_l",
            "siglip2_b": "siglip2_b",
        }[b_type]
        env_name = {
            "dinov3_b": "SGG_DINOV3_B_DIR",
            "dinov3_l": "SGG_DINOV3_L_DIR",
            "siglip2_b": "SGG_SIGLIP2_B_DIR",
        }[b_type]
        local_candidates = [
            Path(os.environ[env_name]).expanduser()
            if os.environ.get(env_name) else None,
            project_root / "checkpoints" / "foundation" / "hf_models" / local_name,
        ]
        local_dir = next(
            (
                path for path in local_candidates
                if path is not None and (path / "config.json").is_file()
            ),
            None,
        )
        if local_dir is None and os.environ.get("SGG_OFFLINE", "0") == "1":
            expected = (
                project_root / "checkpoints" / "foundation" / "hf_models"
                / local_name
            )
            raise FileNotFoundError(
                f"Offline Hugging Face model is missing: {expected}"
            )
        model_ref = str(local_dir.resolve()) if local_dir is not None else model_name
        local_only = local_dir is not None
        if local_dir is not None:
            print(f"[Backbone] Loading {b_type} from local_dir={local_dir}")
            self.pretrained_source = f"local_huggingface:{local_dir.resolve()}"
        model = AutoModel.from_pretrained(
            model_ref, local_files_only=local_only
        )
        if b_type == 'siglip2_b':
            # The fixed-resolution SigLIP2 checkpoint intentionally uses the
            # backwards-compatible SigLIP architecture. Loading it through
            # Siglip2VisionModel (the NaFlex architecture) silently selects an
            # incompatible patch embedding before failing on size mismatches.
            vision_model = getattr(model, "vision_model", None)
            if vision_model is None:
                raise RuntimeError(
                    "SigLIP2 FixRes checkpoint did not expose vision_model"
                )
            return vision_model
        return model

    def _init_sam_vision(self) -> nn.Module:
        try:
            from transformers import SamModel
        except ImportError as exc:
            raise ImportError("SAM ViT-B requires transformers") from exc
        project_root = Path(
            os.environ.get(
                "SGG_PROJECT_ROOT", Path(__file__).resolve().parents[2]
            )
        ).expanduser()
        local_candidates = [
            Path(os.environ["SGG_SAM_VIT_B_DIR"]).expanduser()
            if os.environ.get("SGG_SAM_VIT_B_DIR") else None,
            project_root / "checkpoints" / "foundation" / "hf_models"
            / "sam_vit_b",
        ]
        local_dir = next(
            (
                path for path in local_candidates
                if path is not None and (path / "config.json").is_file()
            ),
            None,
        )
        if local_dir is None and os.environ.get("SGG_OFFLINE", "0") == "1":
            raise FileNotFoundError(
                "Offline SAM ViT-B is missing: "
                f"{project_root / 'checkpoints' / 'foundation' / 'hf_models' / 'sam_vit_b'}"
            )
        model_ref = (
            str(local_dir.resolve()) if local_dir is not None
            else "facebook/sam-vit-base"
        )
        local_only = local_dir is not None
        if local_dir is not None:
            print(f"[Backbone] Loading sam_vit_b from local_dir={local_dir}")
            self.pretrained_source = f"local_huggingface:{local_dir.resolve()}"
        model = SamModel.from_pretrained(
            model_ref, local_files_only=local_only
        )
        return model.vision_encoder

    def _init_radio(self, b_type: str) -> nn.Module:
        version = {
            'radio_v25_b': 'radio_v2.5-b',
            'cradio_v4_so400m': 'c-radio_v4-so400m',
        }[b_type]
        project_root = Path(
            os.environ.get(
                "SGG_PROJECT_ROOT", Path(__file__).resolve().parents[2]
            )
        ).expanduser()
        if b_type == "cradio_v4_so400m":
            local_candidates = [
                Path(os.environ["SGG_CRADIO_V4_SO400M_DIR"]).expanduser()
                if os.environ.get("SGG_CRADIO_V4_SO400M_DIR") else None,
                project_root / "checkpoints" / "foundation" / "hf_models"
                / "cradio_v4_so400m",
            ]
            local_dir = next(
                (
                    path for path in local_candidates
                    if path is not None and (path / "config.json").is_file()
                ),
                None,
            )
            if local_dir is None and os.environ.get("SGG_OFFLINE", "0") == "1":
                raise FileNotFoundError(
                    "Offline C-RADIOv4-SO400M is missing: "
                    f"{project_root / 'checkpoints' / 'foundation' / 'hf_models' / 'cradio_v4_so400m'}"
                )
            model_ref = (
                str(local_dir.resolve()) if local_dir is not None
                else "nvidia/C-RADIOv4-SO400M"
            )
            if local_dir is not None:
                print(
                    "[Backbone] Loading cradio_v4_so400m from "
                    f"local_dir={local_dir}"
                )
                self.pretrained_source = (
                    f"local_huggingface_custom:{local_dir.resolve()}"
                )
                self._radio_hf_wrapper = True
                return self._load_local_cradio(local_dir.resolve())
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "C-RADIOv4 requires transformers and timm"
                ) from exc
            self._radio_hf_wrapper = True
            return AutoModel.from_pretrained(
                model_ref,
                trust_remote_code=True,
                dtype="auto",
            )
        if b_type == "radio_v25_b":
            repo_candidates = [
                Path(os.environ["SGG_RADIO_REPO"]).expanduser()
                if os.environ.get("SGG_RADIO_REPO") else None,
                project_root / "external" / "foundation_repos" / "radio",
            ]
            weight_candidates = [
                Path(os.environ["SGG_RADIO_V25_B_WEIGHTS"]).expanduser()
                if os.environ.get("SGG_RADIO_V25_B_WEIGHTS") else None,
                project_root / "checkpoints" / "foundation" / "radio"
                / "radio-v2.5-b_half.pth.tar",
            ]
            local_repo = next(
                (
                    path for path in repo_candidates
                    if path is not None and (path / "hubconf.py").is_file()
                ),
                None,
            )
            local_weights = next(
                (
                    path for path in weight_candidates
                    if path is not None and path.is_file()
                ),
                None,
            )
            if local_repo is not None and local_weights is not None:
                print(
                    f"[Backbone] Loading RADIOv2.5-B from local repo={local_repo} "
                    f"weights={local_weights}"
                )
                self.pretrained_source = (
                    f"local_torch_hub:{local_repo.resolve()};"
                    f"weights={local_weights.resolve()}"
                )
                return torch.hub.load(
                    str(local_repo), "radio_model", source="local",
                    version=str(local_weights.resolve()), progress=True,
                    skip_validation=True,
                )
            if os.environ.get("SGG_OFFLINE", "0") == "1":
                raise FileNotFoundError(
                    "Offline RADIOv2.5-B requires both "
                    f"{project_root / 'external' / 'foundation_repos' / 'radio'} "
                    "and "
                    f"{project_root / 'checkpoints' / 'foundation' / 'radio' / 'radio-v2.5-b_half.pth.tar'}"
                )
        return torch.hub.load(
            'NVlabs/RADIO', 'radio_model', version=version,
            progress=True, skip_validation=True,
        )

    def _load_local_cradio(self, local_dir: Path) -> nn.Module:
        """Load pinned C-RADIO custom code without the HF module cache.

        Transformers' dynamic-module copier can miss second-order relative
        imports in this repository (notably siglip2_adaptor -> utils). Treating
        the verified snapshot as a local package keeps every relative import
        inside that snapshot and also prevents network access during runs.
        """
        required = ("config.json", "model.safetensors", *CRADIO_LOCAL_CODE_FILES)
        missing = [name for name in required if not (local_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete local C-RADIO snapshot at {local_dir}: {missing}"
            )

        digest = hashlib.sha256(str(local_dir).encode("utf-8")).hexdigest()[:12]
        package_name = f"_sgg_cradio_{digest}"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__file__ = str(local_dir / "__init__.py")
            package.__package__ = package_name
            package.__path__ = [str(local_dir)]
            sys.modules[package_name] = package
        importlib.invalidate_caches()
        module = importlib.import_module(f"{package_name}.hf_model")
        config = module.RADIOConfig.from_pretrained(
            str(local_dir), local_files_only=True
        )
        return module.RADIOModel.from_pretrained(
            str(local_dir), config=config, local_files_only=True, dtype="auto"
        )

    def _init_clip(self, b_type: str) -> nn.Module:
        try:
            import clip
        except ImportError:
            raise ImportError("请安装 OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git")
        model_name = {
            'clip_vitb32': 'ViT-B/32',
            'clip_vitl14': 'ViT-L/14',
        }[b_type]
        # 只取视觉编码器
        model, _ = clip.load(model_name, device='cpu')
        return model.visual

    # ------------------------------------------------------------------
    # 孤立协议
    # ------------------------------------------------------------------

    def freeze(self):
        """
        冻结完整感知模块并固定 BN/LN 统计量。实验只训练 reasoning 层。
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        print(f"[Isolation Active] {self.backbone_type} and projection frozen "
              f"({self.native_dim}→{self.output_dim})")

    # ------------------------------------------------------------------
    # 核心：各 backbone 特征提取，统一输出 [B, native_dim, H_feat, W_feat]
    # ------------------------------------------------------------------

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        根据 backbone 类型，将输入图像 [B, 3, H, W]
        转换为空间特征图 [B, native_dim, H_feat, W_feat]

        统一为空间图输出的原因：
          下游 _extract_roi_features() 需要空间维度来做 ROI 聚合。
          全局池化在这里不做，推迟到 ROI 聚合阶段。
        """
        b_type = self.backbone_type

        # ----- CNN 系列 -----
        if b_type in ('resnet50', 'resnext101'):
            # nn.Sequential 去掉了 avgpool+fc，直接输出 [B, 2048, H/32, W/32]
            return self.backbone(x)

        # ----- ConvNeXt / feature-map timm models -----
        elif b_type in ('convnext_l',):
            feat_list = self.backbone(x)
            feat = feat_list[-1]
            if feat.ndim == 4 and feat.shape[-1] == self.native_dim:
                feat = feat.permute(0, 3, 1, 2)
            return feat.contiguous()

        # ----- Swin 系列 -----
        elif b_type in ('swin_b', 'swin_l'):
            # timm features_only 模式返回 list，取最后一个元素
            # swin_b 输出 [B, 1024, H/32, W/32]，swin_l 输出 [B, 1536, H/32, W/32]
            feat_list = self.backbone(x)
            feat = feat_list[-1]            # [B, C, H, W]（timm ≥0.9 保证是 NCHW）
            if feat.ndim == 4 and feat.shape[-1] == self.native_dim:
                feat = feat.permute(0, 3, 1, 2)
                
            return feat.contiguous() # 确保内存连续，防止后续报错
            # return feat

        # ----- DINOv2 系列 -----
        elif b_type in ('vit_l16', 'openclip_vith14', 'siglip_so400m'):
            return self._timm_vit_feature_map(x)

        elif b_type in ('dinov2_b', 'dinov2_l'):
            # get_intermediate_layers 返回 list，n=1 取最后一层 patch tokens
            # shape: [B, N_patches, D]，N_patches = (H/14) × (W/14)
            patch_tokens = self.backbone.get_intermediate_layers(x, n=1)[0]
            B, N, D = patch_tokens.shape
            grid_h = x.shape[2] // 14
            grid_w = x.shape[3] // 14
            # 安全检查：防止非整除输入尺寸
            if grid_h * grid_w != N:
                # 若不整除，取最近的整数网格（通常因 padding 引起 ±1）
                grid_h = int(math.isqrt(N))
                grid_w = N // grid_h
                patch_tokens = patch_tokens[:, :grid_h * grid_w, :]
            # [B, N, D] → [B, D, grid_h, grid_w]
            feat = patch_tokens.reshape(B, grid_h, grid_w, D).permute(0, 3, 1, 2)
            return feat.contiguous()

        elif b_type in ('dinov3_b', 'dinov3_l', 'siglip2_b'):
            return self._hf_vit_feature_map(x)

        elif b_type == 'sam_vit_b':
            output = self.backbone(x)
            spatial = output.last_hidden_state
            if spatial.ndim != 4:
                raise RuntimeError(
                    f"SAM returned non-spatial features: {tuple(spatial.shape)}"
                )
            return spatial.contiguous()

        elif b_type in ('radio_v25_b', 'cradio_v4_so400m'):
            if getattr(self, "_radio_hf_wrapper", False):
                output = self.backbone(x)
            else:
                output = self.backbone(x, feature_fmt='NCHW')
            if isinstance(output, dict):
                output = output['backbone']
            spatial = (
                output.features if hasattr(output, "features") else output[1]
            )
            if spatial.ndim == 3:
                batch, tokens, channels = spatial.shape
                grid_h = x.shape[2] // 16
                grid_w = x.shape[3] // 16
                if grid_h * grid_w != tokens:
                    grid_h = int(math.isqrt(tokens))
                    grid_w = tokens // grid_h
                    spatial = spatial[:, :grid_h * grid_w]
                spatial = spatial.reshape(
                    batch, grid_h, grid_w, channels
                ).permute(0, 3, 1, 2)
            elif spatial.ndim == 4 and spatial.shape[-1] == self.native_dim:
                spatial = spatial.permute(0, 3, 1, 2)
            if spatial.ndim != 4:
                raise RuntimeError(
                    f"RADIO returned non-spatial features: {tuple(spatial.shape)}"
                )
            return spatial.contiguous()

        # ----- CLIP ViT 系列 -----
        elif b_type in ('clip_vitb32', 'clip_vitl14'):
            # CLIP 视觉编码器默认返回 CLS token [B, D]
            # 我们需要 patch tokens，需要 hook 截取
            patch_tokens = self._clip_extract_patches(x)
            return patch_tokens

        else:
            raise RuntimeError(f"Unhandled backbone in _extract_feature_map: {b_type}")

    def _timm_vit_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert timm ViT-family patch tokens to [B, D, H, W].
        Handles plain ViT, CLIP-pretrained ViT, and SigLIP-style ViT variants.
        """
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            out = out.get('x_norm_patchtokens', out.get('x', next(iter(out.values()))))

        if out.ndim == 4:
            if out.shape[1] != self.native_dim and out.shape[-1] == self.native_dim:
                out = out.permute(0, 3, 1, 2)
            return out.contiguous()

        if out.ndim != 3:
            raise RuntimeError(f"Unexpected timm ViT output shape: {tuple(out.shape)}")

        B, N, D = out.shape
        patch_size = getattr(getattr(self.backbone, 'patch_embed', None), 'patch_size', None)
        if isinstance(patch_size, tuple):
            patch_h, patch_w = patch_size
        elif isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h = patch_w = 14 if ('14' in self.backbone_type or 'siglip' in self.backbone_type) else 16

        grid_h = x.shape[2] // patch_h
        grid_w = x.shape[3] // patch_w
        expected = grid_h * grid_w
        if N == expected + 1:
            out = out[:, 1:, :]
            N -= 1
        elif N > expected:
            out = out[:, -expected:, :]
            N = expected
        elif N != expected:
            grid_h = int(math.isqrt(N))
            grid_w = max(N // grid_h, 1)
            out = out[:, :grid_h * grid_w, :]

        feat = out.reshape(B, grid_h, grid_w, D).permute(0, 3, 1, 2)
        return feat.contiguous()

    def _hf_vit_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=x, return_dict=True)
        tokens = output.last_hidden_state
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Hugging Face vision model returned {tuple(tokens.shape)}"
            )
        patch_size = 16
        grid_h = x.shape[2] // patch_size
        grid_w = x.shape[3] // patch_size
        expected = grid_h * grid_w
        if tokens.size(1) < expected:
            raise RuntimeError(
                f"Expected at least {expected} patch tokens, got {tokens.size(1)}"
            )
        # DINOv3 may prepend class/register tokens; patch tokens are last.
        tokens = tokens[:, -expected:, :]
        return tokens.reshape(
            tokens.size(0), grid_h, grid_w, tokens.size(2)
        ).permute(0, 3, 1, 2).contiguous()

    def _clip_extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        从 CLIP 视觉编码器中提取 patch token 空间特征图
        CLIP 官方接口只暴露 CLS token；通过前向钩子截取最后 Transformer block 前的 patch tokens

        输出: [B, D, grid_h, grid_w]
        """
        visual = self.backbone  # CLIP VisionTransformer

        # CLIP ViT-B/32: patch_size=32, ViT-L/14: patch_size=14
        patch_size = 32 if 'vitb32' in self.backbone_type else 14

        B, C, H, W = x.shape
        grid_h = H // patch_size
        grid_w = W // patch_size

        captured = {}

        def hook_fn(module, input, output):
            # output: [seq_len, B, D] = [1 + N_patches, B, D]（CLIP 使用 seq_first）
            captured['patches'] = output[1:, :, :]  # 去掉 CLS token，[N, B, D]

        # 在最后一个 ResidualAttentionBlock 的输出上挂钩子
        last_block = visual.transformer.resblocks[-1]
        hook = last_block.register_forward_hook(hook_fn)

        with torch.no_grad() if not self.training else torch.enable_grad():
            # CLIP 视觉编码器需要 float16 输入（部分模型），统一 cast 到 float32
            _ = visual(x.float())

        hook.remove()

        patches = captured['patches']           # [N_patches, B, D]
        patches = patches.permute(1, 0, 2)      # [B, N_patches, D]
        D = patches.shape[-1]

        feat = patches.reshape(B, grid_h, grid_w, D).permute(0, 3, 1, 2)
        return feat.contiguous()

    # ------------------------------------------------------------------
    # 结构化抹除（模拟感知偏误）
    # ------------------------------------------------------------------

    def structural_erasure_simulator(self, features: torch.Tensor) -> torch.Tensor:
        """
        对空间特征图做逐元素幅值掩码，模拟稀疏查询下的结构化抹除。
        兼容任意 shape（[B, D, H, W] 或 [N, D]）。
        """
        mask = (features.abs() > self.signal_threshold).float()
        return features * mask

    # ------------------------------------------------------------------
    # 主前向接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        img_ids=None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        输入:  [B, 3, H, W]
        输出:  [B, output_dim, H_feat, W_feat]  空间特征图

        experiment_control._extract_roi_features() 从中切出各物体的 ROI 特征。
        """
        if use_cache and img_ids is not None:
            return self._load_from_cache(img_ids)

        # Step 1: backbone 提取 → [B, native_dim, H_feat, W_feat]
        feat_map = self._extract_feature_map(x)

        # Step 2: 1×1 投影对齐维度 → [B, output_dim, H_feat, W_feat]
        feat_map = self.proj(feat_map)

        # Step 3: 结构化抹除
        feat_map = self.structural_erasure_simulator(feat_map)

        return feat_map

    # ------------------------------------------------------------------
    # 离线缓存
    # ------------------------------------------------------------------

    def _load_from_cache(self, img_ids) -> torch.Tensor:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required only for feature-cache loading. "
                "Install it with `pip install h5py`, or run without use_cache."
            ) from exc
        cache_file = os.path.join(self.cache_dir, f'{self.backbone_type}_feats.h5')
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache not found: {cache_file}. Run cache_all_features() first."
            )
        feats = []
        with h5py.File(cache_file, 'r') as f:
            for img_id in img_ids:
                key = str(img_id)
                if key not in f:
                    raise KeyError(f"img_id={img_id} not in cache.")
                feats.append(torch.from_numpy(f[key][:]))
        return torch.stack(feats)

    def cache_all_features(self, dataloader, device: torch.device):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required only for feature-cache writing. "
                "Install it with `pip install h5py`."
            ) from exc
        print(f"Caching features: {self.backbone_type} ...")
        self.to(device).eval()
        cache_file = os.path.join(self.cache_dir, f'{self.backbone_type}_feats.h5')
        with h5py.File(cache_file, 'w') as f:
            with torch.no_grad():
                for batch in dataloader:
                    imgs    = batch['images'].to(device)
                    img_ids = batch['img_ids']
                    feats   = self.forward(imgs).cpu().numpy()
                    for i, img_id in enumerate(img_ids):
                        key = str(img_id)
                        if key not in f:
                            f.create_dataset(key, data=feats[i])
        print(f"Cache saved: {cache_file}")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @classmethod
    def get_output_dim_for(cls, backbone_type: str) -> int:
        """便捷查询：返回某 backbone 的原生特征维度"""
        if backbone_type not in cls.BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {backbone_type}")
        return cls.BACKBONE_REGISTRY[backbone_type][0]

    @classmethod
    def list_backbones(cls):
        print("\nAvailable backbones:")
        print(f"  {'Name':<15} {'NativeDim':>10} {'InputSize':>10}")
        print("  " + "-" * 38)
        for name, (dim, spatial, size) in cls.BACKBONE_REGISTRY.items():
            print(f"  {name:<15} {dim:>10}  {size:>9}px")
