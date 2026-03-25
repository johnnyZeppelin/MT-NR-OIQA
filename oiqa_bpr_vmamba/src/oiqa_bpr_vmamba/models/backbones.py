from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import importlib.util
import sys
import warnings

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MultiScaleFeatureSpec:
    channels: list[int]


_TIMM_IMPORT_ERROR: Exception | None = None


def _try_import_timm():
    global _TIMM_IMPORT_ERROR
    try:
        import timm  # type: ignore
        return timm
    except Exception as exc:  # pragma: no cover - env-specific
        _TIMM_IMPORT_ERROR = exc
        return None


def _load_vmamba_module(repo_root: str | None = None):
    candidates: list[Path] = []
    if repo_root:
        root = Path(repo_root)
        if root.is_file() and root.name == 'vmamba.py':
            candidates.append(root)
        else:
            candidates.append(root / 'vmamba.py')
    for path in candidates:
        if path.exists():
            try:
                spec = importlib.util.spec_from_file_location('oiqa_external_vmamba', path)
                if spec is None or spec.loader is None:
                    raise ImportError(f'Failed to create module spec for VMamba source: {path}')
                module = importlib.util.module_from_spec(spec)
                sys.modules.setdefault('oiqa_external_vmamba', module)
                spec.loader.exec_module(module)
                return module
            except Exception as exc:
                warnings.warn(f'Failed to import external VMamba source from {path}: {exc}. Falling back to vendored official copy.')

    from oiqa_bpr_vmamba.third_party import vmamba_official
    return vmamba_official


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleHierarchicalCNN(nn.Module):
    """Fallback multi-scale CNN backbone.

    This is intentionally dependency-light so the whole project remains runnable
    even when timm/torchvision/VMamba are unavailable in the environment.
    It returns 3 feature maps, matching the paper's multi-scale fusion design.
    """

    def __init__(self, channels: Sequence[int] = (64, 128, 256, 512), max_input_hw: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.max_input_hw = max_input_hw
        c1, c2, c3, c4 = list(channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = ConvNormAct(c1, c2, stride=2)
        self.stage2 = ConvNormAct(c2, c3, stride=2)
        self.stage3 = ConvNormAct(c3, c4, stride=2)
        self.spec = MultiScaleFeatureSpec(channels=[c2, c3, c4])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.max_input_hw is not None:
            max_h, max_w = self.max_input_hw
            h, w = x.shape[-2:]
            if h > max_h or w > max_w:
                x = F.interpolate(x, size=(max_h, max_w), mode='bilinear', align_corners=False)
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


class TimmFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, out_indices: Sequence[int]) -> None:
        super().__init__()
        timm = _try_import_timm()
        if timm is None:
            raise ImportError('timm import failed')
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        self.spec = MultiScaleFeatureSpec(channels=list(self.model.feature_info.channels()))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.model(x))


class ViTMultiBlockBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        timm = _try_import_timm()
        if timm is None:
            raise ImportError('timm import failed')
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        if not hasattr(self.model, 'blocks') or not hasattr(self.model, 'patch_embed'):
            raise ValueError(f'{model_name} is not a timm ViT model with blocks and patch_embed.')
        embed_dim = getattr(self.model, 'num_features', None) or getattr(self.model, 'embed_dim', None)
        if embed_dim is None:
            raise ValueError(f'Could not infer embed dim from {model_name}.')
        self.spec = MultiScaleFeatureSpec(channels=[embed_dim, embed_dim, embed_dim])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, _, H, W = x.shape
        x = self.model.patch_embed(x)
        if x.dim() == 4:
            Hp, Wp = x.shape[-2], x.shape[-1]
            x = x.flatten(2).transpose(1, 2)
        else:
            patch_size = getattr(self.model.patch_embed, 'patch_size', (16, 16))
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            Hp = max(1, H // patch_size[0])
            Wp = max(1, W // patch_size[1])
        if hasattr(self.model, '_pos_embed'):
            x = self.model._pos_embed(x)
        x = self.model.patch_drop(x) if hasattr(self.model, 'patch_drop') else x
        x = self.model.norm_pre(x) if hasattr(self.model, 'norm_pre') else x

        total_blocks = len(self.model.blocks)
        hook_ids = {max(1, total_blocks // 3) - 1, max(1, 2 * total_blocks // 3) - 1, total_blocks - 1}
        outputs = []
        for idx, blk in enumerate(self.model.blocks):
            x = blk(x)
            if idx in hook_ids:
                feat = x
                if hasattr(self.model, 'num_prefix_tokens') and self.model.num_prefix_tokens > 0:
                    feat = feat[:, self.model.num_prefix_tokens:, :]
                feat = feat.transpose(1, 2).reshape(B, -1, Hp, Wp)
                outputs.append(feat)
        if len(outputs) != 3:
            raise RuntimeError(f'Expected 3 multi-scale outputs from {self.model.__class__.__name__}, got {len(outputs)}.')
        return outputs


_VMAMBA_VARIANTS: dict[str, dict[str, object]] = {
    'vmamba_tiny_s1l8': dict(depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, ssm_ratio=1.0),
    'vmamba_tiny_s2l5': dict(depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2, ssm_ratio=2.0),
    'vmamba_small_s2l15': dict(depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, ssm_ratio=2.0),
    'vmamba_base_s2l15': dict(depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6, ssm_ratio=2.0),
}


class OfficialVMambaBackbone(nn.Module):
    def __init__(
        self,
        variant: str = 'vmamba_tiny_s1l8',
        pretrained_path: str | None = None,
        repo_root: str | None = None,
        out_indices: Sequence[int] = (1, 2, 3),
        channel_first: bool = True,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in _VMAMBA_VARIANTS:
            raise ValueError(f'Unsupported VMamba variant: {variant}. Known: {sorted(_VMAMBA_VARIANTS)}')
        vmamba = _load_vmamba_module(repo_root)
        cfg = dict(_VMAMBA_VARIANTS[variant])
        dims = int(cfg['dims'])
        channels = [dims * (2 ** i) for i in range(4)]
        self.model = vmamba.Backbone_VSSM(
            out_indices=tuple(out_indices),
            pretrained=pretrained_path,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            ssm_d_state=1,
            ssm_ratio=float(cfg['ssm_ratio']),
            ssm_dt_rank='auto',
            ssm_act_layer='silu',
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init='v0',
            forward_type='v05_noz',
            mlp_ratio=4.0,
            mlp_act_layer='gelu',
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            norm_layer=('ln2d' if channel_first else 'ln'),
            downsample_version='v3',
            patchembed_version='v2',
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
            depths=cfg['depths'],
            dims=dims,
            drop_path_rate=float(cfg['drop_path_rate']),
        )
        self.spec = MultiScaleFeatureSpec(channels=[channels[i] for i in out_indices])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.model(x))


class GlobalBackboneFactory:
    @staticmethod
    def build(
        backbone_type: str,
        backbone_name: str,
        pretrained: bool,
        fallback_name: str | None = None,
        pretrained_path: str | None = None,
        vmamba_repo_root: str | None = None,
    ) -> nn.Module:
        backbone_type = backbone_type.lower()
        if backbone_type == 'simple_cnn':
            return SimpleHierarchicalCNN(channels=(32, 64, 128, 256), max_input_hw=(512, 1024))
        if backbone_type == 'vit':
            try:
                return ViTMultiBlockBackbone(backbone_name, pretrained=pretrained)
            except Exception as exc:
                warnings.warn(f'Falling back from ViT backbone {backbone_name} to simple CNN because: {exc}')
                return SimpleHierarchicalCNN(channels=(32, 64, 128, 256), max_input_hw=(512, 1024))
        if backbone_type == 'vmamba':
            try:
                return OfficialVMambaBackbone(
                    variant=backbone_name,
                    pretrained_path=pretrained_path if pretrained else None,
                    repo_root=vmamba_repo_root,
                    out_indices=(1, 2, 3),
                    channel_first=True,
                )
            except Exception as exc:
                warnings.warn(f'Official VMamba backbone {backbone_name} failed: {exc}. Trying fallback backbones.')
                if fallback_name is not None:
                    try:
                        return TimmFeatureBackbone(fallback_name, pretrained=pretrained, out_indices=(1, 2, 3))
                    except Exception as inner_exc:
                        warnings.warn(f'Fallback backbone {fallback_name} also failed: {inner_exc}. Using simple CNN.')
                return SimpleHierarchicalCNN(channels=(32, 64, 128, 256), max_input_hw=(512, 1024))
        if backbone_type in {'timm_hierarchical', 'hierarchical'}:
            try:
                return TimmFeatureBackbone(backbone_name, pretrained=pretrained, out_indices=(1, 2, 3))
            except Exception as exc:
                if fallback_name is not None:
                    try:
                        warnings.warn(f'Falling back from backbone {backbone_name} to {fallback_name} because: {exc}')
                        return TimmFeatureBackbone(fallback_name, pretrained=pretrained, out_indices=(1, 2, 3))
                    except Exception as inner_exc:
                        warnings.warn(f'Fallback backbone {fallback_name} also failed: {inner_exc}. Using simple CNN.')
                else:
                    warnings.warn(f'Backbone {backbone_name} failed: {exc}. Using simple CNN fallback.')
                return SimpleHierarchicalCNN(channels=(32, 64, 128, 256), max_input_hw=(512, 1024))
        raise ValueError(f'Unsupported global backbone type: {backbone_type}')


class LocalResNetBackbone(nn.Module):
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True) -> None:
        super().__init__()
        try:
            self.backbone = TimmFeatureBackbone(model_name, pretrained=pretrained, out_indices=(2, 3, 4))
            self.spec = self.backbone.spec
        except Exception as exc:
            warnings.warn(f'Local backbone {model_name} failed to initialize; using simple CNN fallback. Reason: {exc}')
            self.backbone = SimpleHierarchicalCNN(channels=(32, 64, 128, 256))
            self.spec = self.backbone.spec

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)
