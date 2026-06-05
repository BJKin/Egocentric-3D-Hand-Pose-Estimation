"""
Image backbones, all sourced from timm (Hugging Face).
"""
from pathlib import Path
import timm
import torch
import torch.nn as nn
from loguru import logger

_TIMM_MODELS = {
    "resnet50": "resnet50.tv2_in1k",
    "resnet50-arctic": "resnet50.tv2_in1k",
    "resnet101": "resnet101.tv2_in1k",
    "mobilenet_v3_l": "mobilenetv3_large_100.miil_in21k_ft_in1k",
    "convnext_l": "convnext_large.fb_in22k_ft_in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
    "swinv2_tiny": "swinv2_cr_tiny_ns_224.sw_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
}

ARCTIC_CKPT = Path("../data_reduced/arctic/arctic_sf_allocentric/last.ckpt")
WAVEVIT_CKPT = Path("../data_reduced/wavevit/wavevit_s.pth")

_FEAT_DIM = 2048


class TimmBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, out_channels: int = 2048):
        super().__init__()

        if name not in _TIMM_MODELS:
            raise ValueError(f"Unknown backbone: {name}. Available: {list(_TIMM_MODELS.keys())}")

        self.name = name
        self.timm_id = _TIMM_MODELS[name]
        self.out_channels = out_channels

        self.body = timm.create_model(
            self.timm_id,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        native_channels = self.body.num_features

        if native_channels == out_channels:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.Conv2d(native_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def _to_nchw(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 4:
            native_channels = self.body.num_features

            # Standard CNN output: B, C, H, W
            if feat.shape[1] == native_channels:
                return feat

            # Swin / some transformer output: B, H, W, C
            if feat.shape[-1] == native_channels:
                return feat.permute(0, 3, 1, 2).contiguous()

            return feat

        if feat.ndim == 3:
            # ViT-like output: B, N, C
            b, n, c = feat.shape
            h = int(n ** 0.5)
            w = h

            if h * w != n:
                raise ValueError(f"Cannot reshape token sequence with N={n} into square feature map.")

            return feat.transpose(1, 2).reshape(b, c, h, w).contiguous()

        raise ValueError(f"Unexpected feature shape from backbone {self.name}: {tuple(feat.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body.forward_features(x)

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        feat = self._to_nchw(feat)
        feat = self.adapter(feat)

        return feat


class WaveViTBackbone(nn.Module):
    def __init__(self, pretrained: bool):
        """
        Wraps Wave-ViT-S (vendored, see src/wavevit.py) as a 2048 channel feature extractor.

        Arguments:
            pretrained -- if True, loads the Wave-ViT-S ImageNet weights from WAVEVIT_CKPT
        """
        super().__init__()
        from src.wavevit import wavevit_s  # lazy import: pywt is only needed for this backbone
        self.body = wavevit_s(pretrained=False)
        if pretrained:
            sd = torch.load(WAVEVIT_CKPT, map_location="cpu")
            sd = sd.get("model", sd.get("state_dict", sd))
            sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
            missing, unexpected = self.body.load_state_dict(sd, strict=False)
            loaded = len(self.body.state_dict()) - len(missing)
            logger.info(f"loaded {loaded} Wave-ViT-S keys from {WAVEVIT_CKPT} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")
            assert loaded > 0, f"Wave-ViT-S checkpoint loaded 0 keys (key mismatch): {WAVEVIT_CKPT}"
        native = 448  # wave_vit_s embed_dims[-1]; WaveViT does not expose embed_dims as an attribute
        if native == _FEAT_DIM:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.Conv2d(native, _FEAT_DIM, kernel_size=1, bias=False),
                nn.BatchNorm2d(_FEAT_DIM),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts a 2048-channel feature map from an image batch.

        Replicates Wave-ViT's per-stage loop but keeps the last stage's spatial
        feature map (upstream forward_features pools it into a class token instead).

        Arguments:
            x -- (B, 3, 224, 224) input image batch

        Returns:
            feat -- (B, 2048, 7, 7) feature map
        """
        body, bsz = self.body, x.shape[0]
        for i in range(body.num_stages):
            x, H, W = getattr(body, f"patch_embed{i + 1}")(x)
            for blk in getattr(body, f"block{i + 1}"):
                x = blk(x, H, W)
            x = getattr(body, f"norm{i + 1}")(x)
            x = x.reshape(bsz, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return self.adapter(x)

def _load_arctic_backbone(backbone: "TimmBackbone") -> None:
    """
    Initializes a resnet50 backbone in place from the ARCTIC checkpoint.

    Arguments:
        backbone -- a resnet50 TimmBackbone to load the ARCTIC weights into
    """
    sd = torch.load(ARCTIC_CKPT, map_location="cpu")["state_dict"]
    remapped = {f"body.{k[len('model.backbone.'):]}": v
                for k, v in sd.items() if k.startswith("model.backbone.")}
    backbone.load_state_dict(remapped, strict=False)
    logger.info(f"loaded {len(remapped)} ARCTIC backbone keys from {ARCTIC_CKPT}")

def build_backbone(name: str, pretrained: bool = True) -> nn.Module:

    """
    Builds a WildHands image backbone.

    Arguments:
        name -- one of the keys in _TIMM_MODELS
        pretrained -- if True, loads the configured init weights (ImageNet, or the
                      ARCTIC checkpoint for "resnet50-arctic")

    Returns:
        backbone -- nn.Module mapping (B, 3, 224, 224) -> (B, 2048, 7, 7)
    """
    if name == "wave_vit_s":
        return WaveViTBackbone(pretrained=pretrained)
    if name == "resnet50-arctic" and pretrained:
        backbone = TimmBackbone(name, pretrained=False)
        _load_arctic_backbone(backbone)
        return backbone
    return TimmBackbone(name, pretrained)

