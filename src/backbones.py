"""
Image backbones, all sourced from timm (Hugging Face).
"""
from pathlib import Path
import timm
import torch
import torch.nn as nn
from loguru import logger


# ----------------------------- config -----------------------------
_TIMM_MODELS = {
    "resnet50":        "resnet50.tv2_in1k",
    "resnet50-arctic": "resnet50.tv2_in1k",
    "resnet101":       "resnet101.tv2_in1k",
    "mobilenet_v3_l":  "mobilenetv3_large_100.miil_in21k_ft_in1k",
    "convnext_l":      "convnext_large.fb_in22k_ft_in1k",
    "mobilevit_s":     "mobilevit_s.cvnets_in1k",
}

ARCTIC_CKPT = Path("../data_reduced/arctic/arctic_sf_allocentric/last.ckpt")
WAVEVIT_CKPT = Path("../data_reduced/wavevit/wavevit_s.pth")

_FEAT_DIM = 2048


class TimmBackbone(nn.Module):
    def __init__(self, timm_id: str, pretrained: bool):
        """
        Wraps a timm classification model as a 2048 channel feature extractor.

        Arguments:
            timm_id -- timm model id (with weight tag) to instantiate
            pretrained -- if True, loads the tag's ImageNet weights from the HF Hub
        """
        super().__init__()
        self.body = timm.create_model(
            timm_id, pretrained=pretrained, num_classes=0, global_pool=""
        )
        native = self.body.num_features
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

        Arguments:
            x -- (B, 3, 224, 224) input image batch

        Returns:
            feat -- (B, 2048, 7, 7) feature map
        """
        return self.adapter(self.body.forward_features(x))


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
        backbone = TimmBackbone(_TIMM_MODELS[name], pretrained=False)
        _load_arctic_backbone(backbone)
        return backbone
    return TimmBackbone(_TIMM_MODELS[name], pretrained)
