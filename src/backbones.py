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
    "swinv2_b":        "swinv2_base_window8_256.ms_in1k"
}

ARCTIC_CKPT = Path("../data_reduced/arctic/arctic_sf_allocentric/last.ckpt")

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
        feat = self.body.forward_features(x)
        if feat.ndim == 4 and feat.shape[-1] == self.body.num_features: #Swin / Swin V2 return (B, H, W, C) 
            feat = feat.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W) to match other architectures
        return self.adapter(feat)

        


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
    if name == "resnet50-arctic" and pretrained:
        backbone = TimmBackbone(_TIMM_MODELS[name], pretrained=False)
        _load_arctic_backbone(backbone)
        return backbone
    return TimmBackbone(_TIMM_MODELS[name], pretrained)
