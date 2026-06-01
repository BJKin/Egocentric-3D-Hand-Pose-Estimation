import torch
import torch.nn as nn
import timm


_TIMM_MODELS = {
    "resnet50": "resnet50.tv2_in1k",
    "resnet50-arctic": "resnet50.tv2_in1k",
    "resnet101": "resnet101.tv2_in1k",
    "mobilenet_v3_l": "mobilenetv3_large_100.miil_in21k_ft_in1k",
    "convnext_l": "convnext_large.fb_in22k_ft_in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224.ms_in1k",
}


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


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    return TimmBackbone(name=name, pretrained=pretrained)