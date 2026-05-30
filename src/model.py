"""
WildHands network. Two stream model with a global image backbone and a shared
crop backbone applied to each hand, fused via "center+corner_latent" positional encoding
into a flat 2048-dim feature that drives MANO regression for each hand.
"""
import torch
import torch.nn as nn
from src.common.mano import MANOHead
from src.common.renderer import MANORenderer
from src.heads import HandHMR


# ----------------------------- config -----------------------------
_FEAT_DIM = 2048
_N_FREQ_POS_ENC = 4
_CENTER_PE_DIM = 2 * 2 * _N_FREQ_POS_ENC   
_CORNER_PE_DIM = 8 * 2 * _N_FREQ_POS_ENC   
_POS_ENC_DIM = _CENTER_PE_DIM + _CORNER_PE_DIM  
_FEAT_CONV_IN = _FEAT_DIM + _POS_ENC_DIM  
_POSE_FLAT = 16 * 3 * 3
_N_GRASP_CLASSES = 9


class WildHands(nn.Module):
    def __init__(self, global_backbone: nn.Module, hand_backbone: nn.Module, focal_length: float = 1000.0, img_res: int = 224):
        """
        Builds the WildHands model.

        Arguments:
            global_backbone -- Backbone instance applied to the full image (B, 3, 224, 224)
            hand_backbone -- Backbone instance applied to both right and left hand crops (shared weights)
            focal_length -- WildHands' fixed fallback focal in px (default 1000.0)
            img_res -- patch resolution (default 224)
        """
        super().__init__()
        self.global_backbone = global_backbone
        self.hand_backbone = hand_backbone

        self.feature_conv = nn.Sequential(
            nn.Conv2d(_FEAT_CONV_IN, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, _FEAT_DIM),
            nn.ReLU(inplace=True),
        )

        self.head_r = HandHMR(_FEAT_DIM, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(_FEAT_DIM, is_rhand=False, n_iter=3)
        self.mano_r = MANOHead(is_rhand=True, focal_length=focal_length, img_res=img_res)
        self.mano_l = MANOHead(is_rhand=False, focal_length=focal_length, img_res=img_res)

        self.renderer = MANORenderer(img_res=img_res)

        self.grasp_classifier = nn.Sequential(
            nn.Linear(10 + _POSE_FLAT + _FEAT_DIM, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, _N_GRASP_CLASSES),
        )

    @staticmethod
    def _sin_cos_pos_enc(angle: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-frequency sin/cos positional encoding.

        Arguments:
            angle -- (B, c) angle in radians for each axis; c is 2 for center or 8 for corner

        Returns:
            pe -- (B, c * 2 * n_freq) float positional encoding
        """
        bz, c = angle.shape
        freq = 2 ** torch.arange(_N_FREQ_POS_ENC, device=angle.device, dtype=angle.dtype).view(1, -1, 1)
        a = angle.view(bz, 1, c)
        return torch.stack([torch.sin(freq * a), torch.cos(freq * a)], dim=-1).reshape(bz, -1).float()

    def _fuse(self, hand_feat: torch.Tensor, glb_feat: torch.Tensor, center_angle: torch.Tensor, corner_angle: torch.Tensor) -> torch.Tensor:
        """
        Adds the global feature map onto the hand-crop feature map and concatenates
        the broadcast positional encodings along the channel axis.

        Arguments:
            hand_feat -- (B, 2048, 7, 7) hand crop features
            glb_feat -- (B, 2048, 7, 7) global image features
            center_angle -- (B, 2) center angular position of the crop
            corner_angle -- (B, 8) angular positions of the four crop corners

        Returns:
            fused -- (B, 2128, 7, 7) fused tensor ready for feature_conv
        """
        bz, _, h, w = hand_feat.shape
        center_pe = self._sin_cos_pos_enc(center_angle).view(bz, _CENTER_PE_DIM, 1, 1).expand(-1, -1, h, w)
        corner_pe = self._sin_cos_pos_enc(corner_angle).view(bz, _CORNER_PE_DIM, 1, 1).expand(-1, -1, h, w)
        return torch.cat([hand_feat + glb_feat, center_pe, corner_pe], dim=1)

    def forward(self, inputs: dict, meta_info: dict) -> dict:
        """
        Runs the WildHands forward pass.

        Arguments:
            inputs -- dict with keys img (B,3,224,224), r_img (B,3,224,224), l_img (B,3,224,224),
                      r_center_angle (B,2), l_center_angle (B,2), r_corner_angle (B,8), l_corner_angle (B,8)
            meta_info -- dict with key intrinsics (B,3,3)

        Returns:
            output -- dict with keys
                      mano.{cam_t.wp,cam_t,joints3d,vertices,j3d.cam,v3d.cam,j2d.norm,beta,pose}.{r,l}
                      mano.cam_t.wp.init.{r,l}, grasp.{r,l}, render.{r,l}
        """
        K = meta_info["intrinsics"]
        bz = inputs["img"].shape[0]

        glb_feat = self.global_backbone(inputs["img"])
        feat_vec = glb_feat.view(bz, glb_feat.shape[1], -1).sum(dim=2)

        r_feat = self.hand_backbone(inputs["r_img"])
        l_feat = self.hand_backbone(inputs["l_img"])

        r_feat = self._fuse(r_feat, glb_feat, inputs["r_center_angle"], inputs["r_corner_angle"])
        l_feat = self._fuse(l_feat, glb_feat, inputs["l_center_angle"], inputs["l_corner_angle"])
        r_feat = self.feature_conv(r_feat)
        l_feat = self.feature_conv(l_feat)

        hmr_r = self.head_r(r_feat)
        hmr_l = self.head_l(l_feat)

        mano_r = self.mano_r(hmr_r["pose"], hmr_r["shape"], hmr_r["cam_t.wp"], K)
        mano_l = self.mano_l(hmr_l["pose"], hmr_l["shape"], hmr_l["cam_t.wp"], K)

        output = {f"mano.{k}": v for k, v in mano_r.items()}
        output.update({f"mano.{k}": v for k, v in mano_l.items()})
        output["mano.cam_t.wp.init.r"] = hmr_r["cam_t.wp.init"]
        output["mano.cam_t.wp.init.l"] = hmr_l["cam_t.wp.init"]

        grasp_in_r = torch.cat([hmr_r["shape"], hmr_r["pose"].reshape(bz, -1), feat_vec], dim=1)
        grasp_in_l = torch.cat([hmr_l["shape"], hmr_l["pose"].reshape(bz, -1), feat_vec], dim=1)
        output["grasp.r"] = self.grasp_classifier(grasp_in_r)
        output["grasp.l"] = self.grasp_classifier(grasp_in_l)

        output["render.r"] = self.renderer(output["mano.v3d.cam.r"], K=K, is_right=True)
        output["render.l"] = self.renderer(output["mano.v3d.cam.l"], K=K, is_right=False)

        return output
