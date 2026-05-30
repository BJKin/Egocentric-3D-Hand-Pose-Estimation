"""
Hand-pose visualization.
"""
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Subset
from pytorch3d.renderer import (
    BlendParams, MeshRasterizer, MeshRenderer, PerspectiveCameras, PointLights,
    RasterizationSettings, SoftPhongShader, look_at_view_transform,
)
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from src.backbones import build_backbone
from src.common.data_utils import IMG_NORM_MEAN, IMG_NORM_STD
from src.common.mano import build_mano_aa
from src.datasets.arctic import ArcticDataset
from src.datasets.assembly import AssemblyDataset
from src.datasets.ego_exo import EgoExo4DDataset
from src.datasets.epic_handkps import EPICHandKpsDataset
from src.datasets.h2o import H2ODataset
from src.model import WildHands
from src.common.process import process_data


# ----------------------------- config -----------------------------
IMG_RES = 224
COLOR_R = (100 / 255, 100 / 255, 254 / 255)  # blue
COLOR_L = (183 / 255, 100 / 255, 254 / 255)  # purple
BACKGROUND_GREY = (0.93, 0.93, 0.93)

HAND_AZIM = 0.0
MESH_FOCAL = 3.0
PANEL_GAP = 6      # px between one row
ROW_GAP = 8        # px between sample rows
LABEL_W = 36       # px for the left column holding each row's vertical dataset label
HEADER_H = 28      # px for the top row holding each column's header label
COL_TITLES = ("input", "overlay", "left hand", "right hand")  # one per grid column
GROUP_W = 34       # px for the outer column
GROUP_GAP = 16     # px gap between the group column and the row dataset labels
BORDER_COLOR = (110, 110, 110)  # outline for the label cells
BORDER_W = 1       # label cell outline thickness in px
MARGIN = 6         # px of white padding around the whole figure
IN_DIST_DATASETS = {"arctic", "assembly"}  # rows from these are in distribution, the rest are zero-shot

DATASETS = {
    "arctic":       lambda: ArcticDataset(split="val"),
    "assembly":     lambda: AssemblyDataset(split="val"),
    "h2o":          lambda: H2ODataset(split="val"),
    "epic_handkps": lambda: EPICHandKpsDataset(split="test"),
    "egoexo":       lambda: EgoExo4DDataset(split="val"),
}

# possible backbones: resnet50, resnet50-arctic, resnet101, mobilenet_v3_l, convnext_l, mobilevit_s
BACKBONE = "resnet101"
CKPT = Path("../logs/resnet101/0529-1544_bs32_lr1e-05_ep100_seed1/checkpoints/best.ckpt")
SOURCE = "pred"               # "pred" (model output) or "gt" (ground truth mesh)
SHUFFLE = False               # pick a random sample per dataset instead of the first
SEED = 0
OUTPUT_DIR = f"../vis_out/{BACKBONE}/{datetime.now():%m%d-%H%M}"


def load_checkpoint(model: WildHands, ckpt_path: Path) -> None:
    """
    Loads a trained WildHands checkpoint into the model.

    Arguments:
        model -- the WildHands model to mutate
        ckpt_path -- path to the .ckpt written by scripts/train.py
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Inverts the ImageNet normalization applied by the datasets, returning an
    HxWx3 float image in [0, 1].

    Arguments:
        img_tensor -- (3, H, W) torch tensor

    Returns:
        img -- (H, W, 3) float numpy in [0, 1]
    """
    mean = torch.tensor(IMG_NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(IMG_NORM_STD).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def build_renderer(device: torch.device, img_res: int = IMG_RES) -> MeshRenderer:
    """
    Builds a single MeshRenderer with a SoftPhongShader.

    Arguments:
        device -- target device
        img_res -- output image resolution

    Returns:
        renderer -- pytorch3d MeshRenderer
    """
    raster = RasterizationSettings(image_size=img_res, blur_radius=0.0, faces_per_pixel=1)
    placeholder_lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
    shader = SoftPhongShader(device=device, lights=placeholder_lights, blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)))
    return MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster), shader=shader)


def _set_headlight(renderer, location, device):
    """
    Rebinds the renderer's shader lights so the diffuse component illuminates
    whatever the camera is looking at.

    Arguments:
        renderer -- pytorch3d MeshRenderer to mutate
        location -- (1, 3) camera position
        device -- target device
    """
    renderer.shader.lights = PointLights(
        device=device, location=location,
        ambient_color=((0.35, 0.35, 0.35),),
        diffuse_color=((0.75, 0.75, 0.75),),
        specular_color=((0.15, 0.15, 0.15),),
    )


def _camera_eye(azim_deg, elev_deg, dist):
    """
    Returns the camera position that look_at_view_transform places
    at the given (azim, elev, dist) around the origin.

    Arguments:
        azim_deg -- azimuth in degrees
        elev_deg -- elevation in degrees
        dist -- distance from origin

    Returns:
        eye -- nested list [[x, y, z]]
    """
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    return [[dist * np.cos(el) * np.sin(az), dist * np.sin(el), dist * np.cos(el) * np.cos(az)]]


def _build_combined_mesh(verts_pieces, faces_pieces, colors, device):
    """
    Concatenates each hand's vertices, faces, and the colors assigned to each
    vertex into a single Meshes object (render both hands in one pass).

    Arguments:
        verts_pieces -- list of (N_i, 3) tensors, one per valid hand
        faces_pieces -- list of (F_i, 3) tensors, MANO faces for each hand
        colors -- list of (R, G, B) tuples matching the hands
        device -- target device

    Returns:
        mesh -- pytorch3d Meshes with concatenated geometry and texture per vertex
    """
    verts_list, faces_list, tex_list = [], [], []
    v_offset = 0
    for verts, faces, color in zip(verts_pieces, faces_pieces, colors):
        verts_list.append(verts.to(device))
        faces_list.append(faces.to(device) + v_offset)
        tex_list.append(torch.tensor(color, device=device).expand(verts.shape[0], 3))
        v_offset += verts.shape[0]
    verts_cat = torch.cat(verts_list, dim=0).unsqueeze(0)
    faces_cat = torch.cat(faces_list, dim=0).unsqueeze(0)
    tex_cat = torch.cat(tex_list, dim=0).unsqueeze(0)
    return Meshes(verts=verts_cat, faces=faces_cat, textures=TexturesVertex(verts_features=tex_cat))


def _opencv_verts_to_pytorch3d(verts: torch.Tensor) -> torch.Tensor:
    """
    Converts vertices from OpenCV style camera frame to PyTorch3D-style.

    Arguments:
        verts -- (N, 3) tensor in OpenCV camera coords

    Returns:
        verts -- (N, 3) tensor in PyTorch3D camera coords
    """
    return verts * torch.tensor([-1.0, -1.0, 1.0], device=verts.device)


def render_overlay(renderer, verts_pieces, faces_pieces, colors, K, background, device, img_res=IMG_RES):
    """
    Renders the combined mesh into using the GT intrinsics, then
    alpha-blends with the input image.

    Arguments:
        renderer -- pytorch3d MeshRenderer
        verts_pieces -- list of (N_i, 3) tensors in OpenCV camera coords
        faces_pieces -- list of (F_i, 3) MANO face tensors
        colors -- list of (R, G, B) tuples
        K -- (3, 3) intrinsics
        background -- (H, W, 3) input image in [0, 1]
        device -- target device
        img_res -- patch resolution

    Returns:
        rgb -- (img_res, img_res, 3) numpy uint8
    """
    if not verts_pieces:
        return (background * 255).astype(np.uint8)
    flipped = [_opencv_verts_to_pytorch3d(v) for v in verts_pieces]
    mesh = _build_combined_mesh(flipped, faces_pieces, colors, device)

    intrx_to_ndc = torch.tensor([
        [2 / img_res, 0.0, -1.0],
        [0.0, 2 / img_res, -1.0],
        [0.0, 0.0, 1.0],
    ], device=device)
    K_ndc = (intrx_to_ndc @ K.to(device)).unsqueeze(0)
    focal = torch.diagonal(K_ndc[0], 0)[:2].unsqueeze(0)
    principal = K_ndc[0, :2, 2].unsqueeze(0)
    cameras = PerspectiveCameras(focal_length=focal, principal_point=principal, device=device)

    _set_headlight(renderer, [[0.0, 0.0, 0.0]], device)
    fragments = renderer.rasterizer(mesh, cameras=cameras)
    rgba = renderer.shader(fragments, mesh, cameras=cameras)[0].cpu().numpy()
    alpha = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy().astype(np.float32)[..., None]
    composite = rgba[..., :3] * alpha + background * (1 - alpha)
    return (np.clip(composite, 0, 1) * 255).astype(np.uint8)


def _blank_panel(img_res=IMG_RES) -> np.ndarray:
    """
    Returns a solid grey panel for missing hand cells.

    Arguments:
        img_res -- panel resolution

    Returns:
        a solid grey panel for missing hand cells.
    """
    return (np.ones((img_res, img_res, 3), dtype=np.float32) * np.asarray(BACKGROUND_GREY) * 255).astype(np.uint8)


def render_single_hand(renderer, verts, faces, color, azim, device, mesh_focal=MESH_FOCAL, img_res=IMG_RES) -> np.ndarray:
    """
    Renders one hand mesh alone, centered at its centroid and autoframed so the
    mesh fills most of the panel.

    Arguments:
        renderer -- pytorch3d MeshRenderer
        verts -- (N, 3) tensor in OpenCV camera coords, or None
        faces -- (F, 3) MANO face tensor
        color -- (R, G, B) tuple in [0, 1] 
        azim -- camera azimuth in degrees
        device -- target device
        mesh_focal -- NDC focal length
        img_res -- panel resolution

    Returns:
        rgb -- (img_res, img_res, 3) uint8 numpy
    """
    if verts is None:
        return _blank_panel(img_res)

    flipped = _opencv_verts_to_pytorch3d(verts)
    centroid = flipped.mean(dim=0)
    extent = (flipped - centroid).norm(dim=1).max().item()
    dist = max(extent * 3.0, 0.05)
    centered = flipped - centroid
    mesh = _build_combined_mesh([centered], [faces], [color], device)

    R, T = look_at_view_transform(dist=dist, elev=0.0, azim=azim, device=device)
    focal = torch.tensor([[mesh_focal]], device=device)
    cameras = PerspectiveCameras(focal_length=focal, R=R, T=T, device=device)

    _set_headlight(renderer, _camera_eye(azim, 0.0, dist), device)
    fragments = renderer.rasterizer(mesh, cameras=cameras)
    rgba = renderer.shader(fragments, mesh, cameras=cameras)[0].cpu().numpy()
    alpha = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy().astype(np.float32)[..., None]
    background = np.broadcast_to(np.asarray(BACKGROUND_GREY, dtype=np.float32), (img_res, img_res, 3))
    composite = rgba[..., :3] * alpha + background * (1 - alpha)
    return (np.clip(composite, 0, 1) * 255).astype(np.uint8)


def _collect_hands(pred: dict, targets: dict, source: str, mano_faces_r: torch.Tensor, mano_faces_l: torch.Tensor):
    """
    Pulls (verts, faces, color) tuples for the valid hand sides from either the
    predictions or the process_data GT targets.

    Arguments:
        pred -- model output dict
        targets -- post-process_data targets dict
        source -- 'pred' or 'gt'
        mano_faces_r, mano_faces_l -- MANO face tensors

    Returns:
        verts_pieces, faces_pieces, colors -- three matched lists
    """
    src = pred if source == "pred" else targets
    verts_pieces, faces_pieces, colors = [], [], []
    if float(targets["right_valid"][0]) > 0:
        verts_pieces.append(src["mano.v3d.cam.r"][0])
        faces_pieces.append(mano_faces_r)
        colors.append(COLOR_R)
    if float(targets["left_valid"][0]) > 0:
        verts_pieces.append(src["mano.v3d.cam.l"][0])
        faces_pieces.append(mano_faces_l)
        colors.append(COLOR_L)
    return verts_pieces, faces_pieces, colors


def _load_font(size: int = 18):
    """
    Loads a TrueType font for the row labels.

    Arguments:
        size -- font size in px

    Returns:
        font -- a PIL ImageFont instance
    """
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _vertical_label(text, font, gap_color):
    """
    Renders given text as a RGB image.

    Arguments:
        text -- label string
        font -- PIL ImageFont to render with
        gap_color -- RGB background color

    Returns:
        img -- PIL.Image of the text rotated 90 degrees
    """
    l, t, right, bottom = font.getbbox(text)
    txt = Image.new("RGB", (right - l, bottom - t), gap_color)
    ImageDraw.Draw(txt).text((-l, -t), text, fill=(0, 0, 0), font=font)
    return txt.rotate(90, expand=True)


def _contiguous_runs(values):
    """
    Splits a sequence into runs of consecutive equal values.

    Arguments:
        values -- sequence of hashable values

    Returns:
        runs -- list of (value, start_idx, end_idx_inclusive) tuples
    """
    runs = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        runs.append((values[i], i, j))
        i = j + 1
    return runs


def _grid_compose(rows, labels=None, col_labels=None, groups=None, panel_gap=PANEL_GAP, row_gap=ROW_GAP, label_w=LABEL_W, header_h=HEADER_H, group_w=GROUP_W, group_gap=GROUP_GAP, margin=MARGIN, gap_color=(255, 255, 255)) -> Image.Image:
    """
    Lays out an Nx4 grid of panels (one row per sample: photo, overlay, left mesh,
    right mesh). A gap separates the group column from the label column.

    Arguments:
        rows -- list of N lists, each containing exactly 4 (H, W, 3) panels
        labels -- list of N row labels drawn vertically in the label column
        col_labels -- list of column headers drawn across the top
        groups -- list of N group labels
        panel_gap -- px between the 4 panels in a row
        row_gap -- px between rows
        label_w -- width in px of the per row label column
        header_h -- height in px of the top header row
        group_w -- width in px of the outer group column
        group_gap -- px gap between the group column and the label column
        margin -- px of white padding around the whole composite
        gap_color -- RGB fill color in the gaps and behind the labels

    Returns:
        composite -- PIL.Image
    """
    h, w = rows[0][0].shape[:2]
    cols = len(rows[0])
    gw = group_w if groups else 0
    lw = label_w if labels else 0
    hh = header_h if col_labels else 0
    gg = group_gap if (groups and labels) else 0
    left = gw + gg + lw  
    grid_w = left + cols * w + (cols - 1) * panel_gap
    grid_h = hh + len(rows) * h + (len(rows) - 1) * row_gap
    out = np.full((grid_h, grid_w, 3), gap_color, dtype=np.uint8)
    for r, row_panels in enumerate(rows):
        y = hh + r * (h + row_gap)
        for c, panel in enumerate(row_panels):
            x = left + c * (w + panel_gap)
            out[y:y + h, x:x + w] = panel

    img = Image.fromarray(out)
    font = _load_font()
    draw = ImageDraw.Draw(img)
    boxes = [] 

    def paste_vertical(text, x0, col_w, y0, span_h):
        txt = _vertical_label(text, font, gap_color)
        img.paste(txt, (x0 + max((col_w - txt.width) // 2, 0),
                        y0 + (span_h - txt.height) // 2))

    if col_labels:
        for c, title in enumerate(col_labels):
            x0 = left + c * (w + panel_gap)
            l, t, right, bottom = font.getbbox(title)
            tw, th = right - l, bottom - t
            draw.text((x0 + (w - tw) // 2 - l, (hh - th) // 2 - t), title, fill=(0, 0, 0), font=font)
            boxes.append((x0, 0, x0 + w, hh))

    if labels:
        for r, label in enumerate(labels):
            y0 = hh + r * (h + row_gap)
            paste_vertical(label, gw + gg, lw, y0, h)
            boxes.append((gw + gg, y0, gw + gg + lw, y0 + h))

    if groups:
        for value, r0, r1 in _contiguous_runs(groups):
            y0 = hh + r0 * (h + row_gap)
            span_h = (r1 - r0) * (h + row_gap) + h
            paste_vertical(value, 0, gw, y0, span_h)
            boxes.append((0, y0, gw, y0 + span_h))

    for x0, y0, x1, y1 in boxes:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=BORDER_COLOR, width=BORDER_W)
    return ImageOps.expand(img, border=margin, fill=gap_color)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = WildHands(build_backbone(BACKBONE, pretrained=False), build_backbone(BACKBONE, pretrained=False)).to(device).eval()
    load_checkpoint(model, CKPT)
    mano_r = build_mano_aa(is_rhand=True).to(device)
    mano_l = build_mano_aa(is_rhand=False).to(device)
    mano_faces_r = torch.from_numpy(mano_r.faces.astype("int64")).to(device)
    mano_faces_l = torch.from_numpy(mano_l.faces.astype("int64")).to(device)

    renderer = build_renderer(device)
    rng = np.random.default_rng(SEED)
    logger.info(f"rendering one {SOURCE} sample per dataset ({len(DATASETS)} datasets) -> {out_dir}")

    rows, labels = [], []
    for name, builder in DATASETS.items():
        ds = builder()
        idx = int(rng.integers(len(ds))) if SHUFFLE else 0
        loader = DataLoader(Subset(ds, [idx]), batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
        inputs, targets, meta_info = next(iter(loader))
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}
        meta_info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in meta_info.items()}

        with torch.no_grad():
            pred = model(inputs, meta_info)
            if SOURCE == "gt":
                targets = process_data(mano_r, mano_l, targets, meta_info["intrinsics"])

        verts_pieces, faces_pieces, colors = _collect_hands(pred, targets, SOURCE, mano_faces_r, mano_faces_l)
        background = denormalize_image(inputs["img"][0])
        K = meta_info["intrinsics"][0]

        src = pred if SOURCE == "pred" else targets
        right_valid = float(targets["right_valid"][0]) > 0
        left_valid = float(targets["left_valid"][0]) > 0
        right_verts = src["mano.v3d.cam.r"][0] if right_valid else None
        left_verts = src["mano.v3d.cam.l"][0] if left_valid else None

        with torch.no_grad():
            photo = (background * 255).astype(np.uint8)
            overlay = render_overlay(renderer, verts_pieces, faces_pieces, colors, K, background, device)
            left_panel = render_single_hand(renderer, left_verts, mano_faces_l, COLOR_L, HAND_AZIM, device, mesh_focal=MESH_FOCAL)
            right_panel = render_single_hand(renderer, right_verts, mano_faces_r, COLOR_R, HAND_AZIM, device, mesh_focal=MESH_FOCAL)
        rows.append([photo, overlay, left_panel, right_panel])
        labels.append(name)
        logger.info(f"  {name}: idx={idx}  (r={int(right_valid)} l={int(left_valid)})")

    groups = ["in distribution" if n in IN_DIST_DATASETS else "out of distribution" for n in labels]
    composite = _grid_compose(rows, labels, COL_TITLES, groups)
    ckpt_tag = f"{CKPT.parent.name}-{CKPT.stem}" if CKPT.parent.name else CKPT.stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"all_{BACKBONE}_{SOURCE}_{ckpt_tag}_{timestamp}.png"
    out_path = out_dir / fname
    composite.save(out_path)
    logger.info(f"saved {out_path}  ({composite.size[0]}x{composite.size[1]} px, {len(rows)} datasets)")


if __name__ == "__main__":
    main()
