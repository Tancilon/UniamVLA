"""Shared drawing utilities for UamVLA training visualization.

Provides image tools, 3D projection helpers, and PIL-based drawing
primitives for bounding boxes and coordinate axes.

Camera convention: MuJoCo OpenGL.
    cam_xmat columns are camera-to-world rotation (R).
    p_cam = R^T @ (p_world - t)
    u = fx * x / (-z) + cx
    v = fy * (-y) / (-z) + cy
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Image tools
# ---------------------------------------------------------------------------

def tensor_to_pil(
    tensor: torch.Tensor,
    mean: float = 0.5,
    std: float = 0.5,
) -> Image.Image:
    """Convert a (C, H, W) normalized tensor to a PIL Image.

    Denormalizes: pixel = clamp(tensor * std + mean, 0, 1) * 255.

    Args:
        tensor: (C, H, W) float tensor.
        mean: Normalization mean used during preprocessing.
        std: Normalization std used during preprocessing.

    Returns:
        RGB PIL Image.

    Raises:
        ValueError: If tensor is not 3-dimensional.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"tensor_to_pil expects a 3-D (C, H, W) tensor, got {tensor.ndim}-D"
        )
    # Denormalize
    img_t = tensor.float() * std + mean
    img_t = img_t.clamp(0.0, 1.0)
    # (C, H, W) → (H, W, C), scale to uint8
    arr = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def concat_images_h(images: list[Image.Image]) -> Image.Image:
    """Horizontally concatenate PIL images.

    Shorter images are resized (maintaining aspect ratio) to match the
    maximum height.  A single image returns a copy.

    Args:
        images: Non-empty list of PIL Images.

    Returns:
        New PIL Image with all inputs concatenated horizontally.
    """
    if len(images) == 1:
        return images[0].copy()

    max_h = max(img.height for img in images)
    resized = []
    for img in images:
        if img.height != max_h:
            new_w = int(img.width * max_h / img.height)
            img = img.resize((new_w, max_h), Image.LANCZOS)
        resized.append(img)

    total_w = sum(img.width for img in resized)
    canvas = Image.new("RGB", (total_w, max_h))
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


# ---------------------------------------------------------------------------
# 3D projection (MuJoCo OpenGL convention)
# ---------------------------------------------------------------------------

def world_to_pixel(
    points: np.ndarray,
    intrinsic: dict,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) pixel coordinates.

    Convention:
        p_cam = R^T @ (p_world - t)
        u = fx * x / (-z) + cx
        v = fy * (-y) / (-z) + cy
    Points behind the camera (neg_z <= 0) receive NaN.

    Args:
        points: (N, 3) world-frame points.
        intrinsic: Dict with keys fx, fy, cx, cy.
        cam_R: (3, 3) camera-to-world rotation matrix.
        cam_t: (3,) camera position in world frame.

    Returns:
        (N, 2) pixel coordinates [u, v].
    """
    p_cam = (cam_R.T @ (points - cam_t).T).T  # (N, 3)

    fx, fy = intrinsic["fx"], intrinsic["fy"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]

    neg_z = -p_cam[:, 2]
    behind = neg_z <= 0

    u = np.where(behind, np.nan, fx * p_cam[:, 0] / neg_z + cx)
    v = np.where(behind, np.nan, fy * (-p_cam[:, 1]) / neg_z + cy)

    return np.stack([u, v], axis=-1)


def compute_obb_corners(
    pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Compute 8 OBB corners in world frame from point cloud + pose.

    Args:
        pts: (N, 3) point cloud in world frame.
        R: (3, 3) object rotation (local-to-world).
        t: (3,) object translation (center in world).

    Returns:
        (8, 3) corner vertices in world frame.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        p_local = (R.T @ (pts - t).T).T

    lo = p_local.min(axis=0)
    hi = p_local.max(axis=0)

    corners_local = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ])

    return (R @ corners_local.T).T + t


# ---------------------------------------------------------------------------
# Colorful gradient bbox (cutoop-style)
# ---------------------------------------------------------------------------

# Vertex colors derived from ±1 → RGB mapping.
# Corner index i encodes sign pattern: bit0=x, bit1=y, bit2=z.
# +1 → 255, -1 → 0 for each channel.
# Order matches corners_local in compute_obb_corners:
#   0: (lo,lo,lo)  1: (hi,lo,lo)  2: (lo,hi,lo)  3: (hi,hi,lo)
#   4: (lo,lo,hi)  5: (hi,lo,hi)  6: (lo,hi,hi)  7: (hi,hi,hi)
# Maps: lo=-1→0, hi=+1→255
BBOX_VERTEX_COLORS: np.ndarray = np.array([
    [  0,   0,   0],  # 0: (-,-,-) black
    [255,   0,   0],  # 1: (+,-,-) red
    [  0, 255,   0],  # 2: (-,+,-) green
    [255, 255,   0],  # 3: (+,+,-) yellow
    [  0,   0, 255],  # 4: (-,-,+) blue
    [255,   0, 255],  # 5: (+,-,+) magenta
    [  0, 255, 255],  # 6: (-,+,+) cyan
    [255, 255, 255],  # 7: (+,+,+) white
], dtype=np.uint8)

# 12 edges of a box (index pairs)
BBOX_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]


def draw_gradient_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color0: tuple[int, int, int],
    color1: tuple[int, int, int],
    width: int = 2,
    steps: int = 10,
) -> None:
    """Draw a line as gradient-interpolated segments.

    Args:
        draw: PIL ImageDraw object.
        p0: Start point (x, y).
        p1: End point (x, y).
        color0: RGB color at p0.
        color1: RGB color at p1.
        width: Line width in pixels.
        steps: Number of gradient segments.
    """
    x0, y0 = p0
    x1, y1 = p1
    c0 = np.array(color0, dtype=float)
    c1 = np.array(color1, dtype=float)

    for i in range(steps):
        t_start = i / steps
        t_end = (i + 1) / steps

        sx = x0 + (x1 - x0) * t_start
        sy = y0 + (y1 - y0) * t_start
        ex = x0 + (x1 - x0) * t_end
        ey = y0 + (y1 - y0) * t_end

        seg_color = tuple(
            int(np.clip((c0 * (1 - t_start) + c1 * t_start)[k], 0, 255))
            for k in range(3)
        )
        draw.line([(sx, sy), (ex, ey)], fill=seg_color, width=width)


def draw_bbox_colorful(
    draw: ImageDraw.ImageDraw,
    box_pixels: np.ndarray,
    width: int = 2,
    halo: bool = False,
) -> None:
    """Draw 12 OBB edges with per-vertex gradient colors.

    Edges whose endpoints contain NaN are silently skipped. When ``halo``
    is True, draws solid white lines at the given width instead of
    gradient colors — used for a contrast outline behind colored edges.

    Args:
        draw: PIL ImageDraw object.
        box_pixels: (8, 2) pixel coordinates for box corners.
        width: Line width in pixels.
        halo: If True, draw solid white lines for contrast.
    """
    for i0, i1 in BBOX_EDGES:
        p0 = box_pixels[i0]
        p1 = box_pixels[i1]
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
            continue
        if halo:
            draw.line(
                [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
                fill=(255, 255, 255),
                width=width,
            )
            continue
        c0 = tuple(int(v) for v in BBOX_VERTEX_COLORS[i0])
        c1 = tuple(int(v) for v in BBOX_VERTEX_COLORS[i1])
        draw_gradient_line(
            draw,
            (float(p0[0]), float(p0[1])),
            (float(p1[0]), float(p1[1])),
            c0,
            c1,
            width=width,
        )


# ---------------------------------------------------------------------------
# Rotation axes drawing
# ---------------------------------------------------------------------------

def draw_axes_3d(
    draw: ImageDraw.ImageDraw,
    axes_pixels: np.ndarray,
    width: int = 3,
    tip_radius: int = 4,
) -> None:
    """Draw 3D coordinate axes: X=Red, Y=Green, Z=Blue.

    Args:
        draw: PIL ImageDraw object.
        axes_pixels: (4, 2) array — [origin, x_tip, y_tip, z_tip] in pixels.
        width: Line width in pixels.
        tip_radius: Radius of filled circle drawn at each axis tip.
    """
    origin = axes_pixels[0]
    tips = axes_pixels[1:]  # (3, 2): x, y, z tips
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # R, G, B

    if np.any(np.isnan(origin)):
        return

    ox, oy = float(origin[0]), float(origin[1])

    for tip, color in zip(tips, colors):
        if np.any(np.isnan(tip)):
            continue
        tx, ty = float(tip[0]), float(tip[1])
        draw.line([(ox, oy), (tx, ty)], fill=color, width=width)
        r = tip_radius
        draw.ellipse(
            [tx - r, ty - r, tx + r, ty + r],
            fill=color,
        )
