from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


def load_pil_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_rgb_chw01(path: Path) -> torch.Tensor:
    return to_tensor(load_pil_rgb(path))


def load_rgb_image(
    path: Path,
    *,
    device: torch.device,
    width: int | None = None,
    height: int | None = None,
) -> torch.Tensor:
    image = load_pil_rgb(path)
    if width is not None and height is not None and image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    return to_tensor(image).to(device=device, dtype=torch.float32)


def preprocess_rgb_tensor(rgb: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.shape[-2:] != (out_h, out_w):
        rgb = F.interpolate(rgb, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return rgb.clamp(0.0, 1.0)


def chw_to_hwc_u8(image: torch.Tensor) -> np.ndarray:
    image = torch.clamp(image, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def save_rgb(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(chw_to_hwc_u8(image)).save(path)
