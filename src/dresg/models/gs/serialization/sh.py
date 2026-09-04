from __future__ import annotations

import torch

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0
