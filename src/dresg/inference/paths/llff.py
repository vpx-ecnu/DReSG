from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dresg.data.colmap import ColmapScene
from dresg.inference.paths.geometry import average_pose, viewmatrix
from dresg.inference.paths.geometry import normalize as traj_normalize


def load_forward_facing_bounds(source: ColmapScene) -> NDArray[np.float64]:
    """Load finite positive near/far bounds required by spiral trajectories."""
    path = source.scene_dir / "poses_bounds.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Spiral trajectory requires bounds file: {path}")
    payload = np.load(path, allow_pickle=False)
    if payload.ndim != 2 or payload.shape[0] != len(source) or payload.shape[1] < 2:
        raise ValueError(
            "poses_bounds.npy must have one row per canonical view and at least two columns"
        )
    bounds = np.ascontiguousarray(payload[:, -2:], dtype=np.float64)
    if not np.isfinite(bounds).all():
        raise ValueError("poses_bounds.npy must contain only finite values")
    if np.any(bounds <= 0.0) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each pose bound must satisfy 0 < near < far")
    bounds.setflags(write=False)
    return bounds


def apply_coord_mode(poses: np.ndarray, mode: str) -> np.ndarray:
    poses = poses.copy()
    if mode == "flip_yz":
        poses[:, :3, 1:3] *= -1
    elif mode == "flip_y":
        poses[:, :3, 1] *= -1
    elif mode == "flip_z":
        poses[:, :3, 2] *= -1
    elif mode != "none":
        raise ValueError(f"Unsupported coord mode: {mode}")
    return poses


def normalized_bounds(source: ColmapScene) -> np.ndarray:
    return load_forward_facing_bounds(source).astype(np.float32).T


def render_path_spiral(
    c2w: np.ndarray,
    up: np.ndarray,
    rads: np.ndarray,
    focal: float,
    *,
    zrate: float,
    rots: int,
    n_frames: int,
) -> np.ndarray:
    render_poses = []
    rads_h = np.array(list(rads) + [1.0])
    for theta in np.linspace(0.0, 2.0 * np.pi * rots, n_frames + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
            )
            * rads_h,
        )
        z = traj_normalize(
            c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0]))
        )
        render_poses.append(viewmatrix(z, up, c))
    return np.stack(render_poses)


def build_llff_spiral(
    source: ColmapScene,
    camtoworlds: np.ndarray,
    *,
    coord_mode: str,
    centered_radius: bool,
    radius_scale: float,
    n_frames: int,
) -> np.ndarray:
    poses_llff = apply_coord_mode(camtoworlds, coord_mode)
    c2w = average_pose(poses_llff)
    up = traj_normalize(poses_llff[:, :3, 1].sum(0))
    bds = normalized_bounds(source)
    close_depth = bds.min() * 0.9
    inf_depth = bds.max() * 5.0
    dt = 0.75
    focal = 1.0 / (((1.0 - dt) / close_depth) + (dt / inf_depth))
    centers = poses_llff[:, :3, 3]
    if centered_radius:
        centers = centers - c2w[:3, 3]
    rads = np.percentile(np.abs(centers), 90, 0) * radius_scale
    poses = render_path_spiral(
        c2w,
        up,
        rads,
        focal,
        zrate=0.5,
        rots=2,
        n_frames=n_frames,
    )
    return apply_coord_mode(poses, coord_mode)
