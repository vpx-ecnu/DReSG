from __future__ import annotations

import numpy as np

from dresg.inference.paths.geometry import (
    focus_point,
    homogeneous_poses,
    normalize,
    viewmatrix,
)


def tnt_integrate_weights(w: np.ndarray) -> np.ndarray:
    cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    return np.concatenate([np.zeros(shape), cw, np.ones(shape)], axis=-1)


def tnt_invert_cdf(u: np.ndarray, t: np.ndarray, w_logits: np.ndarray) -> np.ndarray:
    logits = w_logits - np.max(w_logits, axis=-1, keepdims=True)
    w = np.exp(logits)
    w = w / w.sum(axis=-1, keepdims=True)
    cw = tnt_integrate_weights(w)
    return np.interp(u, cw, t)


def tnt_sample(t: np.ndarray, w_logits: np.ndarray, num_samples: int) -> np.ndarray:
    eps = np.finfo(np.float32).eps
    u = np.linspace(0, 1.0 - eps, num_samples)
    u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
    return tnt_invert_cdf(u, t, w_logits)


def tnt_unpad_poses(poses: np.ndarray) -> np.ndarray:
    return poses[..., :3, :4]


def tnt_transform_poses_pca(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = tnt_unpad_poses(transform @ homogeneous_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    scale = np.max(np.abs(poses_recentered[:, :3, 3]))
    if not np.isfinite(scale) or scale <= np.finfo(poses_recentered.dtype).eps:
        raise ValueError("TNT trajectory requires non-degenerate camera positions")
    scale_factor = 1.0 / scale
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform


def build_tnt_ellipse_path(camtoworlds: np.ndarray, n_frames: int, *, ellipse_scale: float = 1.1) -> np.ndarray:
    """Build the TNT ellipse path used by the paper videos."""
    poses = homogeneous_poses(camtoworlds).copy()
    poses[:, :, 1:3] *= -1
    poses, transform = tnt_transform_poses_pca(poses)

    center = focus_point(poses)
    offset = np.array([center[0], center[1], center[2] * 0.0])
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0) * ellipse_scale
    low = -sc + offset
    high = sc + offset

    def get_positions(theta: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                np.zeros(n_frames + 1),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    theta = tnt_sample(
        theta,
        np.log(np.maximum(lengths, np.finfo(np.float32).eps)),
        n_frames + 1,
    )
    positions = get_positions(theta)[:-1]

    avg_up = normalize(poses[:, :3, 1].mean(0))
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    render_poses = []
    for pos in positions:
        render_pose = viewmatrix(pos - center, up, pos)
        if render_pose.shape == (3, 4):
            render_pose = homogeneous_poses(render_pose)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(render_pose)
    return np.stack(render_poses)
