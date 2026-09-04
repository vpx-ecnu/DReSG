"""Shared camera-path geometry and interpolation operations.

Interpolation code is adapted from:
https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/camera_utils.py
"""

import numpy as np


def validate_camera_frames(frames: np.ndarray) -> None:
    """Require right-handed orthogonal frames with one path-wide scale."""
    if not np.issubdtype(frames.dtype, np.floating):
        raise TypeError("Camera frames must use a floating-point dtype")
    if frames.shape[-2:] != (3, 3) or not np.isfinite(frames).all():
        raise ValueError("Camera frames must be finite [..., 3, 3] matrices")
    tolerance = 5.0e-5 if frames.dtype.itemsize <= 4 else 1.0e-7
    canonical = np.asarray(frames, dtype=np.float64)
    gram = np.swapaxes(canonical, -1, -2) @ canonical
    squared_scale = np.trace(gram, axis1=-2, axis2=-1) / 3.0
    if not np.isfinite(squared_scale).all() or np.any(
        squared_scale <= np.finfo(frames.dtype).eps
    ):
        raise ValueError("Camera frames must have a finite positive scale")
    normalized_gram = gram / squared_scale[..., None, None]
    if not np.allclose(
        normalized_gram,
        np.eye(3, dtype=np.float64),
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("Camera frame axes must be orthogonal with one uniform scale")
    scales = np.atleast_1d(np.sqrt(squared_scale))
    if not np.allclose(scales, scales[0], rtol=tolerance, atol=tolerance):
        raise ValueError("All camera frames must share one uniform scale")
    determinants = np.linalg.det(canonical)
    if not np.isfinite(determinants).all() or np.any(determinants <= 0.0):
        raise ValueError("Camera frames must be right-handed")


def homogeneous_poses(poses: np.ndarray) -> np.ndarray:
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] != (3, 4):
        raise ValueError(f"Expected poses ending in (3, 4) or (4, 4), got {poses.shape}")
    bottom = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=poses.dtype),
        (*poses.shape[:-2], 1, 4),
    )
    return np.concatenate([poses, bottom], axis=-2)


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize one finite non-zero trajectory vector."""
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("Trajectory vectors must use a floating-point dtype")
    if not np.isfinite(x).all():
        raise ValueError("Trajectory vectors must contain only finite values")
    denominator = np.linalg.norm(x)
    if denominator <= np.finfo(x.dtype).eps:
        raise ValueError("Cannot normalize a degenerate trajectory vector")
    return x / denominator


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    lhs = mt_m.mean(0)
    rhs = (mt_m @ origins).mean(0)[:, 0]
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def generate_spiral_path(
    poses: np.ndarray,
    bounds: np.ndarray,
    n_frames: int = 120,
    n_rots: int = 2,
    zrate: float = 0.5,
    spiral_scale_f: float = 1.0,
    spiral_scale_r: float = 1.0,
    focus_distance: float = 0.75,
) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of conservative near and far bounds in disparity space.
    near_bound = bounds.min()
    far_bound = bounds.max()
    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 1 / ((1 - focus_distance) / near_bound + focus_distance / far_bound)
    focal = focal * spiral_scale_f

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = radii * spiral_scale_r
    radii = np.concatenate([radii, [1.0]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point(poses)
    # Path height sits at z=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], height])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta: np.ndarray) -> np.ndarray:
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation * (z_low[2] + (z_high - z_low)[2] * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)) + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])


def generate_interpolated_path(
    poses: np.ndarray,
    n_frames: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
) -> np.ndarray:
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_frames: exact number of returned camera poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_frames, 3, 4).
    """

    if poses.ndim != 3 or poses.shape[0] < 2 or poses.shape[1:] not in {
        (3, 4),
        (4, 4),
    }:
        raise ValueError("Interpolated paths require at least two [3, 4] or [4, 4] poses")
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError("Interpolated path n_frames must be a positive integer")

    from scipy import interpolate

    def poses_to_points(poses: np.ndarray, dist: float) -> np.ndarray:
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points: np.ndarray) -> np.ndarray:
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(position - lookat, up - position, position) for position, lookat, up in points])

    def interp(
        points: np.ndarray,
        n: int,
        k: int,
        s: float,
    ) -> np.ndarray:
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points, n_frames, k=spline_degree, s=smoothness)
    return points_to_poses(new_points)
