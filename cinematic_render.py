"""Cinematic Gaussian/point cloud renderer.

This script demonstrates how to plan a simple in-scene camera path for a point
cloud or Gaussian splat stored in a `.ply` file, avoid collisions, smooth the
motion, and render an .mp4 video. It prefers a CUDA Gaussian splat renderer
(`gsplat`) when available and falls back to an Open3D point cloud renderer.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Sequence

np = None  # lazily imported


def _require_numpy():
    global np
    if np is None:
        try:
            import numpy as _numpy
        except ImportError as exc:  # pragma: no cover - core dependency
            raise SystemExit("NumPy is required: pip install numpy") from exc
        np = _numpy
    return np

gsplat = None  # populated lazily


def _require_imageio():
    try:
        import imageio
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("imageio is required for video output: pip install imageio") from exc
    return imageio


def _require_spline():
    try:
        from scipy.interpolate import CubicSpline
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("SciPy is required for spline smoothing: pip install scipy") from exc
    return CubicSpline


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Open3D is required for the fallback renderer: pip install open3d") from exc
    return o3d


def _maybe_import_gsplat():
    global gsplat
    if gsplat is None:
        try:  # Optional CUDA splat renderer
            import gsplat as _gsplat  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            gsplat = None
        else:
            gsplat = _gsplat
    return gsplat


@dataclass
class SceneBounds:
    min_corner: np.ndarray
    max_corner: np.ndarray

    @property
    def center(self) -> np.ndarray:
        np = _require_numpy()
        return (self.min_corner + self.max_corner) / 2.0

    @property
    def extent(self) -> np.ndarray:
        np = _require_numpy()
        return self.max_corner - self.min_corner


@dataclass
class CameraPath:
    waypoints: List[np.ndarray]
    smoothed: np.ndarray


def load_scene(ply_path: str) -> tuple[np.ndarray, SceneBounds]:
    """Load a .ply file and return points plus bounding box."""
    np = _require_numpy()
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)

    if _maybe_import_gsplat() is not None:
        cloud = gsplat.io.load_ply(ply_path)  # type: ignore[attr-defined]
        points = np.asarray(cloud["positions"], dtype=np.float32)
    else:
        o3d = _require_open3d()
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            raise ValueError("Loaded point cloud is empty")
        points = np.asarray(pcd.points, dtype=np.float32)

    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    return points, SceneBounds(min_corner=min_corner, max_corner=max_corner)


def sample_coverage_waypoints(
    bounds: SceneBounds, num_waypoints: int = 6, scene_type: str = "auto"
) -> List[np.ndarray]:
    """Generate interior waypoints that span the scene volume.

    A loose "scene type" hint lets us bias the coverage sampling without
    requiring different code paths for indoor or outdoor scenes:

    * ``auto``: balanced sampling that stays away from boundaries
    * ``indoor``: tighter vertical spread and smaller margins
    * ``outdoor``: larger margins and slightly higher vertical reach
    """
    np = _require_numpy()
    waypoints: List[np.ndarray] = []
    grid = int(math.ceil(num_waypoints ** (1 / 3)))

    if scene_type == "indoor":
        x_span, y_span, z_span = (0.25, 0.75), (0.35, 0.65), (0.25, 0.75)
    elif scene_type == "outdoor":
        x_span, y_span, z_span = (0.15, 0.85), (0.25, 0.75), (0.15, 0.9)
    else:  # auto/balanced
        x_span, y_span, z_span = (0.2, 0.8), (0.3, 0.7), (0.2, 0.8)

    xs = np.linspace(*x_span, grid)
    ys = np.linspace(*y_span, grid)
    zs = np.linspace(*z_span, grid)
    for x in xs:
        for y in ys:
            for z in zs:
                pos = bounds.min_corner + bounds.extent * np.array([x, y, z])
                waypoints.append(pos)
                if len(waypoints) >= num_waypoints:
                    return waypoints
    return waypoints


def avoid_obstacles(
    waypoints: List[np.ndarray], points: np.ndarray, min_distance: float = 0.2
) -> List[np.ndarray]:
    """Push waypoints away from nearby geometry using a KD-tree."""
    np = _require_numpy()
    o3d = _require_open3d()
    kdtree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(points))
    adjusted: List[np.ndarray] = []
    for wp in waypoints:
        _, idx, dists = kdtree.search_knn_vector_3d(wp, 1)
        if idx:
            nearest = points[idx[0]]
            dist = math.sqrt(dists[0])
            if dist < min_distance:
                direction = (wp - nearest)
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([0.0, 1.0, 0.0])
                direction = direction / np.linalg.norm(direction)
                wp = nearest + direction * min_distance
        adjusted.append(wp)
    return adjusted


def smooth_path(waypoints: Sequence[np.ndarray], num_frames: int = 120) -> np.ndarray:
    """Interpolate waypoints with a cubic spline for cinematic motion."""
    np = _require_numpy()
    CubicSpline = _require_spline()
    waypoints_arr = np.stack(waypoints)
    t = np.linspace(0, 1, len(waypoints_arr))
    samples = np.linspace(0, 1, num_frames)
    smoothed = np.stack(
        [CubicSpline(t, waypoints_arr[:, dim], bc_type="clamped")(samples) for dim in range(3)],
        axis=1,
    )
    return smoothed


def plan_camera_path(
    points: np.ndarray,
    bounds: SceneBounds,
    num_waypoints: int = 6,
    num_frames: int = 240,
    scene_type: str = "auto",
) -> CameraPath:
    np = _require_numpy()
    clearance = max(0.1, float(np.linalg.norm(bounds.extent)) * 0.03)
    if scene_type == "indoor":
        clearance *= 0.6
    elif scene_type == "outdoor":
        clearance *= 1.4

    waypoints = sample_coverage_waypoints(bounds, num_waypoints, scene_type=scene_type)
    safe_waypoints = avoid_obstacles(waypoints, points, min_distance=clearance)
    smoothed = smooth_path(safe_waypoints, num_frames=num_frames)
    return CameraPath(waypoints=safe_waypoints, smoothed=smoothed)


def _render_with_gsplat(cloud_path: str, camera_path: np.ndarray, bounds: SceneBounds, size: tuple[int, int]) -> List[np.ndarray]:
    """Render frames using gsplat if available."""
    np = _require_numpy()
    if _maybe_import_gsplat() is None:
        raise RuntimeError("gsplat renderer requested but not available")
    frames: List[np.ndarray] = []
    cloud = gsplat.io.load_ply(cloud_path)  # type: ignore[attr-defined]
    renderer = gsplat.Renderer(width=size[0], height=size[1], use_cuda=True)  # type: ignore[attr-defined]
    for idx, eye in enumerate(camera_path):
        look_at = camera_path[min(idx + 1, len(camera_path) - 1)]
        renderer.set_camera(eye=eye, at=look_at, up=[0, 1, 0])
        frame = renderer.render(cloud)  # type: ignore[attr-defined]
        frames.append(np.asarray(frame))
    return frames


def _render_with_open3d(points: np.ndarray, camera_path: np.ndarray, bounds: SceneBounds, size: tuple[int, int]) -> List[np.ndarray]:
    """Render frames using Open3D's offscreen renderer."""
    np = _require_numpy()
    o3d = _require_open3d()
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(points)
    colors = np.ones_like(points) * 0.8
    mesh.colors = o3d.utility.Vector3dVector(colors)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.5

    renderer = o3d.visualization.rendering.OffscreenRenderer(size[0], size[1])
    scene = renderer.scene
    scene.set_background([0, 0, 0, 1])
    scene.add_geometry("cloud", mesh, material)
    center = bounds.center

    frames: List[np.ndarray] = []
    for idx, eye in enumerate(camera_path):
        target = camera_path[min(idx + 1, len(camera_path) - 1)]
        scene.camera.look_at(center * 0.4 + target * 0.6, eye, np.array([0, 1, 0]))
        frame = np.asarray(renderer.render_to_image())
        frames.append(frame)
    renderer.release()
    return frames


def render_video(
    ply_path: str,
    output_path: str = "output.mp4",
    size: tuple[int, int] = (1280, 720),
    seconds: float = 8.0,
    fps: int = 30,
    scene_type: str = "auto",
    renderer: str = "auto",
) -> None:
    """Load a scene, plan a path, render frames, and save an .mp4."""
    total_frames = max(2, int(round(seconds * fps)))
    points, bounds = load_scene(ply_path)
    camera_path = plan_camera_path(points, bounds, num_frames=total_frames, scene_type=scene_type)

    use_gsplat = (renderer == "gsplat") or (renderer == "auto" and _maybe_import_gsplat() is not None)
    if use_gsplat:
        frames = _render_with_gsplat(ply_path, camera_path.smoothed, bounds, size)
    else:
        frames = _render_with_open3d(points, camera_path.smoothed, bounds, size)

    imageio = _require_imageio()
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264")
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a cinematic fly-through of a .ply point cloud.")
    parser.add_argument("ply", help="Path to input .ply Gaussian splat / point cloud")
    parser.add_argument("--output", default="output.mp4", help="Output video path (mp4)")
    parser.add_argument("--seconds", type=float, default=8.0, help="Length of the video in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    parser.add_argument(
        "--scene-type",
        choices=["auto", "indoor", "outdoor"],
        default="auto",
        help="Hint to bias coverage and clearance for indoor or outdoor scenes",
    )
    parser.add_argument(
        "--renderer",
        choices=["auto", "gsplat", "open3d"],
        default="auto",
        help="Force CUDA gsplat, force Open3D fallback, or auto-select",
    )
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    args = parser.parse_args()

    total_frames = max(2, int(round(args.seconds * args.fps)))
    points, bounds = load_scene(args.ply)
    camera_path = plan_camera_path(
        points, bounds, num_frames=total_frames, scene_type=args.scene_type
    )

    use_gsplat = (args.renderer == "gsplat") or (args.renderer == "auto" and _maybe_import_gsplat() is not None)
    if use_gsplat:
        frames = _render_with_gsplat(args.ply, camera_path.smoothed, bounds, tuple(args.size))
    else:
        frames = _render_with_open3d(points, camera_path.smoothed, bounds, tuple(args.size))

    imageio = _require_imageio()
    imageio.mimwrite(args.output, frames, fps=args.fps, codec="libx264")
    print(f"Saved video to {args.output}")
