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
from typing import List
import warnings

import imageio
import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline

try:  # Optional CUDA splat renderer
    import gsplat  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gsplat = None


@dataclass
class SceneBounds:
    min_corner: np.ndarray
    max_corner: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.min_corner + self.max_corner) / 2.0

    @property
    def extent(self) -> np.ndarray:
        return self.max_corner - self.min_corner


@dataclass
class CameraPath:
    waypoints: List[np.ndarray]
    smoothed: np.ndarray


def load_scene(ply_path: str) -> tuple[np.ndarray, SceneBounds]:
    """Load a .ply file and return points plus bounding box."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(ply_path)

    # Use Open3D for robust point loading; gsplat loaders are only needed for
    # the CUDA renderer and may not be present in all versions.
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise ValueError("Loaded point cloud is empty")
    points = np.asarray(pcd.points, dtype=np.float64)
    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.all():
        removed = int((~finite_mask).sum())
        points = points[finite_mask]
        warnings.warn(f"Dropped {removed} non-finite points from the cloud")
    if len(points) == 0:
        raise ValueError("Point cloud has no finite points after filtering")

    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    return points, SceneBounds(min_corner=min_corner, max_corner=max_corner)


def plan_camera_path(
    points: np.ndarray,
    bounds: SceneBounds,
    num_waypoints: int = 6,
    num_frames: int = 240,
    scene_type: str = "auto",
) -> CameraPath:
    clearance = max(0.1, float(np.linalg.norm(bounds.extent)) * 0.03)
    if scene_type == "indoor":
        clearance *= 0.6
        spans = (0.25, 0.75), (0.35, 0.65), (0.25, 0.75)
    elif scene_type == "outdoor":
        clearance *= 1.4
        spans = (0.15, 0.85), (0.25, 0.75), (0.15, 0.9)
    else:
        spans = (0.2, 0.8), (0.3, 0.7), (0.2, 0.8)

    # Coverage waypoints sampled on a coarse interior grid
    waypoints: List[np.ndarray] = []
    grid = int(math.ceil(num_waypoints ** (1 / 3)))
    xs = np.linspace(*spans[0], grid)
    ys = np.linspace(*spans[1], grid)
    zs = np.linspace(*spans[2], grid)
    for x in xs:
        for y in ys:
            for z in zs:
                pos = bounds.min_corner + bounds.extent * np.array([x, y, z])
                waypoints.append(pos)
                if len(waypoints) >= num_waypoints:
                    break
            if len(waypoints) >= num_waypoints:
                break
        if len(waypoints) >= num_waypoints:
            break

    # Obstacle avoidance
    kdtree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    adjusted: List[np.ndarray] = []
    for wp in waypoints:
        try:
            _, idx, dists = kdtree.search_knn_vector_3d(np.asarray(wp, dtype=np.float64), 1)
        except RuntimeError:
            idx, dists = [], []
        if idx:
            nearest = points[idx[0]]
            dist = math.sqrt(dists[0])
            if dist < clearance:
                direction = (wp - nearest)
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([0.0, 1.0, 0.0])
                direction = direction / np.linalg.norm(direction)
                wp = nearest + direction * clearance
        adjusted.append(wp)

    # Smooth path with cubic splines for cinematic motion
    waypoints_arr = np.stack(adjusted)
    t = np.linspace(0, 1, len(waypoints_arr))
    samples = np.linspace(0, 1, num_frames)
    smoothed = np.stack(
        [CubicSpline(t, waypoints_arr[:, dim], bc_type="clamped")(samples) for dim in range(3)],
        axis=1,
    )
    return CameraPath(waypoints=adjusted, smoothed=smoothed)


def _render_with_gsplat(
    cloud_path: str, camera_path: np.ndarray, bounds: SceneBounds, size: tuple[int, int]
) -> List[np.ndarray]:
    """Render frames using gsplat if available."""
    if gsplat is None:
        raise RuntimeError("gsplat renderer requested but not available")
    frames: List[np.ndarray] = []
    loader = None
    if hasattr(gsplat, "io") and hasattr(gsplat.io, "load_ply"):
        loader = gsplat.io.load_ply  # type: ignore[attr-defined]
    elif hasattr(gsplat, "load_ply"):
        loader = gsplat.load_ply  # type: ignore[attr-defined]
    if loader is None:
        raise RuntimeError(
            "gsplat is installed but has no load_ply; update gsplat or use --renderer open3d"
        )
    try:
        cloud = loader(cloud_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load .ply with gsplat; ensure your gsplat version supports load_ply"
        ) from exc
    renderer = gsplat.Renderer(width=size[0], height=size[1], use_cuda=True)  # type: ignore[attr-defined]
    for idx, eye in enumerate(camera_path):
        look_at = camera_path[min(idx + 1, len(camera_path) - 1)]
        renderer.set_camera(eye=eye, at=look_at, up=[0, 1, 0])
        frame = renderer.render(cloud)  # type: ignore[attr-defined]
        frames.append(np.asarray(frame))
    return frames


def _render_with_open3d(
    points: np.ndarray, camera_path: np.ndarray, bounds: SceneBounds, size: tuple[int, int]
) -> List[np.ndarray]:
    """Render frames using Open3D's offscreen renderer."""
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

    use_gsplat = (renderer == "gsplat") or (renderer == "auto" and gsplat is not None)
    if use_gsplat:
        frames = _render_with_gsplat(ply_path, camera_path.smoothed, bounds, size)
    else:
        frames = _render_with_open3d(points, camera_path.smoothed, bounds, size)

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

    use_gsplat = (args.renderer == "gsplat") or (args.renderer == "auto" and gsplat is not None)
    if use_gsplat:
        frames = _render_with_gsplat(args.ply, camera_path.smoothed, bounds, tuple(args.size))
    else:
        frames = _render_with_open3d(points, camera_path.smoothed, bounds, tuple(args.size))

    imageio.mimwrite(args.output, frames, fps=args.fps, codec="libx264")
    print(f"Saved video to {args.output}")
