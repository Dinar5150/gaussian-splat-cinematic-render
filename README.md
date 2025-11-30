# Gaussian/Point-Cloud Cinematic Renderer

This repository contains a minimal Python example that plans an in-scene camera
path for a Gaussian splat / point cloud stored in a `.ply` file and renders a
cinematic fly-through video. The script prefers a CUDA Gaussian renderer
(`gsplat`) when available and falls back to Open3D's offscreen renderer.

## Features
- Load a `.ply` Gaussian splat or point cloud.
- Compute the scene bounding box.
- Sample interior waypoints that cover the scene volume.
- Simple obstacle avoidance using nearest-neighbor checks to push camera
  positions away from dense geometry.
- Cinematic motion via cubic-spline path smoothing.
- Renders frames and writes an H.264 `.mp4` video.

## Quickstart
```bash
pip install -r requirements.txt
python cinematic_render.py /path/to/scene.ply --output flythrough.mp4 --seconds 8 --fps 30 --size 1280 720
```

> On systems with CUDA and [`gsplat`](https://github.com/ashawkey/gsplat)
> installed, the script will automatically use the faster splat-based renderer.
Otherwise, it defaults to Open3D. Pass ``--renderer gsplat`` to *force* the
CUDA path (the command will exit if gsplat is missing), or ``--renderer open3d``
to force the CPU fallback.

> Note: if you force ``--renderer gsplat``, make sure your gsplat version exposes
> a ``load_ply`` helper (older builds may not); otherwise use ``--renderer open3d``.

### CUDA indoor example (60s @ 30 fps, 1280x720)
Use this command to render an indoor-oriented fly-through on CUDA at 30 fps for
one minute in 720p:

```bash
python cinematic_render.py /path/to/scene.ply \
  --output indoor_flythrough.mp4 \
  --seconds 60 \
  --fps 30 \
  --size 1280 720 \
  --scene-type indoor \
  --renderer gsplat
```

### Indoor vs. outdoor scenes
- ``--scene-type auto`` (default): balanced interior sampling and clearance that
  works well for most scenes.
- ``--scene-type indoor``: narrows the camera's vertical spread and reduces the
  obstacle-clearance margin for tighter spaces.
- ``--scene-type outdoor``: widens sampling toward the boundaries and increases
  clearance to keep the camera farther from large structures.

The setting is just a hint; the bounding box and KD-tree collision checks still
adapt to the actual geometry of your `.ply` file.

## Implementation Notes
- Path planning uses a coarse 3D grid inside the bounding box to cover multiple
  regions, with KD-tree distance checks to avoid collisions.
- Motion is smoothed using cubic splines for professional, easing-friendly
  camera moves.
- The camera looks slightly ahead along the path while biased toward the scene
  center to keep subjects in frame.
