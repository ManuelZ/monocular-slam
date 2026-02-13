# Monocular Visual SLAM (WIP)

Monocular visual odometry — the front end of a SLAM system — estimating camera motion and 3D structure from a single camera. Backend components like loop closure and global optimization are not covered here.

Takes single-camera frames from the KITTI odometry benchmark and:
- Detects and matches features (ORB/SIFT) between consecutive frames
- Estimates relative pose via Essential Matrix (5-point algorithm + USAC_MAGSAC RANSAC)
- Accumulates frame-to-frame transforms into a global trajectory (SE(3))
- Triangulates a sparse, color-sampled point cloud from matched features
- Visualizes trajectory and point cloud in Open3D with ground truth comparison

Translation is recovered only up to an unknown scale factor — the fundamental limitation of monocular SLAM.

**Tech stack:** Python, OpenCV, Open3D, NumPy, pykitti, spatialmath

## Project Structure

| File | Description |
|------|-------------|
| `slam.py` | Two-class pipeline: **Processor** (preprocessing, ORB/SIFT, FLANN/BF matching, essential matrix → R,t) and **VisualOdometry** (main loop, SE(3) accumulation, triangulation, color sampling from cam2) |
| `kitti.py` | Wraps `pykitti.odometry` — frame iterator over cam0 (grayscale) and cam2 (RGB), projection matrices, ground truth poses |
| `utils.py` | Matching helpers (ratio test, cross-check), coordinate transforms, keypoint distribution (SSC, adaptive grid), blur detection |
| `visualization.py` | `Map` class — Open3D GUI on main thread, receives SLAM updates via `post_to_main_thread()`. Estimated path (blue) vs. ground truth (green) |
| `performance.py` | FPS / timing monitor |
| `config.py` | Paths, target resolution, debug flags |

## Usage

### 1. Download the KITTI Odometry Dataset

Download from the [KITTI odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):
- **Grayscale sequences** (required for odometry)
- **Color sequences** (required for point cloud coloring)
- **Ground truth poses**
- **Calibration files**

Expected folder structure (`pykitti` convention):

```
dataset/
└── sequences/
    ├── 00/
    │   ├── image_0/   # cam0 — grayscale left
    │   ├── image_2/   # cam2 — color left
    │   └── calib.txt
    ├── 01/
    │   └── ...
    └── ...
    poses/
    ├── 00.txt
    ├── 01.txt
    └── ...
```

### 2. Configure and run

1. Set `BASE_PATH` (path to `dataset/`) and `KITTI_SEQUENCE` in `config.py`
2. Tune `DETECTOR`, `MATCH_METHOD`, and other parameters as needed
3. Run:

```bash
python slam.py
```

## Dependencies

```
numpy
opencv-python
open3d
imutils
spatialmath-python
pykitti
Pillow
```

Install with:
```bash
pip install -r requirements.txt
```
