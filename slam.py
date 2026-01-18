"""
Monocular Visual SLAM Pipeline.

Architecture
------------
This module implements a two-class design for visual odometry:

    Processor: Low-level image processing and geometry estimation.
        - Frame preprocessing (resize, undistort, grayscale conversion)
        - Feature detection and descriptor extraction (ORB, SIFT, etc.)
        - Feature matching between consecutive frames
        - Essential matrix estimation and relative pose recovery (R, t)

    VisualOdometry: High-level orchestration and state management.
        - Main processing loop (frame capture -> pose -> triangulation -> display)
        - Pose accumulation to build camera trajectory
        - 3D point triangulation from matched features
        - Coordination with Map for visualization

Data Flow
---------
    Frame -> Processor.preprocess_frame() -> detect_and_compute() -> get_matched_points()
          -> pose_estimation_2d2d() -> VisualOdometry accumulates pose -> triangulatePoints()
          -> Map.display()

The Processor maintains state for two frames (current/previous) to enable frame-to-frame
tracking. VisualOdometry drives the pipeline and maintains the global trajectory.
"""

# Standard Library imports
from collections.abc import Sequence
from typing import Literal, TypedDict, cast

# External imports
import cv2
import numpy as np
import numpy.typing as npt
from imutils.feature import FeatureDetector_create
from PIL import Image
from spatialmath import SE3
from spatialmath.base.transforms3d import ishom, isR, trnorm

# Local imports
import config
from kitti import KittiDataset
from utils import (
    check_rotation_mat,
    homogeneous_to_euclidean,
    pil_to_np,
    pixel2cam,
    ratio_test,
    variance_of_laplacian,
    vec_to_skew_symmetric,
)
from visualization import Map


class FrameData(TypedDict):
    """Frame data for previous frame tracking."""

    frame: np.ndarray
    kps: tuple[cv2.KeyPoint, ...]
    des: npt.NDArray[np.uint8]
    pose: np.ndarray | None


class RootSIFT:
    """
    Modified from: https://pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
    """

    def __init__(self):
        # initialize the SIFT feature extractor
        self.extractor = FeatureDetector_create("SIFT")

    def _rootsift(self, descs, eps=1e-7):
        """Apply the Hellinger kernel by first L1-normalizing and taking the square-root."""
        descs /= descs.sum(axis=1, keepdims=True) + eps
        descs = np.sqrt(descs)
        return descs

    def detect(self, image, mask=None):
        """Detect Difference of Gaussian keypoints in the image."""
        return self.extractor.detect(image, mask)

    def compute(self, image, kps, eps=1e-7):
        """Compute RootSIFT descriptors for the given keypoints."""
        kps, des = self.extractor.compute(image, kps)
        if len(kps) == 0:
            return ([], None)
        des = self._rootsift(des, eps)
        return kps, des

    def detectAndCompute(self, image, mask=None, eps=1e-7):
        """Detect keypoints and compute RootSIFT descriptors."""
        kps = self.detect(image, mask)
        return self.compute(image, kps, eps)


def pts_to_kps(points) -> list[cv2.KeyPoint]:
    """Convert (N, 2) point array to a list of cv2.KeyPoint objects."""
    return [cv2.KeyPoint(p[0], p[1], 1) for p in points]


def draw_keypoints(
    frame: np.ndarray, kp: Sequence[cv2.KeyPoint], color=(0, 255, 0)
) -> np.ndarray:
    """Draw keypoints on a frame and return the annotated image."""
    return cv2.drawKeypoints(
        cast(cv2.Mat, frame),
        kp,
        cast(cv2.Mat, None),
        color=color,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )


def show_matches(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    prev_points: npt.NDArray,
    curr_points: npt.NDArray,
    inliers: npt.NDArray[np.uint8],
) -> np.ndarray:
    """Render a side-by-side composite showing inlier matches between two frames."""

    # Filter to inliers only
    mask = inliers.ravel().astype(bool)
    prev_inliers = prev_points[mask]
    curr_inliers = curr_points[mask]

    # Transform into OpenCV KeyPoints
    prev_kps = pts_to_kps(prev_inliers)
    curr_kps = pts_to_kps(curr_inliers)

    # Draw inlier keypoints
    composite = cv2.merge((prev_frame, current_frame, prev_frame))
    composite = draw_keypoints(composite, prev_kps, (0, 0, 255))
    composite = draw_keypoints(composite, curr_kps, (255, 0, 0))

    # Draw lines between inliers
    for k1, k2 in zip(prev_kps, curr_kps):
        pt1 = (int(k1.pt[0]), int(k1.pt[1]))
        pt2 = (int(k2.pt[0]), int(k2.pt[1]))
        cv2.line(composite, pt1, pt2, (255, 0, 0), 1)

    return composite


class Processor:
    """
    Image processing pipeline for monocular visual SLAM.

    Handles the front-end of a visual odometry system: frame preprocessing,
    feature detection/matching, and relative pose estimation between consecutive frames.

    Attributes:
        K: Camera intrinsic matrix (3x3).
        distortion_coeffs: Lens distortion coefficients, or None if pre-calibrated.
        current: Current frame data (frame, keypoints, descriptors).
        prev: Previous frame data including pose.
        E: Essential matrix from the most recent pose estimation.
        inliers: RANSAC inlier mask from essential matrix estimation.
    """

    _MatchMethod = Literal[
        "brute_force", "brute_force_ratio_test", "flann", "flann_ratio_test"
    ]
    _FLANN_INDEX_LSH = 6
    _FLANN_INDEX_KDTREE = 1

    def __init__(
        self,
        K0: np.ndarray,
        distortion_coeffs: np.ndarray | None = None,
        detector_name="orb",
        target_width: int | None = None,
        min_inliers: int = 10,
        e_svd_rtol: float = 1e-3,
        e_svd_atol: float = 1e-6,
        **kwargs,
    ):
        self.prev: FrameData = cast(
            FrameData, {"frame": None, "kps": None, "des": None, "pose": None}
        )
        if detector_name.lower() == "rootsift":
            self.detector = RootSIFT()
        elif detector_name.lower() == "asift":
            self.detector = cv2.AffineFeature_create(
                cv2.SIFT_create(**kwargs), maxTilt=1
            )
        else:
            self.detector = FeatureDetector_create(detector_name, **kwargs)

        self.K: npt.NDArray[np.float32] = K0.copy()
        self.K_inv: npt.NDArray[np.float64] = np.linalg.inv(self.K)
        self.K_hom = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        self.K_is_resized = False
        self.distortion_coeffs = distortion_coeffs
        self.target_width = target_width
        self._resize_ratio = None  # target width / width
        self.inliers: npt.NDArray[np.uint8]  # Inliers found through RANSAC or similar
        self.min_inliers = min_inliers
        self.e_svd_rtol = e_svd_rtol
        self.e_svd_atol = e_svd_atol

    def resize_frame(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Resize frame to the target width while preserving aspect ratio.

        Uses INTER_AREA interpolation for downscaling and INTER_CUBIC for
        upscaling to maintain image quality.

        Args:
            frame: Input image as a numpy array.

        Returns:
            Resized image as a numpy array.
        """
        if self.target_width is None:
            raise RuntimeError(
                "resize_frame() called but target_width was not set. "
                "Pass target_width to the Processor constructor."
            )

        image_h, image_w = frame.shape[:2]
        self._resize_ratio = self.target_width / image_w
        target_height = int(image_h * self._resize_ratio)
        new_dim = (self.target_width, target_height)

        if config.DEBUG:
            print(
                f"Resizing from (H,W) {image_h}x{image_w} to {target_height}x{self.target_width}"
            )

        # From the docs:
        # To shrink an image, it will generally look best with INTER_AREA interpolation,
        # to enlarge an image, it will generally look best with INTER_CUBIC (slow) or INTER_LINEAR
        # (faster but still looks OK).
        interpolation = cv2.INTER_AREA if self._resize_ratio < 1 else cv2.INTER_CUBIC

        return cv2.resize(frame, new_dim, interpolation=interpolation)

    def undistort_points(
        self, points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Correct lens distortion on sparse 2D points.

        Uses `cv2.undistortPoints` to remove radial and tangential distortion
        from a set of 2D pixel coordinates. The `P=self.K` parameter reprojects
        the points back into pixel coordinates so they remain compatible with
        `findEssentialMat` and `triangulatePoints`.

        Args:
            points: Matched keypoints as a (N, 2) float32 array in pixel coordinates.

        Returns:
            Undistorted points as a (N, 2) float32 array in pixel coordinates.
        """
        pts = cv2.undistortPoints(
            np.expand_dims(points, axis=1),
            cameraMatrix=self.K,
            distCoeffs=self.distortion_coeffs,  #  If the vector is NULL/empty, zero distortion coefficients are assumed
            P=self.K,
        )
        return pts.reshape(-1, 2)

    def pixel_to_normalized(
        self, points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Convert pixel coordinates to normalized camera coordinates using K inverse.

        Applies K⁻¹ to transform from pixel space (u, v) to the normalized
        image plane (x, y) where z=1. No distortion correction is performed.

        Args:
            points: 2D pixel coordinates as a (N, 2) float32 array.

        Returns:
            Normalized camera coordinates as a (N, 2) float32 array.
        """

        ones = np.ones((points.shape[0], 1), dtype=points.dtype)
        pts_h = np.concatenate([points, ones], axis=1)  # (N, 3)
        # K is a transformation matrix to go from normalized camera coordinates to pixel coordinates
        pts_cam = (self.K_inv @ pts_h.T).T  # (N, 3)
        return pts_cam[:, :2]

    def detect_and_compute(
        self, frame: npt.NDArray[np.uint8]
    ) -> tuple[tuple[cv2.KeyPoint, ...], npt.NDArray[np.uint8]]:
        """
        Detect keypoints and compute descriptors for a single frame.

        "As a rule of the thumb, 1000 features is a good number for a 640×480-pixel image".
        (https://ieeexplore.ieee.org/document/6153423)
        "The image can be partitioned into a grid, and the feature detector is applied to each cell by tuning
        the detection thresholds until a minimum number of features are found in each subimage."
        """
        kps = self.detector.detect(frame, None)
        kps, des = self.detector.compute(frame, kps)
        return kps, des

    def _rescale_intrinsic_matrix(self):
        """Scale K to match the resized frame dimensions. No-op after the first call."""

        if not self.K_is_resized:
            self.K[0, 0] *= self._resize_ratio
            self.K[1, 1] *= self._resize_ratio
            self.K[0, 2] *= self._resize_ratio
            self.K[1, 2] *= self._resize_ratio
            self.K_is_resized = True
            self.K_inv = np.linalg.inv(self.K)
            self.K_hom = np.concatenate((self.K, np.zeros((3, 1))), axis=1)

    def preprocess_frame(self, frame: npt.NDArray[np.uint8] | Image.Image):
        """Preprocess a frame for feature extraction.

        Converts the input frame to grayscale, optionally resizing and
        undistorting it based on the configured parameters.

        Args:
            frame: Input image as a PIL Image or numpy array (BGR format).

        Returns:
            Grayscale image as a numpy array, ready for feature detection.
        """

        if config.DEBUG:
            print("Preprocessing...")

        if isinstance(frame, Image.Image):
            frame = pil_to_np(frame)

        if self.target_width is not None:
            frame = self.resize_frame(frame)
            self._rescale_intrinsic_matrix()

        return frame

    def get_matched_points(
        self,
        curr_kps: tuple[cv2.KeyPoint, ...],
        curr_des: npt.NDArray[np.uint8],
        method: _MatchMethod,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Match descriptors between the previous and current frame.

        Args:
            curr_kps: Keypoints detected in the current frame.
            curr_des: Descriptors for the current frame keypoints.
            method: Matching strategy. One of: "brute_force", "brute_force_ratio_test",
                    "flann", "flann_ratio_test".

        Returns:
            Tuple of (prev_points, curr_points), each an (N, 2) float32 array of
            matched pixel coordinates in the previous and current frame respectively.
        """

        prev_kps = self.prev["kps"]

        matches = self.match_descriptors(curr_des, method=method)
        prev_frame_points = np.array(
            [prev_kps[m.queryIdx].pt for m in matches], dtype=np.float32
        )
        curr_frame_points = np.array(
            [curr_kps[m.trainIdx].pt for m in matches], dtype=np.float32
        )

        return (prev_frame_points, curr_frame_points)

    @staticmethod
    def _norm_type(descriptors) -> int:
        """Return the appropriate cv2 norm for the given descriptors.

        Binary descriptors (uint8, e.g. ORB/BRIEF/BRISK) -> NORM_HAMMING.
        Float descriptors (float32, e.g. SIFT/SURF)      -> NORM_L2.
        """
        if descriptors.dtype == np.uint8:
            return cv2.NORM_HAMMING
        return cv2.NORM_L2

    def match_descriptors(self, curr_des, method: _MatchMethod):
        """Match current descriptors against previous frame descriptors.

        Notes:
            Most feature-matching techniques assume linear illumination changes, pure camera
            rotation and scaling (zoom), or affine distortion [1].

            The threshold for the ratio test can only be set heuristically and an unlucky guess
            might remove correct matches. In many cases, it might be beneficial to skip the ratio
            test and let RANSAC handle the outliers [1].

            [1] https://ieeexplore.ieee.org/document/6153423

        The correct distance norm (Hamming vs L2) is inferred automatically
        from the descriptor dtype, so this works with any detector.

        Args:
            curr_des: Descriptors from the current frame.
            method: Matching strategy. One of:
                - "brute_force_ratio_test": BFMatcher with Lowe's ratio test.
                - "flann": FLANN-based matcher. Selects LSH for binary
                  descriptors or KDTree for float descriptors.
                - "brute_force": BFMatcher with cross-check, sorted by
                  distance.

        Returns:
            List of cv2.DMatch objects that passed the filtering criteria.
        """

        norm = self._norm_type(curr_des)

        if method in ("brute_force", "brute_force_ratio_test"):
            if method == "brute_force":
                bf = cv2.BFMatcher(norm, crossCheck=True)
                matches = bf.match(self.prev["des"], curr_des)
                return matches
            elif method == "brute_force_ratio_test":
                bf = cv2.BFMatcher(norm, crossCheck=False)
                matches = bf.knnMatch(self.prev["des"], curr_des, k=2)
                return ratio_test(matches)

        elif method in ("flann", "flann_ratio_test"):
            # Check descriptor type to select appropriate FLANN algorithm
            # Binary descriptors (ORB, BRIEF, BRISK): uint8, use LSH with Hamming distance
            # Float descriptors (SIFT, SURF): float32, use KDTree with Euclidean distance
            # https://github.com/opencv/opencv/blob/6950bedb5ce1827bc025bea7c1b23df6e947a437/modules/flann/include/opencv2/flann/defines.h#L70
            if curr_des.dtype == np.uint8:
                # LSH (Locality Sensitive Hashing) for binary descriptors.
                # Hashes similar descriptors into the same buckets using Hamming distance.
                # Parameters:
                # - table_number: number of hash tables; more = better recall, slower
                # - key_size: hash key bits (2^12 = 4096 buckets); larger = faster, less recall
                # - multi_probe_level: check neighboring buckets; higher = better recall, slower
                # Tuning:
                # - More matches needed: increase table_number or multi_probe_level
                # - Faster matching: decrease table_number or increase key_size
                index_params = dict(
                    algorithm=self._FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1,
                )
            else:
                index_params = dict(algorithm=self._FLANN_INDEX_KDTREE, trees=5)

            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            if method == "flann":
                matches = flann.match(self.prev["des"], curr_des)
                return matches

            elif method == "flann_ratio_test":
                matches = flann.knnMatch(self.prev["des"], curr_des, k=2)
                return ratio_test(matches)

        raise ValueError(
            f"Unknown matching method {method!r}. "
            "Valid options: 'brute_force', 'brute_force_ratio_test', 'flann', 'flann_ratio_test'"
        )

    def pose_estimation_2d2d(
        self, prev_pts: npt.NDArray[np.float32], curr_pts: npt.NDArray[np.float32]
    ):
        """
        Estimate camera pose from 2D-2D point correspondences.

        Recovers relative camera motion (rotation and translation) between two frames
        by computing the essential matrix from matched feature points using RANSAC,
        then decomposing it into the rotation matrix and translation vector.

        Parameters
        ----------
        prev_pts : np.ndarray[np.float32]
            2D points from the first image, shape (N, 2).
        curr_pts : np.ndarray[np.float32]
            Corresponding 2D points from the second image, shape (N, 2).

        Returns
        -------
        R : np.ndarray
            3x3 rotation matrix representing the camera rotation from view A to view B.
        t : np.ndarray
            3x1 unit translation vector. Note: translation is only recovered up to scale.

        Notes
        -----
        - Sets self.inliers to the inlier mask from RANSAC.
        - The essential matrix satisfies: y2.T @ E @ y1 = 0 (epipolar constraint).
        - E = t^ @ R, where t^ is the skew-symmetric matrix of t.
        """

        # cv2.USAC_MAGSAC: robust RANSAC variant; see https://opencv.org/blog/evaluating-opencvs-new-ransacs
        # Five-point algorithm: D. Nister, "An efficient solution to the five-point relative pose problem"
        # https://github.com/opencv/opencv/blob/fe38fc608f6acb8b68953438a62305d8318f4fcd/modules/calib3d/src/five-point.cpp#L442
        # https://github.com/opencv/opencv/blob/fe38fc608f6acb8b68953438a62305d8318f4fcd/modules/calib3d/src/usac/essential_solver.cpp#L27
        essential_matrix, inliers = cv2.findEssentialMat(
            prev_pts,
            curr_pts,
            cameraMatrix=self.K,
            method=cv2.USAC_MAGSAC,
            prob=config.RANSAC_PROB,
            threshold=config.RANSAC_THRESHOLD,
            maxIters=config.RANSAC_MAX_ITERS,
        )

        self.inliers = cast(npt.NDArray[np.uint8], inliers)

        U, S, Vt = np.linalg.svd(essential_matrix)  # type: ignore[arg-type]

        # Any rank-2 matrix is a possible fundamental matrix. An essential matrix has the additional property that the
        # two nonzero singular values are equal. (D. Nister)
        if not np.isclose(S[0], S[1], rtol=self.e_svd_rtol) or S[2] > self.e_svd_atol:
            # From "Introduction to Visual Slam - From Theory to Practice", p. 142:
            # According to E = t∧R, the singular value of the essential matrix E must be in the form of [σ, σ, 0]^T.
            #
            # If that's not the case, there are two options:
            #  - Projecting the calculated essential matrix onto the manifold where E is located with
            #      E = U diag((σ1+σ2)/2, (σ1+σ2)/2 , 0) V^T
            #  - Take the singular value matrix as diag(1, 1, 0), due to E's scale equivalence
            #
            # The first option is used in:
            # https://github.com/scikit-image/scikit-image/blob/e9f965243a4f30f9fe5cdc9ef7afe631aaf378d7/src/skimage/transform/_geometric.py#L901
            print(f"Singular values of E: {S}")
            S[0] = (S[0] + S[1]) / 2
            S[1] = S[0]
            S[2] = 0
            print(f"WARNING: E singular values {S} deviate from [σ, σ, 0]. Fixing.")

            # Reconstruct the matrix from its SVD components
            essential_matrix = U @ np.diag(S) @ Vt

        # recoverPose: decomposes E into 4 candidate (R, t) solutions and selects the one where triangulated points are
        # in front of both cameras (chirality). Modifies `inliers` in-place.
        num_inliers, R, t, _triangulated = cv2.recoverPose(  # type: ignore[call-overload]
            essential_matrix, prev_pts, curr_pts, self.K, mask=inliers
        )

        # TODO: add parallax check here as well

        # Pure rotation handling: When the camera undergoes pure rotation (t ≈ 0), E = [t]×R ≈ 0, the essential matrix
        # is undefined, no translation means no epipolar geometry and triangulation is impossible.
        # Compare the number of inliers that passed the chirality check. A low count indicates a degenerate
        # configuration.
        if num_inliers < self.min_inliers:
            print(
                f"WARNING: recoverPose returned only {num_inliers} chirality inliers (need {self.min_inliers}). Skipping frame."
            )
            return np.eye(3), np.zeros((3, 1))

        if config.DEBUG:
            t_skew = vec_to_skew_symmetric(t)

            # Note: the recovered t has unit length, probably due to an assumption made in the 5-point algorithm.
            # "...it is assumed that the first camera matrix is [I|0] and that t is of unit length". (D. Nister)
            # Verify that neither E nor t^R are equal to zero. That would be a way (an incorrect way) of satisfying the
            # epipolar constraints.
            print(f"t:\n{t}")
            print(f"t length:\n{np.linalg.norm(t)}")
            print("E: \n", essential_matrix)
            print("E = t^R:\n", t_skew @ R)
            print(
                "E/(t^R): (this should be a constant!)\n",
                essential_matrix / (t_skew @ R),
            )

            for pt1, pt2 in zip(prev_pts, curr_pts):
                pt1 = pixel2cam(pt1, self.K)
                y1 = np.array([[pt1[0]], [pt1[1]], [1.0]])  # Homogeneous coordinates

                pt2 = pixel2cam(pt2, self.K)
                y2 = np.array([[pt2[0]], [pt2[1]], [1.0]])  # Homogeneous coordinates

                # Show epipolar constraints.
                # These values should be close to zero, because that's the constraint, an equality to zero:
                #   y2.T @ t_skew @ R @ y1 == 0
                d = y2.T @ t_skew @ R @ y1
                print(f"Epipolar constraint = {d.item()}")

        return R, t


class VisualOdometry:
    """
    Main visual odometry loop for monocular SLAM.

    Orchestrates frame capture, feature tracking, pose estimation, and 3D point
    triangulation. Maintains camera trajectory and sparse 3D map.

    Attributes:
        processor: Processor instance for feature extraction and pose estimation.
        map: Map instance for 3D visualization.
        cap: Video capture source (cv2.VideoCapture or KittiDataset).
    """

    def __init__(
        self,
        processor: Processor,
        _map: Map,
        cap: KittiDataset,
        P2,
        min_matches: int = 8,
        min_parallax_deg: float = 0.8,
        blur_threshold: float = 25,
    ):
        """
        Args:
            P2: 3x4 projection matrix of the color camera (KITTI P_rect_20).
                Projects 3D points in the cam0 reference frame into cam2 pixel
                coordinates: pixel_C2 = P2 @ point_C0_hom. Note this is a
                projection matrix (3x4, includes intrinsics), not a rigid body
                transformation. The extrinsic transformation T_C2_C0 is
                extracted internally via _decompose_projection.
            min_matches: Minimum descriptor matches for essential matrix estimation.
            min_parallax_deg: Minimum parallax angle (degrees) for triangulated points.
            blur_threshold: Variance of Laplacian below this skips the frame.
        """
        self.processor = processor
        self.map = _map
        self.cap = cap
        self.min_matches = min_matches
        self.min_parallax_deg = min_parallax_deg
        self.blur_threshold = blur_threshold
        self.K2, self.T_C2_C0 = self._decompose_projection(P2)

        # Performance monitoring for the processing loop
        # self.perf = PerformanceMonitor(report_interval=50)

    @staticmethod
    def _decompose_projection(P):
        """Build a 4x4 extrinsic transformation from a 3x4 projection matrix.

        Decomposes P = K[R|t] and constructs the 4x4 rigid body transformation
        that maps points from the reference camera frame to this camera's frame.

        Note: cv2.decomposeProjectionMatrix returns the camera center C in
        world (homogeneous) coordinates, not the extrinsic translation t.
        The extrinsic translation is t = -R @ C.

        Refer to "Multiple View Geometry" Ch 6.2.4, by Richard Hartley.
        """
        K, R, C_hom, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        C = (C_hom[:3] / C_hom[3]).flatten()
        t = (-R @ C).flatten()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return K, T

    def _sample_colors(self, points_3D, T_C0_W, color_frame):
        """Sample RGB colors from the color camera for 3D world points.

        Reproject world-frame points into the color camera's (cam2) pixel space by chaining:
          world → cam0 (via T_C0_W) → cam2 (via T_C2_C0) → pixels (via K2).
        Then samples colors from color_frame at the projected locations.

        Args:
            points_3D: 3xN array of 3D points in world coordinates.
            T_C0_W: 4x4 world-to-cam0 transformation (i.e. the inverse of the cam0 pose, which expresses world points
            in cam0's frame).
            color_frame: HxWx3 BGR image from the color camera.

        Returns:
            3xN float32 array of RGB colors normalized to [0, 1].
        """
        # World points → cam0 frame → cam2 frame
        p_W_hom = np.vstack([points_3D, np.ones((1, points_3D.shape[1]))])
        p_C0 = T_C0_W @ p_W_hom
        p_C2 = (self.T_C2_C0 @ p_C0)[:3, :]

        # Project into the color camera's pixel space
        projected = self.K2 @ p_C2
        px = (projected[:2, :] / projected[2, :]).astype(int).T  # Nx2

        # Clamp pixel coordinates to stay within image bounds
        h, w = color_frame.shape[:2]
        px[:, 0] = np.clip(px[:, 0], 0, w - 1)
        px[:, 1] = np.clip(px[:, 1], 0, h - 1)

        # Sample colors and normalize to [0, 1]
        point_colors = (color_frame[px[:, 1], px[:, 0]].astype(np.float32) / 255.0).T
        return point_colors  # 3xN

    @staticmethod
    def _chirality_mask(hom_points, inv_prev_pose, inv_curr_pose):
        """Return a boolean mask selecting points in front of both cameras.

        A triangulated point passes the chirality check when its depth (Z
        coordinate) is positive in both camera frames. Points that fail are
        behind at least one camera and are geometrically invalid.

        Each inverse pose transforms points from world frame to camera frame.
        The Z component of the resulting homogeneous vector, divided by its W
        component, gives the Euclidean depth.

        Args:
            hom_points: 4xN array of triangulated points in homogeneous world
                coordinates.
            inv_prev_pose: 4x4 world frame expressed in the previous camera frame.
            inv_curr_pose: 4x4 world frame expressed in the current camera frame.

        Returns:
            Boolean array of shape (N,).
        """

        # Re-express points from world frame into the camera frame
        points_prev_cam = inv_prev_pose @ hom_points
        points_curr_cam = inv_curr_pose @ hom_points

        # Convert depth from homogeneous to euclidean
        z_prev = points_prev_cam[2, :] / points_prev_cam[3, :]
        z_curr = points_curr_cam[2, :] / points_curr_cam[3, :]
        return (z_prev > 0) & (z_curr > 0)

    @staticmethod
    def _parallax_mask(eucl_points, prev_pose, curr_pose, min_parallax_deg):
        """Return a boolean mask selecting points with sufficient parallax.

        Parallax: two 3D points that project to the same pixel in one view (i.e. lie on the
        same camera ray) will generally project to different pixels once the camera translates,
        because translation breaks the ray alignment. This relative displacement between their
        images is called "parallax". ("Multiple View Geometry in Computer Vision", Ch 8.4.5)

        The parallax is measured as the angle between the rays from each camera center to a
        triangulated point. When this angle is small the two rays are nearly parallel, so a
        tiny error in image coordinates causes a large error in depth.

        Args:
            eucl_points: 3xN array of triangulated 3D points.
            prev_pose: 4x4 previous camera frame expressed in the world frame.
            curr_pose: 4x4 current camera frame expressed in the world frame.
            min_parallax_deg: minimum parallax angle in degrees.

        Returns:
            Boolean array of shape (N,).
        """
        # Extract camera centers (translation component of each pose)
        prev_center = prev_pose[:3, 3].reshape(3, 1)
        curr_center = curr_pose[:3, 3].reshape(3, 1)

        # Build unit-length rays from each camera center to every 3D point
        # a · b = |a||b|cos(θ)  =>  cos(θ) = â · b̂ (unit vectors)
        ray_prev = eucl_points - prev_center
        ray_curr = eucl_points - curr_center
        ray_prev = ray_prev / np.linalg.norm(ray_prev, axis=0, keepdims=True)
        ray_curr = ray_curr / np.linalg.norm(ray_curr, axis=0, keepdims=True)

        cos_angle = np.sum(ray_prev * ray_curr, axis=0)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        parallax_deg = np.degrees(np.arccos(cos_angle))
        return parallax_deg >= min_parallax_deg

    def _projection_matrix(self, world_to_cam):
        """Build a 3x4 projection matrix (also known as camera matrix) P = K @ [R|t].

        Projection matrices map points in world coordinates into pixel coordinates:
          - The extrinsic matrix [R|t] maps from world coordinates to camera normalized coordinates.
          - The intrinsic matrix K maps from camera normalized coordinates to pixel coordinates.

        Args:
            world_to_cam: 4x4 world frame expressed in the camera frame.

        Returns:
            3x4 projection matrix.

        NOTE:
            recoverPose cannot return the real translation, so [R|t] isn't a real extrinsic matrix.
            To get a real extrinsic matrix we would need the scale of the world obtained by some
            measurement with detected tags or with an IMU.
        """
        P = self.processor.K_hom @ world_to_cam
        assert P.shape == (3, 4), P.shape
        return P

    def _estimate_and_accumulate_pose(
        self,
        prev_matched_pts: npt.NDArray[np.float32],
        curr_matched_pts: npt.NDArray[np.float32],
        curr_pose: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Estimate relative pose from matches and chain it onto the accumulated world pose.

        Calls pose_estimation_2d2d to recover (R, t), builds the SE3 relative transform,
        and accumulates it with the current pose in world coordinates.
        """
        R_rel, t_rel = self.processor.pose_estimation_2d2d(
            prev_matched_pts, curr_matched_pts
        )

        assert isR(R_rel)

        # T_2_1: pose of Camera1 expressed in Camera2's coordinates (https://stackoverflow.com/a/58504392).
        # T_1_2 = inv(T_2_1): pose of Camera2 expressed in Camera1's coordinates.
        #
        # Post-multiplying the world-frame pose by T_1_2 chains the relative motion in the camera's own frame
        # (subscript cancellation: pose_W_C2 = pose_W_C1 @ T_C1_C2).
        T_2_1 = SE3.Rt(R_rel, t_rel)
        T_1_2 = T_2_1.inv()
        assert curr_pose.shape == (4, 4)
        curr_pose = curr_pose @ T_1_2.A  # 4x4

        # Check if T is still a proper SE(3) matrix, normalize if not
        if not ishom(curr_pose, check=True):
            print("WARNING - current pose is not SE(3), normalizing...")
            curr_pose = trnorm(curr_pose)

        return curr_pose

    def _triangulate_and_add_points(
        self,
        prev_pose: npt.NDArray[np.floating],
        curr_pose: npt.NDArray[np.floating],
        prev_matched_pts: npt.NDArray[np.float32],
        curr_matched_pts: npt.NDArray[np.float32],
        color_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Triangulate inlier matches between two frames and add good points to the map.

        Filters matched points by the RANSAC/chirality inlier mask, triangulates them into
        3D, applies chirality and parallax checks, samples colors, and appends the surviving
        points to self.map.
        """
        assert prev_pose.shape == (4, 4)
        assert curr_pose.shape == (4, 4)
        assert self.processor.K_hom.shape == (3, 4)

        # Filter to RANSAC inliers (inliers mask includes results from the recoverPose's chirality check!)
        inlier_mask = self.processor.inliers.ravel().astype(bool)

        assert prev_matched_pts.shape[1] == 2
        assert curr_matched_pts.shape[1] == 2
        prev_inlier_pts = prev_matched_pts[inlier_mask]
        curr_inlier_pts = curr_matched_pts[inlier_mask]

        if len(prev_inlier_pts) == 0:
            print("Warning: no inliers for triangulation")
            return

        # with self.perf.time("triangulate_and_filter"):
        inv_prev_pose = np.linalg.inv(prev_pose)
        inv_curr_pose = np.linalg.inv(curr_pose)

        P_prev = self._projection_matrix(inv_prev_pose)
        P_curr = self._projection_matrix(inv_curr_pose)

        # 4xN array of reconstructed points in homogeneous coordinates (world frame)
        hom_points = cv2.triangulatePoints(
            P_prev, P_curr, prev_inlier_pts.T, curr_inlier_pts.T
        )

        eucl_points = homogeneous_to_euclidean(hom_points)

        in_front = self._chirality_mask(hom_points, inv_prev_pose, inv_curr_pose)
        good_parallax = self._parallax_mask(
            eucl_points, prev_pose, curr_pose, self.min_parallax_deg
        )

        # Combined mask: points in front of both cameras and with enough parallax
        mask = in_front & good_parallax
        good_eucl_points = eucl_points[:, mask]  # 3xN
        assert good_eucl_points.shape[0] == 3

        if good_eucl_points.shape[1] > self.processor.min_inliers:
            point_colors = self._sample_colors(
                good_eucl_points, inv_curr_pose, color_frame
            )

            self.map.point_colors.append(point_colors)
            self.map.points_3D.append(good_eucl_points)

    def main(self):
        """
        Entry point for the SLAM pipeline.

        Initializes the Open3D GUI viewer and passes _processing_loop as a
        callback to run in a background thread. create_viewer() blocks on
        app.run(), which drives the GUI event loop on the main thread until
        the viewer window is closed.

        The main thread must own the GUI event loop because OS windowing
        systems (Cocoa on macOS, Win32 on Windows) require it.

        See visualization_architecture.md for a detailed explanation.
        """

        self.map.create_viewer(callback=self._processing_loop)

    def _processing_loop(self):
        """Main SLAM loop: reads frames, estimates pose, triangulates points, and updates the map."""

        # A rigid body transformation from camera to world (static frame)
        # Initial pose of the camera wrt the world
        curr_pose = np.eye(4)  # curr_pose_S_C
        frame_idx = -1  # KITTI frame index, incremented for every frame read

        while self.cap.isOpened():
            self.map.wait_if_paused()
            frame_ok, frame, frame_rgb = self.cap.read()

            if frame is None or frame_rgb is None:
                print("No gray or color frame received")
                break

            # Incremented for every frame read, even skipped ones, to stay aligned with KITTI GT poses
            frame_idx += 1

            if not frame_ok:
                print("Problem reading frame")
                continue

            if (blur_val := variance_of_laplacian(frame)) < self.blur_threshold:
                print(f"Skipping blurry frame: {blur_val:.2f} VoL")
                continue

            preprocessed_frame = self.processor.preprocess_frame(frame)
            color_frame = pil_to_np(frame_rgb)
            if self.processor.target_width is not None:
                color_frame = self.processor.resize_frame(color_frame)
            kps, des = self.processor.detect_and_compute(preprocessed_frame)

            if self.processor.prev["frame"] is None:
                self.processor.prev = {
                    "frame": preprocessed_frame,
                    "kps": kps,
                    "des": des,
                    "pose": None,
                }

                continue

            prev_matched_pts, curr_matched_pts = self.processor.get_matched_points(
                kps, des, method=config.MATCH_METHOD
            )

            # Essential matrix estimation requires at least 5 points (5-point algorithm)
            if len(prev_matched_pts) < self.min_matches:
                print(
                    f"Skipping frame: only {len(prev_matched_pts)} matches (need {self.min_matches})"
                )
                continue

            # The world frame is located at the origin of camera 0 during the first time step
            # The coordinate system assumes Z is forward, X is right, and Y is down.
            curr_pose = self._estimate_and_accumulate_pose(
                prev_matched_pts, curr_matched_pts, curr_pose
            )

            if (prev_pose := self.processor.prev.get("pose")) is not None:
                self._triangulate_and_add_points(
                    prev_pose,
                    curr_pose,
                    prev_matched_pts,
                    curr_matched_pts,
                    color_frame,
                )

            self.processor.prev = {
                "frame": preprocessed_frame,
                "kps": kps,
                "des": des,
                "pose": curr_pose,
            }

            self.map.cam_poses.append(curr_pose)

            self.map.display(frame_idx=frame_idx)
            composite = show_matches(
                self.processor.prev["frame"],
                preprocessed_frame,
                prev_matched_pts,
                curr_matched_pts,
                self.processor.inliers,
            )
            self.map.update_images(composite, color_frame)

            if config.DEBUG:
                num_inliers = np.sum(self.processor.inliers, axis=0)[0]
                print(f"Inliers: {num_inliers}")
                if num_inliers < self.processor.min_inliers:
                    print(f"Too few inliers: {num_inliers}")

                pressed_key = cv2.waitKey(1)
                if pressed_key in [ord("q"), 27]:
                    break

        print("Finished.")
        self.cap.release()


def main():
    cap = KittiDataset(config.BASE_PATH, sequence=config.KITTI_SEQUENCE)
    K0, K1, K2, K3 = cap.get_intrinsics()
    gt_poses = cap.get_poses()
    P0, P1, P2, P3 = cap.get_projection_mats()

    processor = Processor(
        K0=K0,
        distortion_coeffs=None,
        detector_name=config.DETECTOR,
        target_width=config.W_TARGET,
        min_inliers=config.MIN_INLIERS,
        e_svd_rtol=config.E_SVD_RTOL,
        e_svd_atol=config.E_SVD_ATOL,
        **config.DETECTOR_PARAMS,
    )

    _map = Map(
        gt_poses=gt_poses, max_length=500, window_size=(1500, 720), follow_distance=1
    )
    vo = VisualOdometry(
        processor,
        _map,
        cap,
        P2=P2,
        min_matches=config.MIN_MATCHES,
        min_parallax_deg=config.MIN_PARALLAX_DEG,
        blur_threshold=config.BLUR_THRESHOLD,
    )
    vo.main()


if __name__ == "__main__":
    main()
