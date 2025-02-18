# Standard Library imports
import time

# External imports
import cv2
import numpy as np
import PIL
import PIL.Image
from imutils.feature import FeatureDetector_create
from spatialmath import SE3

# Local imports
import config
from kitti import KittiDataset
from visualization import Map
from utils import (
    pil_to_np,
    variance_of_laplacian,
    vec_to_skew_symmetric,
    homogeneous_to_euclidean,
    check_rotation_mat,
    ratio_test,
    pixel2cam
)

print(f"OpenCV version: {cv2.__version__}")


class Processor:
    """
    Handle image processing operations for visual SLAM, including feature detection, matching, and pose estimation.
    """

    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray | None = None,
        detector_name="orb",
        **kwargs,
    ):
        self.current = {"frame": None, "kps": None, "des": None}
        self.prev = {"frame": None, "kps": None, "des": None, "pose": None}
        self.detector = FeatureDetector_create(detector_name, **kwargs)
        self.computer = FeatureDetector_create(detector_name)
        self.resize_ratio = None
        self.K = K  # intrinsic matrix
        self.K_is_resized = False
        self.distortion_coeffs = dist
        self.E = None  # Essential matrix, filled in pose_estimation_2d
        self.inliers = None  # Inliers found through RANSAC or similar

    def resize_frame(self, frame, w_target: int):
        """ """
        h, w = frame.shape[:2]
        self.resize_ratio = w_target / w
        h_target = int(h * self.resize_ratio)
        w_target = int(w_target)
        new_dim = (w_target, h_target)
        if config.DEBUG:
            print(f"Resizing from (H,W) {h}x{w} to {h_target}x{w_target}")

        # From the docs:
        # To shrink an image, it will generally look best with INTER_AREA interpolation,
        # to enlarge an image, it will generally look best with INTER_CUBIC (slow) or INTER_LINEAR
        # (faster but still looks OK).
        interpolation = cv2.INTER_AREA if self.resize_ratio < 1 else cv2.INTER_CUBIC

        return cv2.resize(frame, new_dim, interpolation=interpolation)

    def undistort(self, frame):
        """
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html#autotoc_md1177

        Note:
        Be aware of the existence of `cv2.undistortPoints`, which operates on a sparse set of points instead of a
        raster image.
        """
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.distortion_coeffs, (w, h), 1, (w, h)
        )
        return cv2.undistort(frame, self.K, self.distortion_coeffs, None, newcameramtx)

    def detect_and_compute(self, frame):
        """ """

        try:
            kps = self.detector.detect(frame, None)
        except Exception as e:
            print("error", e)
            kps = self.detector.detect(frame)

        kps, des = self.computer.compute(frame, kps)
        return kps, des

    def _rescale_intrinsic_matrix(self):
        if not self.K_is_resized:
            self.K[0, 0] *= self.resize_ratio
            self.K[1, 1] *= self.resize_ratio
            self.K[0, 2] *= self.resize_ratio
            self.K[1, 2] *= self.resize_ratio
            self.K_is_resized = True

    def preprocess_frame(self, frame, resize=False):
        """ """

        if config.DEBUG:
            print("Preprocessing...")

        if isinstance(frame, PIL.Image.Image):
            frame = pil_to_np(frame)

        if resize:
            frame = self.resize_frame(frame, config.W_TARGET)
            self._rescale_intrinsic_matrix()
            self.h, self.w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.distortion_coeffs is not None:
            gray = self.undistort(gray)

        return gray

    def draw_keypoints(self, frame: np.ndarray, kp: list[cv2.KeyPoint], color=(0, 255, 0)):
        """
        """
        
        return cv2.drawKeypoints(
            frame,
            kp,
            outImage=None,
            color=color,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

    def get_matched_points(self, curr_kps, curr_des, method="brute_force_ratio_test"):
        """
        Extract matched keypoints from two sets of keypoints: the previous frame and the current frame.
        Return two lists containing only the keypoints that have corresponding matches.
        """

        if config.DEBUG:
            print("Getting points...")

        matches = self.match_descriptors(curr_des, method=method)
        indices = map(lambda x: (x.queryIdx, x.trainIdx), matches)

        prev_frame_points = []
        curr_frame_points = []
        for queryIdx, trainIdx in indices:
            # TODO: instead of using kps maybe use pts? Do I have that?
            prev_frame_points.append(self.prev["kps"][queryIdx].pt)
            curr_frame_points.append(curr_kps[trainIdx].pt)

        prev_frame_points = np.array(prev_frame_points, dtype=np.float32)
        curr_frame_points = np.array(curr_frame_points, dtype=np.float32)

        return (prev_frame_points, curr_frame_points)

    @staticmethod
    def pts_to_kps(points):
        return [cv2.KeyPoint(p[0], p[1], 1) for p in points]

    def match_descriptors(self, curr_des, method: str):
        """ """

        if config.DEBUG:
            print("Matching points...")

        if method == "brute_force_ratio_test":
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.prev["des"], curr_des, k=2)
            return ratio_test(matches)

        elif method == "flann":
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass an empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(self.prev["des"], curr_des, k=2)
            return ratio_test(matches)

        else:  # brute_force
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.prev["des"], curr_des)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches

    def pose_estimation_2d(
        self, pA: np.ndarray[np.float32], pB: np.ndarray[np.float32]
    ):
        """
        Parameters
        ----------
        pA, pB are matched points
        """

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # USAC_MAGSAC: https://opencv.org/blog/evaluating-opencvs-new-ransacs/
        essential_matrix, inliers = cv2.findEssentialMat(
            pA, pB, fx, (cx, cy), method=cv2.USAC_MAGSAC
        )

        self.E = essential_matrix
        self.inliers = inliers

        # https://stackoverflow.com/a/74030087/1253729
        assert np.linalg.matrix_rank(essential_matrix) == 2

        retval, R, t, valid_triangulared_points = cv2.recoverPose(
            essential_matrix, pA, pB, self.K
        )

        if config.DEBUG:
            t_skew = vec_to_skew_symmetric(t)

            # E has scale equivalence; thus, t and R could also have scale equivalence. However, R has its own constraints,
            # so only t can be multiplied by any non-zero constant while keeping E = t^R valid.
            # t has been normalized to a length of 1, but the actual length is not known.
            # Note: `t^` means "t using the form of a skew-symmetric matrix".

            # Verify that neither E nor t^R are equal to zero. That would be a way (an incorrect way) of satisfying the
            # epipolar constraints.
            print("E: \n", essential_matrix)
            print("E = t^R:\n", t_skew @ R)
            print(
                "E/(t^R): (this should be a constant!)\n",
                essential_matrix / (t_skew @ R),
            )

            for pt1, pt2 in zip(pA, pB):

                pt1 = pixel2cam(pt1, self.K)
                y1 = np.array([[pt1[0]], [pt1[1]], [1.0]])  # Homogeneous coordinates

                pt2 = pixel2cam(pt2, self.K)
                y2 = np.array([[pt2[0]], [pt2[1]], [1.0]])  # Homogeneous coordinates

                # Show epipolar constraints.
                # These values should be close to zero, because that's the constraint, an equality to zero:
                # y2.T @ t_skew @ R @ y1 = 0
                d = y2.T @ t_skew @ R @ y1
                print(f"Epipolar constraint = {d}")

        return R, t

    def show_matches(self, current_frame, prev_points, points, cvshow=True):
        """
        """

        # Transform into OpenCV KeyPoints
        prev_kps = self.pts_to_kps(prev_points)
        curr_kps = self.pts_to_kps(points)

        # Draw Keypoints in the images (before matching)
        composite = cv2.merge((self.prev["frame"], current_frame, self.prev["frame"]))
        composite = self.draw_keypoints(composite, prev_kps, (0, 0, 255))
        composite = self.draw_keypoints(composite, curr_kps, (255, 0, 0))

        # Draw lines between keypoints
        for i, (k1, k2) in enumerate(zip(prev_kps, curr_kps)):

            if self.inliers[i, :] == 0:
                continue

            pt1 = (int(k1.pt[0]), int(k1.pt[1]))
            pt2 = (int(k2.pt[0]), int(k2.pt[1]))

            composite = cv2.line(composite, pt1, pt2, (255, 0, 0), 1)

        if cvshow:
            cv2.imshow("Composite", composite)

        num_inliers = np.sum(self.inliers, axis=0)[0]
        if config.DEBUG:
            print(f"Inliers: {num_inliers}")

        if num_inliers < 10:
            print(f"Too few inliers: {num_inliers}")

        return composite


class VisualOdometry:
    def __init__(self, processor: Processor, map, cap):
        self.processor = processor
        self.map = map
        self.cap = cap

    def map_world_points_to_camera(self, prev_matched_pts, curr_matched_pts):
        """Transform points from world coordinates to camera coordinates"""

        prev_pts = []
        curr_pts = []
        for pt1, pt2 in zip(prev_matched_pts, curr_matched_pts):

            pt1 = pixel2cam(pt1, processor.K)
            prev_pts.append(pt1)
            y1 = np.array([[pt1[0]], [pt1[1]], [1.0]])  # Homogeneous coordinates

            pt2 = pixel2cam(pt2, processor.K)
            curr_pts.append(pt2)
            y2 = np.array([[pt2[0]], [pt2[1]], [1.0]])  # Homogeneous coordinates

        prev_pts = np.array(prev_pts).T
        curr_pts = np.array(curr_pts).T

        return prev_pts, curr_pts

    def main(self, cvshow=True):
        """ """

        self.map.create_viewer()

        time.sleep(5)

        homogeneous_K = np.concatenate((processor.K, np.zeros((3, 1))), axis=1)

        # Initial pose
        pose = np.eye(4)
        pose[:3, :3] = np.eye(3)
        pose[:3, 3] = np.array([0, 0, 0]).T

        while self.cap.isOpened():

            frame_ok, frame = self.cap.read()

            if frame is None:
                print("No frame received")
                break

            if not frame_ok:
                print("Problem reading frame")
                continue

            if blur_val := variance_of_laplacian(frame) < 25:
                print(f"Skipping blurry frame: {blur_val:.2f} VoL")
                continue

            preprocessed_frame = self.processor.preprocess_frame(frame)
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
                kps, des
            )

            # Missing undistorting the prev_matched_pts, curr_matched_pts with
            # cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None) ?
            # I don't think so, because the points are undistorted for the whole image in preprocess_frame.

            # The world frame is located in the origin of the camera 0 in the first time step.
            R, t = self.processor.pose_estimation_2d(prev_matched_pts, curr_matched_pts)

            assert check_rotation_mat(R)

            # - Is this the pose of the Camera?
            #   -> This is the pose of Camera1 expressed in Camera2's coordinates.
            #      Camera1 is the previous camera, Camera2 is the current camera.
            # - Are these extrinsics that map from world to camera?
            #   -> The first pose is also the world frame. These extrinsics
            # - How does the multiplication between poses work?
            T_2_1 = SE3.Rt(R, t)

            T_1_2 = T_2_1.inv()
            pose = pose @ T_1_2.A  # 4x4

            if (prev_pose := self.processor.prev.get("pose")) is not None:

                prev_pts, curr_pts = self.map_world_points_to_camera(
                    prev_matched_pts, curr_matched_pts
                )

                # Projection matrices, also known as Camera matrices.
                #
                # They allow to map a world point into pixel coordinates.
                #   - The extrinsic matrix [R|t] allows to map from world coordinates to camera coordinates.
                #   - The intrinsic matrix K allows to map from camera normalized coordinates into pixel coordinates.
                # Note that recoverPose cannot give me the real translation, so [R|t] won't be a real extrinsic matrix.
                # To get a real extrinsic matrix I would need the scale of the world obtained by some measurement with
                # detected tags or with an IMU.
                # FIXME: when I don't use homogeneous_K, I get good results when plotting
                prev_P = homogeneous_K @ prev_pose
                curr_P = homogeneous_K @ pose

                assert prev_P.shape == (3, 4), prev_P.shape
                assert curr_P.shape == (3, 4), curr_P.shape

                hom_points = cv2.triangulatePoints(prev_P, curr_P, prev_pts, curr_pts)
                points = homogeneous_to_euclidean(hom_points)
                self.map.points_3D.append(points)

            self.processor.prev = {
                "frame": preprocessed_frame,
                "kps": kps,
                "des": des,
                "pose": pose,
            }

            # TODO: Fix the triangulated points, they don't look good

            self.map.cam_poses.append(pose)
            self.map.display()

            self.processor.show_matches(
                preprocessed_frame, prev_matched_pts, curr_matched_pts, cvshow
            )

            pressed_key = cv2.waitKey(1)
            if pressed_key in [ord("q"), 27]:
                break

        print("Finished.")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    cap = KittiDataset(config.BASE_PATH, "04")
    K0, K1 = cap.get_intrinsics()

    orb_parameters = {"nfeatures": 1500}
    processor = Processor(K0, dist=None, detector_name="orb", **orb_parameters)

    mapp = Map()
    vo = VisualOdometry(processor, mapp, cap)
    vo.main()
