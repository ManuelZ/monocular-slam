# Standard Library imports
from multiprocessing import Process, Queue

# External imports
import PIL.Image
import cv2
import numpy as np
import open3d as o3d
import PIL
from imutils.feature import FeatureDetector_create
import spatialmath.base as sm

# Local imports
import config
from kitti import KittiDataset
from visualization import Map


print(f"OpenCV version: {cv2.__version__}")


def pil_to_np(image_pil):
    """ """
    pil_data = image_pil.convert("RGB")
    return np.array(pil_data)[:, :, ::-1].copy()


def variance_of_laplacian(image) -> float:
    """
    From: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    # Compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    if isinstance(image, PIL.Image.Image):
        image = pil_to_np(image)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def ratio_test(matches, list=False):
    """
    Test by D.Lowe
    Source: https://stackoverflow.com/a/51020550/1253729
    """
    good = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
            good_without_list.append(m)
    if list:
        return good
    return good_without_list


def pixel2cam(p, K):
    """
    Convert pixel coordinates to normalized camera coordinates using the intrinsic matrix K.

    Parameters:
    - p: A tuple or list representing the pixel coordinates (u, v).
    - K: A 3x3 intrinsic camera matrix.

    Returns:
    - A tuple representing the normalized camera coordinates (x', y').
    """
    x_normalized = (p[0] - K[0, 2]) / K[0, 0]
    y_normalized = (p[1] - K[1, 2]) / K[1, 1]
    return (x_normalized, y_normalized)


def vec_to_skew_symmetric(v):
    """
    Convert a 3D vector to a 3x3 skew-symmetric matrix.

    Parameters
    ----------
    v : array-like, shape (3,1)
        Input 3D vector.

    Returns
    -------
    skew_matrix : ndarray, shape (3, 3)
        The corresponding skew-symmetric matrix.
    """
    assert v.shape == (3, 1)
    x = v[0][0]
    y = v[1][0]
    z = v[2][0]
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def skew_symmetric_to_vec(M):
    """
    Convert a 3x3 skew-symmetric matrix to a 3D vector.

    Parameters
    ----------
    M : ndarray, shape (3, 3)
        Input 3x3 skew-symmetric matrix.

    Returns
    -------
    v : ndarray, shape (3,)
        The corresponding 3D vector.
    """
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class Processor:
    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray | None = None,
        detector_name="orb",
        **kwargs,
    ):
        self.current = {"frame": None, "kps": None, "des": None}
        self.prev = {"frame": None, "kps": None, "des": None}
        self.detector = FeatureDetector_create(detector_name, **kwargs)
        self.computer = FeatureDetector_create(detector_name)
        self.resize_ratio = None
        self.K = K  # intrinsic matrix
        self.K_is_resized = False
        self.distortion_coeffs = dist

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
        """
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.distortion_coeffs, (w, h), 1, (w, h)
        )
        return cv2.undistort(frame, self.K, self.distortion_coeffs, None, newcameramtx)

    def detect_and_compute(self, frame):
        """ """

        # TODO: revise if I need to do this

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

    def draw_keypoints(self, frame, kp, color=(0, 255, 0)):
        return cv2.drawKeypoints(
            frame,
            kp,
            outImage=None,
            color=color,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

    def get_matched_points(self, kps, des, method="brute_force_ratio_test"):
        """
        Filter keypoints based on the provided matches and return matched points.

        This method extracts matched keypoints from two sets of keypoints: the
        previous frame and the current frame. It returns two lists containing
        only the keypoints that have corresponding matches.
        """
        if config.DEBUG:
            print("Getting points...")

        matches = self.match_descriptors(des, method=method)

        indices = map(lambda x: (x.queryIdx, x.trainIdx), matches)

        frameA_points = []
        frameB_points = []
        for queryIdx, trainIdx in indices:
            frameA_points.append(self.prev["kps"][queryIdx].pt)
            frameB_points.append(kps[trainIdx].pt)

        frameA_points = np.array(frameA_points, dtype=np.float32)
        frameB_points = np.array(frameB_points, dtype=np.float32)

        return (frameA_points, frameB_points)

    def pts_to_kps(self, points):
        return [cv2.KeyPoint(p[0], p[1], 1) for p in points]

    def match_descriptors(self, desB, method):
        """ """

        if config.DEBUG:
            print("Matching points...")

        if method == "brute_force_ratio_test":
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.prev["des"], desB, k=2)
            return ratio_test(matches)

        elif method == "flann":
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(self.prev["des"], desB, k=2)
            return ratio_test(matches)

        else:  # brute_force
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.prev["des"], desB)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches

    def pose_estimation_2d(
        self, kpA: np.ndarray[np.float32], kpB: np.ndarray[np.float32]
    ):
        """
        Parameters
        ----------
        kpA, kpB are matched keypoints
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # # TODO: Do I need to use RANSAC  here?
        # fundamental_matrix, inliers = cv2.findFundamentalMat(
        #     kpA, kpB, method=cv2.USAC_MAGSAC
        # )

        essential_matrix, inliers = cv2.findEssentialMat(
            kpA, kpB, fx, (cx, cy), method=cv2.USAC_MAGSAC
        )

        # From: https://youtu.be/6KIPusOv5fA?list=PLggLP4f-rq01NLHOh2vVPPJZ0rxkbVFNc
        #
        # A rotation matrix can be used for three purposes (Lynch & Park, 2017, p. 71):
        #     - To represent an orientation
        #     - To change the reference frame in which a vector or a frame is represented (as an operator)
        #     - To rotate a vector or a frame (as an operator)
        # To express a vector in a different rotated frame of reference, two rotation approaches can produce the same outcome:
        #     Keep the vector fixed while rotating the Frame of reference θ degrees.
        #     Rotate the vector -θ degrees while keeping the Frame of reference fixed.
        #
        # Here, the recovered pose can be used to change the reference frame in which points are expressed.
        # Specifically, it allows to re-express the points that are originally expressed in camera 2, into camera 1.
        # Does it?

        #
        # Apparently, the essential matrix cannot use points that are coplanar?
        #
        
        # `.recoverPose` uses `.decomposeEssentialMat`, its documentation says:
        #
        #     In general, four possible poses exist for the decomposition of E. (...) any of the tuples (R1, t), 
        #     (R1, -t), (R2, t), (R2, -t) is a change of basis from the first camera's coordinate system to the second 
        #     camera's coordinate system.
        #
        # The obtained transformation will act as an operator that will rotate/translate the frame of reference of the 
        # camera at t=0 into the frame of reference of the camera at t=1? Or is it just the new pose?
        # What is obtained? A frame relative to another frame, right?
        #
        # What recoverPose does is giving the pose of frame A expressed in frame B: T_B_A
        #
        retval, R, t, inliers = cv2.recoverPose(essential_matrix, kpA, kpB, self.K)

        if config.DEBUG:
            t_skew = vec_to_skew_symmetric(t)

            # E has scale equivalence; thus, t and R could also have scale equivalence. However, R has its own constraints,
            # so only t can be multiplied by any non-zero constant while keeping E = t^R valid.
            # t has been normalized to a length of 1, but the actual length is not known.
            # Note: `t^` means "t using the form of a skew-symmetric matrix".

            # Verify that neither E or t^R are equal to zero. That would be a way (an incorrect way) of satisfying the
            # epipolar constraints.
            print("E: \n", essential_matrix)
            print("E = t^R:\n", t_skew @ R)
            print(
                "E/(t^R): (this should be a constant!)\n",
                essential_matrix / (t_skew @ R),
            )

            for pt1, pt2 in zip(kpA, kpB):

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

        Inliers is a bool matrix where 1 represents an inlier.
        """
        # _, inliers = cv2.estimateAffinePartial2D(prev_points, points, method=cv2.RANSAC)
        _, inliers = cv2.findEssentialMat(
            prev_points, points, self.K, method=cv2.USAC_MAGSAC
        )

        prev_kps = self.pts_to_kps(prev_points)
        kps = self.pts_to_kps(points)

        # Draw keypoints in the images (before matching)
        composite = cv2.merge((self.prev["frame"], current_frame, self.prev["frame"]))
        composite = self.draw_keypoints(composite, prev_kps, (0, 0, 255))
        composite = self.draw_keypoints(composite, kps, (255, 0, 0))

        # Draw lines between keypoints
        for i, (k1, k2) in enumerate(zip(prev_kps, kps)):

            if inliers[i, :] == 0:
                continue

            pt1 = (int(k1.pt[0]), int(k1.pt[1]))
            pt2 = (int(k2.pt[0]), int(k2.pt[1]))

            composite = cv2.line(composite, pt1, pt2, (255, 0, 0), 1)

        if cvshow:
            cv2.imshow("Composite", composite)

        num_inliers = np.sum(inliers, axis=0)[0]
        if config.DEBUG:
            print(f"Inliers: {num_inliers}")

        if num_inliers < 10:
            print(f"Too few inliers: {num_inliers}")

        return composite


class VisualOdometry:
    def __init__(self, processor, map):
        self.processor = processor
        self.map = map


def main(input):
    if isinstance(input, str):
        cap = cv2.VideoCapture(input)
    else:
        cap = input

    # w_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    K0, K1 = cap.get_intrinsics()

    mapp = Map()
    mapp.create_viewer()

    orb_parameters = {"nfeatures": 1500}
    processor = Processor(K0, dist=None, detector_name="orb", **orb_parameters)

    # Initial pose
    pose = np.eye(4)
    pose[:3, :3] = np.eye(3)
    pose[:3, 3] = np.array([0, 0, 0]).T

    while cap.isOpened():
        frame_ok, frame = cap.read()

        if frame is None:
            print("No frame received")
            break

        if not frame_ok:
            print("Problem reading frame")
            continue

        if blur_val := variance_of_laplacian(frame) < 25:
            print(f"Skipping blurry frame: {blur_val:.2f} VoL")
            continue

        preprocessed_frame = processor.preprocess_frame(frame)
        kps, des = processor.detect_and_compute(preprocessed_frame)

        if processor.prev["frame"] is not None:
            prev_matched_pts, curr_matched_pts = processor.get_matched_points(kps, des)

            processor.show_matches(
                preprocessed_frame, prev_matched_pts, curr_matched_pts
            )

            # What is R? A transformation from the pose of camera0 in the second time step to the first time step?
            #  - Yes, I think so. A transformation to change the reference frame in which a vector/frame is represented
            #
            # How do I express them in the world frame?
            # - The world frame is located in the origin of the camera 0 in the first time step.
            # -
            #
            # This pose, goes from frame 1 to frame 0 or from frame 0 to frame 1?
            R, t = processor.pose_estimation_2d(prev_matched_pts, curr_matched_pts)

            # What is monocular initialization? Where does it happen? Does it happen only once?
            #
            # I think is just doing the pose estimation, triangulation and putting them in an empty map?
            # the global coordinate frame can be considerede, for example, the coordinate frame of the camera's first
            # position

            # Construct transformation matrix
            T = sm.rt2tr(R, t)
            T_inv = np.linalg.inv(T)
            pose = pose @ T_inv

            mapp.poses.append(pose)

        processor.prev = {"frame": preprocessed_frame, "kps": kps, "des": des}

        pressed_key = cv2.waitKey(1)
        if pressed_key in [ord("q"), 27]:
            break

    print("Finished.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    base_path = r"F:\DATASETS\KITTI\dataset"
    cap = KittiDataset(base_path, "04")

    main(cap)
