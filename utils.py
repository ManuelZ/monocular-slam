import numpy as np
import cv2
import PIL
import numpy.typing as npt
from PIL.Image import Image


def pil_to_np(image_pil: Image) -> npt.NDArray:
    """ """
    if image_pil.mode == "L":
        return np.array(image_pil)
    pil_data = image_pil.convert("RGB")
    return np.array(pil_data).copy()


def variance_of_laplacian(image) -> float:
    """
    From: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    # Compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    if isinstance(image, PIL.Image.Image):
        image = pil_to_np(image)
    return cv2.Laplacian(image, cv2.CV_64F).var()


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


def GetTriangulatedPts(img1pts, img2pts, K, R, t):
    """From SFM_Starter.ipynb by Jeremy Cohen."""
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:, 0, :]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:, 0, :]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:, 0, :]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:, 0, :]

    pts4d = cv2.triangulatePoints(
        np.eye(3, 4), np.hstack((R, t)), img1ptsNorm.T, img2ptsNorm.T
    )
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:, 0, :]

    return pts3d


def homogeneous_to_euclidean(points: npt.NDArray):
    """
    Convert points from Homogeneous coordinates to Euclidean coordinates.

    Args:
        points (np.ndarray): A 4xN array of points in homogeneous coordinates.

    Returns:
        np.ndarray: A 3xN array of points in Euclidean coordinates.
    """
    # Check if the input has 4 rows (for homogeneous coordinates)
    if points.shape[0] != 4:
        raise ValueError("Input should be a 4xN array of homogeneous coordinates.")

    # Divide the first three rows by the fourth row
    euclidean_points = points[:3, :] / points[3, :]

    assert euclidean_points.shape[0] == 3

    return euclidean_points


def check_rotation_mat(R: np.ndarray, EPS=1e-7) -> bool:
    """Check if matrix is a valid rotation matrix by verifying det(R) = 1."""
    # TODO: anything else to check?
    if abs(np.linalg.det(R) - 1.0) > EPS:
        print("Rotation matrix is invalid")
        return False
    return True


def ratio_test(matches, threshold=0.7, as_list=False):
    """
    Test by D.Lowe
    """
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)
    if as_list:
        return [[m] for m in good]
    return good


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


def undistort(frame, K, distortion_coeffs):
    """Correct lens distortion on a full frame using the camera intrinsic matrix and distortion coefficients.

    Computes an optimal new camera matrix via ``cv2.getOptimalNewCameraMatrix``
    (with ``alpha=1`` to retain all original pixels) and applies ``cv2.undistort``
    to remap the image.

    Args:
        frame: BGR image as a NumPy array (H, W, 3).

    Returns:
        Undistorted image with the same shape and dtype as *frame*.

    Note:
        For undistorting a sparse set of points rather than a full raster image,
        use ``cv2.undistortPoints`` instead.

    See Also:
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html#autotoc_md1177
    """
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        K, distortion_coeffs, (w, h), 1, (w, h)
    )
    return cv2.undistort(frame, K, distortion_coeffs, None, newcameramtx)
