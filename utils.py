import numpy as np
import cv2
import PIL


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


def homogeneous_to_euclidean(points):
    """
    Convert points from homogeneous coordinates to Euclidean coordinates.

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

    return euclidean_points


def check_rotation_mat(R: np.ndarray, EPS=1e-7) -> bool:
    if abs(np.linalg.det(R) - 1.0) > EPS:
        print("Rotation matrix is invalid")
        return False
    return True


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
