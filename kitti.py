# External imports
import cv2
import pykitti
from PIL.Image import Image


class KittiDataset:
    def __init__(self, base_path, sequence="04"):
        self.odom = pykitti.odometry(base_path, sequence)
        self.cam0 = self.odom.cam0  # monochrome left
        self.cam1 = self.odom.cam1  # monochrome right
        self.cam2 = self.odom.cam2  # rgb left
        self.is_opened = True

    def isOpened(self):
        return self.is_opened

    def release(self):
        pass

    def read(self) -> tuple[bool, Image | None, Image | None]:
        try:
            return True, next(self.cam0), next(self.cam2)
        except StopIteration:
            return True, None, None

    def get_intrinsics(self):
        """ """

        # Projection matrices of the two gray cameras
        # P_cam_0 = self.odom.calib.P_rect_00
        # P_cam_1 = self.odom.calib.P_rect_10

        # Extract the intrinsic (K) and extrinsic (R + t) matrices
        # Note how these cameras are positioned w.r.t cam0 (gray)
        # K0, R0, t0, _, _, _, _ = cv2.decomposeProjectionMatrix(P_cam_0)
        # K1, R1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P_cam_1)

        K0 = self.odom.calib.K_cam0
        K1 = self.odom.calib.K_cam1

        return K0, K1

    def get_poses(self):
        """ """
        return self.odom.poses


if __name__ == "__main__":
    base_path = r"C:\Users\Manuel\Desktop\Documentos\1.PROJECTS\COMPUTER VISION\Monocular SLAM\kitti_data_sample"
    odom = pykitti.odometry(base_path, "04")

    # Projection matrices of the two color cameras
    P_cam_2 = odom.calib.P_rect_20
    P_cam_3 = odom.calib.P_rect_30

    # Extract the intrinsic (K) and extrinsic (R + t) matrices
    # Note how these cameras are positioned w.r.t cam0 (gray)
    K2, R2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(P_cam_2)
    K3, R3, t3, _, _, _, _ = cv2.decomposeProjectionMatrix(P_cam_3)

    print(f"Camera2 R:\n{R2}")
    print(f"Camera2 t:\n{t2 / t2[3]}")
