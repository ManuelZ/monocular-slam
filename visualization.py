# Standard Library imports
from multiprocessing import Process, Queue

# External imports
import numpy as np
import open3d as o3d


class Map:
    """
    Modified from https://learnopencv.com/monocular-slam-in-python/
    """

    def __init__(self):
        self.cam_poses = []
        self.points_3D = []
        self.state = None
        self.queue = None  # for inter-process communication
        self.curr_pose = None
        self.prev_pose = None

    def create_viewer(self):
        """ """

        # Queue for inter-process communication
        self.queue = Queue()

        # Run viewer in parallel
        p = Process(target=self.viewer_thread, args=(self.queue,))

        # The process will terminate when the main program exits
        # p.daemon = True

        p.start()

    def viewer_thread(self, queue):
        """ """

        print("Starting viewer thread in a new process.")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Viewer", width=1280, height=720)

        # Add coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coordinate_frame)

        # Initialize point cloud and line set for visualization
        point_cloud = o3d.geometry.PointCloud()
        vis.add_geometry(point_cloud)

        line_set = o3d.geometry.LineSet()
        vis.add_geometry(line_set)

        while vis.poll_events():

            if not queue.empty():
                pose, pts = queue.get()

                self.prev_pose = self.curr_pose
                self.curr_pose = pose

                # # Update point cloud
                if pts.size > 0:
                    # Vector3dVector expects shape (n,3)
                    point_cloud.points = o3d.utility.Vector3dVector(pts.T)
                    vis.update_geometry(point_cloud)

                # Add frame representing a pose
                frame = self.create_frame(pose)
                vis.add_geometry(frame)

                # Connect the previous and current frames with a line
                if self.prev_pose is not None:
                    previous_point = self.prev_pose[:3, 3]
                    current_point = self.curr_pose[:3, 3]
                    line = self.create_line(previous_point, current_point)
                    vis.add_geometry(line)

                vis.update_renderer()

    def display(self):
        """ """

        if self.queue is None:
            return

        # Gather camera poses and map points
        pose = np.array(self.cam_poses[-1])

        # Add all the points each time
        if len(self.points_3D) > 0:
            pts = np.hstack(self.points_3D)
        else:
            pts = np.array([])

        self.queue.put((pose, pts))

    @staticmethod
    def create_frame(transform, size=0.75):
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=transform[:3, 3])

    @staticmethod
    def create_line(p1, p2, color=[0, 0, 1]):
        """
        Function to create a line connecting two points
        """
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([p1, p2]),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        line_set.colors = o3d.utility.Vector3dVector([color])
        return line_set
