# Standard Library imports
from collections import deque
from threading import Event, Thread

# External imports
import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy.typing as npt


class Map:
    """
    3D visualization manager for monocular SLAM using Open3D GUI API.

    Manages camera poses and 3D points, rendering them via an Open3D GUI
    viewer. The viewer displays camera trajectory as connected coordinate
    frames and accumulated map points as a point cloud.

    The GUI runs on the main thread. Processing (the callback passed to
    create_viewer) runs in a background thread and communicates scene
    updates via post_to_main_thread.

    Attributes:
        cam_poses: List of 4x4 camera pose matrices.
        points_3D: List of 3D point arrays to visualize.

    Based on: https://learnopencv.com/monocular-slam-in-python/
    """

    PANEL_POSITIONS = ("top", "bottom", "left", "right")

    def __init__(
        self,
        panel_position="top",
        gt_poses=None,
        max_length=100,
        window_size=(1280, 720),
    ):
        """
        Args:
            panel_position: Where to place the image panel relative to the 3D viewer.
                One of "top", "bottom", "left", "right". Defaults to "top".
            window_size: (width, height) in pixels for the viewer window.
        """
        if panel_position not in self.PANEL_POSITIONS:
            raise ValueError(
                f"panel_position must be one of {self.PANEL_POSITIONS}, got '{panel_position}'"
            )
        self.cam_poses = []
        self.points_3D = deque(maxlen=max_length)
        self.point_colors = deque(maxlen=max_length)
        self.is_first_frame = True
        self.panel_position = panel_position
        self.window_size = window_size

        # GUI references (initialized in create_viewer)
        self.app = None
        self.window = None
        self.widget = None

        # Materials (initialized in create_viewer)
        self.mat_point = None
        self.mat_line = None
        self.mat_mesh = None

        # Image panel (initialized in create_viewer)
        self.panel = None
        self.image_widget = None
        self.rgb_image_widget = None

        # Controls panel (initialized in create_viewer)
        self.controls_panel = None

        # Trajectory state
        self.trajectory_points = []
        self.frame_count = 0
        self.gt_frame_count = 0

        self.gt_poses = gt_poses

        # Pause control: starts paused, spacebar toggles
        self._running = Event()
    def create_viewer(self, callback):
        """
        Initialize the Open3D GUI viewer and launch processing in a background thread.

        Args:
            callback: A callable (the SLAM processing loop) to run in a
                background thread. It should call self.display() to push
                updates to the viewer.
        """

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("3D Viewer", *self.window_size)

        # SceneWidget + Open3DScene
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self.widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS,
            [0.577, -0.577, -0.577],
        )

        # Materials
        self.mat_point = rendering.MaterialRecord()
        self.mat_point.shader = "defaultUnlit"
        self.mat_point.point_size = 3.0

        self.mat_line = rendering.MaterialRecord()
        self.mat_line.shader = "unlitLine"
        self.mat_line.line_width = 5.0

        self.mat_mesh = rendering.MaterialRecord()
        self.mat_mesh.shader = "defaultLit"

        # Origin coordinate frame
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        origin.compute_vertex_normals()
        self.widget.scene.add_geometry("origin", origin, self.mat_mesh)

        # Image panel
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        # Labels row
        labels_row = gui.Horiz(0.5 * em)
        labels_row.add_child(gui.Label("RGB"))
        labels_row.add_stretch()
        labels_row.add_child(gui.Label("Matches"))
        labels_row.add_stretch()
        self.panel.add_child(labels_row)

        # Images side by side
        images_row = gui.Horiz(0.5 * em)
        self.rgb_image_widget = gui.ImageWidget()
        images_row.add_child(self.rgb_image_widget)
        self.image_widget = gui.ImageWidget()
        images_row.add_child(self.image_widget)
        self.panel.add_child(images_row)

        # Controls panel
        self.controls_panel = self._build_controls_panel(em, margin)

        self.widget.set_view_controls(gui.SceneWidget.Controls.FLY)

        # Layout
        self.window.add_child(self.widget)
        self.window.add_child(self.panel)
        self.window.add_child(self.controls_panel)
        self.window.set_on_layout(self._on_layout)

        # Enable rotation around clicked point
        self.widget.set_on_mouse(self._on_mouse)

        # Spacebar toggles pause/resume
        self.window.set_on_key(self._on_key)

        # Initial camera (Z-forward, Y-down for KITTI)
        bbox = o3d.geometry.AxisAlignedBoundingBox([-50, -50, -50], [50, 50, 50])
        self.widget.setup_camera(60.0, bbox, [0, 0, 0])
        self.widget.look_at(
            [0, 0, -1],  # center: look along -Z
            [0, -10, -10],  # eye: above and behind
            [0, -1, 0],  # up: -Y (Y points down in KITTI)
        )

        # Launch processing in background thread, then app.run() blocks on GUI
        thread = Thread(target=callback, daemon=True)
        thread.start()
        self.app.run()  # does not return until the window is closed

    def display(self):
        """
        Send the latest pose and accumulated points to the viewer.

        Called from the background processing thread. Schedules a scene
        update on the main thread via post_to_main_thread.
        """

        if self.window is None:
            return

        # Gather camera poses and map points (pre-pack as contiguous Nx3 float64
        # on the background thread so the main/UI thread gets fast memcpy into Open3D)
        pose = np.array(self.cam_poses[-1])
        if len(self.points_3D) > 0:
            pts = np.ascontiguousarray(np.hstack(self.points_3D), dtype=np.float64)
            colors = np.ascontiguousarray(
                np.hstack(self.point_colors), dtype=np.float64
            )
        else:
            pts = np.array([])
            colors = np.array([])

        # The scene must be updated in the UI thread (the main thread)
        # https://github.com/isl-org/Open3D/issues/2300#issuecomment-689196594
        # https://github.com/isl-org/Open3D/blob/f38360cf/examples/python/visualization/video.py#L112
        self.app.post_to_main_thread(
            self.window, lambda: self._update_scene(pose, pts, colors)
        )

    def _update_scene(self, pose: npt.NDArray, pts: npt.NDArray, colors: npt.NDArray):
        """
        Update the 3D scene with new point cloud, trajectory, and camera frame. Runs on the main thread.
        """

        scene = self.widget.scene

        # Update point cloud
        if pts.size > 0:
            # pts is 3xN float64 (pre-packed in display()); transpose to Nx3.
            # np.ascontiguousarray ensures the C-contiguous layout that
            # Vector3dVector needs for its fast bulk-copy path.
            pts_3d = np.ascontiguousarray(pts.T)
            colors_3d = np.ascontiguousarray(colors.T)

            # Distance-based filtering: remove points too far from the current camera
            cam_pos = pose[:3, 3]
            dists = np.linalg.norm(pts_3d - cam_pos, axis=1)
            median_dist = np.median(dists)
            dist_mask = dists < median_dist * 5.0
            pts_3d = np.ascontiguousarray(pts_3d[dist_mask])
            if colors_3d.size > 0:
                colors_3d = np.ascontiguousarray(colors_3d[dist_mask])

            if len(pts_3d) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_3d)
                if colors_3d.size > 0 and len(colors_3d) == len(pts_3d):
                    pcd.colors = o3d.utility.Vector3dVector(colors_3d)

                pcd = pcd.voxel_down_sample(voxel_size=0.5)
                if scene.has_geometry("map_points"):
                    scene.remove_geometry("map_points")
                scene.add_geometry("map_points", pcd, self.mat_point)

        # Add current camera position to trajectory
        t = pose[:3, 3]
        self.trajectory_points.append(t)

        # Update trajectory line
        if len(self.trajectory_points) > 1:
            line_set = o3d.geometry.LineSet()
            lines = [[i, i + 1] for i in range(len(self.trajectory_points) - 1)]
            line_set.points = o3d.utility.Vector3dVector(self.trajectory_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(lines))
            if scene.has_geometry("trajectory"):
                scene.remove_geometry("trajectory")
            scene.add_geometry("trajectory", line_set, self.mat_line)

        # Update ground truth trajectory
        if (self.gt_poses is not None) and len(self.trajectory_points) <= len(
            self.gt_poses
        ):
            gt_points = [p[:3, 3] for p in self.gt_poses[: len(self.trajectory_points)]]
            if len(gt_points) > 1:
                gt_line_set = o3d.geometry.LineSet()
                gt_lines = [[i, i + 1] for i in range(len(gt_points) - 1)]
                gt_line_set.points = o3d.utility.Vector3dVector(gt_points)
                gt_line_set.lines = o3d.utility.Vector2iVector(gt_lines)
                gt_line_set.colors = o3d.utility.Vector3dVector(
                    [[0, 1, 0]] * len(gt_lines)  # RGB: green
                )
                if scene.has_geometry("gt_trajectory"):
                    scene.remove_geometry("gt_trajectory")
                scene.add_geometry("gt_trajectory", gt_line_set, self.mat_line)

        if (self.gt_poses is not None) and len(self.trajectory_points) <= len(
            self.gt_poses
        ):
            # Add coordinate frame at ground truth pose (yellow to distinguish from estimated)
            gt_idx = len(self.trajectory_points) - 1
            gt_pose = self.gt_poses[gt_idx]
            if len(self.trajectory_points) == 1:
                gt_frame = self.create_frame(gt_pose, size=2.0)
                gt_frame.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
                gt_frame.compute_vertex_normals()
                scene.add_geometry(
                    f"gt_frame_{self.gt_frame_count}", gt_frame, self.mat_mesh
                )
                self.gt_frame_count += 1
            elif len(self.trajectory_points) % 5 == 0:
                gt_frame = self.create_frame(gt_pose)
                gt_frame.paint_uniform_color([1.0, 1.0, 0.0])
                gt_frame.compute_vertex_normals()
                scene.add_geometry(
                    f"gt_frame_{self.gt_frame_count}", gt_frame, self.mat_mesh
                )
                self.gt_frame_count += 1

        # Add coordinate frame at camera pose
        if self.is_first_frame:
            frame = self.create_frame(pose, size=2.0)
            frame.compute_vertex_normals()
            scene.add_geometry(f"frame_{self.frame_count}", frame, self.mat_mesh)
            self.is_first_frame = False
            self.frame_count += 1
        elif len(self.trajectory_points) % 5 == 0:
            frame = self.create_frame(pose)
            frame.compute_vertex_normals()
            scene.add_geometry(f"frame_{self.frame_count}", frame, self.mat_mesh)
            self.frame_count += 1

        self.window.post_redraw()

    def _build_controls_panel(self, em, margin):
        """Build the controls panel with camera mode, scene toggles, and point size."""
        panel = gui.Vert(0.5 * em, gui.Margins(margin))
        panel.add_child(gui.Label("Controls"))

        # Camera mode
        panel.add_child(gui.Label("Camera Mode"))
        mode_combo = gui.Combobox()
        camera_modes = [
            ("Fly", gui.SceneWidget.Controls.FLY),
            ("Rotate Camera", gui.SceneWidget.Controls.ROTATE_CAMERA),
        ]
        for label, _ in camera_modes:
            mode_combo.add_item(label)

        def on_camera_mode(text, index):
            self.widget.set_view_controls(camera_modes[index][1])

        mode_combo.set_on_selection_changed(on_camera_mode)
        panel.add_child(mode_combo)

        # Show axes
        axes_cb = gui.Checkbox("Show Axes")
        axes_cb.checked = False
        axes_cb.set_on_checked(lambda checked: self.widget.scene.show_axes(checked))
        panel.add_child(axes_cb)

        # Show ground plane
        ground_cb = gui.Checkbox("Show Ground Plane")
        ground_cb.checked = False
        ground_cb.set_on_checked(
            lambda checked: self.widget.scene.show_ground_plane(
                checked, rendering.Scene.GroundPlane.XZ
            )
        )
        panel.add_child(ground_cb)

        # Point size
        panel.add_child(gui.Label("Point Size"))
        size_slider = gui.Slider(gui.Slider.DOUBLE)
        size_slider.set_limits(1.0, 10.0)
        size_slider.double_value = self.mat_point.point_size

        def on_point_size(value):
            self.mat_point.point_size = value

        size_slider.set_on_value_changed(on_point_size)
        panel.add_child(size_slider)

        return panel

    def _on_mouse(self, event):
        """On left-button down, pick the 3D point under the cursor and set it as the center of rotation."""
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
            gui.MouseButton.LEFT
        ):
            # Capture coordinates at event time
            click_x, click_y = event.x, event.y

            def on_depth(depth_image):
                try:
                    arr = np.asarray(depth_image)
                    h, w = arr.shape[:2]
                    x = int(click_x)
                    y = int(click_y)
                    if 0 <= x < w and 0 <= y < h:
                        depth = arr[y, x]
                        if depth < 1.0:
                            world_pt = self.widget.scene.camera.unproject(
                                x,
                                y,
                                depth,
                                self.widget.frame.width,
                                self.widget.frame.height,
                            )
                            self.widget.center_of_rotation = world_pt
                except Exception:
                    pass  # silently ignore pick failures

            self.widget.scene.scene.render_to_depth_image(on_depth)
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key(self, event):
        """Toggle pause/resume when spacebar is pressed."""
        if event.key == gui.KeyName.SPACE and event.type == gui.KeyEvent.UP:
            if self._running.is_set():
                self._running.clear()
                print("PAUSED — press Space to resume")
            else:
                self._running.set()
                print("RESUMED")
            return True
        return False

    def wait_if_paused(self):
        """Block the calling thread until the pipeline is unpaused."""
        self._running.wait()

    def _on_layout(self, layout_context, panel_ratio=0.35, controls_width=200):
        """Layout the SceneWidget, image panel, and controls panel.

        Args:
            panel_ratio: Fraction of the window dimension allocated to the image panel (0.0–1.0).
            controls_width: Fixed pixel width for the controls panel on the right.
        """
        r = self.window.content_rect
        pos = self.panel_position

        # Controls panel: fixed width on the right
        cw = min(controls_width, int(r.width * 0.2))
        self.controls_panel.frame = gui.Rect(r.x + r.width - cw, r.y, cw, r.height)

        # Remaining area for image panel + 3D viewer
        main_w = r.width - cw

        if pos in ("top", "bottom"):
            panel_size = int(r.height * panel_ratio)
        else:
            panel_size = int(main_w * panel_ratio)

        if pos == "top":
            self.panel.frame = gui.Rect(r.x, r.y, main_w, panel_size)
            self.widget.frame = gui.Rect(
                r.x, r.y + panel_size, main_w, r.height - panel_size
            )
        elif pos == "bottom":
            self.widget.frame = gui.Rect(r.x, r.y, main_w, r.height - panel_size)
            self.panel.frame = gui.Rect(
                r.x, r.y + r.height - panel_size, main_w, panel_size
            )
        elif pos == "left":
            self.panel.frame = gui.Rect(r.x, r.y, panel_size, r.height)
            self.widget.frame = gui.Rect(
                r.x + panel_size, r.y, main_w - panel_size, r.height
            )
        else:  # right
            self.widget.frame = gui.Rect(r.x, r.y, main_w - panel_size, r.height)
            self.panel.frame = gui.Rect(
                r.x + main_w - panel_size, r.y, panel_size, r.height
            )

    def update_images(self, matches_image, rgb_image):
        """Update both the matches and RGB image panels. Called from the background thread."""
        if self.window is None:
            return
        matches_rgb = cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)
        o3d_matches = o3d.geometry.Image(matches_rgb)
        o3d_rgb = o3d.geometry.Image(rgb_image)
        self.app.post_to_main_thread(
            self.window,
            lambda m=o3d_matches, r=o3d_rgb: self._do_update_images(m, r),
        )

    def _do_update_images(self, o3d_matches, o3d_rgb):
        """Replace both panel images. Runs on the main thread."""
        self.image_widget.update_image(o3d_matches)
        self.rgb_image_widget.update_image(o3d_rgb)
        self.window.post_redraw()

    @staticmethod
    def create_frame(transform, size=0.75):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(transform)
        return frame
