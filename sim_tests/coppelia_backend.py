from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import numpy as np
import cv2
import array
import math
import time


class CoppeliaBackend:

    def __init__(self):

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")

        # Get simulation objects
        self.sensor = self.sim.getObject("/Vision_sensor")
        self.robot_base = self.sim.getObject("/UR5")
        self.target = self.sim.getObject("/UR5/target")

        # Get camera resolution
        _, resolution = self.sim.getVisionSensorImg(self.sensor)

        self.width, self.height = resolution

        # ---------------------------------------
        # CAMERA PROPERTIES
        # ---------------------------------------

        view_angle = self.sim.getObjectFloatParam(
            self.sensor,
            self.sim.visionfloatparam_perspective_angle
        )

        self.horizontal_fov = view_angle

        self.vertical_fov = 2 * math.atan(
            (self.height / self.width)
            * math.tan(self.horizontal_fov / 2)
        )

        self.cx = self.width / 2
        self.cy = self.height / 2

        self.fx = self.width / (
            2 * math.tan(self.horizontal_fov / 2)
        )

        self.fy = self.height / (
            2 * math.tan(self.vertical_fov / 2)
        )

        # Camera coordinate system -> robot coordinate system
        self.camera_to_robot = self.sim.getObjectMatrix(
            self.sensor,
            self.robot_base
        )

        print("Connected to CoppeliaSim")


    # =========================================================
    # GET CAMERA RGB + DEPTH
    # =========================================================

    def get_camera_data(self):

        # ----------------
        # RGB IMAGE
        # ----------------

        image, resolution = self.sim.getVisionSensorImg(
            self.sensor
        )

        width, height = resolution

        frame = np.frombuffer(
            image,
            dtype=np.uint8
        )

        frame = frame.reshape(
            height,
            width,
            3
        )

        # CoppeliaSim -> OpenCV orientation
        frame = cv2.flip(frame, 0)

        # RGB -> BGR
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # ----------------
        # DEPTH IMAGE
        # ----------------

        depth_bytes, _ = self.sim.getVisionSensorDepth(
            self.sensor,
            1
        )

        depth_array = array.array("f")
        depth_array.frombytes(depth_bytes)

        depth = np.array(
            depth_array,
            dtype=np.float32
        )

        depth = depth.reshape(
            height,
            width
        )

        # Make depth orientation match RGB/OpenCV
        depth = cv2.flip(depth, 0)

        return frame, depth

    # =========================================================
    # PIXEL + DEPTH -> CAMERA XYZ
    # =========================================================

    def pixel_to_camera(self, u, v, depth):

        D = float(depth[v, u])

        x_cam = (
            -(u - self.cx)
            * D
            / self.fx
        )

        y_cam = (
            -(v - self.cy)
            * D
            / self.fy
        )

        z_cam = D

        return [
            x_cam,
            y_cam,
            z_cam
        ]


    # =========================================================
    # CAMERA XYZ -> ROBOT XYZ
    # =========================================================

    def camera_to_robot_coords(self, camera_point):

        Xc, Yc, Zc = camera_point

        m = self.camera_to_robot

        x_robot = (
            m[0] * Xc
            + m[1] * Yc
            + m[2] * Zc
            + m[3]
        )

        y_robot = (
            m[4] * Xc
            + m[5] * Yc
            + m[6] * Zc
            + m[7]
        )

        z_robot = (
            m[8] * Xc
            + m[9] * Yc
            + m[10] * Zc
            + m[11]
        )

        return [
            x_robot,
            y_robot,
            z_robot
        ]


    # =========================================================
    # PIXEL -> ROBOT XYZ
    # =========================================================

    def pixel_to_robot(self, u, v, depth):

        camera_point = self.pixel_to_camera(
            u,
            v,
            depth
        )

        robot_point = self.camera_to_robot_coords(
            camera_point
        )

        return robot_point


    # =========================================================
    # SMOOTH ROBOT TARGET MOVEMENT
    # =========================================================

    def move_target_smoothly(
        self,
        destination,
        steps=100,
        delay=0.01
    ):

        start = self.sim.getObjectPosition(
            self.target,
            self.robot_base
        )

        for i in range(1, steps + 1):

            t = i / steps

            new_position = [
                start[0]
                + (destination[0] - start[0]) * t,

                start[1]
                + (destination[1] - start[1]) * t,

                start[2]
                + (destination[2] - start[2]) * t
            ]

            self.sim.setObjectPosition(
                self.target,
                new_position,
                self.robot_base
            )

            time.sleep(delay)


    # =========================================================
    # MOVE ABOVE OBJECT
    # =========================================================

    def move_above(
        self,
        robot_point,
        approach_height=0.20
    ):

        approach = [
            robot_point[0],
            robot_point[1],
            robot_point[2] + approach_height
        ]

        self.move_target_smoothly(
            approach
        )

        return approach


    # =========================================================
    # DEBUG FUNCTION
    # =========================================================

    def get_object_position(self, object_path):

        obj = self.sim.getObject(
            object_path
        )

        return self.sim.getObjectPosition(
            obj,
            self.robot_base
        )