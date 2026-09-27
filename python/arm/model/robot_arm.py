import math as math
import time
import numpy as np
from typing import Any, TypeAlias

from coppeliasim_zmqremoteapi_client import RemoteAPIClient # type: ignore

from arm.simulation.connection import SimulationConnection
from arm.vision.simulation_camera import SimulationCamera
from arm.config_loader import ArmConfig

# Shorthand for typing a 3D point
Point: TypeAlias = tuple[float, float, float]

class RobotArm:
    def __init__(self, connection: SimulationConnection, simulation_camera: SimulationCamera, config: ArmConfig) -> None:
        # Storing connection for any methods for SimulationConnection to be used (so far none)
        self._connection = connection
        self._client, self._sim = connection.client, connection.sim
        self._simulation_camera = simulation_camera

        # Retrieve sensor handle from simulation camera
        self._sensor_handle = simulation_camera.sensor_handle
      
        # Get simulation objects using config values
        self._robot_base = self._sim.getObject(config.model_path)
        self._target = self._sim.getObject(config.target_path)

        # Other robot arm configuration values
        self._steps_count = config.steps_count
        self._approach_height_offset = config.approach_height_offset

        self._width, self._height = simulation_camera.resolution

        self._horizontal_fov = self._sim.getObjectFloatParam(
            self._sensor_handle,
            self._sim.visionfloatparam_perspective_angle
        )
        self._vertical_fov = 2 * math.atan(
            (self._height / self._width)
            * math.tan(self._horizontal_fov / 2)
        )

        self._cx = self._width / 2
        self._cy = self._height / 2

        self._fx = self._width / (2 * math.tan(self._horizontal_fov / 2))     
        self._fy = self._height / (2 * math.tan(self._vertical_fov / 2))

        # Linear transformation matrix for mapping camera to robot frame
        self._camera_to_robot = self._sim.getObjectMatrix(
            self._sensor_handle,
            self._robot_base
        )

    def is_simulation_running(self) -> bool:
        """Return whether CoppeliaSim client connection is active."""
        return self._connection.is_simulation_running()

    def read_depth(self) -> np.ndarray:
        """
        Return the raw metric depth map (distance in meters) for this arm's
        own connection, mirroring SimulationCamera.read_depth().
        """
        try:
            # options=1 returns true metric distances in meters as floats
            depth_bytes, resolution = self._sim.getVisionSensorDepth(self._sensor_handle, 1)
        except Exception as e:
            raise RuntimeError(
                f"ARM_SIM_ERROR: Failed to retrieve depth buffer: {e}"
            ) from e
 
        depth_map: np.ndarray = np.frombuffer(depth_bytes, dtype=np.float32).reshape(
            (resolution[1], resolution[0])
        )
        return np.flipud(depth_map)

    # Applying the matrix transformation to a 2d point
    def pixel_to_robot(self, u: float, v: float) -> Point:
        # Linear transformation shorthand
        m = self._camera_to_robot
        depth = self.read_depth()

        D = float(depth[v, u])

        # Pixel -> Camera XYZ
        x_cam = -(u - self._cx) * D / self._fx
        y_cam = -(v - self._cy) * D / self._fy
        z_cam = D

        # Camera XYZ -> Robot XYZ
        x_robot = m[0] * x_cam + m[1] * y_cam + m[2] * z_cam + m[3]
        y_robot = m[4] * x_cam + m[5] * y_cam + m[6] * z_cam + m[7]
        z_robot = m[8] * x_cam + m[9] * y_cam + m[10] * z_cam + m[11]

        return [x_robot, y_robot, z_robot]

    # Moving the target smoothly to a destination
    def move_target_smoothly(self, destination: Point) -> None:
        start = self._sim.getObjectPosition(self._target, self._robot_base)

        for i in range(1, self._steps_count + 1):
            t = i / self._steps_count

            new_position = [
                start[0] + (destination[0] - start[0]) * t,
                start[1] + (destination[1] - start[1]) * t,
                start[2] + (destination[2] - start[2]) * t
            ]

            self._sim.setObjectPosition(self._target, new_position, self._robot_base)

            time.sleep(0.01)

    # Going into the "pick" position
    def move_above(self, robot_point: Point) -> Point:
        approach = [
            robot_point[0],
            robot_point[1],
            robot_point[2] + self._approach_height_offset
        ]

        self.move_target_smoothly(approach)

        return approach