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
        self._tip = self._sim.getObject("/UR5/tip")
        self._pickup_sensor = self._sim.getObject("/UR5/proximitySensor")
        self._drop_target = self._sim.getObject("/DropTarget")

        # Other robot arm configuration values
        self._steps_count = config.steps_count
        self._step_size = config.step_size
        self._approach_height_offset = config.approach_height_offset

        self._default_position: Point = (
            -0.18495,
            -0.010,
            0.86255,
        )

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

    def move_above(self, robot_point: Point) -> Point:
        approach = [
            robot_point[0],
            robot_point[1],
            robot_point[2] + self._approach_height_offset
        ]

        self.move_target_smoothly(approach)

        return approach
    
    def pick_and_place(self, robot_point: Point) -> None:
        """Combine private functions to carry out a full pick and place based on a robot point"""
        object_handle = self._pick_object(robot_point)
        self._drop_object(object_handle)
        self.return_to_default_position()

    def return_to_default_position(self) -> None:
        """Return the arm to a default position that is hard coded."""
        self.move_target_smoothly(self._default_position)

    # Lower the TCP until the proximity sensor detects an object
    def _lower_until_detected(
        self,
        max_descent: float = 0.30,
    ) -> int:

        start = self._sim.getObjectPosition(self._target, self._robot_base)

        minimum_z = start[2] - max_descent

        while True:
            result, distance, point, detected_object, normal = (self._sim.readProximitySensor(self._pickup_sensor))

            # Stop lowering once an object is detected
            if result == 1:
                print(
                    f"Detected object {detected_object} "
                    f"at {distance:.4f} m"
                )

                return detected_object

            current = self._sim.getObjectPosition(self._target, self._robot_base)

            # Safety check so the robot cannot keep lowering forever
            if current[2] <= minimum_z:
                raise RuntimeError(
                    "Pickup failed: no object detected."
                )

            # Lower the target by 2 mm
            current[2] -= self._step_size

            self._sim.setObjectPosition(
                self._target,
                current,
                self._robot_base
            )

            time.sleep(0.02)

    def _attach_object(self, object_handle: int) -> None:

        self._sim.setObjectParent(
            object_handle,
            self._tip,
            True
        )

    def _release_object(self, object_handle: int) -> None:

        self._sim.setObjectParent(
            object_handle,
            -1,
            True
        )

    def _lift(self, height: float | None = None) -> Point:

        if height is None:
            height = self._approach_height_offset

        current = self._sim.getObjectPosition(
            self._target,
            self._robot_base
        )

        destination = [
            current[0],
            current[1],
            current[2] + height
        ]

        self.move_target_smoothly(destination)

        return destination

    def _pick_object(self, robot_point: Point) -> int:

        self.move_above(robot_point)
        detected_object = self._lower_until_detected()
        self._attach_object(detected_object)
        self._lift()

        return detected_object
    
    def _move_to_drop_target(self) -> Point:

        drop_position = self._sim.getObjectPosition(
            self._drop_target,
            self._robot_base
        )

        destination = [
            drop_position[0],
            drop_position[1],
            drop_position[2]
        ]

        self.move_target_smoothly(destination)

        return destination

    def _drop_object(self, object_handle: int) -> None:

        self._move_to_drop_target()
        self._release_object(object_handle)
        self._lift()