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
    # Hardcoded for now, these move into robot_arm.yaml in a later commit

    # Home pose in degrees for joints 1 to 6, bent and pointing down (not a singular pose)
    HOME_JOINTS_DEG = (-90.0, 0.0, -100.0, 10.0, 90.0, 0.0)
    HOME_STEPS = 60

    # Heights and distances in meters
    TRAVEL_HEIGHT = 0.30
    MAX_DESCENT = 0.30
    GRIP_GAP = 0.005
    MAX_REACH = 0.82

    # Straight line movement
    LINEAR_STEP = 0.01
    STEP_DELAY = 0.02
    MAX_TRACKING_ERROR = 0.005

    # Proximity sensor offset along the tool axis in meters
    SENSOR_OFFSET_Z = 0.005

    # IK solver
    IK_DAMPING = 0.02
    IK_MAX_ITERATIONS = 20
    IK_RETRIES = 3

    def __init__(self, connection: SimulationConnection, simulation_camera: SimulationCamera, config: ArmConfig) -> None:
        # Storing connection for any methods for SimulationConnection to be used (so far none)
        self._connection = connection
        self._client, self._sim = connection.client, connection.sim
        self._simulation_camera = simulation_camera

        # Retrieve sensor handle from simulation camera
        self._sensor_handle = simulation_camera.sensor_handle

        # Get simulation objects
        self._robot_base = self._sim.getObject(config.model_path)
        self._target = self._sim.getObject(config.target_path)
        self._tip = self._sim.getObject("/UR5/tip")
        self._pickup_sensor = self._sim.getObject("/UR5/proximitySensor")
        self._drop_target = self._sim.getObject("/Drop_Target")

        # The table is optional, it is only used to ignore it when detecting objects
        self._table = self._sim.getObject("/table", {"noError": True})

        # The six UR5 joints and every object that belongs to the robot
        self._joints = list(
            self._sim.getObjectsInTree(
                self._robot_base,
                self._sim.object_joint_type,
                0
            )
        )

        if len(self._joints) != 6:
            raise RuntimeError(f"Expected 6 UR5 joints, found {len(self._joints)}")

        self._robot_objects = set(
            self._sim.getObjectsInTree(
                self._robot_base,
                self._sim.handle_all,
                0
            )
        )

        # Other robot arm configuration values
        self._steps_count = config.steps_count
        self._step_size = config.step_size
        self._approach_height_offset = config.approach_height_offset

        # Motion values
        self._travel_height = self.TRAVEL_HEIGHT
        self._max_descent = self.MAX_DESCENT
        self._grip_gap = self.GRIP_GAP
        self._linear_step = self.LINEAR_STEP
        self._step_delay = self.STEP_DELAY
        self._max_tracking_error = self.MAX_TRACKING_ERROR
        self._max_reach = self.MAX_REACH

        # Home pose is stored in radians, the constant is in degrees
        self._home_joints = [math.radians(angle) for angle in self.HOME_JOINTS_DEG]
        self._home_steps = self.HOME_STEPS

        # Tool and IK values
        self._sensor_offset_z = self.SENSOR_OFFSET_Z
        self._ik_damping = self.IK_DAMPING
        self._ik_max_iterations = self.IK_MAX_ITERATIONS
        self._ik_retries = self.IK_RETRIES

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

        # Tool orientation (pointing down), set the first time the arm goes home
        self._tool_quaternion: list[float] | None = None

        # Remember whether a carried object was static so it can be restored on release
        self._carried_was_static: dict[int, int] = {}

        # Prepare the robot in the scene, then create the IK solver
        self._make_robot_kinematic()
        self._fix_tool_frames()
        self._setup_ik()

    def _make_robot_kinematic(self) -> None:
        """Switch the joints to kinematic mode and the links to static so the arm never lags."""

        for joint in self._joints:
            try:
                mode = self._sim.getJointMode(joint)
                mode = mode[0] if isinstance(mode, (list, tuple)) else mode

                if mode != self._sim.jointmode_kinematic:
                    self._sim.setJointMode(joint, self._sim.jointmode_kinematic, 0)
                    print(f"Joint {joint}: switched to kinematic mode")

            except Exception as e:
                print(f"WARNING: could not set joint {joint} to kinematic: {e}")

        shapes = self._sim.getObjectsInTree(
            self._robot_base,
            self._sim.object_shape_type,
            0
        )

        for shape in shapes:
            try:
                if self._sim.getObjectInt32Param(shape, self._sim.shapeintparam_static) == 0:
                    self._sim.setObjectInt32Param(shape, self._sim.shapeintparam_static, 1)
                    self._sim.resetDynamicObject(shape)

            except Exception as e:
                print(f"WARNING: could not make shape {shape} static: {e}")

    def _fix_tool_frames(self) -> None:
        """Place the tip on the flange and the proximity sensor on the tip, both along the tool axis."""

        flange = self._sim.getObjectParent(self._tip)

        # Tip sits exactly on the flange, so its Z axis is the tool axis
        self._sim.setObjectPose(
            self._tip,
            [0, 0, 0, 0, 0, 0, 1],
            flange
        )

        if self._sim.getObjectParent(self._pickup_sensor) != self._tip:
            self._sim.setObjectParent(self._pickup_sensor, self._tip, True)

        # Sensor looks along the tool axis, so it points down whenever the tool does
        self._sim.setObjectPose(
            self._pickup_sensor,
            [0, 0, self._sensor_offset_z, 0, 0, 0, 1],
            self._tip
        )

    def _setup_ik(self) -> None:
        """Create the IK solver that holds both position and orientation of the tip."""

        try:
            self._simIK = self._client.require("simIK")
        except Exception:
            self._simIK = self._client.getObject("simIK")

        self._ik_env = self._simIK.createEnvironment()
        self._ik_group = self._simIK.createGroup(self._ik_env)

        self._simIK.setGroupCalculation(
            self._ik_env,
            self._ik_group,
            self._simIK.method_damped_least_squares,
            self._ik_damping,
            self._ik_max_iterations
        )

        # Pose constraint keeps the wrist orientation fixed while the target moves
        self._simIK.addElementFromScene(
            self._ik_env,
            self._ik_group,
            self._robot_base,
            self._tip,
            self._target,
            self._simIK.constraint_pose
        )

    def close(self) -> None:
        """Release the IK solver."""
        try:
            self._simIK.eraseEnvironment(self._ik_env)
        except Exception:
            pass

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

        D = float(depth[int(v), int(u)])

        # Pixel -> Camera XYZ
        x_cam = -(u - self._cx) * D / self._fx
        y_cam = -(v - self._cy) * D / self._fy
        z_cam = D

        # Camera XYZ -> Robot XYZ
        x_robot = m[0] * x_cam + m[1] * y_cam + m[2] * z_cam + m[3]
        y_robot = m[4] * x_cam + m[5] * y_cam + m[6] * z_cam + m[7]
        z_robot = m[8] * x_cam + m[9] * y_cam + m[10] * z_cam + m[11]

        return (x_robot, y_robot, z_robot)

    def _tip_position(self) -> np.ndarray:
        """Return the current TCP position in the robot frame."""
        return np.array(self._sim.getObjectPosition(self._tip, self._robot_base))

    def _snap_target_to_tip(self) -> None:
        """Put the target exactly on the tip so the IK solver has nothing left to chase."""

        pose = self._sim.getObjectPose(self._tip, self._robot_base)

        self._sim.setObjectPose(self._target, pose, self._robot_base)

    def _set_target(self, position: Point) -> None:
        """Move the target to a position while keeping the tool orientation pointing down."""

        if self._tool_quaternion is None:
            raise RuntimeError("Call go_home() first so the tool orientation is known.")

        pose = [float(position[0]), float(position[1]), float(position[2])] + list(self._tool_quaternion)

        self._sim.setObjectPose(self._target, pose, self._robot_base)

    def _solve_ik(self) -> bool:
        """Run the IK solver once and return whether it succeeded."""

        try:
            result = self._simIK.handleGroup(
                self._ik_env,
                self._ik_group,
                {"syncWorlds": True}
            )
            code = result[0] if isinstance(result, (list, tuple)) else result

            return code == self._simIK.result_success

        except Exception:
            # Older simIK API (CoppeliaSim 4.2)
            self._simIK.applyIkEnvironmentToScene(self._ik_env, self._ik_group)

            return True

    def _check_reachable(self, point: Point) -> None:
        """Refuse points that are too far from the base for the tool to reach pointing down."""

        radius = math.hypot(point[0], point[1])

        if radius > self._max_reach:
            raise RuntimeError(
                f"Point {tuple(round(p, 3) for p in point)} is {radius:.2f} m from the base, "
                f"the maximum with the tool pointing down is {self._max_reach} m."
            )

    def _move_linear(self, destination: Point) -> None:
        """Move the tip in a straight line to a point, solving IK in small steps."""

        destination = np.array(destination, dtype=float)

        self._check_reachable(destination)

        start = self._tip_position()

        steps = max(1, math.ceil(np.linalg.norm(destination - start) / self._linear_step))

        for i in range(1, steps + 1):
            waypoint = start + (destination - start) * (i / steps)

            self._set_target(waypoint)

            # Give the solver a few attempts at each waypoint
            for _ in range(self._ik_retries):
                if self._solve_ik():
                    break

            # Stop instead of continuing if the tip cannot follow the target
            error = np.linalg.norm(self._tip_position() - waypoint)

            if error > self._max_tracking_error:
                self._snap_target_to_tip()

                raise RuntimeError(
                    f"IK could not follow the path (error {error:.3f} m at "
                    f"{waypoint.round(3).tolist()}). Stopped safely."
                )

            time.sleep(self._step_delay)

    def _move_via_travel_height(self, destination: Point) -> None:
        """Go up to the travel height, across, then down so the tip never drags through the table."""

        current = self._tip_position()

        travel_z = max(self._travel_height, current[2], destination[2])

        self._move_linear((current[0], current[1], travel_z))
        self._move_linear((destination[0], destination[1], travel_z))
        self._move_linear(destination)

    def move_to(self, robot_point: Point) -> None:
        """Move the tip in a straight line onto a robot frame point."""
        self._move_linear(robot_point)

    def move_target_smoothly(self, destination: Point) -> None:
        """Move to a destination via the travel height, keeping the tool orientation fixed."""
        self._move_via_travel_height(destination)

    def move_above(self, robot_point: Point) -> Point:
        """Move to the approach position above a robot frame point."""

        approach = (
            robot_point[0],
            robot_point[1],
            robot_point[2] + self._approach_height_offset
        )

        print("Detected robot point:", [round(c, 4) for c in robot_point])
        print("Approach point:", [round(c, 4) for c in approach])

        self._move_via_travel_height(approach)

        print("Final TCP:", self._tip_position().round(4).tolist())

        return approach

    def pick_and_place(self, robot_point: Point) -> None:
        """Combine private functions to carry out a full pick and place based on a robot point"""

        # The tool orientation is only known once the arm has been home
        if self._tool_quaternion is None:
            self.go_home()

        object_handle = self._pick_object(robot_point)
        self._drop_object(object_handle)
        self.return_to_default_position()

        print("PICK AND PLACE COMPLETE")

    def go_home(self) -> None:
        """Move the joints to the bent home pose and remember the tool orientation."""

        start = [self._sim.getJointPosition(joint) for joint in self._joints]

        for i in range(1, self._home_steps + 1):
            t = i / self._home_steps

            for joint, a, b in zip(self._joints, start, self._home_joints):
                self._sim.setJointPosition(joint, a + (b - a) * t)

            time.sleep(self._step_delay)

        self._snap_target_to_tip()

        # The first time home is reached, the tool is pointing down so store that orientation
        if self._tool_quaternion is None:
            self._tool_quaternion = list(
                self._sim.getObjectQuaternion(self._tip, self._robot_base)
            )

        print("HOME reached. TCP:", self._tip_position().round(4).tolist())

    def return_to_default_position(self) -> None:
        """Return the arm to the home pose."""
        self.go_home()

    # Lower the TCP until the proximity sensor detects an object
    def _lower_until_detected(self) -> int:

        start = self._tip_position()

        minimum_z = start[2] - self._max_descent

        while True:
            result, distance, point, detected_object, normal = (
                self._sim.checkProximitySensor(
                    self._pickup_sensor,
                    self._sim.handle_all
                )
            )

            # Ignore the table and the robot itself, only real objects count
            if (
                result == 1
                and detected_object != self._table
                and detected_object not in self._robot_objects
            ):
                print(
                    f"Detected object {detected_object} "
                    f"at {distance:.4f} m"
                )

                # Close the remaining gap so the tip stops right at the object
                gap = max(0.0, distance - self._grip_gap)

                current = self._tip_position()

                self._move_linear((current[0], current[1], current[2] - gap))

                return detected_object

            current = self._tip_position()

            # Safety check so the robot cannot keep lowering forever
            if current[2] - self._step_size < minimum_z:
                raise RuntimeError(
                    "Pickup failed: no object detected."
                )

            # Lower the tip by one step
            self._move_linear((current[0], current[1], current[2] - self._step_size))

    def _attach_object(self, object_handle: int) -> None:

        # Make the object static while carried so physics cannot fight the parenting
        try:
            was_static = self._sim.getObjectInt32Param(
                object_handle,
                self._sim.shapeintparam_static
            )

            self._carried_was_static[object_handle] = was_static

            self._sim.setObjectInt32Param(
                object_handle,
                self._sim.shapeintparam_static,
                1
            )
            self._sim.resetDynamicObject(object_handle)

        except Exception:
            # Not a shape, parenting still works
            pass

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

        # Restore the object's original physics state
        was_static = self._carried_was_static.pop(object_handle, None)

        if was_static is not None:
            self._sim.setObjectInt32Param(
                object_handle,
                self._sim.shapeintparam_static,
                was_static
            )
            self._sim.resetDynamicObject(object_handle)

    def _lift(self, height: float | None = None) -> Point:

        if height is None:
            height = self._approach_height_offset

        current = self._tip_position()

        # Never lift to less than the travel height
        destination = (
            current[0],
            current[1],
            max(self._travel_height, current[2] + height)
        )

        self._move_linear(destination)

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

        destination = (
            drop_position[0],
            drop_position[1],
            drop_position[2]
        )

        self._move_via_travel_height(destination)

        return destination

    def _drop_object(self, object_handle: int) -> None:

        self._move_to_drop_target()
        self._release_object(object_handle)
        self._lift()