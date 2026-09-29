from __future__ import annotations

from collections.abc import Callable

from model.robot_arm import Point, RobotArm
from PySide6.QtCore import QObject, Signal, Slot
from vision.detect import Detection


class RobotArmWorker(QObject):
    """
    RobotArmWorker is responsible for owning the live link to the CoppeliaSim
    robot arm and running its movement commands in a Qt compatible manner.
    Movement (move_above / move_to / pick_and_place) blocks for the duration of
    the motion via time.sleep(), so this worker is meant to run on its own
    QThread, the same way CameraWorker does, so it never freezes the GUI thread.
    It emits signals for linked, link_broken, error and movement progress.
    """

    linked = Signal()
    link_broken = Signal()
    error = Signal(str)

    movement_started = Signal()
    movement_finished = Signal(object)

    # Emitted with the robot frame point once a drop location has been chosen
    drop_point_set = Signal(object)

    # Emitted with a message when a clicked drop point cannot be used (arm stays linked)
    drop_point_rejected = Signal(str)

    def __init__(
        self,
        # Create a new RobotArm instance when called
        robot_arm_factory: Callable[[], RobotArm],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        # Store the factory rather than a RobotArm instance so linking can be
        # deferred until link() is called on this worker's own thread
        self._robot_arm_factory = robot_arm_factory

        self._robot_arm: RobotArm | None = None

        # Where the next pick should be dropped, None means the drop target in the scene
        self._drop_point: Point | None = None

    @property
    def arm(self) -> RobotArm | None:
        """Return the current RobotArm instance."""
        return self._robot_arm

    @property
    def robot_arm(self) -> RobotArm | None:
        """Expose the active RobotArm instance so the main thread can check link status."""
        return self._robot_arm

    @Slot()
    def link(self) -> None:
        """Create a RobotArm instance and connect it to the running simulation."""
        if self._robot_arm is not None:
            return

        try:
            self._robot_arm = self._robot_arm_factory()

            # Move to the home pose straight away so the tool orientation is locked
            self._robot_arm.go_home()

            self.linked.emit()

        except Exception as error:
            self._robot_arm = None
            self.error.emit(str(error))

    @Slot()
    def break_link(self) -> None:
        """Drop the current RobotArm instance, breaking the link to the simulation."""
        if self._robot_arm is None:
            return

        # Release the IK solver before dropping the arm
        self._robot_arm.close()

        self._robot_arm = None
        self.link_broken.emit()

    # Returning to the home pose
    @Slot()
    def go_home(self) -> None:
        """Move the arm back to its home pose."""
        if self._robot_arm is None:
            self.error.emit("Cannot move: robot arm is not linked.")
            return

        try:
            self.movement_started.emit()

            self._robot_arm.go_home()

            self.movement_finished.emit(None)

        except Exception as error:
            self.error.emit(str(error))

    # Choosing where the next pick will be dropped, e.g. from a click on the camera view
    @Slot(int, int)
    def set_drop_pixel(self, u: int, v: int) -> None:
        """Convert a clicked pixel into a drop point used by the next pick."""
        if self._robot_arm is None:
            self.drop_point_rejected.emit("robot arm is not linked.")
            return

        try:
            self._drop_point = self._robot_arm.pixel_to_drop_point(u, v)

            self.drop_point_set.emit(self._drop_point)

        except Exception as error:
            # Keep the previous drop point cleared and tell the UI, this is not a link failure
            self._drop_point = None
            self.drop_point_rejected.emit(str(error))

    @Slot(object)
    def pick_target(self, target: Detection) -> None:
        """Pick up and place a detected target using its bounding box centre."""

        if target is None:
            self.error.emit("Cannot pick: no target detected.")
            return

        if self._robot_arm is None:
            self.error.emit("Cannot pick: robot arm is not linked.")
            return

        # Get the centre pixel of the detected object's bounding box
        x1, y1, x2, y2 = target.x1, target.y1, target.x2, target.y2
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # The chosen drop point applies to this one pick only, even if the pick fails
        drop_point = self._drop_point
        self._drop_point = None

        try:
            self.movement_started.emit()

            # Convert camera pixel into robot coordinates
            robot_point = self._robot_arm.pixel_to_robot(u, v)

            # Run the complete pickup and drop routine
            self._robot_arm.pick_and_place(robot_point, drop_point)

            self.movement_finished.emit(robot_point)

        except Exception as error:
            self.error.emit(str(error))

    # Moving above a target point (the "approach" position)
    @Slot(object)
    def move_above(self, robot_point: Point) -> None:
        """Move the arm target to the approach position above a robot frame point."""
        if self._robot_arm is None:
            self.error.emit("Cannot move: robot arm is not linked.")
            return

        try:
            self.movement_started.emit()

            approach = self._robot_arm.move_above(robot_point)

            self.movement_finished.emit(approach)

        except Exception as error:
            self.error.emit(str(error))

    # Moving directly onto a target point, e.g. descending onto a pick position
    @Slot(object)
    def move_to(self, robot_point: Point) -> None:
        """Smoothly move the arm target onto a robot frame point."""
        if self._robot_arm is None:
            self.error.emit("Cannot move: robot arm is not linked.")
            return

        try:
            self.movement_started.emit()

            self._robot_arm.move_target_smoothly(robot_point)

            self.movement_finished.emit(robot_point)

        except Exception as error:
            self.error.emit(str(error))

    # Full pick approach: convert a detected pixel to a robot frame point,
    # move above it, then descend onto it
    @Slot(int, int)
    def move_to_pixel(self, u: int, v: int) -> None:
        """Convert a pixel coordinate to a robot point, approach it, then descend onto it."""
        if self._robot_arm is None:
            self.error.emit("Cannot move: robot arm is not linked.")
            return

        try:
            self.movement_started.emit()

            robot_point = self._robot_arm.pixel_to_robot(u, v)

            self._robot_arm.move_above(robot_point)

            # Descend straight down onto the point with the tool orientation fixed
            self._robot_arm.move_to(robot_point)

            self.movement_finished.emit(robot_point)

        except Exception as error:
            self.error.emit(str(error))
