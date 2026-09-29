from __future__ import annotations

from collections.abc import Callable
from PySide6.QtCore import QObject, Signal, Slot

from model.robot_arm import RobotArm, Point
from vision.detect import Detection

class RobotArmWorker(QObject):
    """
    RobotArmWorker is responsible for owning the live link to the CoppeliaSim
    robot arm and running its movement commands in a Qt compatible manner.
    Movement (move_target_smoothly / move_above) blocks for the duration of
    the motion via time.sleep(), so this worker is meant to run on its own
    QThread, the same way CameraWorker does, so it never freezes the GUI thread.
    It emits signals for linked, link_broken, error and movement progress.
    """

    linked = Signal()
    link_broken = Signal()
    error = Signal(str)

    movement_started = Signal()
    movement_finished = Signal(object)

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
            self.linked.emit()

        except Exception as error:
            self._robot_arm = None
            self.error.emit(str(error))

    @Slot()
    def break_link(self) -> None:
        """Drop the current RobotArm instance, breaking the link to the simulation."""
        if self._robot_arm is None:
            return

        self._robot_arm = None
        self.link_broken.emit()

    @Slot(object)
    def pick_target(self, target: Detection) -> None:
        """Move to and pick up a detected target, using its pixel bounding box centre."""
        if target is None:
            self.error.emit("Cannot pick: no target detected.")
            return
 
        x1, y1, x2, y2 = target.x1, target.y1, target.x2, target.y2
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
 
        self.move_to_pixel(u, v)
        
        # Final action is to pick up the target and drop

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
            self._robot_arm.move_target_smoothly(robot_point)

            self.movement_finished.emit(robot_point)

        except Exception as error:
            self.error.emit(str(error))