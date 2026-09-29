from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cv2 import COLOR_BGR2RGB, cvtColor
from PySide6.QtCore import QObject, QTimer, Signal, Slot  # type: ignore
from vision.cameras.simulation_camera import SimulationCamera
from vision.cameras.types import Frame
from vision.detect import Detection, Detector, highlight_objects


class CameraWorker(QObject):
    """
    CameraWorker is responsible for managing the camera and capturing frames in a Qt compatible manner.
    It emits signals for frame ready, started, stopped, and error, allowing the worker to run in a separate thread from the main application thread.
    """

    frame_ready = Signal(object)
    started = Signal()
    stopped = Signal()
    error = Signal(str)
    target_detected = Signal(object)
    grounding_dino_requested = Signal(object, str)

    def __init__(
        self,
        # Create a new camera instance when called
        camera_factory: Callable[[], SimulationCamera],
        capture_fps: int,
        grounding_dino_config: str | Path,
        grounding_dino_weights: str | Path,
        yolo_model_name: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        # Set up the camera and timer (capture interval based on the desired FPS)
        self._camera_factory = camera_factory
        self._interval_ms = max(1, round(1000 / capture_fps))

        self._camera: SimulationCamera | None = None
        self._timer: QTimer | None = None

        self._full_detection_enabled = False

        # Storing the result of detect() as an attribute
        # Easily accessible by the main thread
        self._objects_detected: list[Detection] = []

        self._detection_description: str | None = None
        self._target_object: Detection | None = None

        self._grounding_dino_config = grounding_dino_config
        self._grounding_dino_weights = grounding_dino_weights

        # Required for threading the grounding dino model
        self._grounding_dino_ready = False
        self._grounding_dino_busy = False
        self._grounded_detections: list[Detection] = []

        self._yolo_model_name = yolo_model_name

        self._vision_model: Detector | None = None

    @property
    def camera(self) -> SimulationCamera | None:
        """Expose the active camera instance for spatial calibration."""
        return self._camera

    @Slot()
    def start(self) -> None:
        """Open the camera and begin reading frames."""
        if self._camera is not None:
            return

        # Create a new camera instance on start and set a timer
        try:
            # Creating the model on start requiring all config values for both
            # The grounding dino and yolo model
            if self._vision_model is None:
                self._vision_model = Detector(
                    config_path=self._grounding_dino_config,
                    weights_path=self._grounding_dino_weights,
                    yolo_model_name=self._yolo_model_name,
                    box_threshold=0.35,
                    text_threshold=0.25,
                )

            self._camera = self._camera_factory()

            if self._timer is None:
                self._timer = QTimer(self)
                self._timer.setInterval(self._interval_ms)
                self._timer.timeout.connect(self._read_frame)

            self._timer.start()
            self.started.emit()

        except Exception as error:
            self._close_camera()
            self.error.emit(str(error))

    @Slot()
    def stop(self) -> None:
        """Stop frame capture and release the camera."""
        was_running = self._camera is not None

        if self._timer is not None:
            self._timer.stop()

        self._close_camera()

        if was_running:
            self.stopped.emit()

    @Slot(bool)
    def set_full_detection_enabled(self, enabled: bool) -> None:
        """Enable or disable object detection and highlighting."""
        self._full_detection_enabled = enabled

    @Slot(str)
    def set_detection_description(self, description: str) -> None:
        """Set the natural-language description of the target object."""

        description = description.strip()
        new_description = description or None

        # If the description remains the same, do nothing
        if new_description == self._detection_description:
            return

        self._detection_description = new_description
        self._grounded_detections = []
        self._target_object = None

        self.target_detected.emit(None)

    def clear_detections(self) -> None:
        """Clear all detections included target object for when turning camera off."""
        self._detection_description = None
        self._target_object = None
        self._objects_detected.clear()
        try:
            self.target_detected.emit(None)
        except RuntimeError:
            # QObject is already deleted during camera shutdown
            pass

    @Slot()
    def set_grounding_dino_ready(self) -> None:
        """Mark Grounding DINO as ready to receive requests."""

        self._grounding_dino_ready = True

    @Slot(str)
    def accept_grounding_dino_error(
        self,
        description: str,
    ) -> None:
        """Handle a Grounding DINO failure by resetting the busy state."""
        self._grounding_dino_busy = False

        # Ignore a result belonging to an older command
        if description != self._detection_description:
            return

        # Clear detections since the model failed to process the request
        self._grounded_detections = self._grounded_detections
        self._target_object = None

        self.target_detected.emit(None)

    @Slot(object, str)
    def accept_grounding_dino_result(
        self,
        detections: list[Detection],
        description: str,
    ) -> None:
        """Cache the newest Grounding DINO result."""

        self._grounding_dino_busy = False

        # Ignore results from an older description.
        if description != self._detection_description:
            return

        self._grounded_detections = detections

        target = max(
            detections,
            key=lambda detection: detection.confidence,
            default=None,
        )

        self._grounded_detections = detections
        self._target_object = target

        self.target_detected.emit(target)

    @Slot()
    def _read_frame(self) -> None:
        """
        Read and emit a single frame while handling multiple threads for the
        separate grounding dino model and camera.
        """
        if self._camera is None:
            return

        try:
            frame = self._camera.read()

            should_request_grounding = (
                self._detection_description is not None
                and self._grounding_dino_ready
                and not self._grounding_dino_busy
            )

            if should_request_grounding:
                self._grounding_dino_busy = True

                self.grounding_dino_requested.emit(
                    frame.copy(),
                    self._detection_description,
                )

            updated_frame = self._process_frame(frame)
            self.frame_ready.emit(updated_frame)

        except Exception as error:
            self.stop()
            self.error.emit(str(error))

    def _close_camera(self) -> None:
        """Close and discard the current camera instance."""
        if self._camera is None:
            return

        self.target_detected.emit(None)
        self._camera.close()
        self._camera = None

    def _process_frame(self, frame: Frame) -> Frame:
        """Apply YOLO and/or Grounding DINO detection to a camera frame."""

        has_description = self._detection_description is not None

        if not self._full_detection_enabled and not has_description:
            self._objects_detected = []
            return cvtColor(frame, COLOR_BGR2RGB)

        highlighted_frame = frame.copy()

        yolo_detections: list[Detection] = []

        if self._full_detection_enabled and self._vision_model is not None:
            yolo_detections = self._vision_model.analyse(frame)

            highlighted_frame = highlight_objects(
                frame=highlighted_frame,
                objects=yolo_detections,
            )

        if has_description:
            highlighted_frame = highlight_objects(
                frame=highlighted_frame,
                objects=self._grounded_detections,
                target_object=self._target_object,
            )

        self._objects_detected = yolo_detections + self._grounded_detections

        return cvtColor(
            highlighted_frame,
            COLOR_BGR2RGB,
        )
