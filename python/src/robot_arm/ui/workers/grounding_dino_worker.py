from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot

from vision.detect import Detector
from vision.cameras.types import Frame

class GroundingDinoWorker(QObject):
    ready = Signal()
    detection_complete = Signal(object, str)
    error = Signal(str)

    def __init__(
        self,
        config_path: str | Path,
        weights_path: str | Path,
        yolo_model_name: str,
    ) -> None:
        # Do not give this worker a parent
        super().__init__()

        self._config_path = config_path
        self._weights_path = weights_path
        self._yolo_model_name = yolo_model_name

        self._detector: Detector | None = None

    @Slot()
    def initialize(self) -> None:
        """Load Grounding DINO inside this worker's QThread."""

        try:
            self._detector = Detector(
                config_path = self._config_path,
                weights_path = self._weights_path,
                yolo_model_name = self._yolo_model_name,
                box_threshold = 0.35,
                text_threshold = 0.25,
            )

            self.ready.emit()

        except Exception as error:
            self.error.emit(str(error))

    @Slot(object, str)
    def detect(
        self,
        frame: Frame,
        description: str,
    ) -> None:
        """Run one Grounding DINO inference."""

        if self._detector is None:
            self.error.emit(
                "Grounding DINO received a request before initialization."
            )
            return

        try:
            detections = self._detector.detect(
                frame = frame,
                description = description,
            )

            # Return the description so stale results can be rejected
            self.detection_complete.emit(
                detections,
                description,
            )

        except Exception as error:
            self.error.emit(str(error))