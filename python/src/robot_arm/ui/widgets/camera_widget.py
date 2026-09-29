from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel

from vision.cameras.types import Frame

class CameraWidget(QLabel):
    """
    Camera widget for setting, clearing frames and updating displays

    Stores the frame as QPixmap for maximum editability on the image
    """

    # Store a QPixmap for the newest frame to be added
    # YOLO model embedded into the camera widget, required for object detection
    def __init__(self, style_settings: str) -> None:
        super().__init__("Camera stopped")

        self._source_pixmap: QPixmap | None = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1280, 720)
        self.setMaximumSize(1280, 720)
        self.setStyleSheet(style_settings)

    # Frame is received and processed under object highlighting
    # Essential attributes are set before setting the source pixel map
    @Slot(object)
    def set_frame(self, frame: Frame) -> None:
        """Convert and display an OpenCV BGR frame"""

        height, width, channels = frame.shape
        bytes_per_line = frame.strides[0]

        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        
        self._source_pixmap = QPixmap.fromImage(image)
        self._update_display()

    @Slot()
    def clear_frame(self) -> None:
        """Clear the displayed frame"""
        self._source_pixmap = None
        self.clear()
        self.setText("Camera stopped")

    # Resize the widget screen to the maximum size (set in attributes)
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Resize the camera window from min size to max size when camera is started"""
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self) -> None:
        if self._source_pixmap is None:
            return

        displayed_pixmap = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.setPixmap(displayed_pixmap)