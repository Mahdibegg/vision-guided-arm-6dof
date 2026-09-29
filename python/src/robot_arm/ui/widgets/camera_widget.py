from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import QLabel
from vision.cameras.types import Frame


class CameraWidget(QLabel):
    """
    Camera widget for setting, clearing frames and updating displays

    Stores the frame as QPixmap for maximum editability on the image
    """

    # Emitted with the clicked pixel in camera frame coordinates (not widget coordinates)
    pixel_clicked = Signal(int, int)

    # Store a QPixmap for the newest frame to be added
    # YOLO model embedded into the camera widget, required for object detection
    def __init__(self, style_settings: str) -> None:
        super().__init__("Camera stopped")

        self._source_pixmap: QPixmap | None = None

        # Frame pixel of the chosen drop point, drawn on top of every frame
        self._drop_marker: tuple[int, int] | None = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1280, 720)
        self.setMaximumSize(1280, 720)
        self.setStyleSheet(style_settings)
        self.setCursor(Qt.CursorShape.CrossCursor)

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
        self._drop_marker = None
        self.clear()
        self.setText("Camera stopped")

    # Show or remove the marker for where the next pick will be dropped
    @Slot(int, int)
    def set_drop_marker(self, u: int, v: int) -> None:
        """Draw a marker on the frame pixel where the object will be dropped."""
        self._drop_marker = (u, v)
        self._update_display()

    @Slot()
    def clear_drop_marker(self) -> None:
        """Remove the drop marker."""
        self._drop_marker = None
        self._update_display()

    # Turn a left click into a pixel of the original camera frame
    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)

        if event.button() != Qt.MouseButton.LeftButton:
            return

        pixel = self._widget_to_frame_pixel(event.position())

        if pixel is not None:
            self.pixel_clicked.emit(pixel[0], pixel[1])

    def _widget_to_frame_pixel(self, position: QPointF) -> tuple[int, int] | None:
        """Map a widget position to a frame pixel, or None if it is outside the image."""
        displayed = self.pixmap()

        if self._source_pixmap is None or displayed is None or displayed.isNull():
            return None

        # The scaled image is centred in the label, so remove the empty border first
        x = position.x() - (self.width() - displayed.width()) / 2
        y = position.y() - (self.height() - displayed.height()) / 2

        if not (0 <= x < displayed.width() and 0 <= y < displayed.height()):
            return None

        # Scale from the displayed size back up to the real camera frame size
        u = int(x * self._source_pixmap.width() / displayed.width())
        v = int(y * self._source_pixmap.height() / displayed.height())

        return (
            min(u, self._source_pixmap.width() - 1),
            min(v, self._source_pixmap.height() - 1),
        )

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

        if self._drop_marker is not None:
            self._paint_drop_marker(displayed_pixmap)

        self.setPixmap(displayed_pixmap)

    def _paint_drop_marker(self, pixmap: QPixmap) -> None:
        """Draw a green target on the scaled pixmap at the drop pixel."""
        if self._source_pixmap is None or self._drop_marker is None:
            return

        x = int(self._drop_marker[0] * pixmap.width() / self._source_pixmap.width())
        y = int(self._drop_marker[1] * pixmap.height() / self._source_pixmap.height())

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#00E676"), 3))
        painter.drawEllipse(QPoint(x, y), 14, 14)
        painter.drawLine(x - 22, y, x + 22, y)
        painter.drawLine(x, y - 22, x, y + 22)
        painter.end()
