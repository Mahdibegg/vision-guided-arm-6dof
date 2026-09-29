from __future__ import annotations

from datetime import datetime
from PySide6.QtCore import Slot
from PySide6.QtGui import QTextCharFormat, QColor,  QTextCursor
from PySide6.QtWidgets import QPlainTextEdit
from enum import Enum

BASE_MAX_LINE = 128

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    CMD = "CMD"

class Colour(Enum):
    WHITE = 0
    RED = 1
    YELLOW = 2
    GREEN = 3
    BLUE = 4

class LogWidget(QPlainTextEdit):
    """
    Console log widget for displaying any application output and messages.
    """

    def __init__(self, style_settings: str, max_lines: int = BASE_MAX_LINE) -> None:
        super().__init__()

        # Have a limit to number of lines that you can see
        # Otherwise application may lag
        self._max_lines = max_lines
        self.document().setMaximumBlockCount(self._max_lines)
        
        self.setMaximumHeight(1050)
        self.setMinimumHeight(1050)
        
        self.setStyleSheet(style_settings)
        
        self.setPlainText("")
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    
        self._colour_map = {
            Colour.WHITE: QColor("#E8ECF2"),
            Colour.RED: QColor("#F47067"),
            Colour.YELLOW: QColor("#E5C07B"),
            Colour.GREEN: QColor("#56D364"),
            Colour.BLUE: QColor("#58A6FF"),
        }

    # Add a new line of text to the log widget externally by changing the colour
    @Slot(str)
    def addLine(self, log_level: LogLevel, message: str, colour: Colour = Colour.WHITE) -> None:
        """Add a new line with timestamp and message with max line number limits"""
        
        # Get current time for timestamp with log entry formatting
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {log_level.value} - {message}"

        # An explicitly supplied colour overrides the log level colour
        if colour is not None:
            text_colour = self._colour_map[colour]
        else:
            text_colour = self._level_colour_map.get(log_level, QColor("#E8ECF2"))

        text_format = QTextCharFormat()
        text_format.setForeground(text_colour)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if not self.document().isEmpty():
            cursor.insertBlock()

        cursor.insertText(log_entry, text_format)

        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def clear_log(self):
        # Reset the entire log and set to default
        self.setPlainText("")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())