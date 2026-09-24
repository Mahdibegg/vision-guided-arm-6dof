from __future__ import annotations

from PySide6.QtCore import (
    QMetaObject,
    QThread,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QWidget,
    QHBoxLayout,
)

from arm.vision.simulation_camera import SimulationCamera
from arm.vision.detect import Detection, is_valid_detection

from arm.ui.grounding_dino_worker import GroundingDinoWorker
from arm.ui.camera_widget import CameraWidget
from arm.ui.camera_worker import CameraWorker
from arm.ui.log_widget import LogWidget, LogLevel, Colour
from arm.ui.sanitize_input import SanitizeInput

from arm.config_loader import (
    load_camera_config,
    load_app_config,
    load_grounding_dino_config,
    load_grounding_dino_weights,
    load_yolo_model,
    load_simulation_camera_config
)

APP_CONFIG = load_app_config()
CAMERA_CONFIG = load_camera_config()
SIMULATION_CAMERA_CONFIG = load_simulation_camera_config()
YOLO_CONFIG = load_yolo_model()
GROUNDING_DINO_CONFIG = load_grounding_dino_config()
GROUNDING_DINO_WEIGHTS = load_grounding_dino_weights()

class MainWindow(QMainWindow):
    start_camera_requested = Signal()
    stop_camera_requested = Signal()
    full_detection_requested = Signal(bool)

    detection_description_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("6-DOF Arm Controller")
        self.resize(1920, 1080)

        self._target_reported = False
        self._camera_running = False
        self._strict_detection = False

        self._create_widgets()
        self._create_layout()
        self._create_camera_worker()
        self._connect_signals()

    # Private layout/widget setup functions

    # Keep the init function small by having all the widgets in a private function
    def _create_widgets(self) -> None:
        self._camera_widget = CameraWidget(APP_CONFIG.styles.camera_widget)

        self._log_widget = LogWidget(
            APP_CONFIG.styles.log_widget,
            APP_CONFIG.log.max_lines
        )

        # Input box for user input
        self._command_input = QLineEdit()
        self._command_input.setObjectName("commandInput")
        self._command_input.setPlaceholderText("Enter command")

        # Buttons for the controller
        self._submit_button = QPushButton("Submit")
        self._submit_button.setObjectName("submitButton")
        self._submit_button.setEnabled(False)

        self._detection_button = QPushButton("Full Detection")
        self._detection_button.setObjectName("detectionButton")
        self._detection_button.setCheckable(True)
        self._detection_button.setChecked(False)
        self._detection_button.setEnabled(False)

        self._strict_detection_button = QPushButton("Strict Detection")
        self._strict_detection_button.setObjectName("detectionButton")
        self._strict_detection_button.setCheckable(True)
        self._strict_detection_button.setChecked(False)
        self._strict_detection_button.setEnabled(False)

        self._clear_log_button = QPushButton("Clear Log")
        self._link_arm_button = QPushButton("Link Arm")

        self._start_button = QPushButton("Start Camera")
        self._stop_button = QPushButton("Stop Camera")
        # Set stop button initially to false, so that it is disabled by default
        self._stop_button.setEnabled(False)

    def _create_layout(self) -> None:
        main_layout = QGridLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setHorizontalSpacing(14)
        main_layout.setVerticalSpacing(14)

        # Main camera and log area
        main_layout.addWidget(self._camera_widget, 0, 0)
        main_layout.addWidget(self._log_widget, 0, 1)

        # Embedded command bar
        command_bar = QWidget()
        command_bar.setObjectName("commandBar")
        command_bar.setFixedHeight(58)

        command_layout = QHBoxLayout(command_bar)
        command_layout.setContentsMargins(0, 0, 0, 0)
        command_layout.setSpacing(0)

        command_layout.addWidget(self._command_input, 1)
        command_layout.addWidget(self._submit_button)

        main_layout.addWidget(command_bar, 1, 0)

        # Main control buttons
        controls_widget = QWidget()
        controls_layout = QGridLayout(controls_widget)

        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(14)
        controls_layout.setVerticalSpacing(12)

        controls_layout.addWidget(self._start_button, 0, 0)
        controls_layout.addWidget(self._detection_button, 0, 1)
        controls_layout.addWidget(self._link_arm_button, 0, 2)
        controls_layout.addWidget(self._stop_button, 1, 0)
        controls_layout.addWidget(self._clear_log_button, 1, 2)
        controls_layout.addWidget(self._strict_detection_button, 1, 1)

        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 1)

        main_layout.addWidget(controls_widget, 2, 0)

        # Camera area is approximately twice as wide as the log.
        main_layout.setColumnStretch(0, 2)
        main_layout.setColumnStretch(1, 1)

        # Let the camera/log row consume extra vertical space.
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 0)
        main_layout.setRowStretch(2, 0)

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        central_widget.setLayout(main_layout)

        central_widget.setStyleSheet(
            "\n".join(
                (
                    APP_CONFIG.styles.app_widget,
                    APP_CONFIG.styles.button_widget,
                    APP_CONFIG.styles.command_bar,
                )
            )
        )

        self.setCentralWidget(central_widget)

    # Separating the camera requests, buttons and the thread to run the camera display
    def _connect_signals(self) -> None:
        # Camera requests
        self.start_camera_requested.connect(self._camera_worker.start)
        self.stop_camera_requested.connect(self._camera_worker.stop)

        # Button signals
        self._start_button.clicked.connect(self.start_camera_requested.emit)
        self._stop_button.clicked.connect(self.stop_camera_requested.emit)

        self._command_input.textChanged.connect(self._update_submit_button)
        self._submit_button.clicked.connect(self._on_submit_button_clicked)
        self._clear_log_button.clicked.connect(self._log_widget.clear_log)
        self._link_arm_button.clicked.connect(self._on_link_arm_button_clicked)
        self._detection_button.toggled.connect(self._on_full_detection_toggled)
        self._strict_detection_button.toggled.connect(self._on_strict_detection_toggled)

        # Camera worker functionality
        self._camera_worker.frame_ready.connect(self._camera_widget.set_frame)
        self._camera_worker.started.connect(self._on_camera_started)
        self._camera_worker.stopped.connect(self._on_camera_stopped)
        self._camera_worker.error.connect(self._on_camera_error)
        self.full_detection_requested.connect(self._camera_worker.set_full_detection_enabled)
        self.detection_description_requested.connect(self._camera_worker.set_detection_description)

        self._camera_worker.target_detected.connect(self._on_target_detected)

    # Camera worker functionality

    # Create a camera worker private to main window using config values
    def _create_camera_worker(self) -> None:
        self._camera_thread = QThread(self)

        # Receive a configured camera class
        # And receive all model configs from main window (so one single import from main_window)
        self._camera_worker = CameraWorker(
            lambda: SimulationCamera(
                config = SIMULATION_CAMERA_CONFIG,
            ),
            SIMULATION_CAMERA_CONFIG.fps,
            GROUNDING_DINO_CONFIG,
            GROUNDING_DINO_WEIGHTS,
            YOLO_CONFIG,
        )

        self._camera_worker.moveToThread(self._camera_thread)

        self._camera_thread.finished.connect(self._camera_worker.deleteLater)
        
        self._grounding_dino_thread = QThread(self)

        # Do not give this worker a parent because it must be moved
        # from the GUI thread into the Grounding DINO thread
        self._grounding_dino_worker = GroundingDinoWorker(
            config_path = GROUNDING_DINO_CONFIG,
            weights_path = GROUNDING_DINO_WEIGHTS,
            yolo_model_name = YOLO_CONFIG,
        )

        self._grounding_dino_worker.moveToThread(self._grounding_dino_thread)

        self._grounding_dino_thread.finished.connect(self._grounding_dino_worker.deleteLater)

        # Load the model after the worker enters its own thread.
        self._grounding_dino_thread.started.connect(self._grounding_dino_worker.initialize)

        self._camera_worker.grounding_dino_requested.connect(self._grounding_dino_worker.detect)

        self._grounding_dino_worker.ready.connect(self._camera_worker.set_grounding_dino_ready)

        self._grounding_dino_worker.detection_complete.connect(self._camera_worker.accept_grounding_dino_result)

        self._grounding_dino_worker.error.connect(self._camera_worker.accept_grounding_dino_error)

        self._camera_thread.start()
        self._grounding_dino_thread.start()

    # Camera start button functionality
    @Slot()
    def _on_camera_started(self) -> None:
        # For all information on connection
        config = SIMULATION_CAMERA_CONFIG

        self._log_widget.addLine(
            LogLevel.INFO,
            f"successfully connected to CoppeliaSim\n                  at @ {config.host}:{config.port} on '{config.sensor_path}'"
        )

        if not self._camera_worker.is_simulation_running():
            self._log_widget.addLine(
                LogLevel.WARNING,
                "start CoppeliaSim simulation to render frames",
                Colour.YELLOW
            )
        else:
            self._log_widget.addLine(
                LogLevel.INFO,
                "camera running",
            )

        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._detection_button.setEnabled(True)
        self._strict_detection_button.setEnabled(True)

        self._command_input.clear()
        self._camera_running = True

    # Camera end button functionality
    @Slot()
    def _on_camera_stopped(self) -> None:
        self._log_widget.addLine(LogLevel.INFO, "camera stopped", Colour.YELLOW)
        self._camera_worker.clear_detections()
        self._camera_widget.clear_frame()
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._detection_button.setChecked(False)
        self._detection_button.setEnabled(False)

        self._strict_detection_button.setChecked(False)
        self._strict_detection_button.setEnabled(False)

        self._command_input.clear()
        self._camera_running = False

    @Slot(str)
    def _on_camera_error(self) -> None:
        config = SIMULATION_CAMERA_CONFIG

        self._log_widget.addLine(
            LogLevel.ERROR,
            f"failed to connect to {config.host}:{config.port} on '{config.sensor_path}'",
            Colour.RED
        )
        
        self._camera_worker.clear_detections()
        self._camera_widget.clear_frame()
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._detection_button.setChecked(False)
        self._detection_button.setEnabled(False)
        self._strict_detection_button.setChecked(False)
        self._strict_detection_button.setEnabled(False)

    # Command input functionality

    def _update_submit_button(self) -> None:
        has_command = bool(self._command_input.text().strip())

        self._submit_button.setEnabled(self._camera_running and has_command)

    @Slot()
    def _on_submit_button_clicked(self) -> None:
        input_text = self._command_input.text()

        if not input_text:
            return
        
        sanitized_text = SanitizeInput(input_text).process_input()
        
        if not sanitized_text.is_valid or sanitized_text.input is None:
            self._log_widget.addLine(
                LogLevel.ERROR,
                "invalid object description",
                Colour.RED,
            )
            self._command_input.clear()
            return
        
        input_text = sanitized_text.input
        
        self._target_reported = False
        
        self.detection_description_requested.emit(input_text)

        self._log_widget.addLine(LogLevel.CMD, input_text)
        self._log_widget.addLine(LogLevel.DEBUG, "parsing command data...")
        
        self._command_input.clear()

    # Link arm button functionality
    def _on_link_arm_button_clicked(self) -> None:
        # For now just output a message
        self._log_widget.addLine(LogLevel.DEBUG, "linking arm...")
        self._log_widget.addLine(LogLevel.ERROR, "failed to connect to arm", Colour.RED)

    # Full detection mode for identifying all objects
    @Slot(bool)
    def _on_full_detection_toggled(self, enabled: bool) -> None:
        self.full_detection_requested.emit(enabled)

        if enabled:
            self._log_widget.addLine(LogLevel.INFO, "full detection enabled", Colour.BLUE)
        else:
            self._log_widget.addLine(LogLevel.INFO, "full detection disabled", Colour.YELLOW)

    # Strict detection mode for filtering out low confidence level objects
    def _on_strict_detection_toggled(self) -> None:

        self._strict_detection = not self._strict_detection

        self._camera_worker.clear_detections()
        
        if self._strict_detection:
            self._log_widget.addLine(LogLevel.INFO, "strict detection enabled", Colour.BLUE)
        else:
            self._log_widget.addLine(LogLevel.INFO, "strict detection disabled", Colour.YELLOW)

    @Slot(object)
    def _on_target_detected(
        self,
        target: Detection | None,
    ) -> None:
        """Report a successfully detected target."""

        if target is None or self._target_reported:
            return
        
        self._target_reported = True

        valid_target = is_valid_detection(target)

        if valid_target or not self._strict_detection:
            self._log_widget.addLine(LogLevel.INFO,
                "Detection complete: "
                f"{target.description or target.class_name} "
                f"found with {target.confidence:.0%} confidence",
                Colour.GREEN
            )
        else:
            # Use the available target to report error then clear the target data so nothing is drawn
            self._log_widget.addLine(LogLevel.ERROR,
                "Detection failure: "
                f"{target.description or target.class_name} "
                f"could not be found",
                Colour.RED
            )

            self._camera_worker.clear_detections()

    @Slot(str)
    def request_grounding_dino_detection(self, description: str) -> None:
        """Request Grounding DINO detection for a specific object description."""
        if self._camera_worker is not None:
            # Set the description in camera worker
            self._camera_worker.set_detection_description(description)
            
            # Trigger detection in background thread
            self._camera_worker._grounding_dino_worker.run_detection()

    @Slot()
    def clear_grounding_dino(self) -> None:
        """Clear Grounding DINO detection state."""
        if self._camera_worker is not None:
            self._camera_worker.clear_detections()
        
    # Close any threads that are running when exiting the program
    def closeEvent(self, event: QCloseEvent) -> None:

        camera_thread = getattr(self, "_camera_thread", None)
        grounding_thread = getattr(
            self,
            "_grounding_dino_thread",
            None,
        )

        if camera_thread is not None and camera_thread.isRunning():
            QMetaObject.invokeMethod(
                self._camera_worker,
                "stop",
                Qt.ConnectionType.BlockingQueuedConnection,
            )

            camera_thread.quit()
            camera_thread.wait()

        if grounding_thread is not None and grounding_thread.isRunning():
            grounding_thread.quit()
            grounding_thread.wait()

        event.accept()