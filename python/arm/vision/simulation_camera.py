import cv2 as cv
import numpy as np
import warnings
from typing import Any

from coppeliasim_zmqremoteapi_client import RemoteAPIClient # type: ignore
from arm.config_loader import SimulationCameraConfig
from .camera_types import Frame

class SimulationCamera:
    """
    Manage frame acquisition from a CoppeliaSim Vision Sensor via ZeroMQ Remote API.

    Connects to an active CoppeliaSim instance, verifies sensor resolution,
    transforms the raw OpenGL buffer to OpenCV BGR format, and mirrors the
    Camera interface for modular use across vision and UI workers.
    """

    def __init__(self, config: SimulationCameraConfig) -> None:
            self._config = config
            self._sensor_path = config.sensor_path
            self._expected_width = config.width
            self._expected_height = config.height
            self._fps = config.fps

            try:
                # Connect to CoppeliaSim Remote API server using loaded host and port
                self._client = RemoteAPIClient(host=config.host, port=config.port)
                self._sim: Any = self._client.require("sim")
            except Exception as e:
                raise RuntimeError(
                    f"CAM_SIM_ERROR: Could not establish connection to CoppeliaSim at "
                    f"{config.host}:{config.port} - {e}"
                ) from e

            try:
                # Query object handle for the vision sensor
                self._sensor_handle = self._sim.getObject(self._sensor_path)
            except Exception as e:
                raise RuntimeError(
                    f"CAM_SIM_ERROR: Could not find vision sensor '{self._sensor_path}' in scene: {e}"
                ) from e

            # Query and validate configured resolution on the sensor
            res_x = self._sim.getObjectInt32Param(
                self._sensor_handle,
                self._sim.visionintparam_resolution_x
            )
            res_y = self._sim.getObjectInt32Param(
                self._sensor_handle,
                self._sim.visionintparam_resolution_y
            )

            if res_x != self._expected_width or res_y != self._expected_height:
                warnings.warn(
                    f"CAM_SIM_WARNING: Sensor resolution ({res_x}x{res_y}) does not match "
                    f"config specification ({self._expected_width}x{self._expected_height}).",
                    RuntimeWarning,
                )

    @property
    def is_open(self) -> bool:
        """Return whether CoppeliaSim client connection is active."""
        try:
            # Check if API responds to basic ping/status query
            return self._sim is not None and self._sim.getSimulationState() is not None
        except Exception:
            return False

    def read(self) -> Frame:
        """
        Capture and return a singular BGR image frame from the vision sensor.
        
        Transforms the raw OpenGL buffer (bottom-to-top RGB) into top-to-bottom BGR
        compatible with OpenCV and YOLO/Grounding DINO detectors.
        """
        try:
            # options=0 -> RGB 3 bytes per pixel
            img_bytes, resolution = self._sim.getVisionSensorImg(self._sensor_handle, 0)
        except Exception as e:
            raise RuntimeError(
                f"CAM_SIM_ERROR: Failed to retrieve vision sensor buffer: {e}"
            ) from e

        if not img_bytes:
            raise RuntimeError(
                "CAM_SIM_ERROR: Received empty frame buffer. Ensure simulation is running."
            )

        # Convert raw binary buffer to uint8 NumPy array with (H, W, C) shape
        raw_frame: Frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
            (resolution[1], resolution[0], 3)
        )

        # CoppeliaSim OpenGL buffer origin is bottom-left; flip vertically for OpenCV/PySide6
        flipped_frame = np.flipud(raw_frame)

        # Convert RGB buffer to BGR
        bgr_frame: Frame = cv.cvtColor(flipped_frame, cv.COLOR_RGB2BGR)

        # Resize if sensor internal resolution diverges from requested frame size
        if bgr_frame.shape[1] != self._expected_width or bgr_frame.shape[0] != self._expected_height:
            bgr_frame = cv.resize(bgr_frame, (self._expected_width, self._expected_height))

        return bgr_frame

    def read_depth(self) -> np.ndarray:
        """
        Return the raw metric depth map (distance in meters) for LiDAR/depth operations.
        """
        try:
            # options=1 returns true metric distances in meters as floats
            depth_bytes, resolution = self._sim.getVisionSensorDepth(self._sensor_handle, 1)
        except Exception as e:
            raise RuntimeError(
                f"CAM_SIM_ERROR: Failed to retrieve depth buffer: {e}"
            ) from e

        depth_map: np.ndarray = np.frombuffer(depth_bytes, dtype=np.float32).reshape(
            (resolution[1], resolution[0])
        )
        return np.flipud(depth_map)

    def display(self, frame: Frame) -> bool:
            """Display camera frame on an OpenCV window; returns False if 'q' is pressed."""
            cv.namedWindow("Simulation Camera", cv.WINDOW_NORMAL)
            cv.imshow("Simulation Camera", frame)
            key = cv.waitKey(30) & 0xFF
            if key == ord("q"):
                return False
            return True

    def close(self) -> None:
        """Close external OpenCV preview windows and detach."""
        cv.destroyAllWindows()

    def is_sim_running(self) -> bool:
        """Returns True if the simulation is actively running and advancing."""
        try:
            state = self._sim.getSimulationState()
            # 0: stopped, 1: paused, >=2: advancing/running
            stopped = getattr(self._sim, "simulation_stopped", 0)
            paused = getattr(self._sim, "simulation_paused", 1)

            return state not in (stopped, paused)
        except Exception:
            return False