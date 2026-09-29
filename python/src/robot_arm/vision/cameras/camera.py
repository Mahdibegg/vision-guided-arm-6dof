import cv2 as cv
import warnings

from .types import Frame

class Camera:
    """
    Manage frame capture from an OpenCV-compatible camera.

    Configures the requested resolution and frame rate, validates the
    resulting camera settings, and manages frame acquisition and release.
    """

    # Obtain the camera device settings 
    # Set properties from camera config and validate those properties
    def __init__(self,
        device: int,
        width: int,
        height: int,
        fps: int,
        format: str = None
    ) -> None:
        self._capture = cv.VideoCapture(device)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"CAM_ERROR: Could not open camera {device}"
            )

        # This is the camera format, the initial camera used for testing had two, YUYV and MJPG
        # This selects a specific format set in camera.format config
        # It allows for correct setting of width/height/fps, using the camera to the full extent
        if format is not None:
            self._capture.set(
                cv.CAP_PROP_FOURCC,
                cv.VideoWriter_fourcc(*format),
            )
        
        # Identify and set the different camera properties
        # These include, width, height
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv.CAP_PROP_FPS, fps)

        # Obtain camera property values using property ID
        actual_width = self._capture.get(cv.CAP_PROP_FRAME_WIDTH)
        actual_height = self._capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._capture.get(cv.CAP_PROP_FPS)
    
        # Incorrect width and height means calibration will not be correct
        if actual_width != width or actual_height != height:
            raise RuntimeError(
                "CAM_ERROR: Could not set width and height to 720p settings" \
                f"Expected: {width}x{height}, Got: {actual_width}x{actual_height}"
            )
        
        # Raise warning for incorrect fps as it wont affect results
        # Incorrect fps is anything less than the config value
        if actual_fps < fps:
            warnings.warn(
                "CAM_WARNING: Camera FPS differs from requested FPS" \
                f"Expected: {fps}, Got: {actual_fps}",
                RuntimeWarning
            )

    @property
    def is_open(self) -> bool:
        """Return whether the camera device is open"""
        return self._capture.isOpened()

    # Use the _capture attribute method to display camera output on another window
    # Press a certain key to quit the application, hence the function returns false
    def display(self,
        frame: Frame
    ) -> bool:
        """Display camera on a separate window, return false if quitting application"""

        cv.imshow('window', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            return False
        
        return True

    # Returns a single frame, if it cannot then raise an error
    def read(self) -> Frame:
        """Return one singular image frame"""
        success, frame = self._capture.read()

        if not success:
            raise RuntimeError(
                "CAM_ERROR: Could not return frame from camera"
            )
        
        return frame
    
    def close(self) -> None:
        """Free camera usage by clearing capture property and quit separate window for displaying camera"""
        if self.is_open:
            self._capture.release()