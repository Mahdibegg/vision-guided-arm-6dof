import cv2 as cv

from ultralytics import YOLO
from arm.vision.camera import Camera
from arm.vision.detect import Detector, highlight_objects
from arm.config_loader import (
    load_camera_config,
    load_grounding_dino_config,
    load_grounding_dino_weights,
    load_yolo_model
)

def main() -> None:
    
    # Load configs
    camera_config = load_camera_config()

    # These aren't actually used, they are just needed for passing
    dino_config = load_grounding_dino_config()
    dino_weights = load_grounding_dino_weights()

    # Yolo model for real time detection
    yolo_model = load_yolo_model()

    camera = Camera(
        camera_config.device,
        camera_config.width,
        camera_config.height,
        camera_config.fps,
        # Required depending on the camera you are using
        camera_config.format
    )

    # Initialize detection system with the loaded configs and model
    detection_system = Detector(
        dino_config,
        dino_weights,
        yolo_model
    )

    # Loop can be set to false if user presses the button to quit camera application
    loop = True

    try:
        while loop:
            frame = camera.read()
            objects_identified = detection_system.analyse(frame)
            loop = camera.display(highlight_objects(frame, objects_identified))
    finally:
        camera.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()