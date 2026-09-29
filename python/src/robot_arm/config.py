import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UTF8 = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use this common path to yaml configs directory
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"

CAMERA_CONFIG_PATH = CONFIG_DIRECTORY / "camera.yaml"
APP_CONFIG_PATH = CONFIG_DIRECTORY / "app.yaml"
YOLO_CONFIG_PATH = CONFIG_DIRECTORY / "detection.yaml"
ARM_CONFIG_PATH = CONFIG_DIRECTORY / "robot_arm.yaml"
SIMULATION_CONFIG_PATH = CONFIG_DIRECTORY / "simulation.yaml"

# Path to grounding dino model configs/weights
GROUNDING_DINO_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "grounding_dino"
)
GROUNDING_DINO_CONFIG_PATH = (
    GROUNDING_DINO_DIRECTORY
    / "GroundingDINO_SwinT_OGC.py"
)

GROUNDING_DINO_WEIGHTS_PATH = (
    GROUNDING_DINO_DIRECTORY
    / "groundingdino_swint_ogc.pth"
)

# Helper function used to load yaml files as well as error handle non existent paths
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding=UTF8) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"CONFIG_ERROR: Invalid YAML structure in {path}")

    return data

# Loading camera configuration values

# When returning this type to main, configuration values are stored together
@dataclass(frozen=True, slots=True)
class CameraConfig:
    device: int
    width: int
    height: int
    fps: int
    format: str

@dataclass(frozen=True, slots=True)
class SimulationCameraConfig:
  sensor_path: str
  width: int
  height: int
  fps: int

def load_camera_config() -> CameraConfig:
    data = load_yaml(CAMERA_CONFIG_PATH)["camera"]

    return CameraConfig(
        device = data["device"],
        width = data["width"],
        height = data["height"],
        fps = data["fps"],
        format = data["format"]
    )

def load_simulation_camera_config() -> SimulationCameraConfig:
    data = load_yaml(CAMERA_CONFIG_PATH)["simulation_camera"]
    
    return SimulationCameraConfig(
        sensor_path = data["sensor_path"],
        width = data["width"],
        height = data["height"],
        fps = data["fps"]
    )

# Loading arm configuration values

@dataclass(frozen=True, slots=True)
class ArmConfig:
    model_path: str
    target_path: str

    serial_port: str
    baudrate: int

    approach_height_offset: float
    max_velocity: float
    step_size: float
    steps_count: int

def load_arm_config() -> ArmConfig:
    data = load_yaml(ARM_CONFIG_PATH)["arm"]
       
    return ArmConfig(
        model_path = data["model_path"],
        target_path = data["target_path"],
        serial_port = data["serial_port"],
        baudrate = data["baudrate"],
        approach_height_offset = data["approach_height_offset"],
        max_velocity = data["max_velocity"],
        step_size = data["step_size"],
        steps_count = data["steps_count"]
    )

# Loading simulation configuration values

@dataclass(frozen=True, slots=True)
class SimulationConfig:
    host: str
    port: int

def load_simulation_config() -> SimulationConfig:
    data = load_yaml(SIMULATION_CONFIG_PATH)["connection_details"]
    
    return SimulationConfig(
        host = data["host"],
        port = data["port"]
    )

# Loading detection model values

def load_grounding_dino_config() -> Path:
    """Return the path to the Grounding DINO model configuration."""

    if not GROUNDING_DINO_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "Grounding DINO configuration was not found at "
            f"{GROUNDING_DINO_CONFIG_PATH}"
        )

    return GROUNDING_DINO_CONFIG_PATH

def load_grounding_dino_weights() -> Path:
    """Return the path to the Grounding DINO model weights."""

    if not GROUNDING_DINO_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            "Grounding DINO weights were not found at "
            f"{GROUNDING_DINO_WEIGHTS_PATH}"
        )

    return GROUNDING_DINO_WEIGHTS_PATH

def load_yolo_model() -> str:
    """Return the YOLO model str to pass into YOLO model."""

    if not YOLO_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "YOLO model was not found at "
            f"{YOLO_CONFIG_PATH}"
        )
    
    data = load_yaml(YOLO_CONFIG_PATH)
    return data["model"]["name"]

# App configuration values

@dataclass(frozen=True, slots=True)
class LogConfig:
    max_lines: int

@dataclass(frozen=True, slots=True)
class StyleConfig:
    camera_widget: str
    log_widget: str
    app_widget: str
    button_widget: str
    command_bar: str

@dataclass(frozen=True, slots=True)
class AppConfig:
    log: LogConfig
    styles: StyleConfig

def load_app_config() -> AppConfig:
    data = load_yaml(APP_CONFIG_PATH)

    log = LogConfig(
        max_lines = data["log"]["max_lines"]
        )
    
    styles = StyleConfig(
        camera_widget = data["styles"]["camera_widget"],
        log_widget = data["styles"]["log_widget"],
        app_widget = data["styles"]["app_widget"],
        button_widget = data["styles"]["button_widget"],
        command_bar = data["styles"]["command_bar"],
    )

    return AppConfig(
        log,
        styles,
    )