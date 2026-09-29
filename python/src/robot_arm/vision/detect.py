from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import groundingdino.datasets.transforms as T
import torch
from groundingdino.util.inference import load_model, predict
from PIL import Image
from torch import Tensor
from torchvision.ops import box_convert
from ultralytics import YOLO

from .cameras.types import Frame

TARGET_CONFIDENCE_THRESHOLD = 0.60


# Custom data type to represent Detection
@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int

    # Phrase matched by the grounding dino model
    description: str | None = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def centre(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )


class Detector:
    """Detect objects in camera frames using natural-language descriptions."""

    # All model configs passed into Detector class
    # Keep all models private to Detector
    def __init__(
        self,
        config_path: str | Path,
        weights_path: str | Path,
        yolo_model_name: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

        self._model = load_model(
            str(config_path),
            str(weights_path),
            device=self._device,
        )

        self._yolo_model = YOLO(yolo_model_name)

        self._transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    # YOLO model used for frame analysing since its quicker in real time than grounding dino
    def analyse(
        self,
        frame: Frame,
    ) -> list[Detection]:
        """Identify all objects in the current frame using YOLO."""

        results = self._yolo_model(frame)
        detections = []

        # Only some sections of the data class are required
        for result in results:
            for box in result.boxes:
                class_name = self._yolo_model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(class_name, confidence, x1, y1, x2, y2))
        return detections

    def detect(
        self,
        frame: Frame,
        description: str,
    ) -> list[Detection]:
        """Detect objects matching a natural-language description."""

        if frame.size == 0:
            return []

        description = description.strip()

        if not description:
            return []

        image = self._prepare_frame(frame)

        boxes, logits, phrases = predict(
            model=self._model,
            image=image,
            caption=description,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            device=self._device,
        )

        return self._create_detections(
            frame=frame,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )

    def _prepare_frame(self, frame: Frame) -> Tensor:
        """Convert an OpenCV BGR frame into a Grounding DINO tensor."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        image, _ = self._transform(pil_image, None)
        return image

    @staticmethod
    def _create_detections(
        frame: Frame,
        boxes: Tensor,
        logits: Tensor,
        phrases: list[str],
    ) -> list[Detection]:
        """Convert normalized Grounding DINO boxes into pixel coordinates."""

        frame_height, frame_width = frame.shape[:2]

        boxes = box_convert(
            boxes=boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        scale = torch.tensor(
            [frame_width, frame_height, frame_width, frame_height],
            device=boxes.device,
        )

        pixel_boxes = (boxes * scale).round().to(torch.int32).cpu()
        confidences = logits.detach().float().cpu()

        detections: list[Detection] = []

        for box, confidence, phrase in zip(
            pixel_boxes,
            confidences,
            phrases,
            strict=True,
        ):
            x1, y1, x2, y2 = box.tolist()

            # Ensure coordinates remain inside the frame
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    class_name=phrase,
                    description=phrase,
                    confidence=float(confidence.item()),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections


def is_valid_detection(
    detection: Detection | None,
) -> bool:
    """Return whether a detection is suitable as a target."""

    return (
        detection is not None
        and detection.confidence >= TARGET_CONFIDENCE_THRESHOLD
        and detection.width > 0
        and detection.height > 0
    )


def highlight_objects(
    frame: Frame, objects: list[Detection], target_object: Detection | None = None
) -> Frame:
    """Highlight the all objects detected on the camera window"""

    colour: tuple[int, int, int]

    highlighted_frame = frame.copy()

    # Every object detected will be pinpointed on the camera window
    for obj in objects:
        # Limit confidence to 0-1.0
        confidence = max(0.0, min(obj.confidence, 1.0))

        # Highlight target object with blue
        if target_object == obj:
            colour = (255, 0, 0)
        else:
            # Change the colour based on the confidence
            colour = (0, round(confidence * 255), round(255 - confidence * 255))

        highlighted_frame = cv2.rectangle(
            highlighted_frame, (obj.x1, obj.y1), (obj.x2, obj.y2), colour, 3
        )
        highlighted_frame = cv2.putText(
            highlighted_frame,
            obj.class_name,
            (obj.x1 + 5, obj.y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return highlighted_frame
