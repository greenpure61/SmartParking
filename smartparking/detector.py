import cv2
import logging
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# List of COCO classes that represent a vehicle
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    67: 'cell phone',
}


class Detector:
    """
    Loads and runs the YOLO model for vehicle detection.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        :param model_path: Path to the trained YOLO model file (e.g., 'yolov8n.pt').
        :param confidence_threshold: Minimum confidence score for a detection to be considered.
        """
        try:
            # Load the model using Ultralytics
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            raise

        self.confidence_threshold = confidence_threshold

    def detect(self, frame: cv2.Mat) -> List[Dict[str, Any]]:
        """
        Performs detection on a single frame and filters for vehicles.

        :param frame: The BGR image frame from OpenCV.
        :return: A list of dictionaries, each representing a vehicle detection.
                 [{'box': (x1, y1, x2, y2), 'class': 'car', 'conf': 0.95}, ...]
        """
        # Run inference on the frame
        results: List[Results] = self.model(frame, verbose=False, conf=self.confidence_threshold)

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # --- Midlertidig debugging: Print ALLE detektioner ---
                class_name = self.model.names[class_id] if self.model.names else f"Class_{class_id}"
                logger.info(f"Detected: {class_name}, Conf: {confidence:.2f}, Box: {(x1, y1, x2, y2)}")
                # --- Slut på midlertidig debugging ---

                # Filter by vehicle classes (denne linje er som før)
                if class_id in VEHICLE_CLASSES:
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'class': VEHICLE_CLASSES[class_id],
                        'conf': confidence
                    })

        return detections