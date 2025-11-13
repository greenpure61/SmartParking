import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from smartparking.parking_lot import ParkingLot


class OccupancyLogic:
    """
    Component to determine parking space occupancy based on vehicle detections.
    """

    def __init__(self, parking_lot: ParkingLot):
        self.parking_lot = parking_lot

    def process_frame(self, detections: List[Dict[str, Any]]):
        """
        Checks for overlap between detections and parking polygons and updates state.
        :param detections: List of vehicle detections (from Detector.detect).
        """
        for space_id, space in self.parking_lot.spaces.items():
            is_occupied = False
            parking_poly = np.array(space.polygon, dtype=np.int32)

            # Simple heuristic: Check if the center of the detection box falls inside the parking polygon.
            for det in detections:
                x1, y1, x2, y2 = det['box']
                # Calculate the center point of the detection box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                center_point = (center_x, center_y)

                # Use OpenCV's pointPolygonTest to check if the point is inside the polygon
                # Returns +1 if inside, 0 if on edge, -1 if outside
                # Since we pass True for measure_dist, it returns the distance.
                # Distance > 0 means the point is strictly inside.
                distance = cv2.pointPolygonTest(parking_poly, center_point, measureDist=False)

                if distance >= 0:
                    is_occupied = True
                    # Optimization: once one vehicle is found, the spot is occupied
                    break

                    # Update the parking space state with temporal smoothing
            space.update_state(is_occupied, self.parking_lot.smoothing_frames)