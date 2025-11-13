import cv2
import numpy as np
from smartparking.parking_lot import ParkingLot
from typing import List, Dict, Any


class Visualizer:
    """
    Handles drawing the visualization overlays on the video frame.
    """

    def __init__(self, parking_lot: ParkingLot):
        self.parking_lot = parking_lot
        self.FONT = cv2.FONT_HERSHEY_DUPLEX
        self.FONT_SCALE = 0.7
        self.FONT_THICKNESS = 1

    def draw(self, frame: cv2.Mat, detections: List[Dict[str, Any]]) -> cv2.Mat:
        """
        Draws parking spaces, detections, and status text on the frame.
        :param frame: The frame to draw on.
        :param detections: The list of vehicle detections.
        :return: The frame with visualizations applied.
        """
        # 1. Draw Parking Spaces
        for space_id, space in self.parking_lot.spaces.items():

            # Define color based on state
            if space.state == "occupied":
                color = (0, 0, 255)  # RED (BGR)
            else:
                color = (0, 255, 0)  # GREEN (BGR)

            # Convert polygon to NumPy array for cv2.polylines
            pts = np.array(space.polygon, np.int32)
            # Reshape for polylines (must be a list of contours)
            pts = pts.reshape((-1, 1, 2))

            # Draw the polygon boundary
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # Add text label to the center of the space
            # Find approximate center for text
            M = cv2.moments(pts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(frame, space_id, (cx - 20, cy + 10), self.FONT, 0.6, color, self.FONT_THICKNESS)

        # 2. Draw Vehicle Detections (Optional but helpful)
        for det in detections:
            x1, y1, x2, y2 = det['box']
            # Draw bounding box (e.g., in yellow)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 3. Draw Global Status Text
        status = self.parking_lot.get_status()
        text = f"OCCUPIED: {status['occupied']} | FREE: {status['free']} | TOTAL: {status['total']}"

        # Draw background rectangle for status bar
        (text_w, text_h), baseline = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], text_h + baseline + 10), (0, 0, 0), -1)

        # Draw the status text
        cv2.putText(frame, text, (10, text_h + 5), self.FONT, self.FONT_SCALE, (255, 255, 255), self.FONT_THICKNESS)

        return frame

    def show_frame(self, frame: cv2.Mat, window_name: str = "SmartParking"):
        """Displays the frame in an OpenCV window."""
        cv2.imshow(window_name, frame)

    def wait_for_key(self, delay: int = 1) -> bool:
        """
        Waits for a key press.
        :param delay: Milliseconds to wait.
        :return: True if 'q' was pressed, False otherwise.
        """
        key = cv2.waitKey(delay) & 0xFF
        return key == ord('q')

    def close_window(self, window_name: str = "SmartParking"):
        """Destroys all OpenCV windows."""
        cv2.destroyWindow(window_name)