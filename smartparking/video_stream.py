import cv2
import logging
from typing import Union, Generator

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Handles capturing frames from a webcam, video file, or RTSP stream.
    """

    def __init__(self, source: Union[int, str]):
        """
        Initializes the video capture.
        :param source: Camera index (int) or file/URL path (str).
        """
        self.source = source
        # Convert index '0' string to int 0, otherwise keep as string
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        self.cap = None

    def __enter__(self):
        """Opens the video source."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Error: Could not open video source {self.source}"
            logger.error(error_msg)
            # Raise an exception to be handled by the main loop
            raise IOError(error_msg)

        logger.info(f"Successfully opened video source: {self.source}")
        return self

    def frame_generator(self) -> Generator[tuple[bool, cv2.Mat], None, None]:
        """
        Generates frames one by one from the video stream.
        :return: Generator yielding (ret, frame) tuples.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"End of stream or read error from source {self.source}")
                break
            yield ret, frame

    def get_fps(self) -> float:
        """Returns the frame rate of the stream."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Releases the video capture when exiting the context."""
        if self.cap:
            self.cap.release()
            logger.info(f"Released video source: {self.source}")