import argparse
import logging
import time
import sys
import os

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('SmartParking')

# --- Dynamic imports from the smartparking package ---
try:
    from smartparking.video_stream import VideoStream
    from smartparking.detector import Detector
    from smartparking.parking_lot import ParkingLot
    from smartparking.occupancy_logic import OccupancyLogic
    from smartparking.visualizer import Visualizer
    from smartparking.api import run_api_server, init_api
except ImportError as e:
    logger.error(
        "Failed to import SmartParking modules. Ensure you are running from the project root and requirements are installed.")
    logger.error(f"Import Error: {e}")
    sys.exit(1)


def parse_args():
    """Parses command line arguments."""

    HARDCODED_VIDEO_PATH = "parkingvideo.mp4"

    parser = argparse.ArgumentParser(description="SmartParking: Real-time parking lot occupancy monitoring.")

    parser.add_argument("--source", type=str, default=HARDCODED_VIDEO_PATH,
                        help="Video source: '0' for webcam, path to video file, or RTSP URL.")

    parser.add_argument("--config", type=str, default="config/parking_zones.yaml",
                        help="Path to the YAML configuration file defining parking zones.")

    parser.add_argument("--model", type=str, default="models/yolov8n.pt",
                        help="Path to the trained YOLO model file (e.g., yolov8n.pt).")

    parser.add_argument("--api-port", type=int, default=8000,
                        help="Port for the FastAPI status API.")

    parser.add_argument("--smoothing-frames", type=int, default=5,
                        help="Number of consecutive frames required for a state change (temporal smoothing).")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialization and Setup
    logger.info("--- SmartParking System Starting ---")

    if not os.path.exists(args.model):
        logger.error(
            f"YOLO model not found at: {args.model}. Please download a YOLOv8 model (e.g., yolov8n.pt) and place it in the models/ directory.")
        sys.exit(1)

    try:
        # Load Parking Lot configuration
        parking_lot = ParkingLot(config_path=args.config, smoothing_frames=args.smoothing_frames)
        if not parking_lot.spaces:
            logger.error("No valid parking spaces defined. Exiting.")
            sys.exit(1)

        # Initialize core components
        detector = Detector(model_path=args.model)
        occupancy_logic = OccupancyLogic(parking_lot=parking_lot)
        visualizer = Visualizer(parking_lot=parking_lot)

    except Exception as e:
        logger.error(f"Application setup failed: {e}")
        sys.exit(1)

    # 2. Start API Server in a separate thread
    init_api(parking_lot)
    api_thread = run_api_server(port=args.api_port)

    # 3. Main Video Processing Loop
    try:
        # Use context manager for graceful video stream release
        with VideoStream(source=args.source) as video_stream:

            # Use source FPS to calculate a better frame processing delay
            # If FPS is 30, delay should be around 1000/30 = 33ms
            fps = video_stream.get_fps()
            delay_ms = int(1000 / fps) if fps > 0 else 1

            logger.info(f"Video source FPS: {fps:.2f}. Using delay of {delay_ms}ms.")

            for ret, frame in video_stream.frame_generator():
                if not ret:
                    break

                start_time = time.time()

                # A. Detection
                detections = detector.detect(frame)

                # B. Occupancy Logic
                occupancy_logic.process_frame(detections)

                # C. Visualization
                visual_frame = visualizer.draw(frame, detections)
                visualizer.show_frame(visual_frame)

                # D. Control (Quit check)
                if visualizer.wait_for_key(delay=delay_ms):
                    break

                # Logging processing time (optional, for optimization)
                processing_time = (time.time() - start_time) * 1000  # in ms
                # logger.debug(f"Frame processed in {processing_time:.2f}ms")

    except IOError as e:
        logger.error(f"Video stream error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        logger.info("Main processing loop finished. Cleaning up.")
        # Ensure OpenCV windows are destroyed
        visualizer.close_window()

        # NOTE: The API thread is a daemon thread, so it will exit when the main thread exits.
        logger.info("--- SmartParking System Shut Down ---")


if __name__ == "__main__":
    # Ensure the correct path structure for relative config/model paths
    # Note: When running a development script, it's often run from the project root.
    # We will assume 'main.py' is run from the 'smartparking/' directory's parent.
    main()