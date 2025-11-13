import threading
import uvicorn
import logging
from fastapi import FastAPI
from smartparking.parking_lot import ParkingLot

logger = logging.getLogger(__name__)

# Global variable to hold the FastAPI application instance
# We use this global access to the ParkingLot instance
app = FastAPI(title="SmartParking API")
global_parking_lot: ParkingLot = None


def init_api(parking_lot_instance: ParkingLot):
    """Initializes the global parking lot instance for the API to use."""
    global global_parking_lot
    global_parking_lot = parking_lot_instance


@app.get("/status")
def get_parking_status():
    """
    Returns the current occupancy status of the entire parking lot.
    """
    if global_parking_lot is None:
        return {"error": "Parking lot data not initialized"}, 500

    # ParkingLot.get_status() already returns the required JSON structure
    return global_parking_lot.get_status()


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Runs the FastAPI server in a separate thread.
    """

    def start_server():
        logger.info(f"Starting API server on http://{host}:{port}")
        # Note: log_level is set to 'warning' to prevent cluttering the video processing logs
        uvicorn.run(app, host=host, port=port, log_level="warning")

    # Start the server in a daemon thread so it automatically closes when the main thread exits
    api_thread = threading.Thread(target=start_server, daemon=True)
    api_thread.start()
    return api_thread