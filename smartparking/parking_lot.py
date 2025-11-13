import yaml
import logging
from typing import Dict, List, Tuple, Literal

logger = logging.getLogger(__name__)

# Define a type alias for the state
ParkingState = Literal["occupied", "free"]


class ParkingSpace:
    """Represents a single parking space with its ID, polygon, and current state."""

    def __init__(self, space_id: str, polygon: List[Tuple[int, int]]):
        self.id = space_id
        # Ensure polygon coordinates are integers
        self.polygon = [tuple(map(int, p)) for p in polygon]
        self.state: ParkingState = "free"
        self.state_history: List[ParkingState] = []  # For temporal smoothing

    def update_state(self, is_occupied_current_frame: bool, smoothing_frames: int):
        """
        Updates the state using temporal smoothing.
        :param is_occupied_current_frame: Whether a vehicle was detected in the space this frame.
        :param smoothing_frames: The number of consecutive frames required for a state change.
        """
        # Record current frame's raw state
        raw_state = "occupied" if is_occupied_current_frame else "free"

        # Keep a history of the last N frames' raw states
        self.state_history.append(raw_state)
        if len(self.state_history) > smoothing_frames:
            self.state_history.pop(0)

        # Only change the official state if the history is consistent
        if len(self.state_history) == smoothing_frames:

            # Check for a transition to 'occupied'
            if raw_state == "occupied" and all(s == "occupied" for s in self.state_history):
                self.state = "occupied"

            # Check for a transition to 'free'
            elif raw_state == "free" and all(s == "free" for s in self.state_history):
                self.state = "free"

            # Otherwise, the state remains unchanged (temporal smoothing in effect)


class ParkingLot:
    """Manages all parking spaces and their collective status."""

    def __init__(self, config_path: str, smoothing_frames: int = 5):
        self.spaces: Dict[str, ParkingSpace] = {}
        self.smoothing_frames = smoothing_frames
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Loads parking space definitions from the YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            spaces_data = config.get('parking_spaces', {})
            if not spaces_data:
                logger.error(f"Configuration file {config_path} has no 'parking_spaces' defined.")
                return

            for space_id, data in spaces_data.items():
                polygon = data.get('polygon')
                if polygon and len(polygon) >= 3:
                    self.spaces[space_id] = ParkingSpace(space_id, polygon)
                else:
                    logger.warning(f"Skipping parking space {space_id}: Invalid polygon definition.")

            logger.info(f"Loaded {len(self.spaces)} parking spaces from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load parking configuration from {config_path}: {e}")
            raise

    def get_status(self) -> Dict:
        """Returns the current occupancy status summary."""
        occupied_count = sum(1 for space in self.spaces.values() if space.state == "occupied")
        total_count = len(self.spaces)
        free_count = total_count - occupied_count

        return {
            "total": total_count,
            "occupied": occupied_count,
            "free": free_count,
            "spaces": [{"id": space.id, "state": space.state} for space in self.spaces.values()]
        }