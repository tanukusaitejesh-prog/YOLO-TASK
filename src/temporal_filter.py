"""
Temporal Consensus Filter for Robotics Perception
==================================================
Implements a safety-asymmetric sliding-window temporal filter
to prevent transient single-frame Closed -> Open misclassifications
from triggering collision hazards in autonomous mobile robots (AMRs).

Safety Policy:
- State transitions to OPEN require K=3 consecutive agreeing frames.
- State transitions to CLOSED occur immediately on 1 frame (asymmetric fail-safe).
"""

import argparse
import sys
from collections import deque
from typing import List, Optional

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None


class TemporalConsensusFilter:
    """Sliding-window temporal filter for door state stability."""

    def __init__(self, window_size: int = 3, open_threshold: int = 3):
        """
        Args:
            window_size: Number of consecutive history frames to maintain (K=3).
            open_threshold: Consecutive OPEN detections required to declare traversability (M=3).
        """
        self.window_size = window_size
        self.open_threshold = open_threshold
        self.history = deque(maxlen=window_size)
        self.current_state = "UNKNOWN"

    def update(self, raw_prediction: Optional[str]) -> str:
        """
        Update the filter with the latest single-frame detection.

        Args:
            raw_prediction: 'door_open', 'door_closed', or None (no detection)

        Returns:
            Filtered consensus state: 'OPEN' (traversable), 'CLOSED' (obstacle), or 'HOLD'
        """
        self.history.append(raw_prediction)

        # Count occurrences in the current window
        open_count = sum(1 for s in self.history if s == "door_open")
        closed_count = sum(1 for s in self.history if s == "door_closed")

        # Asymmetric Safety Policy:
        # 1. Closed door detected in the window -> immediately hold/revert to CLOSED (safe)
        if closed_count > 0:
            self.current_state = "CLOSED"
        # 2. Fully confirmed open across K consecutive frames -> transition to OPEN
        elif len(self.history) == self.window_size and open_count >= self.open_threshold:
            self.current_state = "OPEN"
        # 3. Warming up or ambiguous -> preserve previous state or fail-safe to CLOSED
        else:
            if self.current_state == "UNKNOWN":
                self.current_state = "CLOSED"

        return self.current_state

    def reset(self):
        """Clear history buffer upon robot re-localization or scene jump."""
        self.history.clear()
        self.current_state = "UNKNOWN"


def run_unit_tests():
    """Verify consensus filter transitions against noisy simulation sequences."""
    print("=" * 65)
    print("  Running Temporal Consensus Filter Unit Tests")
    print("=" * 65)

    filter_obj = TemporalConsensusFilter(window_size=3, open_threshold=3)

    # Test 1: Transient 1-frame glitch (Closed -> Open glitch -> Closed)
    glitch_sequence = ["door_closed", "door_closed", "door_open", "door_closed"]
    expected_1 = ["CLOSED", "CLOSED", "CLOSED", "CLOSED"]
    actual_1 = [filter_obj.update(x) for x in glitch_sequence]
    assert actual_1 == expected_1, f"Glitch test failed: {actual_1} vs {expected_1}"
    print("[PASS] Test 1: Transient 1-frame OPEN glitch safely suppressed.")

    # Test 2: Genuine opening sequence (3 consecutive opens)
    filter_obj.reset()
    open_sequence = ["door_open", "door_open", "door_open"]
    expected_2 = ["CLOSED", "CLOSED", "OPEN"]
    actual_2 = [filter_obj.update(x) for x in open_sequence]
    assert actual_2 == expected_2, f"Opening test failed: {actual_2} vs {expected_2}"
    print("[PASS] Test 2: Genuine door opening confirms OPEN on 3rd frame.")

    # Test 3: Instant fail-safe stop on door closing (1 frame reaction)
    filter_obj.update("door_closed")
    assert filter_obj.current_state == "CLOSED", "Immediate closing reaction failed"
    print("[PASS] Test 3: Immediate fail-safe fallback to CLOSED on first closed detection.")

    print("=" * 65)
    print("  All Temporal Filter Unit Tests Passed Successfully!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Test temporal consensus filter for robot perception.")
    parser.add_argument("--test", action="store_true", default=True, help="Run simulation test suite.")
    args = parser.parse_args()

    if args.test:
        run_unit_tests()


if __name__ == "__main__":
    main()
