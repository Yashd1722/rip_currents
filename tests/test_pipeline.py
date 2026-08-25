from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import select_candidates


def test_select_candidates_filters_by_altitude_and_height() -> None:
    records = [
        {"altitude_m": 20.0, "person_height_px": 100},
        {"altitude_m": 31.0, "person_height_px": 100},
        {"altitude_m": 25.0, "person_height_px": 60},
        {"altitude_m": 15.0, "person_height_px": 90},
    ]

    selected = select_candidates(records)

    assert len(selected) == 2
    assert selected[0]["person_height_px"] == 100
    assert selected[1]["person_height_px"] == 90
