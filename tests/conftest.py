"""Optional packages never hide the core/controller/agent test directories."""

import importlib.util


def pytest_report_header() -> list[str]:
    missing = [
        name
        for name in ("rerun", "cv2", "mujoco", "i2rt")
        if importlib.util.find_spec(name) is None
    ]
    return (
        [
            "optional runtime packages absent (individual integration tests may skip): "
            f"{', '.join(missing)}"
        ]
        if missing
        else []
    )
