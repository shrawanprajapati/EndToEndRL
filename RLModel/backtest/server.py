"""Compatibility wrapper for the canonical backend server.

The FastAPI app now lives in backend/main.py. This module only exists so older
commands that import backtest.server:app keep working while pointing at the
same backend application.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app  # noqa: E402,F401


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
