"""
Logger utility for Kaggriculture agent.
Configured for zero overhead when debugging is disabled.
"""

import sys

DEBUG_MODE = False

def log(msg: str) -> None:
    if DEBUG_MODE:
        sys.stderr.write(f"[KAGGRICULTURE] {msg}\n")
        sys.stderr.flush()
