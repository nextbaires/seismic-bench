"""Evaluation protocol: grid, rolling windows, consistency tests, runner."""

from .grid import CALIFORNIA, Grid
from .split import Window, rolling_windows

__all__ = ["Grid", "CALIFORNIA", "Window", "rolling_windows"]
