"""Reference models. Beat these before claiming skill."""

from .base import Forecaster, validate_forecast
from .etas import ETASBaseline
from .poisson import PoissonBaseline

__all__ = ["Forecaster", "validate_forecast", "PoissonBaseline", "ETASBaseline"]
