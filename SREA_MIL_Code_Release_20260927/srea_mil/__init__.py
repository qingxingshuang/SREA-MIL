"""Public SREA-MIL research code."""

from .model import MODEL_SPECS, SREAConfig, TwoStreamSREAMIL, build_model

__all__ = ["MODEL_SPECS", "SREAConfig", "TwoStreamSREAMIL", "build_model"]

__version__ = "0.1.0"
