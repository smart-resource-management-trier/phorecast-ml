"""
phorecast_ml – Toolkit für PV-Daten, Vorverarbeitung und Forecasting.
"""

from __future__ import annotations

# ----------------------------------------------------------
# Version (funktioniert auch, wenn das Wheel installiert ist)
# ----------------------------------------------------------
try:
    from importlib.metadata import version as _version
    __version__ = _version("phorecast_ml")
except Exception:
    __version__ = "0.0.0"   # Fallback für Editable Installs

# ----------------------------------------------------------
# Öffentliche API
# ----------------------------------------------------------
# Die Submodule selbst werden NICHT importiert, nur bereitgestellt.
# So bleibt das Package schnell importierbar.
__all__ = [
    "metrics",
    "model",
    "preprocessing",
    # "pipeline",
    "__version__"
]

import numpy as np
_global_seed: int | None = None
_global_rng = np.random.default_rng()
_global_rng_mode = "default"

def set_rng_mode(mode: str = "default"):
    global _global_rng_mode
    _global_rng_mode = mode

def set_seed(seed: int) -> None:
    """

    :param seed:
    :return:
    """
    global _global_seed
    global _global_rng

    _global_seed = int(seed)
    _global_rng = np.random.default_rng(seed)


def get_seed() -> int | None:
    """

    :return:
    """

    return _global_seed


def get_rng() -> np.random.Generator:
    """

    :return:
    """
    match _global_rng_mode:
        case "shared":
            return _global_rng

        case "independent" | "default":
            return np.random.default_rng(_global_seed)

        case _:
            raise ValueError(f"Unknown RNG mode: {_global_rng_mode}")

# ----------------------------------------------------------
# Lazy Imports bei Zugriff
# ----------------------------------------------------------
# Dadurch werden die Unterpakete importiert, wenn der Benutzer darauf zugreift:
#     import phorecast_core.models
#     from phorecast_core import preprocessing
#
# Aber beim Laden des Pakets selbst werden keine schweren Module (z. B. TensorFlow)
# geladen → schneller & ressourcenschonender.
# ----------------------------------------------------------
import importlib

def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")