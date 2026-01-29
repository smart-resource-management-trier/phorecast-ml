"""
phorecast_core – Toolkit für PV-Daten, Vorverarbeitung und Forecasting.
"""

from __future__ import annotations

# ----------------------------------------------------------
# Version (funktioniert auch, wenn das Wheel installiert ist)
# ----------------------------------------------------------
try:
    from importlib.metadata import version as _version
    __version__ = _version("phorecast-core")
except Exception:
    __version__ = "0.0.0"   # Fallback für Editable Installs

# ----------------------------------------------------------
# Öffentliche API
# ----------------------------------------------------------
# Die Submodule selbst werden NICHT importiert, nur bereitgestellt.
# So bleibt das Package schnell importierbar.
__all__ = [
    "model",
    "preprocessing",
    "pipeline",
    "__version__"
]

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