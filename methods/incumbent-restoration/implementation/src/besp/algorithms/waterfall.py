"""Waterfall algorithm exports."""

from ..algorithms.jax.deterministic import WaterfallBESP_PE, waterfall_besp_pe
from ..algorithms.jax.deterministic import GEO_ALPHA, GeoWaterfallBESP_PE, geo_waterfall_besp_pe
from ..algorithms.jax.deterministic import DD_M_MAX, DDWaterfallBESP_PE, dd_waterfall_besp_pe
from ..domains.control import WaterfallISP_Control, waterfall_isp_control
from ..domains.control import GeoWaterfallISP_Control, geo_waterfall_isp_control
from ..domains.control import DDWaterfallISP_Control, dd_waterfall_isp_control


H_INIT = 1e6
H_MIN = 1.0
STALL_PATIENCE = 3
DECAY_FACTOR = 0.1
DIVERGENCE_MULT = 1e4

__all__ = [
    "DECAY_FACTOR",
    "DDWaterfallBESP_PE",
    "DDWaterfallISP_Control",
    "DD_M_MAX",
    "DIVERGENCE_MULT",
    "GEO_ALPHA",
    "GeoWaterfallBESP_PE",
    "GeoWaterfallISP_Control",
    "H_INIT",
    "H_MIN",
    "STALL_PATIENCE",
    "WaterfallBESP_PE",
    "WaterfallISP_Control",
    "dd_waterfall_besp_pe",
    "dd_waterfall_isp_control",
    "geo_waterfall_besp_pe",
    "geo_waterfall_isp_control",
    "waterfall_besp_pe",
    "waterfall_isp_control",
]
