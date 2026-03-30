"""Memory helpers, momentum filter, and detector constants."""

import numpy as np
import psutil


# ─── PMT / MC constants ───────────────────────────────────────────────────────
PMT_RADIUS = 110.0   # mm, cathode radius used for MC overlap fractions
MC_SAMPLES = 10000   # Monte Carlo samples for fractional PMT overlap

# ─── SSD UID constants ────────────────────────────────────────────────────────
SSD_UID_WALL = 1965
SSD_UID_PIT  = 1966
SSD_UID_BOT  = 1967
SSD_UID_TOP  = 1968


def get_chunk_size() -> int:
    """Determine optimal chunk size based on available memory."""
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if available_mem_gb > 400:
        return 50_000
    elif available_mem_gb > 200:
        return 30_000
    elif available_mem_gb > 50:
        return 20_000
    elif available_mem_gb > 30:
        return 15_000
    elif available_mem_gb > 20:
        return 10_000
    return 5_000


def checkRadialMomentumVectorized(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    px: np.ndarray, py: np.ndarray, pz: np.ndarray,
) -> np.ndarray:
    """Return True where radial momentum component points outward (barrel region)."""
    return (x * px + y * py) >= 0
