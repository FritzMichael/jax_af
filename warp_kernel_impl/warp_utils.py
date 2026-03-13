import warp as wp
import numpy as np

def convert_rp_to_vec2(rp: np.ndarray) -> np.ndarray:
    """Convert complex range profiles to vec2 format for warp."""
    return rp.astype(np.complex64).view(np.float32).reshape(rp.shape + (2,))

@wp.func
def complex_mul(a: wp.vec2f, b: wp.vec2f) -> wp.vec2f:
    """Complex multiplication using vec2f: (x=real, y=imag)"""
    return wp.vec2f(
        a[0] * b[0] - a[1] * b[1],
        a[0] * b[1] + a[1] * b[0],
    )

@wp.func
def interp_linear(
    range_profiles: wp.array2d(dtype=wp.vec2f),
    chirp: wp.int32,
    range_idx: wp.float32,
) -> wp.vec2f:
    """
    Linear interpolation into range profiles.
    Returns zero if index is out of bounds.
    """
    idx_floor = wp.int32(wp.floor(range_idx))
    idx_ceil = idx_floor + 1
    alpha = range_idx - wp.float32(idx_floor)

    chirp_length = range_profiles.shape[1]

    # Bounds check - return zero if out of range
    if idx_floor < 0 or idx_ceil >= chirp_length:
        return wp.vec2f(0.0, 0.0)

    val_floor = range_profiles[chirp, idx_floor]
    val_ceil = range_profiles[chirp, idx_ceil]

    return (1.0 - alpha) * val_floor + alpha * val_ceil