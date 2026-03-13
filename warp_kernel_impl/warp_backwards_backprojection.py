import warp as wp
from functools import lru_cache
from .constants import SarConstants
from .warp_utils import complex_mul, interp_linear


@lru_cache(maxsize=None)
def get_backward_kernel(constants: SarConstants = SarConstants()):
    c_0 = wp.constant(constants.c_0)
    B = wp.constant(constants.b)
    TP = wp.constant(constants.tp)
    F_0 = wp.constant(constants.f_0)
    DR = wp.constant(constants.dr)
    MU = wp.constant(constants.b / constants.tp)
    DR_recip = wp.constant(1.0 / constants.dr)
    C_half_recip = wp.constant(2.0 / constants.c_0)
    F_0_over_MU = wp.constant(constants.f_0 / MU)
    F_0_over_C_0 = wp.constant(constants.f_0 / constants.c_0)
    
    @wp.kernel
    def sar_backward_trajectory_explicit(
        # Primal Inputs needed for re-computation
        pixel_pos: wp.array2d(dtype=wp.vec3f),
        range_profiles: wp.array2d(dtype=wp.vec2f),
        radar_pos: wp.array(dtype=wp.vec3f),
        radar_vel: wp.array(dtype=wp.vec3f),
        # Gradients from the Tape (dLoss/dImage)
        adj_image: wp.array2d(dtype=wp.vec2),
        # Configuration
        num_pixels_total: wp.int32,
        # Output: The gradient for the Trajectory
        grad_radar_pos: wp.array(dtype=wp.vec3),
    ):
        # Dim 0: The Chirp Index
        # Dim 1: The Pixel Block Index (e.g., if we have 1M pixels and block size 256, this is 0..4095)
        chirp_idx, pixel_block_idx = wp.tid()

        # Configuration for tiling
        block_size = 256  # Should match block_dim used in launch
        start_pixel = pixel_block_idx * block_size

        # Early exit if out of bounds
        if start_pixel >= num_pixels_total:
            return

        # 1. Load Chirp Data (Invariant for this thread)
        r_pos = radar_pos[chirp_idx]
        r_vel = radar_vel[chirp_idx]

        # Initialize Local Accumulator (Register)
        # This prevents hitting global memory for every pixel
        grad_accum = wp.vec3(0.0, 0.0, 0.0)

        # 2. Loop over the chunk of pixels assigned to this thread
        for i in range(block_size):
            p_idx = start_pixel + i
            if p_idx >= num_pixels_total:
                break

            # Convert linear index to 2D (assuming square grid for simplicity, or pass stride)
            # Assuming pixel_pos is (width, height)
            width = pixel_pos.shape[0]
            px = p_idx % width
            py = p_idx // width

            # --- RECOMPUTE FORWARD STATE ---
            p_pos = pixel_pos[px, py]
            disp = p_pos - r_pos
            dist = wp.length(disp)

            # Recompute Phase
            tau = dist * C_half_recip
            val = tau * F_0
            angle = -2.0 * wp.PI * (val - wp.floor(val))
            s_hyp = wp.vec2(wp.cos(angle), wp.sin(angle))

            # Recompute Sample
            v_relative = wp.dot(r_vel, disp / dist)  # radial velocity
            range_idx = (dist - (F_0_over_MU) * v_relative ) * DR_recip
            sample = interp_linear(range_profiles, chirp_idx, range_idx)

            # --- BACKWARD PROPAGATION ---

            # 1. Load incoming gradients (dL/dImage)
            adj = adj_image[px, py]
        

            # 2. Gradient through Complex Multiply
            # val = s_hyp * sample
            # dL/ds_hyp_real = sample.real * adj_r + sample.imag * adj_i
            # dL/ds_hyp_imag = -sample.imag * adj_r + sample.real * adj_i
            d_shyp_real = sample[0] * adj[0] + sample[1] * adj[1]
            d_shyp_imag = -sample[1] * adj[0] + sample[0] * adj[1]

            # 3. Gradient through Phase (Euler)
            # s_hyp = cos(a) + i*sin(a)
            # d(real)/da = -sin(a); d(imag)/da = cos(a)
            d_angle = d_shyp_real * (-s_hyp[1]) + d_shyp_imag * (s_hyp[0])

            # 4. Gradient through Phase History Formula
            # angle = -2pi * frac(dist * const)
            # d_angle/d_dist = -2pi * (2/C * FC)
            d_dist = d_angle * (-4.0 * wp.PI * F_0_over_C_0)

            # 5. Gradient through Geometry
            # dist = length(pixel - radar)
            # d_dist/d_radar = -(pixel - radar) / dist (normalized direction pointing TO pixel)
            dir_vec = disp / dist

            # Accumulate into register
            # Chain rule: dL/dRadar = dL/dDist * dDist/dRadar
            #                       = d_dist   * (-dir_vec)
            grad_accum += dir_vec * (-d_dist)

        # 3. Atomic Add the accumulated chunk to the global array
        # This reduces atomic collisions by a factor of 256 (or block_size)
        wp.atomic_add(grad_radar_pos, chirp_idx, grad_accum)

    return sar_backward_trajectory_explicit