import warp as wp
from functools import lru_cache
from .constants import SarConstants
from .warp_utils import complex_mul, interp_linear

@lru_cache(maxsize=None)
def get_forward_kernel(constants: SarConstants = SarConstants()):
    c_0 = wp.constant(constants.c_0)
    B = wp.constant(constants.b)
    TP = wp.constant(constants.tp)
    F_0 = wp.constant(constants.f_0)
    DR = wp.constant(constants.dr)
    mu = constants.b / constants.tp
    DR_recip = wp.constant(1.0 / constants.dr)
    two_over_c = wp.constant(2.0 / constants.c_0)
    F_0_over_MU = wp.constant(constants.f_0 / mu)
    
    @wp.kernel
    def sar_image_kernel_local_accum(
        pixel_pos: wp.array2d(dtype=wp.vec3f),  # (u_dim, v_dim) grid of 3D positions
        range_profiles: wp.array2d(
            dtype=wp.vec2f
        ),  # (num_radar_pos, chirp_length) complex as vec2f
        radar_pos: wp.array(dtype=wp.vec3f),  # (num_radar_pos,) radar positions
        radar_vel: wp.array(dtype=wp.vec3f),  # (num_radar_pos,) radar velocities
        num_radar_pos: wp.int32,  # number of radar positions to loop over
        left_side_only: wp.bool,  # if True, only project pixels on the left side of the aircraft
        output_image: wp.array2d(dtype=wp.vec2f),  # (u_dim, v_dim) output image real part
    ):
        """
        SAR backprojection imaging kernel with local accumulation.

        Thread mapping: 2D grid (x, y) where:
        - x: azimuth index (0 to u_dim-1)
        - y: range index (0 to v_dim-1)

        Each thread loops over all radar positions and accumulates locally in registers,
        then writes the final result once (no atomic operations needed).

        This version trades parallelism for reduced memory contention.
        """
        y, x = wp.tid()

        # Get current pixel position
        current_pixel_pos = pixel_pos[x, y]

        # Local accumulators (in registers)
        accum = wp.vec2f(0.0, 0.0)

        # Loop over all radar positions
        for chirp in range(num_radar_pos):
            current_radar_pos = radar_pos[chirp]
            current_radar_vel = radar_vel[chirp]

            # Compute displacement and distance
            displacement = current_radar_pos - current_pixel_pos
            distance = wp.length(displacement)

            # Check if pixel is on the left side of the aircraft (if flag is set)
            if left_side_only:
                # In NED: left_vector = velocity × down = velocity × (0, 0, 1)
                # We are not using NED though, z is up; down becomes (0, 0, -1), so left_vector = velocity × (0, 0, -1) = (-vy, vx, 0)
                left_vector = wp.vec3f(-current_radar_vel[1], current_radar_vel[0], 0.0)
                left_vec_norm = wp.length(left_vector)
                left_unit_vector = left_vector / left_vec_norm

                # If dot product with displacement is negative, pixel is on the right side -> skip
                if (wp.dot(displacement / distance, left_unit_vector)) < 0.17:
                    continue

            v_relative = wp.dot(current_radar_vel, displacement / distance)
            # Compute time delay (velocity term neglected)
            tau = distance * two_over_c
            d_hyp = distance - (F_0_over_MU) * v_relative  

            # Compute phase for matched filter
            tau_f0 = tau * F_0
            s_hyp_angle = -2.0 * wp.pi * (tau_f0 - wp.floor(tau_f0))

            # Compute complex exponential (s_hyp = cos + i*sin)
            s_hyp = wp.vec2f(wp.cos(s_hyp_angle), wp.sin(s_hyp_angle))

            # Compute range bin index for interpolation
            range_idx = d_hyp * DR_recip

            # Get interpolated range profile value
            range_val = interp_linear(range_profiles, chirp, range_idx)

            # Complex multiply: pixel_val = s_hyp * range_val
            pixel_val = complex_mul(s_hyp, range_val)

            # Accumulate locally (no atomic needed)
            accum += pixel_val

        # Write final result (single write per pixel)
        output_image[x, y] = accum
    
    return sar_image_kernel_local_accum