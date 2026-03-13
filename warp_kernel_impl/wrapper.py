import numpy as np
import warp as wp
from .warp_utils import convert_rp_to_vec2
from .warp_forwards_backprojection import get_forward_kernel
from .warp_backwards_backprojection import get_backward_kernel
from warp.jax_experimental import jax_kernel
import jax
import jax.numpy as jnp
from .constants import SarConstants
from functools import partial 

@jax.jit
def calc_entropy(image):
    power = jnp.abs(image) ** 2
    power_sum = jnp.sum(power)
    prob = power / (power_sum)  # Avoid division by zero
    entropy = -jnp.sum(prob * jnp.log(prob + 1e-12))  # Avoid log(0)
    return entropy

def run_BP(
    pixel_pos: np.ndarray,
    range_profiles: np.ndarray,
    radar_pos: np.ndarray,
    radar_vel: np.ndarray,
    ):
    forward_kernel = get_forward_kernel()
    
    # convert inputs to wp arrays
    chirp_length = range_profiles.shape[-1]
    num_radar_pos = radar_pos.shape[0]


    pixel_pos_wp = wp.array(pixel_pos, dtype=wp.vec3)
    range_profiles_wp = wp.array(convert_rp_to_vec2(range_profiles), dtype=wp.vec2)
    radar_pos_wp = wp.array(np.ascontiguousarray(radar_pos), dtype=wp.vec3)
    radar_vel_wp = wp.array(np.ascontiguousarray(radar_vel), dtype=wp.vec3)

    # call kernel
    u_dim, v_dim = pixel_pos.shape[0], pixel_pos.shape[1]
    output_image = wp.zeros((u_dim, v_dim), dtype=wp.vec2)

    print(f"kernel grid: {u_dim}, {v_dim} | num_radar_pos {num_radar_pos}")
    print(f"range profiles shape {range_profiles_wp.shape}")

    wp.launch(
        kernel=forward_kernel,
        dim=(u_dim, v_dim),
        inputs = [
            pixel_pos_wp,
            range_profiles_wp,
            radar_pos_wp,
            radar_vel_wp,
            num_radar_pos,
            False,  # left_side_only
        ],
        outputs = [
            output_image,
        ]
    )

    return output_image.numpy()

def get_jax_BP_fwd(constants: SarConstants = SarConstants()):
    forward_kernel = get_forward_kernel(constants)
    jax_BP = jax_kernel(
        forward_kernel,
        num_outputs=1,
        enable_backward=False,
        has_side_effect=False
        )
    return jax_BP

def get_jax_BP_backward(launch_dims_, constants: SarConstants = SarConstants()):
    backward_kernel = get_backward_kernel(constants)
    jax_BP_backward = jax_kernel(
        backward_kernel,
        num_outputs=1,
        enable_backward=False,
        has_side_effect=False,
        in_out_argnames=["grad_radar_pos"],
        launch_dims = launch_dims_
        )
    return jax_BP_backward

def get_jax_BP(constants: SarConstants = SarConstants()):
    bwd_launch_dims = (1024, ((1024 * 1024 + 255) // 256))
    print(f"Backward kernel launch dims: {bwd_launch_dims}")
    fwd_ffi_BP = get_jax_BP_fwd(constants)
    bwd_ffi_BP = get_jax_BP_backward(bwd_launch_dims, constants)

    @partial(jax.custom_vjp, nondiff_argnums=(4, 5))
    def jax_BP(
        pixel_pos: jnp.ndarray,
        range_profiles: jnp.ndarray,
        radar_pos: jnp.ndarray,
        radar_vel: jnp.ndarray,
        num_radar_pos: int,
        left_side_only: bool,
    ):
        return fwd_ffi_BP(
            pixel_pos,
            range_profiles,
            radar_pos,
            radar_vel,
            num_radar_pos,
            left_side_only
        )
    
    def _sar_BP_fwd(
        pixel_pos,
        range_profiles,
        radar_pos,
        radar_vel,
        num_radar_pos,
        left_side_only,
    ):
        output_image = fwd_ffi_BP(
            pixel_pos,
            range_profiles,
            radar_pos,
            radar_vel,
            num_radar_pos,
            left_side_only
        )

        # Keep only differentiable primal inputs in residuals. Nondiff args are
        # passed explicitly to the backward rule when using nondiff_argnums.
        res = (pixel_pos, range_profiles, radar_pos, radar_vel)
        return output_image, res

    def _sar_BP_bwd(num_radar_pos, left_side_only, res, adj_output_image):
        (pixel_pos,
        range_profiles,
        radar_pos,
        radar_vel) = res

        pixel_total = pixel_pos.shape[0] * pixel_pos.shape[1]
        
        radar_pos_grad = jnp.zeros_like(radar_pos)
        grad_radar_pos = bwd_ffi_BP(
            pixel_pos,
            range_profiles,
            radar_pos,
            radar_vel,
            adj_output_image[0],
            pixel_total,
            radar_pos_grad,
            output_dims=radar_pos.shape
        )

        # Return cotangents only for differentiable args:
        # (pixel_pos, range_profiles, radar_pos, radar_vel)
        return (None, None, grad_radar_pos[0], None)

    jax_BP.defvjp(_sar_BP_fwd, _sar_BP_bwd)
    return jax_BP
    

def run_jax_BP(
    pixel_pos: np.ndarray,
    range_profiles: np.ndarray,
    radar_pos: np.ndarray,
    radar_vel: np.ndarray,
    constants: SarConstants = SarConstants(),
    loss_fn: callable = calc_entropy,
):
    jax_BP = get_jax_BP(constants)
    # convert inputs to jax arrays
    chirp_length = range_profiles.shape[-1]
    num_radar_pos = int(radar_pos.shape[0])
    left_side_only = False

    pixel_pos_jax = jnp.array(pixel_pos, dtype=jnp.float32)
    range_profiles_jax = jnp.array(convert_rp_to_vec2(range_profiles), dtype=jnp.float32)
    radar_pos_jax = jnp.array(np.ascontiguousarray(radar_pos), dtype=jnp.float32)
    radar_vel_jax = jnp.array(np.ascontiguousarray(radar_vel), dtype=jnp.float32)

    u_dim, v_dim = pixel_pos.shape[0], pixel_pos.shape[1]

    def _run_jax_BP(
        pixel_pos,
        range_profiles,
        radar_pos,
        radar_vel,
        num_radar_pos,
        left_side_only,
        loss_fn = calc_entropy
    ):
        image = jax_BP(
            pixel_pos,
            range_profiles,
            radar_pos,
            radar_vel,
            num_radar_pos,
            left_side_only,
        )[0]

        entropy = loss_fn(image)
        return entropy

    output_image = jax_BP(
        pixel_pos_jax,
        range_profiles_jax,
        radar_pos_jax,
        radar_vel_jax,
        num_radar_pos,
        left_side_only,
    )[0]

    # reinterpret output as complex64
    output_image = output_image.view(jnp.complex64).reshape(output_image.shape[0], output_image.shape[1])

    # calc image, entropy and grad for testing
    entropy, grad = jax.jit(
        jax.value_and_grad(
            partial(_run_jax_BP, loss_fn=loss_fn), 
            argnums = 2),
        static_argnums=(4, 5),
    )(
        pixel_pos_jax,
        range_profiles_jax,
        radar_pos_jax,
        radar_vel_jax,
        num_radar_pos,
        left_side_only,
    )

    return output_image, entropy, grad