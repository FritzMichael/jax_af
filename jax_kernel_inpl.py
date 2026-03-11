from typing import NamedTuple
import jax.numpy as jnp
import jax

class RadarParams(NamedTuple):
    F0: float  # Starting frequency of the chirp
    DR: float  # Range resolution
    C: float   # Speed of light
    dt: float = 1/1300  # Time step between samples (default 1 microsecond)


@jax.jit
def sample_range_profile(chirp, idx):
    safe_idx = jnp.clip(idx, 0, len(chirp) - 1)
    return chirp[safe_idx]

@jax.jit
def backproject_single(pixel_pos, radar_pos, chirp, params: RadarParams):
    displacement = pixel_pos - radar_pos
    distance = jnp.linalg.norm(displacement, axis=-1)

    range_idx = distance / params.DR

    range_val = (sample_range_profile(chirp, jnp.round(range_idx).astype(int)))
    tau = 2.0 * distance / params.C
    tau_f0 = tau * params.F0
    s_hyp_angle = -2.0 * jnp.pi * tau_f0
    s_hyp = jnp.exp(1j * s_hyp_angle)

    pixel_val = range_val * s_hyp
    return pixel_val

@jax.jit
def backproject_sum(pixel_positions, radar_positions, chirps, params):
    # pixel_positions: (1024, 1024, 3)
    # radar_positions: (1024, 3)
    # chirps: (1024, 8192)

    # f for a single pixel, summed over all radar positions via scan
    def pixel_sum(pixel_pos):
        def body(carry, args):
            radar_pos, chirp = args
            return carry + backproject_single(pixel_pos, radar_pos, chirp, params), None
        
        total, _ = jax.lax.scan(body, jnp.zeros((), dtype=jnp.complex64), (radar_positions, chirps))
        return total

    # vmap over both spatial axes
    return jax.vmap(jax.vmap(pixel_sum))(pixel_positions)

@jax.jit
def calc_entropy(image):
    power = jnp.abs(image) ** 2
    power_sum = jnp.sum(power)
    prob = power / (power_sum)  # Avoid division by zero
    entropy = -jnp.sum(prob * jnp.log(prob + 1e-12))  # Avoid log(0)
    return entropy

@jax.jit
def calc_energy(image):
    return -jnp.sum(jnp.abs(image) ** 6)

@jax.jit
def integrate_velocity(positions, velocities, dt):
    return positions[0] + jnp.cumsum(velocities, axis=0) * dt

@jax.jit
def entropy_grad_wrt_positions(pixel_positions, radar_positions, radar_velocities, chirps, params):
    positions_integrated = integrate_velocity(radar_positions, radar_velocities, params.dt)
    
    # Phase 1: forward-only scan to build the full image (no AD storage)
    image = backproject_sum(pixel_positions, positions_integrated, chirps, params)

    # Phase 2: entropy value + adjoint vector (grad of entropy w.r.t. image)
    entropy, adjoint = jax.value_and_grad(calc_entropy)(image)

    # Phase 3: sequential per-chirp VJPs — each step allocates and frees
    def per_chirp_grad(carry, x):
        radar_pos, chirp = x

        def chirp_contribution(pos):
            return backproject_single(pixel_positions, pos, chirp, params)

        _, vjp_fn = jax.vjp(chirp_contribution, radar_pos)
        (pos_grad,) = vjp_fn(adjoint)
        return carry, pos_grad

    _, grad_positions = jax.lax.scan(
        per_chirp_grad, None, (positions_integrated, chirps)
    )

    # get velocity gradients by propagating position gradients through the integration step
    _, grad_velocities_vjp_fn = jax.vjp(
        lambda vels: integrate_velocity(radar_positions, vels, params.dt),
        radar_velocities
    )
    grad_velocities = grad_velocities_vjp_fn(grad_positions)[0]

    return image, entropy, grad_velocities, grad_positions
