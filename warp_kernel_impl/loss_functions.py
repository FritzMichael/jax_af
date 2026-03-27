import jax.numpy as jnp
import jax

@jax.jit
def calc_entropy(image):
    power = jnp.abs(image) ** 2
    power_sum = jnp.sum(power)
    prob = power / (power_sum)  # Avoid division by zero
    entropy = -jnp.sum(prob * jnp.log(prob + 1e-12))  # Avoid log(0)
    return entropy

@jax.jit
def calc_entropy_clip(image):
    power = jnp.abs(image)**2
    
    # --- Soft compression of top 5% ---
    p95 = jnp.percentile(power, 99.9)
    # How aggressively to compress above the threshold (higher = tighter squish)
    knee = 0.5
    # Values below p95 pass through; above p95 get tanh-squished
    excess = jnp.maximum(power - p95, 0.0)
    #power = p95 + (p95 / knee) * jnp.tanh(knee * excess / (p95 + 1e-12))

    power_sum = jnp.sum(power)
    prob = power / (power_sum)  # Avoid division by zero
    entropy = -jnp.sum(prob * jnp.log(prob + 1e-12))  # Avoid log(0)
    return entropy

@jax.jit
def l2_norm(image):
    return -jnp.sum(jnp.abs(image) ** 2)

@jax.jit
def l4_norm(image):
    return -jnp.sum(jnp.abs(image) ** 4)

@jax.jit
def calc_entropy_quadrants(image):
    """
    Split image into 4 quadrants, calculate entropy for each,
    and return the sum of all quadrant entropies.
    """
    h, w = image.shape
    h_mid = h // 2
    w_mid = w // 2
    
    # Split into 4 quadrants
    q1 = image[:h_mid, :w_mid]      # Top-left
    q2 = image[:h_mid, w_mid:]      # Top-right
    q3 = image[h_mid:, :w_mid]      # Bottom-left
    q4 = image[h_mid:, w_mid:]      # Bottom-right
    
    # Calculate entropy for each quadrant
    entropy_q1 = calc_entropy(q1)
    entropy_q2 = calc_entropy(q2)
    entropy_q3 = calc_entropy(q3)
    entropy_q4 = calc_entropy(q4)
    
    # Sum all entropies
    total_entropy = entropy_q1 + entropy_q2 + entropy_q3 + entropy_q4
    return total_entropy