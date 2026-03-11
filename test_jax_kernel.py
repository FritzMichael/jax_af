from data_reader import open_data_reader
from grid import create_pixel_grid
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax_kernel_inpl as jk
import numpy as np
import scipy.signal as signal
import optax

offset = 20_000
aperture_length = 1024
img_resolution = 512*4
path_to_h5 = ""

def calc_image_entropy(image):
    power = np.abs(image) ** 2
    power_sum = np.sum(power)
    prob = power / (power_sum + 1e-12)  # Avoid division by zero
    entropy = -np.sum(prob * np.log(prob + 1e-12))  # Avoid log(0)
    return entropy

def debug_plot(positions, scene_center, extents = (400, 400)):
    plt.figure(figsize=(6, 6))
    plt.scatter(positions[:, 0], positions[:, 1], s=1, label="Radar Positions")
    plt.scatter(positions[0, 0], positions[0, 1], color='green', marker='o', label="First Position")
    plt.scatter(scene_center[0], scene_center[1], color='red', marker='x', label="Scene Center")
    # draw extent box around scene center
    extent_x, extent_y = extents
    rect = plt.Rectangle((scene_center[0] - extent_x / 2, scene_center[1] - extent_y / 2), extent_x, extent_y, 
                         edgecolor='red', facecolor='none', linestyle='--', label="Scene Extent")
    plt.gca().add_patch(rect)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Radar Trajectory and Scene Center")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    scene_center = (170.0, -200.0)
    grid = create_pixel_grid(
        u_dim=img_resolution,
        v_dim=img_resolution,
        extent_x=300.0,
        extent_y=300.0,
        height=-200.0,
        center=scene_center,
    )

    radar_params = jk.RadarParams(
        F0=9.5e9,
        DR=0.0657581761289589,
        C=299792458.0,
        dt=1/1300,
    )
    reader = open_data_reader(path_to_h5)
    positions = jnp.array(reader.read_positions(offset, offset + aperture_length))
    velocities = jnp.array(reader.read_velocities(offset, offset + aperture_length))
    chirps = jnp.array(
        reader.read_range_profiles(offset, offset + aperture_length)
        * signal.windows.hann(aperture_length)[:, None]  # Apply Hann window to range profiles
    )

    debug_plot(positions, scene_center)

    result = jk.backproject_sum(
            grid, positions, chirps, radar_params
        )
    
    entropy_history = []
    iters = 300

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(velocities)

    for i in range(iters):
        image, entropy, vel_gradients, pos_gradients = jk.entropy_grad_wrt_positions(
            grid,
            positions,
            velocities,
            chirps,
            radar_params,
        )
        if i == 0:
            initial_image = np.array(image.reshape(img_resolution, img_resolution))
        if i == iters-1:
            final_image = np.array(image.reshape(img_resolution, img_resolution))
        entropy_history.append(entropy)
        print(entropy)
        updates, opt_state = optimizer.update(vel_gradients, opt_state)
        velocities = optax.apply_updates(velocities, updates)

    print(f"Entropy: {entropy}")
    plt.plot(entropy_history, label="Entropy")
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    img = 20*jnp.log10(jnp.abs(initial_image))
    vmax = jnp.percentile(img, 99.9)
    vmin = vmax - 40 
    axs[0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Initial Image (log scale) - Entropy {calc_image_entropy(initial_image):.4f}")
    img_after = 20*jnp.log10(jnp.abs(final_image))
    vmax = jnp.percentile(img_after, 99.9)
    vmin = vmax - 40 
    axs[1].imshow(img_after, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Final Image (log scale) - Entropy {calc_image_entropy(final_image):.4f}")
    plt.show()