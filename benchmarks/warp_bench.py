#%%
import os
from pathlib import Path
# add the parent directory to the Python path to allow imports from the main project
import sys
_base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.append(str(_base_dir.parent))

import numpy as np
import jax.numpy as jnp
from data_reader import open_data_reader
from grid import create_pixel_grid, create_pixel_grid_side
from warp_kernel_impl.wrapper import run_BP, run_jax_BP
from warp_kernel_impl.loss_functions import calc_entropy
import matplotlib.pyplot as plt
import nvtx
import jax


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(_base_dir.parent / ".env")
# %%
OFFSET = 144_000
aperture_length = 1024
data_path = os.getenv("SAR_DATA_PATH")
reader = open_data_reader(data_path)
# %%


rp = reader.read_range_profiles(OFFSET, OFFSET+aperture_length)
rad_pos = reader.read_positions(OFFSET, OFFSET+aperture_length)
rad_vel = reader.read_velocities(OFFSET, OFFSET+aperture_length)

grid = create_pixel_grid_side(
    traj_window=rad_pos,
    u_dim=512,
    v_dim=512,
    extent_x=400.0,
    extent_y=400.0,
    height=-200.0,
    near_edge=50.0,
)

# just forwards pass
img = run_BP(
    pixel_pos=grid,
    range_profiles=rp,
    radar_pos=rad_pos,
    radar_vel=rad_vel)

# running jax BP with warp kernel and calculating entropy and gradient for the image
output_image, entropy, grad, (jitted_function, jax_args) = run_jax_BP(
    pixel_pos=grid,
    range_profiles=rp,
    radar_pos=rad_pos,
    radar_vel=rad_vel,
    loss_fn = lambda x: calc_entropy(x),
)
# %%
for i in range(100):
    with nvtx.annotate(f"JITted function iteration {i+1}", color="blue"):
        entropy, grad = jitted_function(*jax_args)
# %%
