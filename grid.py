import numpy as np

def create_pixel_grid(
    u_dim: int,
    v_dim: int,
    extent_x: float,
    extent_y: float,
    height: float,
    center: tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """
    Create a 2D grid of 3D pixel positions, optionally rotated.
    
    Args:
        u_dim: Number of pixels in azimuth (x) direction
        v_dim: Number of pixels in range (y) direction
        extent_x: Total extent in x direction (centered at 0)
        extent_y: Total extent in y direction (centered at 0)
        height: Z coordinate (height) of the image plane
        center: Center of the grid (x, y)
        rotation_deg: Rotation angle in degrees (counter-clockwise)
        
    Returns:
        Array of shape (u_dim, v_dim, 3) with 3D positions
    """
    # Create local coordinates centered at origin
    u = np.linspace(-extent_x / 2, extent_x / 2, u_dim, dtype=np.float32)
    v = np.linspace(-extent_y / 2, extent_y / 2, v_dim, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing='ij')

    if rotation_deg != 0.0:
        rad = np.radians(rotation_deg)
        cos_a = np.float32(np.cos(rad))
        sin_a = np.float32(np.sin(rad))
        xx = uu * cos_a - vv * sin_a + np.float32(center[0])
        yy = uu * sin_a + vv * cos_a + np.float32(center[1])
    else:
        xx = uu + np.float32(center[0])
        yy = vv + np.float32(center[1])

    zz = np.full_like(xx, height, dtype=np.float32)
    return np.stack([xx, yy, zz], axis=-1)

def create_pixel_grid_side(
    traj_window, 
    u_dim: int,
    v_dim: int,
    extent_x: float,
    extent_y: float,
    height: float = 0.0,
    near_edge: float = 50,
) -> np.ndarray:
    """
    Create a 2D grid of 3D pixel positions for a side-looking configuration.
    
    This creates a grid to the left of a trajectory:
    1. Compute the direction of the trajectory in x-y (from first to last radar position)
    2. Create x-axis of grid along the trajectory direction from -extent_x/2 to extent_x/2
    3. Create y-axis of grid perpendicular to trajectory from near_edge to near_edge + extent_y
    """
    # Compute trajectory direction
    traj_start = traj_window[0, :2]
    traj_end = traj_window[-1, :2]
    traj_dir = traj_end - traj_start
    traj_dir = traj_dir / np.linalg.norm(traj_dir)
    
    # Perpendicular direction (rotate 90 degrees counter-clockwise)
    perp_dir = np.array([-traj_dir[1], traj_dir[0]], dtype=np.float32)
    
    # Compute center of trajectory
    center = np.mean(traj_window[:, :2], axis=0)
    
    # Create local grid coordinates
    u = np.linspace(-extent_x / 2, extent_x / 2, u_dim, dtype=np.float32)
    v = np.linspace(near_edge, near_edge + extent_y, v_dim, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    
    # Transform to world coordinates
    xx = center[0] + uu * traj_dir[0] + vv * perp_dir[0]
    yy = center[1] + uu * traj_dir[1] + vv * perp_dir[1]
    zz = np.full_like(xx, height, dtype=np.float32)
    
    return np.stack([xx, yy, zz], axis=-1)



def get_grid_parameters(
    u_dim: int,
    v_dim: int,
    extent_x: float,
    extent_y: float,
    height: float,
    center: tuple = (0.0, 0.0),
) -> dict:
    """
    Get grid parameters for on-the-fly pixel position computation.
    
    This returns the parameters needed to compute pixel positions without
    storing the entire grid in memory. For a general (possibly rotated) grid,
    the pixel at index [i, j] has position:
        position = origin + i * u_step + j * v_step
    
    where u_step and v_step are the step vectors in the u and v directions.
    
    Args:
        u_dim: Number of pixels in azimuth (u) direction (corresponds to first index)
        v_dim: Number of pixels in range (v) direction (corresponds to second index)
        extent_x: Total extent in x direction
        extent_y: Total extent in y direction
        height: Z coordinate (height) of the image plane
        center: Center of the grid (x, y)
        
    Returns:
        Dictionary with keys:
        - origin_x: X coordinate of pixel [0, 0]
        - origin_y: Y coordinate of pixel [0, 0]
        - origin_z: Z coordinate of pixel [0, 0]
        - u_step_x: X component of step in u direction (from [i,j] to [i+1,j])
        - u_step_y: Y component of step in u direction
        - u_step_z: Z component of step in u direction
        - v_step_x: X component of step in v direction (from [i,j] to [i,j+1])
        - v_step_y: Y component of step in v direction
        - v_step_z: Z component of step in v direction
        - u_dim: Number of pixels in U direction
        - v_dim: Number of pixels in V direction
    """
    # For axis-aligned grids, the step vectors are aligned with x and y axes
    # Calculate origin (position of pixel [0, 0])
    origin_x = center[0] - extent_x / 2
    origin_y = center[1] - extent_y / 2
    origin_z = height
    
    # Calculate step vectors between adjacent pixels
    # u direction: changes first index [i, j] -> [i+1, j]
    u_step_x = extent_x / (u_dim - 1) if u_dim > 1 else 0.0
    u_step_y = 0.0
    u_step_z = 0.0
    
    # v direction: changes second index [i, j] -> [i, j+1]
    v_step_x = 0.0
    v_step_y = extent_y / (v_dim - 1) if v_dim > 1 else 0.0
    v_step_z = 0.0
    
    return {
        'origin_x': np.float32(origin_x),
        'origin_y': np.float32(origin_y),
        'origin_z': np.float32(origin_z),
        'u_step_x': np.float32(u_step_x),
        'u_step_y': np.float32(u_step_y),
        'u_step_z': np.float32(u_step_z),
        'v_step_x': np.float32(v_step_x),
        'v_step_y': np.float32(v_step_y),
        'v_step_z': np.float32(v_step_z),
        'u_dim': u_dim,
        'v_dim': v_dim,
    }