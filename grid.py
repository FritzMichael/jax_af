import numpy as np

def create_pixel_grid(
    u_dim: int,
    v_dim: int,
    extent_x: float,
    extent_y: float,
    height: float,
    center: tuple = (0.0, 0.0),
) -> np.ndarray:
    """
    Create a 2D grid of 3D pixel positions.
    
    Args:
        u_dim: Number of pixels in azimuth (x) direction
        v_dim: Number of pixels in range (y) direction
        extent_x: Total extent in x direction (centered at 0)
        extent_y: Total extent in y direction (centered at 0)
        height: Z coordinate (height) of the image plane
        
    Returns:
        Array of shape (u_dim, v_dim, 3) with 3D positions
    """
    x = np.linspace(center[0] - extent_x / 2, center[0] + extent_x / 2, u_dim, dtype=np.float32)
    y = np.linspace(center[1] - extent_y / 2, center[1] + extent_y / 2, v_dim, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing='ij')
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