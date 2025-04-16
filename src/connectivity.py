import numpy as np
from scipy.ndimage import label

# ===========================================
# Connectivity Check
# ===========================================
def is_solid_connected(voxels):
    """check if the solid part (value=1) is connected (only face-connected voxels count as connected)"""
    # 6-connectivity: only face-connected voxels are considered connected
    structure = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    ], dtype=int)
    solid = (voxels == 1)
    if not np.any(solid):
        return False  # no solid, not valid
    # label the connected components with 6-connectivity
    labeled, num_features = label(solid, structure=structure)
    return num_features == 1

def is_connected(voxels):
    """check if the empty space is connected (only face-connected voxels count as connected)"""
    # 6-connectivity: only face-connected voxels are considered connected
    structure = np.array([
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    ], dtype=int)
    # create a mask, representing the empty space
    empty_space = (voxels == 0)
    # if there is no empty space, then it is connected
    if not np.any(empty_space):
        return True
    # label the connected components with 6-connectivity
    labeled, num_features = label(empty_space, structure=structure)
    return num_features == 1

def ensure_connectivity(voxels):
    """Helper function to try to ensure both the solid and empty parts are connected"""
    fixed_voxels = voxels.copy()
    resolution = voxels.shape[0]
    center = resolution // 2
    
    # First check if the empty space is connected
    if not is_connected(fixed_voxels):
        # Try to connect unconnected empty spaces to the main cavity
        # 1. Find the largest connected component of empty space
        from scipy.ndimage import label, find_objects
        empty_space = (fixed_voxels == 0)
        labeled, num_features = label(empty_space, structure=np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ]))
        
        if num_features > 1:
            # Find sizes of components
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest_component = np.argmax(component_sizes) + 1
            
            # Connect smaller components to the largest one
            for i in range(1, resolution-1):
                for j in range(1, resolution-1):
                    for k in range(1, resolution-1):
                        if labeled[i, j, k] > 0 and labeled[i, j, k] != largest_component:
                            # Create a path to the largest component
                            # Find the nearest point in the largest component
                            min_dist = float('inf')
                            nearest = None
                            
                            for ii in range(1, resolution-1):
                                for jj in range(1, resolution-1):
                                    for kk in range(1, resolution-1):
                                        if labeled[ii, jj, kk] == largest_component:
                                            dist = abs(i-ii) + abs(j-jj) + abs(k-kk)
                                            if dist < min_dist:
                                                min_dist = dist
                                                nearest = (ii, jj, kk)
                            
                            if nearest:
                                # Create a straight path
                                xi, yi, zi = i, j, k
                                xf, yf, zf = nearest
                                
                                # X direction
                                while xi != xf:
                                    xi += 1 if xf > xi else -1
                                    if 1 <= xi < resolution-1:
                                        fixed_voxels[xi, yi, zi] = False
                                
                                # Y direction
                                while yi != yf:
                                    yi += 1 if yf > yi else -1
                                    if 1 <= yi < resolution-1:
                                        fixed_voxels[xi, yi, zi] = False
                                
                                # Z direction
                                while zi != zf:
                                    zi += 1 if zf > zi else -1
                                    if 1 <= zi < resolution-1:
                                        fixed_voxels[xi, yi, zi] = False
                            
                            # Only do this for one voxel of the component
                            break
    
    # Check if the solid part is connected
    if not is_solid_connected(fixed_voxels):
        # Same approach - connect unconnected solid parts
        from scipy.ndimage import label, find_objects
        solid = (fixed_voxels == 1)
        labeled, num_features = label(solid, structure=np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ]))
        
        if num_features > 1:
            # Find sizes of components
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest_component = np.argmax(component_sizes) + 1
            
            # If the largest component doesn't include the shell, that's a problem
            shell_component = labeled[0, 0, 0]
            if shell_component != largest_component:
                largest_component = shell_component
            
            # Connect smaller components to the largest one
            for i in range(1, resolution-1):
                for j in range(1, resolution-1):
                    for k in range(1, resolution-1):
                        if labeled[i, j, k] > 0 and labeled[i, j, k] != largest_component:
                            # Create a path to the largest component
                            # Find the nearest point in the largest component
                            min_dist = float('inf')
                            nearest = None
                            
                            for ii in range(resolution):
                                for jj in range(resolution):
                                    for kk in range(resolution):
                                        if labeled[ii, jj, kk] == largest_component:
                                            dist = abs(i-ii) + abs(j-jj) + abs(k-kk)
                                            if dist < min_dist:
                                                min_dist = dist
                                                nearest = (ii, jj, kk)
                            
                            if nearest:
                                # Create a straight path
                                xi, yi, zi = i, j, k
                                xf, yf, zf = nearest
                                
                                # X direction
                                while xi != xf:
                                    xi += 1 if xf > xi else -1
                                    if 0 <= xi < resolution:
                                        fixed_voxels[xi, yi, zi] = True
                                
                                # Y direction
                                while yi != yf:
                                    yi += 1 if yf > yi else -1
                                    if 0 <= yi < resolution:
                                        fixed_voxels[xi, yi, zi] = True
                                
                                # Z direction
                                while zi != zf:
                                    zi += 1 if zf > zi else -1
                                    if 0 <= zi < resolution:
                                        fixed_voxels[xi, yi, zi] = True
                            
                            # Only do this for one voxel of the component
                            break
    
    return fixed_voxels 