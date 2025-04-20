import numpy as np
from collections import deque

def create_outer_layer_mask(voxels):
    """
    Returns a boolean mask of the outer shell (walls) of the voxel grid.
    """
    res = voxels.shape[0]
    mask = np.zeros_like(voxels, dtype=bool)
    # z faces
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    # y faces
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    # x faces
    mask[0, :, :] = True
    mask[-1, :, :] = True
    return mask

def rotate_to_bottom(voxels, face_idx):
    """
    Rotate the voxel grid so that the specified face index
    (0: Z-, 1: Z+, 2: Y-, 3: Y+, 4: X-, 5: X+) becomes the bottom (z=0).
    """
    v = voxels.copy()
    if face_idx == 0:
        return v
    if face_idx == 1:
        return v[:, :, ::-1]
    if face_idx == 2:
        return v.transpose(0, 2, 1)
    if face_idx == 3:
        return v[:, ::-1, :].transpose(0, 2, 1)
    if face_idx == 4:
        return v.transpose(1, 2, 0)
    if face_idx == 5:
        return v[::-1, :, :].transpose(1, 2, 0)
    return v


def remove_floating_overhangs(voxels):
    """
    Remove any internal voxels not connected to the base (z=0) layer,
    eliminating small floating islands while preserving outer walls.
    """
    res = voxels.shape[0]
    outer_mask = create_outer_layer_mask(voxels)
    supported = np.zeros_like(voxels, dtype=bool)
    q = deque()
    # Initialize with all base voxels (including walls)
    for i in range(res):
        for j in range(res):
            if voxels[i, j, 0]:
                supported[i, j, 0] = True
                q.append((i, j, 0))
    # BFS across all connected interior voxels
    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < res and 0 <= ny < res and 0 <= nz < res:
                if voxels[nx, ny, nz] and not supported[nx, ny, nz]:
                    supported[nx, ny, nz] = True
                    q.append((nx, ny, nz))
    # Preserve original outer walls regardless
    supported[outer_mask] = voxels[outer_mask]
    return supported


def enforce_vertical_support(voxels):
    """
    Remove any internal voxels that lack direct support from the layer beneath,
    ensuring no overhangs, but keep outer walls intact.
    """
    res = voxels.shape[0]
    outer_mask = create_outer_layer_mask(voxels)
    supported = voxels.copy()
    # Iterate from the second layer upward
    for z in range(1, res):
        # Only internal voxels require support
        for x in range(1, res-1):
            for y in range(1, res-1):
                if supported[x, y, z] and not supported[x, y, z-1]:
                    supported[x, y, z] = False
    # Reapply walls
    supported[outer_mask] = voxels[outer_mask]
    return supported

def remove_overhang(voxels, face_idx):
    voxels_oriented = rotate_to_bottom(voxels, face_idx)
    voxels_supported = remove_floating_overhangs(voxels_oriented)
    voxels_clean = enforce_vertical_support(voxels_supported)
    return voxels_clean