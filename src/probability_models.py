import numpy as np

def calculate_solid_angles(voxels):
    """Calculate centroid solid angles from center of mass to cube faces"""
    resolution = voxels.shape[0]
    x = y = z = np.linspace(0.5 / resolution, 1 - 0.5 / resolution, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    total_mass = np.sum(voxels)
    if total_mass == 0:
        return np.ones(6) * (4 * np.pi / 6)

    center_x = np.sum(X * voxels) / total_mass
    center_y = np.sum(Y * voxels) / total_mass
    center_z = np.sum(Z * voxels) / total_mass
    center = np.array([center_x, center_y, center_z])

    face_centers = np.array([
        [0.5, 0.5, 0], [0.5, 0.5, 1],       # Bottom, Top
        [0.5, 0, 0.5], [0.5, 1, 0.5],       # Back, Front
        [0, 0.5, 0.5], [1, 0.5, 0.5]        # Left, Right
    ])
    normals = np.array([
        [0, 0, -1], [0, 0, 1],
        [0, -1, 0], [0, 1, 0],
        [-1, 0, 0], [1, 0, 0]
    ])
    face_area = 1.0

    solid_angles = np.zeros(6)
    for i in range(6):
        r_vec = face_centers[i] - center
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-9:
            continue
        cos_theta = np.dot(r_vec / r_mag, normals[i])
        if cos_theta > 0:
            solid_angles[i] = (face_area * cos_theta) / (r_mag ** 2)

    solid_angles = np.maximum(solid_angles, 0)
    if np.sum(solid_angles) < 1e-6:
        return np.ones(6) * (4 * np.pi / 6)

    return solid_angles

def calculate_centroid_heights(voxels):
    """Calculate center of mass height for each face-down orientation"""
    resolution = voxels.shape[0]
    x = y = z = np.linspace(0.5 / resolution, 1 - 0.5 / resolution, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    coords = np.stack([X, Y, Z], axis=-1)
    mass = voxels.astype(float)

    heights = np.zeros(6)
    for i, axis in enumerate(['z', 'z', 'y', 'y', 'x', 'x']):
        if axis == 'z':
            flipped = i == 1  # top face
            heights[i] = np.sum(Z * mass) / np.sum(mass)
            if flipped:
                heights[i] = 1.0 - heights[i]
        elif axis == 'y':
            flipped = i == 3  # front face
            heights[i] = np.sum(Y * mass) / np.sum(mass)
            if flipped:
                heights[i] = 1.0 - heights[i]
        elif axis == 'x':
            flipped = i == 5  # right face
            heights[i] = np.sum(X * mass) / np.sum(mass)
            if flipped:
                heights[i] = 1.0 - heights[i]

    return heights

def calculate_probabilities(voxels, p=2.427):
    """Calculate probability distribution using centroid solid angle & height^p"""
    solid_angles = calculate_solid_angles(voxels)
    heights = calculate_centroid_heights(voxels)

    # Avoid divide-by-zero
    heights = np.maximum(heights, 1e-6)

    weights = solid_angles / (heights ** p)
    total = np.sum(weights)

    if total < 1e-6:
        return np.ones(6) / 6

    return weights / total
