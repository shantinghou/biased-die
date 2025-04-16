import numpy as np

def calculate_solid_angles(voxels):
    """Calculate solid angles from centroid to cube faces"""
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
        [0.5, 0.5, 0], [0.5, 0.5, 1], # Bottom, Top (Z)
        [0.5, 0, 0.5], [0.5, 1, 0.5], # Back, Front (Y)
        [0, 0.5, 0.5], [1, 0.5, 0.5]  # Left, Right (X)
    ])
    
    normals = np.array([
        [0, 0, -1], [0, 0, 1],
        [0, -1, 0], [0, 1, 0],
        [-1, 0, 0], [1, 0, 0]
    ])
    
    face_area = 1.0 * 1.0

    solid_angles = np.zeros(6)
    for i in range(6):
        r_vec = face_centers[i] - center
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-9: continue

        cos_theta = np.dot(r_vec / r_mag, normals[i])

        if cos_theta > 0:
            solid_angles[i] = (face_area * cos_theta) / (r_mag**2)

    solid_angles = np.maximum(solid_angles, 0)

    if np.sum(solid_angles) < 1e-6:
       return np.ones(6) * (4 * np.pi / 6)

    return solid_angles

def calculate_probabilities(voxels):
    """Calculate probability distribution from voxels"""
    solid_angles = calculate_solid_angles(voxels)
    total_solid_angle = np.sum(solid_angles)
    if total_solid_angle > 1e-6:
        return solid_angles / total_solid_angle
    else:
        return np.ones(6) / 6 