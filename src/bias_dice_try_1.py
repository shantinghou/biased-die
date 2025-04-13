# -*- coding: utf-8 -*-
# Required libraries: numpy, scipy, igl, meshio, open3d, scikit-image, matplotlib, trimesh
# Install them using pip:
# pip install numpy scipy meshio open3d scikit-image igl matplotlib trimesh

import numpy as np
import igl
from scipy.ndimage import label
import meshio
import open3d as o3d
from skimage import measure # For Marching Cubes
import trimesh

# ===========================================
# Helper Functions
# ===========================================

def create_outer_layer_mask(resolution):
    """Create a mask for the outer layer of voxels"""
    mask = np.zeros((resolution, resolution, resolution), dtype=bool)
    mask[0, :, :] = mask[-1, :, :] = True
    mask[:, 0, :] = mask[:, -1, :] = True
    mask[:, :, 0] = mask[:, :, -1] = True
    return mask

def check_connectivity(voxels):
    """Check if voxel grid maintains a single connected component"""
    labeled_array, num_features = label(voxels)
    return num_features == 1

def is_well_connected(voxels, x, y, z):
    """Check if removing a voxel maintains good connectivity"""
    if not voxels[x, y, z]:
        return True
    
    temp_voxels = voxels.copy()
    temp_voxels[x, y, z] = False
    
    neighbors_count = 0
    offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    resolution = voxels.shape[0]
    for dx, dy, dz in offsets:
        nx, ny, nz = x + dx, y + dy, z + dz
        if (0 <= nx < resolution and 0 <= ny < resolution and 
            0 <= nz < resolution and temp_voxels[nx, ny, nz]):
            neighbors_count += 1
    
    if neighbors_count < 3:
        return False
    
    neighbors = np.zeros((3, 3, 3), dtype=bool)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                nx, ny, nz = x + dx, y + dy, z + dz
                if (0 <= nx < resolution and 0 <= ny < resolution and 
                    0 <= nz < resolution and temp_voxels[nx, ny, nz]):
                    neighbors[dx+1, dy+1, dz+1] = True
    
    labeled_array, num_features = label(neighbors)
    return num_features <= 1 or neighbors_count >= 4

def select_random_internal_voxel(voxels, outer_layer_mask):
    """Select a random internal voxel"""
    internal_voxels = np.logical_and(voxels, ~outer_layer_mask)
    indices = np.where(internal_voxels)
    if len(indices[0]) == 0:
        return None
    idx = np.random.randint(len(indices[0]))
    return indices[0][idx], indices[1][idx], indices[2][idx]

def get_removable_voxels(voxels, outer_layer_mask):
    """Get all internal voxels that can be safely removed"""
    internal_voxels = np.logical_and(voxels, ~outer_layer_mask)
    indices = np.where(internal_voxels)
    
    removable = []
    for i in range(len(indices[0])):
        x, y, z = indices[0][i], indices[1][i], indices[2][i]
        temp_voxels = voxels.copy()
        temp_voxels[x, y, z] = False
        if check_connectivity(temp_voxels) and is_well_connected(voxels, x, y, z):
            removable.append((x, y, z))
    
    return removable

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

def evaluate_solution(voxels, target_probabilities):
    """Evaluate MSE between current and target probabilities"""
    solid_angles = calculate_solid_angles(voxels)
    total_solid_angle = np.sum(solid_angles)
    if total_solid_angle < 1e-6:
         return 1.0

    predicted_probabilities = solid_angles / total_solid_angle
    return np.mean((predicted_probabilities - target_probabilities)**2)

# ===========================================
# Voxel Optimization
# ===========================================

def optimize_biased_dice(target_probabilities, resolution=20, max_iterations=10000, T_initial=0.01, T_min=1e-6, alpha=0.99, smooth_factor=0.7):
    """Optimize voxel shape using simulated annealing"""
    voxels = np.ones((resolution, resolution, resolution), dtype=bool)
    outer_layer_mask = create_outer_layer_mask(resolution)

    T = T_initial
    current_score = evaluate_solution(voxels, target_probabilities)
    best_voxels = voxels.copy()
    best_score = current_score

    print(f"Initial Score: {current_score:.6f}")

    for iteration in range(max_iterations):
        if T <= T_min:
            print("Temperature below minimum threshold. Stopping.")
            break

        removable_voxels = get_removable_voxels(voxels, outer_layer_mask)
        if not removable_voxels:
            print("No more removable internal voxels. Stopping.")
            break

        voxel_idx = np.random.randint(len(removable_voxels))
        x, y, z = removable_voxels[voxel_idx]
        
        temp_voxels = voxels.copy()
        temp_voxels[x, y, z] = False

        new_score = evaluate_solution(temp_voxels, target_probabilities)
        delta_E = new_score - current_score

        acceptance_prob = np.exp(-delta_E / (T * (1 + smooth_factor * is_well_connected(voxels, x, y, z))))
        if delta_E < 0 or np.random.random() < acceptance_prob:
                voxels = temp_voxels
                current_score = new_score

                if current_score < best_score:
                    best_voxels = voxels.copy()
                    best_score = current_score

        T *= alpha

        if iteration % 500 == 0:
            print(f"Iteration {iteration}, Temp: {T:.6f}, Current Score: {current_score:.6f}, Best Score: {best_score:.6f}")

    print(f"Optimization finished. Final Best Score: {best_score:.6f}")
    
    print("Post-processing: Removing isolated voxels...")
    best_voxels = remove_isolated_voxels(best_voxels)

    return best_voxels

def remove_isolated_voxels(voxels):
    """Remove isolated voxels with weak connectivity"""
    result = voxels.copy()
    resolution = voxels.shape[0]
    
    changed = True
    while changed:
        changed = False
        for x in range(resolution):
            for y in range(resolution):
                for z in range(resolution):
                    if result[x, y, z]:
                        neighbors_count = 0
                        offsets = [
                            (1, 0, 0), (-1, 0, 0),
                            (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)
                        ]
                        
                        for dx, dy, dz in offsets:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < resolution and 0 <= ny < resolution and 
                                0 <= nz < resolution and result[nx, ny, nz]):
                                neighbors_count += 1
                        
                        if neighbors_count < 2:
                            temp = result.copy()
                            temp[x, y, z] = False
                            if check_connectivity(temp):
                                result[x, y, z] = False
                                changed = True
    
    return result

# ===========================================
# Mesh Generation
# ===========================================

def create_mesh_from_voxels_marching_cubes(voxels, level=0.5, voxel_size=1.0):
    """Create a smooth mesh using Marching Cubes algorithm"""
    resolution = voxels.shape[0]
    scale_factor = voxel_size / resolution

    padded_voxels = np.pad(voxels.astype(float), pad_width=1, mode='constant', constant_values=0.0)

    try:
        verts, faces, normals, values = measure.marching_cubes(
            volume=padded_voxels,
            level=level,
            spacing=(scale_factor, scale_factor, scale_factor)
        )

        verts -= scale_factor

        print(f"Marching Cubes generated: {len(verts)} vertices, {len(faces)} faces")
        return verts, faces

    except Exception as e:
        print(f"Error during Marching Cubes: {e}")
        if np.all(voxels == False):
            print("Warning: Voxel grid is empty. Marching Cubes cannot generate a surface.")
        elif np.all(voxels == True):
             print("Warning: Voxel grid is full. Marching Cubes might only generate boundary surface.")
        else:
             print(f"Voxels shape: {voxels.shape}, dtype: {voxels.dtype}, min/max: {np.min(voxels)}/{np.max(voxels)}")
        return np.array([]), np.array([])

def create_blocky_mesh_from_voxels(voxels, voxel_size=1.0):
    """Create a blocky mesh preserving voxel cube geometry using Trimesh"""
    try:
        from trimesh.creation import box
        from trimesh.scene import Scene
        
        resolution = voxels.shape[0]
        scale_factor = voxel_size / resolution
        
        scene = Scene()
        
        filled_voxels = np.argwhere(voxels)
        
        print(f"Processing {len(filled_voxels)} filled voxels...")
        
        for idx, (i, j, k) in enumerate(filled_voxels):
            cube = box(extents=[scale_factor, scale_factor, scale_factor])
            
            position = np.array([i, j, k]) * scale_factor
            cube.apply_translation(position)
            
            scene.add_geometry(cube, name=f"voxel_{idx}")
            
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx}/{len(filled_voxels)} voxels...")
        
        combined_mesh = trimesh.util.concatenate(scene.dump())
        
        vertices = np.array(combined_mesh.vertices)
        faces = np.array(combined_mesh.faces)
        
        print(f"Blocky mesh generated: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces
        
    except Exception as e:
        print(f"Error generating blocky mesh: {e}")
        return np.array([]), np.array([])

# ===========================================
# Mesh Validation and Fixing
# ===========================================

def validate_and_fix_mesh(vertices, faces):
    """Validate and fix mesh issues"""
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        print("Warning: Mesh is empty, skipping validation.")
        return np.array([]), np.array([])

    print(f"Validating mesh: {len(vertices)} vertices, {faces.shape} faces shape")

    if faces.ndim != 2 or faces.shape[1] != 3:
        print(f"Error: Faces array has wrong shape {faces.shape}. Expected (N, 3).")
        if faces.ndim == 1 and len(faces) % 3 == 0:
            try:
                faces = faces.reshape(-1, 3)
                print("Reshaped faces array to (N, 3).")
            except Exception as reshape_err:
                 print(f"Error reshaping faces: {reshape_err}")
                 return vertices, np.array([])
        else:
            return vertices, np.array([])

    if not np.issubdtype(faces.dtype, np.integer):
        print(f"Warning: Faces dtype is {faces.dtype}, attempting to convert to int.")
        try:
            faces = faces.astype(np.int32)
        except Exception as type_err:
            print(f"Error converting faces dtype: {type_err}")
            return vertices, np.array([])

    max_index = np.max(faces)
    min_index = np.min(faces)
    num_vertices = len(vertices)

    if max_index >= num_vertices or min_index < 0:
        print(f"Error: Face indices out of bounds (min: {min_index}, max: {max_index}, num_vertices: {num_vertices}). Attempting to remove invalid faces.")
        valid_face_indices = np.all((faces >= 0) & (faces < num_vertices), axis=1)
        faces = faces[valid_face_indices]
        print(f"Removed {np.sum(~valid_face_indices)} invalid faces. Remaining faces: {len(faces)}")
        if len(faces) == 0:
            print("Error: No valid faces remaining after index check.")
            return vertices, np.array([])

    try:
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d.remove_degenerate_triangles()

        if not mesh_o3d.has_triangles():
             print("Error: Mesh has no triangles after Open3D cleaning.")
             return np.array([]), np.array([])

        mesh_o3d.orient_triangles()

        final_vertices = np.asarray(mesh_o3d.vertices)
        final_faces = np.asarray(mesh_o3d.triangles)
        print(f"Mesh after Open3D validation: {len(final_vertices)} vertices, {len(final_faces)} faces")
        return final_vertices, final_faces

    except Exception as e:
        print(f"Error during Open3D validation/fixing: {e}")
        print("Warning: Returning mesh before Open3D fixing step.")
        return vertices, faces

# ===========================================
# STL Export
# ===========================================

def save_to_stl(vertices, faces, filename="biased_dice_mc.stl"):
    """Save mesh to STL file"""
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        print("Error: Cannot save empty mesh.")
        return

    print(f"Saving mesh to {filename}: {len(vertices)} vertices, {len(faces)} faces")

    try:
        mesh_to_save = meshio.Mesh(
            points=np.array(vertices, dtype=np.float64),
            cells=[("triangle", np.array(faces, dtype=np.int32))]
        )
        mesh_to_save.write(filename)
        print(f"Successfully saved mesh to {filename}")

    except Exception as e:
        print(f"Error saving STL file: {e}")
        print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")

# ===========================================
# Main Function
# ===========================================

def design_biased_dice(target_probabilities, resolution=20, max_iterations=5000, connectivity_strength='medium', mesh_type='smooth'):
    """Main function to execute optimization, mesh generation, validation and return results"""
    if not isinstance(target_probabilities, np.ndarray):
        target_probabilities = np.array(target_probabilities)

    if len(target_probabilities) != 6:
        raise ValueError("Target probabilities must be a list or array of length 6.")
    if not np.isclose(np.sum(target_probabilities), 1.0):
         print(f"Warning: Target probabilities sum to {np.sum(target_probabilities)}, normalizing.")
         target_probabilities = target_probabilities / np.sum(target_probabilities)
    
    # Select parameters based on connectivity strength
    if connectivity_strength == 'low':
        T_initial = 0.01
        T_min = 1e-6
        alpha = 0.995
        smooth_factor = 0.5
    elif connectivity_strength == 'high':
        T_initial = 0.005
        T_min = 1e-7
        alpha = 0.99
        smooth_factor = 0.9
    else:  # medium (default)
        T_initial = 0.007
        T_min = 1e-6
        alpha = 0.992
        smooth_factor = 0.7
        
    print(f"Using connectivity strength: {connectivity_strength}")
    print(f"Optimization parameters: T_initial={T_initial}, T_min={T_min}, alpha={alpha}, smooth_factor={smooth_factor}")
    print(f"Mesh type: {mesh_type}")

    print("Starting voxel optimization...")
    optimized_voxels = optimize_biased_dice(
        target_probabilities, 
        resolution=resolution, 
        max_iterations=max_iterations,
        T_initial=T_initial,
        T_min=T_min,
        alpha=alpha,
        smooth_factor=smooth_factor
    )

    # Generate mesh based on selected type
    if mesh_type.lower() == 'blocky':
        print("\nGenerating blocky mesh using Trimesh...")
        vertices, faces = create_blocky_mesh_from_voxels(optimized_voxels, voxel_size=1.0)
    else:
        print("\nGenerating smooth mesh using Marching Cubes...")
        vertices, faces = create_mesh_from_voxels_marching_cubes(optimized_voxels, level=0.5, voxel_size=1.0)

    if len(vertices) > 0 and len(faces) > 0:
        print("\nValidating and fixing generated mesh...")
        vertices, faces = validate_and_fix_mesh(vertices, faces)
    else:
        print("Warning: Mesh generation failed, returning empty mesh.")
        vertices, faces = np.array([]), np.array([])

    # Calculate final probabilities
    print("\nCalculating final probabilities...")
    final_solid_angles = calculate_solid_angles(optimized_voxels)
    total_solid_angle = np.sum(final_solid_angles)
    if total_solid_angle > 1e-6:
        final_probabilities = final_solid_angles / total_solid_angle
    else:
        final_probabilities = np.ones(6) / 6 # Fallback if no solid angle
        print("Warning: Total solid angle is near zero, using uniform probability.")

    final_error = np.mean((final_probabilities - target_probabilities)**2)

    print("\n--- Results ---")
    print(f"Target Probabilities: {np.round(target_probabilities, 4)}")
    print(f"Achieved Probabilities: {np.round(final_probabilities, 4)}")
    print(f"Mean Squared Error: {final_error:.6f}")
    print(f"Generated Mesh: {len(vertices)} vertices, {len(faces)} faces")
    print("---------------")

    return vertices, faces, optimized_voxels, final_probabilities

# ===========================================
# Main Execution
# ===========================================

if __name__ == "__main__":
    try:
        print("--- Biased Dice Design Tool ---")
        
        # Parameter setup
        print("\nSelect probability distribution:")
        print("1. Uniform [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]")
        print("2. High six [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
        print("3. High one and six [0.3, 0.1, 0.1, 0.1, 0.1, 0.3]")
        print("4. Linear increase [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]")
        print("5. Custom probabilities")
        
        try:
            choice = int(input("Select option (1-5): ") or "2")
            if choice == 1:
                target_probs = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
            elif choice == 2:
                target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
            elif choice == 3:
                target_probs = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
            elif choice == 4:
                target_probs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
            elif choice == 5:
                print("Enter 6 probability values separated by spaces (sum will be normalized):")
                probs_input = input("Probability values: ")
                try:
                    probs_list = [float(p) for p in probs_input.split()]
                    if len(probs_list) != 6:
                        print("Must enter 6 values, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
                        target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
                    else:
                        target_probs = np.array(probs_list)
                except:
                    print("Invalid input, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
                    target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
            else:
                print("Invalid selection, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
                target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
        except:
            print("Input error, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
            target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
            
        # Normalize target probabilities
        target_probs = target_probs / np.sum(target_probs)
        print(f"\nTarget probability distribution: {np.round(target_probs, 4)}")
            
        # Set voxel resolution
        try:
            voxel_resolution = int(input("\nVoxel resolution (15-30, recommended 20): ") or "20")
            if voxel_resolution < 10 or voxel_resolution > 50:
                print("Resolution out of range, using default 20")
                voxel_resolution = 20
        except:
            print("Invalid input, using default 20")
            voxel_resolution = 20
            
        print(f"Using voxel resolution: {voxel_resolution}x{voxel_resolution}x{voxel_resolution}")
        
        # Set iteration count
        try:
            max_iter = int(input("\nMaximum iterations (1000-10000, recommended 5000): ") or "5000")
            if max_iter < 100 or max_iter > 20000:
                print("Iteration count out of range, using default 5000")
                max_iter = 5000
        except:
            print("Invalid input, using default 5000")
            max_iter = 5000
            
        print(f"Maximum iterations: {max_iter}")
        
        # Set connectivity strength
        print("\nConnectivity strength options:")
        print("1. Low (allows more irregular shapes)")
        print("2. Medium (balances shape and target probability)")
        print("3. High (prioritizes shape connectivity and regularity)")
        
        try:
            conn_choice = int(input("Select connectivity strength (1-3): ") or "2")
            if conn_choice == 1:
                connectivity = 'low'
            elif conn_choice == 2:
                connectivity = 'medium'
            elif conn_choice == 3:
                connectivity = 'high'
            else:
                print("Invalid selection, using default 'medium'")
                connectivity = 'medium'
        except:
            print("Input error, using default 'medium'")
            connectivity = 'medium'
            
        # Set mesh generation type
        print("\nMesh type options:")
        print("1. Smooth mesh (using Marching Cubes for smooth surface)")
        print("2. Blocky mesh (preserves original cube shapes of voxels)")
        
        try:
            mesh_choice = int(input("Select mesh type (1-2): ") or "1")
            if mesh_choice == 1:
                mesh_type = 'smooth'
            elif mesh_choice == 2:
                mesh_type = 'blocky'
            else:
                print("Invalid selection, using default 'smooth'")
                mesh_type = 'smooth'
        except:
            print("Input error, using default 'smooth'")
            mesh_type = 'smooth'
            
        print(f"Mesh type: {mesh_type}")
        
        # Set output filename
        try:
            output_file = input("\nOutput STL filename (default biased_dice.stl): ") or "biased_dice.stl"
            if not output_file.lower().endswith('.stl'):
                output_file += '.stl'
        except:
            print("Invalid input, using default filename")
            output_file = "biased_dice.stl"
            
        print(f"Output file: {output_file}")
        
        print("\n--- Starting Design Process ---")
            
        # Run design process
        final_vertices, final_faces, final_voxels, achieved_probs = design_biased_dice(
            target_probs,
            resolution=voxel_resolution,
            max_iterations=max_iter,
            connectivity_strength=connectivity,
            mesh_type=mesh_type
        )

        # Save if mesh generation was successful
        if final_vertices is not None and final_faces is not None and len(final_vertices) > 0 and len(final_faces) > 0:
            save_to_stl(final_vertices, final_faces, filename=output_file)
        else:
            print("\nNo valid mesh was generated.")

        print("\n--- Design Process Complete ---")

    except ImportError as e:
        print(f"\nError: Missing required libraries. {e}")
        print("Please install all required libraries:")
        print("pip install numpy scipy meshio open3d scikit-image igl matplotlib trimesh")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
