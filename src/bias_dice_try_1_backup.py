# -*- coding: utf-8 -*-
# Required libraries: numpy, scipy, igl, meshio, open3d, trimesh, matplotlib
# Install them using pip:
# pip install numpy scipy meshio open3d igl trimesh matplotlib

import numpy as np
import igl
from scipy.ndimage import label
import meshio
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def is_connected(voxels):
    """check if the empty space is connected (only face-connected voxels count as connected)"""
    from scipy.ndimage import label
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

def is_cube_shell_intact(voxels):
    """check if the cube shell is intact"""
    resolution = voxels.shape[0]
    return (
        np.all(voxels[0, :, :] == 1) and
        np.all(voxels[-1, :, :] == 1) and
        np.all(voxels[:, 0, :] == 1) and
        np.all(voxels[:, -1, :] == 1) and
        np.all(voxels[:, :, 0] == 1) and
        np.all(voxels[:, :, -1] == 1)
    )

def select_random_internal_voxel(voxels, outer_layer_mask):
    """Select a random internal voxel"""
    internal_voxels = np.logical_and(voxels, ~outer_layer_mask)
    indices = np.where(internal_voxels)
    if len(indices[0]) == 0:
        return None
    idx = np.random.randint(len(indices[0]))
    return indices[0][idx], indices[1][idx], indices[2][idx]

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

def evaluate_solution(voxels, target_probabilities):
    """Evaluate MSE between current and target probabilities"""
    # check the constraints
    if not is_cube_shell_intact(voxels) or not is_connected(voxels) or not is_solid_connected(voxels):
        return float('inf')  # not satisfied, return infinity
        
    # Calculate probability distribution
    probs = calculate_probabilities(voxels)
    
    # Simple MSE calculation (original scoring mechanism)
    return np.mean((probs - target_probabilities)**2)

def is_solid_connected(voxels):
    """check if the solid part (value=1) is connected (only face-connected voxels count as connected)"""
    from scipy.ndimage import label
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

# ===========================================
# Voxel Optimization
# ===========================================

def optimize_biased_dice(target_probabilities, resolution=20, max_iterations=15000, T_initial=0.05, T_min=1e-7, alpha=0.997):
    """Optimize voxel shape using simulated annealing"""
    voxels = np.ones((resolution, resolution, resolution), dtype=bool)
    outer_layer_mask = create_outer_layer_mask(resolution)

    # initialize a cube with a cavity, ensuring connectivity
    center = resolution // 2
    
    # Try multiple initial patterns and pick the best one
    initial_patterns = []
    
    # Pattern 1: Medium central cavity (larger than original)
    pattern1 = voxels.copy()
    cavity_size = resolution // 3  # Increase cavity size
    for i in range(center-cavity_size, center+cavity_size+1):
        for j in range(center-cavity_size, center+cavity_size+1):
            for k in range(center-cavity_size, center+cavity_size+1):
                if 1 <= i < resolution-1 and 1 <= j < resolution-1 and 1 <= k < resolution-1:
                    pattern1[i, j, k] = False
    
    # Pattern 2: Extended cavity in the bias direction with gradual tapering
    pattern2 = voxels.copy()
    probs = list(target_probabilities)
    max_prob_idx = probs.index(max(probs))
    directions = [
        (0, 0, -1), (0, 0, 1),   # Bottom, Top (Z) - 索引 0,1
        (0, -1, 0), (0, 1, 0),   # Back, Front (Y) - 索引 2,3
        (-1, 0, 0), (1, 0, 0)    # Left, Right (X) - 索引 4,5
    ]
    bias_dir = directions[max_prob_idx]
    
    # Create a larger cavity in the direction opposite to bias
    opposite_dir = (-bias_dir[0], -bias_dir[1], -bias_dir[2])
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                # Compute distance from center
                di = i - center
                dj = j - center
                dk = k - center
                
                # Manhattan distance to center
                manhattan_dist = abs(di) + abs(dj) + abs(dk)
                
                # Projected distance in direction of bias
                proj_dist = bias_dir[0] * di + bias_dir[1] * dj + bias_dir[2] * dk
                
                # Create elongated cavity
                # Keep voxels in the bias direction, remove in opposite direction
                # Use a larger cavity threshold
                cavity_threshold = resolution // 2
                if manhattan_dist < cavity_threshold and proj_dist < 0:
                    # Gradually increase probability of being cavity as we get further in opposite direction
                    cavity_prob = min(1.0, abs(proj_dist) / (cavity_threshold/2))
                    if np.random.random() < cavity_prob:
                        pattern2[i, j, k] = False
    
    # Pattern 3: Tube-like cavity for extreme bias
    pattern3 = voxels.copy()
    # Create a tube along the axis opposite to the bias direction
    axis = np.argmax(np.abs(bias_dir))
    tube_radius = resolution // 4
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                # Compute distance from center along the specified axis
                coords = [i, j, k]
                axis_dist = coords[axis] - center
                
                # Compute distance from the axis
                coords[axis] = center  # Project onto the axis
                di = i - coords[0]
                dj = j - coords[1]
                dk = k - coords[2]
                
                # Distance from axis (perpendicular to the tube)
                radial_dist = np.sqrt(di**2 + dj**2 + dk**2)
                
                # If bias_dir[axis] is positive, create cavity on negative side
                # If bias_dir[axis] is negative, create cavity on positive side
                on_correct_side = (axis_dist * bias_dir[axis]) < 0
                
                # Create a tube-like cavity
                if radial_dist < tube_radius and on_correct_side:
                    pattern3[i, j, k] = False
    
    # Evaluate each pattern and fix connectivity issues
    patterns = [pattern1, pattern2, pattern3]
    valid_patterns = []
    
    for idx, pattern in enumerate(patterns):
        # Check if the pattern has valid empty space
        if np.any(pattern == 0):
            # Try to fix connectivity
            fixed_pattern = ensure_connectivity(pattern)
            valid_patterns.append((fixed_pattern, idx))
    
    # Score the valid patterns
    pattern_scores = []
    for pattern, idx in valid_patterns:
        if is_cube_shell_intact(pattern) and is_connected(pattern) and is_solid_connected(pattern):
            score = evaluate_solution(pattern, target_probabilities)
            pattern_scores.append((score, idx))
        else:
            # The pattern is invalid even after fixing
            continue
    
    # If no valid patterns, create a simpler pattern
    if not pattern_scores:
        print("No valid initial patterns, creating a simple cavity")
        simple_pattern = voxels.copy()
        # Create a small central cavity
        for i in range(center-1, center+2):
            for j in range(center-1, center+2):
                for k in range(center-1, center+2):
                    simple_pattern[i, j, k] = False
        
        # Create a path from center to one side
        side = max_prob_idx // 2  # Convert to axis (0=z, 1=y, 2=x)
        direction = max_prob_idx % 2  # 0=negative, 1=positive
        
        path_length = resolution // 3
        coords = [center, center, center]
        
        # Determine step direction (-1 or 1)
        step = 1 if direction else -1
        
        # Create path
        for offset in range(path_length):
            coords[side] = center + step * offset
            if 1 <= coords[0] < resolution-1 and 1 <= coords[1] < resolution-1 and 1 <= coords[2] < resolution-1:
                simple_pattern[coords[0], coords[1], coords[2]] = False
        
        voxels = simple_pattern
        current_score = evaluate_solution(voxels, target_probabilities)
    else:
        # Sort by score and select the best
        pattern_scores.sort()
        best_pattern_idx = pattern_scores[0][1]
        
        print(f"Selected initial pattern {best_pattern_idx+1} with score {pattern_scores[0][0]:.6f}")
        voxels = patterns[best_pattern_idx].copy()
        current_score = pattern_scores[0][0]
    
    best_voxels = voxels.copy()
    best_score = current_score
    
    # 初始化温度T (修复T未定义的错误)
    T = T_initial

    print(f"Initial Score: {current_score:.6f}")
    
    # Calculate and display initial probabilities
    initial_probs = calculate_probabilities(voxels)
    print(f"Initial probabilities: {np.round(initial_probs, 4)}")
    print(f"Target probabilities:  {np.round(target_probabilities, 4)}")
    print(f"Initial cavity size: {np.sum(voxels == 0)} voxels")

    # Track progress for plotting
    iteration_history = []
    score_history = []
    temp_history = []
    
    # Phase 1: Rapid exploration
    phase1_iterations = max_iterations // 3

    for iteration in range(max_iterations):
        if T <= T_min:
            print("Temperature below minimum threshold. Stopping.")
            break

        # Change annealing strategy between phases
        if iteration == phase1_iterations:
            print(f"Switching to fine-tuning phase at iteration {iteration}")
            T = max(T, 0.01)  # Reset temperature for fine-tuning
            
        # Select multiple voxels with probability proportional to temperature
        num_voxels_to_modify = max(1, int(np.random.poisson(T * 10)))
        num_voxels_to_modify = min(num_voxels_to_modify, 5)  # Cap at 5 voxels
        
        temp_voxels = voxels.copy()
        valid_modification = False
        
        for _ in range(num_voxels_to_modify):
            # select a random internal voxel
            internal_voxel = select_random_internal_voxel(~outer_layer_mask, outer_layer_mask)
            if internal_voxel is None:
                print("No more internal voxels to modify. Stopping.")
                break
                
            x, y, z = internal_voxel
            
            # try flipping the state of this voxel
            temp_voxels[x, y, z] = not temp_voxels[x, y, z]
            
            # Check intermediate validity for performance
            if not is_cube_shell_intact(temp_voxels):
                temp_voxels[x, y, z] = not temp_voxels[x, y, z]  # Revert
                continue
        
        # check if the modified cube satisfies all constraints
        if not is_cube_shell_intact(temp_voxels) or not is_connected(temp_voxels) or not is_solid_connected(temp_voxels):
            continue  # not satisfied, skip this modification
        
        valid_modification = True
        
        if valid_modification:
            new_score = evaluate_solution(temp_voxels, target_probabilities)
            delta_E = new_score - current_score

            # according to the simulated annealing algorithm, decide whether to accept the modification
            # Use adaptive acceptance based on iteration phase
            acceptance_probability = np.exp(-delta_E / T)
            
            # In later iterations, be more selective about accepting worse solutions
            if iteration > phase1_iterations:
                acceptance_probability *= 0.8
                
            if delta_E < 0 or np.random.random() < acceptance_probability:
                voxels = temp_voxels
                current_score = new_score

                if current_score < best_score:
                    best_voxels = voxels.copy()
                    best_score = current_score

                    # Calculate and show the current best probabilities
                    if iteration % 1000 == 0:
                        best_probs = calculate_probabilities(best_voxels)
                        print(f"Current best probabilities: {np.round(best_probs, 4)}")
                        print(f"Target probabilities:      {np.round(target_probabilities, 4)}")
                        print(f"Current cavity size: {np.sum(best_voxels == 0)} voxels")
    
            # Cool down temperature
            if iteration < phase1_iterations:
                T *= alpha  # Standard cooling
            else:
                T *= (alpha ** 1.2)  # Faster cooling in fine-tuning phase
                
            # Record history every 100 iterations
            if iteration % 100 == 0:
                iteration_history.append(iteration)
                score_history.append(current_score)
                temp_history.append(T)
    
            # Reporting progress
        if iteration % 500 == 0:
            print(f"Iteration {iteration}, Temp: {T:.6f}, Current Score: {current_score:.6f}, Best Score: {best_score:.6f}")

    print(f"Optimization finished. Final Best Score: {best_score:.6f}")
    
    # Final probability analysis
    final_probs = calculate_probabilities(best_voxels)
    print("\nFinal probability analysis:")
    print(f"Target probabilities: {np.round(target_probabilities, 4)}")
    print(f"Achieved probabilities: {np.round(final_probs, 4)}")
    diff = np.abs(final_probs - target_probabilities)
    print(f"Absolute differences: {np.round(diff, 4)}")
    print(f"Mean absolute error: {np.mean(diff):.4f}")
    print(f"Final cavity size: {np.sum(best_voxels == 0)} voxels")

    return best_voxels

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

# ===========================================
# Mesh Generation
# ===========================================

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
        
        if len(filled_voxels) == 0:
            print("Error: No filled voxels found!")
            return np.array([]), np.array([])
            
        # 创建更简单的方式来生成blocky mesh
        boxes = []
        for idx, (i, j, k) in enumerate(filled_voxels):
            position = np.array([i, j, k]) * scale_factor
            box_mesh = box(extents=[scale_factor, scale_factor, scale_factor], transform=np.array([
                [1, 0, 0, position[0]],
                [0, 1, 0, position[1]],
                [0, 0, 1, position[2]],
                [0, 0, 0, 1]
            ]))
            boxes.append(box_mesh)
            
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx}/{len(filled_voxels)} voxels...")
        
        if not boxes:
            print("Error: Failed to create any boxes!")
            return np.array([]), np.array([])
            
        # 合并所有立方体网格
        try:
            combined_mesh = trimesh.util.concatenate(boxes)
            print(f"Successfully merged {len(boxes)} boxes into one mesh")
            
            # 直接返回trimesh网格的顶点和面
            vertices = np.array(combined_mesh.vertices)
            faces = np.array(combined_mesh.faces)
            
            print(f"Blocky mesh generated: {len(vertices)} vertices, {len(faces)} faces")
            return vertices, faces
        except Exception as merge_err:
            print(f"Error merging boxes: {merge_err}")
            # 尝试另一种方式
            print("Trying alternative method...")
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            
            for box_mesh in boxes:
                v = np.array(box_mesh.vertices)
                f = np.array(box_mesh.faces) + vertex_offset
                all_vertices.append(v)
                all_faces.append(f)
                vertex_offset += len(v)
            
            if all_vertices and all_faces:
                vertices = np.vstack(all_vertices)
                faces = np.vstack(all_faces)
                print(f"Alternative method generated: {len(vertices)} vertices, {len(faces)} faces")
                return vertices, faces
            else:
                print("Alternative method also failed")
                return np.array([]), np.array([])
        
    except Exception as e:
        print(f"Error generating blocky mesh: {e}")
        import traceback
        traceback.print_exc()
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

    # 检查面的形状
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

    # 检查面的数据类型
    if not np.issubdtype(faces.dtype, np.integer):
        print(f"Warning: Faces dtype is {faces.dtype}, attempting to convert to int.")
        try:
            faces = faces.astype(np.int32)
        except Exception as type_err:
            print(f"Error converting faces dtype: {type_err}")
            return vertices, np.array([])

    # 检查面的索引是否有效
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

    # 尝试使用Open3D修复网格，但如果失败则返回原始网格
    try:
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d.remove_degenerate_triangles()

        if not mesh_o3d.has_triangles():
             print("Error: Mesh has no triangles after Open3D cleaning. Returning original mesh instead.")
             return vertices, faces

        mesh_o3d.orient_triangles()

        final_vertices = np.asarray(mesh_o3d.vertices)
        final_faces = np.asarray(mesh_o3d.triangles)
        print(f"Mesh after Open3D validation: {len(final_vertices)} vertices, {len(final_faces)} faces")
        return final_vertices, final_faces

    except Exception as e:
        print(f"Error during Open3D validation/fixing: {e}")
        print("Warning: Returning original mesh.")
        return vertices, faces

# ===========================================
# STL Export
# ===========================================

def save_to_stl(vertices, faces, filename="biased_dice_blocky.stl"):
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
# Visualization Functions
# ===========================================

def visualize_dice(voxels, probabilities, filename="dice_visualization.png"):
    """
    Visualize the dice with labeled faces based on standard dice numbering
    
    Args:
        voxels: The voxel representation of the dice
        probabilities: The probability distribution for each face
        filename: Output file name for the visualization
    """
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get voxel dimensions
    resolution = voxels.shape[0]
    
    # Standard dice numbering convention (opposite faces sum to 7)
    # Direction order: -z, +z, -y, +y, -x, +x
    # This corresponds to: Bottom, Top, Back, Front, Left, Right
    face_numbers = {
        0: 1,  # Bottom face (-z) = 1
        1: 6,  # Top face (+z) = 6
        2: 3,  # Back face (-y) = 3
        3: 4,  # Front face (+y) = 4
        4: 2,  # Left face (-x) = 2
        5: 5   # Right face (+x) = 5
    }
    
    # Direction vectors for the faces
    directions = [
        (0, 0, -1),  # Bottom (-z)
        (0, 0, 1),   # Top (+z)
        (0, -1, 0),  # Back (-y)
        (0, 1, 0),   # Front (+y)
        (-1, 0, 0),  # Left (-x)
        (1, 0, 0)    # Right (+x)
    ]
    
    # Face positions in 3D space (for text labels)
    face_positions = [
        (resolution/2, resolution/2, 0),             # Bottom
        (resolution/2, resolution/2, resolution),    # Top
        (resolution/2, 0, resolution/2),             # Back
        (resolution/2, resolution, resolution/2),    # Front
        (0, resolution/2, resolution/2),             # Left
        (resolution, resolution/2, resolution/2)     # Right
    ]

    # Plot the voxels
    ax.voxels(voxels, facecolors='lightgray', edgecolors='gray', alpha=0.7)
    
    # Add text labels for the faces with dice numbers
    for i, (pos, prob) in enumerate(zip(face_positions, probabilities)):
        face_num = face_numbers[i]
        direction = directions[i]
        
        # Position the text slightly outside the cube 
        text_pos = (
            pos[0] + direction[0] * 2,
            pos[1] + direction[1] * 2,
            pos[2] + direction[2] * 2
        )
        
        # Calculate the text size based on probability (higher prob = larger text)
        text_size = 10 + 20 * prob
        
        # Add the face number and probability
        ax.text(
            text_pos[0], text_pos[1], text_pos[2],
            f"{face_num}\n({prob:.3f})",
            color='black', fontsize=text_size,
            horizontalalignment='center',
            verticalalignment='center'
        )
    
    # Highlight the center of mass
    filled_positions = np.argwhere(voxels > 0)
    if len(filled_positions) > 0:
        center_of_mass = filled_positions.mean(axis=0)
        ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], 
                  color='red', s=100, label='Center of Mass')
    
    # Highlight the geometric center
    geo_center = np.array(voxels.shape) / 2
    ax.scatter(geo_center[0], geo_center[1], geo_center[2], 
              color='blue', s=100, label='Geometric Center')
    
    # Draw a line between centers to visualize bias
    if len(filled_positions) > 0:
        ax.plot([center_of_mass[0], geo_center[0]], 
                [center_of_mass[1], geo_center[1]], 
                [center_of_mass[2], geo_center[2]], 
                'purple', linewidth=2, label='Bias Vector')
    
    # Add title and legend
    ax.set_title('Biased Dice Visualization', fontsize=16)
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Add axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add probability information
    info_text = "Face Probabilities:\n"
    for i, prob in enumerate(probabilities):
        info_text += f"Face {face_numbers[i]}: {prob:.3f}\n"
    
    # Calculate bias magnitude
    if len(filled_positions) > 0:
        bias_vector = center_of_mass - geo_center
        bias_magnitude = np.linalg.norm(bias_vector)
        info_text += f"\nBias Magnitude: {bias_magnitude:.3f}"
    
    # Add text box with probability information
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
    
    # Show the figure
    plt.show()
    
    return fig, ax

# ===========================================
# Main Function
# ===========================================

def design_biased_dice(target_probabilities, resolution=20, max_iterations=5000):
    """Main function to execute optimization, mesh generation, validation and return results"""
    if not isinstance(target_probabilities, np.ndarray):
        target_probabilities = np.array(target_probabilities)

    if len(target_probabilities) != 6:
        raise ValueError("Target probabilities must be a list or array of length 6.")
    if not np.isclose(np.sum(target_probabilities), 1.0):
         print(f"Warning: Target probabilities sum to {np.sum(target_probabilities)}, normalizing.")
         target_probabilities = target_probabilities / np.sum(target_probabilities)
    
    # optimize parameters
    T_initial = 0.01
    T_min = 1e-6
    alpha = 0.995
    
    print(f"Optimization parameters: T_initial={T_initial}, T_min={T_min}, alpha={alpha}")
    print("Using blocky mesh representation")

    print("Starting voxel optimization...")
    optimized_voxels = optimize_biased_dice(
        target_probabilities, 
        resolution=resolution, 
        max_iterations=max_iterations,
        T_initial=T_initial,
        T_min=T_min,
        alpha=alpha
    )

    # Always generate blocky mesh
    print("\nGenerating blocky mesh using Trimesh...")
    vertices, faces = create_blocky_mesh_from_voxels(optimized_voxels, voxel_size=1.0)

    if len(vertices) > 0 and len(faces) > 0:
        print("\nValidating and fixing generated mesh...")
        try:
            fixed_vertices, fixed_faces = validate_and_fix_mesh(vertices, faces)
            if len(fixed_vertices) > 0 and len(fixed_faces) > 0:
                vertices, faces = fixed_vertices, fixed_faces
            else:
                print("Validation removed all faces, using original mesh instead")
        except Exception as val_err:
            print(f"Error during validation: {val_err}, using original mesh")
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

    # 添加max_prob_idx的定义
    max_prob_idx = np.argmax(final_probabilities)

    final_error = np.mean((final_probabilities - target_probabilities)**2)

    print("\n--- Results ---")
    print(f"Target Probabilities: {np.round(target_probabilities, 4)}")
    print(f"Achieved Probabilities: {np.round(final_probabilities, 4)}")
    print(f"Mean Squared Error: {final_error:.6f}")
    print(f"Generated Mesh: {len(vertices)} vertices, {len(faces)} faces")
    print("---------------")

    # 在现有代码的最后，添加以下可视化调用
    print("\nGenerating visualization...")
    visualize_dice(optimized_voxels, final_probabilities, filename=f"biased_dice_viz_{max_prob_idx+1}.png")

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
        print("2. High six [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]")
        print("3. High five and six [0.1, 0.1, 0.1, 0.1, 0.3, 0.3]")
        print("4. Linear increase [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]")
        print("5. Custom probabilities")
        
        try:
            choice = int(input("Select option (1-5): ") or "2")
            if choice == 1:
                target_probs = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
            elif choice == 2:
                target_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
            elif choice == 3:
                target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3])
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
            voxel_resolution = int(input("\nVoxel resolution (5-30, recommended 20): ") or "20")
            if voxel_resolution < 5 or voxel_resolution > 50:
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
            max_iterations=max_iter
        )

        # Save if mesh generation was successful
        if final_vertices is not None and final_faces is not None and len(final_vertices) > 0 and len(final_faces) > 0:
            save_to_stl(final_vertices, final_faces, filename=output_file)
            
            # Also generate a final visualization with custom filename
            print("\nGenerating final visualization...")
            visualize_dice(final_voxels, achieved_probs, filename=f"{output_file.split('.')[0]}_viz.png")
        else:
            print("\nNo valid mesh was generated.")

        print("\n--- Design Process Complete ---")

    except ImportError as e:
        print(f"\nError: Missing required libraries. {e}")
        print("Please install all required libraries:")
        print("pip install numpy scipy meshio open3d igl trimesh matplotlib")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
