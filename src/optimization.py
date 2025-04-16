import numpy as np

try:
    # When running from project root
    from src.connectivity import is_connected, is_solid_connected, ensure_connectivity
    from src.probability_models import calculate_probabilities
except ImportError:
    # When running from src directory
    from connectivity import is_connected, is_solid_connected, ensure_connectivity
    from probability_models import calculate_probabilities

# ===========================================
# Helper Functions
# ===========================================

def create_outer_layer_mask(resolution, wall_thickness=1):
    """Create a mask for the outer layer of voxels with specified thickness
    
    Args:
        resolution: The size of the voxel grid
        wall_thickness: The thickness of the outer wall (in voxels)
    
    Returns:
        A boolean array where True indicates voxels that should remain fixed (wall voxels)
    """
    mask = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    # For each layer of the wall thickness
    for layer in range(wall_thickness):
        mask[layer, :, :] = mask[resolution-1-layer, :, :] = True
        mask[:, layer, :] = mask[:, resolution-1-layer, :] = True
        mask[:, :, layer] = mask[:, :, resolution-1-layer] = True
    
    return mask

def is_cube_shell_intact(voxels, wall_thickness=1):
    """Check if the cube shell is intact with specified thickness
    
    Args:
        voxels: 3D numpy array representing the voxel model
        wall_thickness: The required thickness of the outer wall (in voxels)
    
    Returns:
        True if the cube shell has the required thickness, False otherwise
    """
    resolution = voxels.shape[0]
    
    # Check each layer of the wall
    for layer in range(wall_thickness):
        # Check if any voxel in the current wall layer is empty (0)
        if (
            np.any(voxels[layer, :, :] == 0) or
            np.any(voxels[resolution-1-layer, :, :] == 0) or
            np.any(voxels[:, layer, :] == 0) or
            np.any(voxels[:, resolution-1-layer, :] == 0) or
            np.any(voxels[:, :, layer] == 0) or
            np.any(voxels[:, :, resolution-1-layer] == 0)
        ):
            return False
    
    return True

def select_random_internal_voxel(voxels, outer_layer_mask):
    """Select a random internal voxel that is not part of the wall"""
    internal_voxels = np.logical_and(voxels, ~outer_layer_mask)
    indices = np.where(internal_voxels)
    if len(indices[0]) == 0:
        return None
    idx = np.random.randint(len(indices[0]))
    return indices[0][idx], indices[1][idx], indices[2][idx]

def evaluate_solution(voxels, target_probabilities, wall_thickness=1):
    """Evaluate MSE between current and target probabilities"""
    # check the constraints
    if not is_cube_shell_intact(voxels, wall_thickness) or not is_connected(voxels) or not is_solid_connected(voxels):
        return float('inf')  # not satisfied, return infinity
        
    # Calculate probability distribution
    probs = calculate_probabilities(voxels)
    
    # Simple MSE calculation (original scoring mechanism)
    return np.mean((probs - target_probabilities)**2)

# ===========================================
# Voxel Optimization
# ===========================================

def optimize_biased_dice(target_probabilities, resolution=20, max_iterations=15000, wall_thickness=1, T_initial=0.05, T_min=1e-7, alpha=0.997, progress_callback=None):
    """Optimize voxel shape using simulated annealing
    
    Args:
        target_probabilities: Target probability for each face (array of 6 values)
        resolution: Size of the voxel grid
        max_iterations: Maximum number of optimization iterations
        wall_thickness: Thickness of the outer wall in voxels
        T_initial: Initial temperature for simulated annealing
        T_min: Minimum temperature
        alpha: Cooling rate
        progress_callback: Function to report progress
        
    Returns:
        Optimized voxel model
    """
    # Validate wall thickness
    if wall_thickness * 2 + 2 > resolution:
        raise ValueError(f"Wall thickness {wall_thickness} is too large for resolution {resolution}. Maximum allowed is {(resolution - 2) // 2}.")
    
    voxels = np.ones((resolution, resolution, resolution), dtype=bool)
    outer_layer_mask = create_outer_layer_mask(resolution, wall_thickness)

    # initialize a cube with a cavity, ensuring connectivity
    center = resolution // 2
    
    # Try multiple initial patterns and pick the best one
    initial_patterns = []
    
    # Pattern 1: Medium central cavity (larger than original)
    pattern1 = voxels.copy()
    cavity_size = max(1, resolution // 3 - wall_thickness)  # Adjust cavity size based on wall thickness
    for i in range(center-cavity_size, center+cavity_size+1):
        for j in range(center-cavity_size, center+cavity_size+1):
            for k in range(center-cavity_size, center+cavity_size+1):
                if wall_thickness <= i < resolution-wall_thickness and wall_thickness <= j < resolution-wall_thickness and wall_thickness <= k < resolution-wall_thickness:
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
    for i in range(wall_thickness, resolution-wall_thickness):
        for j in range(wall_thickness, resolution-wall_thickness):
            for k in range(wall_thickness, resolution-wall_thickness):
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
                cavity_threshold = resolution // 2 - wall_thickness
                if manhattan_dist < cavity_threshold and proj_dist < 0:
                    # Gradually increase probability of being cavity as we get further in opposite direction
                    cavity_prob = min(1.0, abs(proj_dist) / (cavity_threshold/2))
                    if np.random.random() < cavity_prob:
                        pattern2[i, j, k] = False
    
    # Pattern 3: Tube-like cavity for extreme bias
    pattern3 = voxels.copy()
    # Create a tube along the axis opposite to the bias direction
    axis = np.argmax(np.abs(bias_dir))
    tube_radius = max(1, resolution // 4 - wall_thickness)
    
    for i in range(wall_thickness, resolution-wall_thickness):
        for j in range(wall_thickness, resolution-wall_thickness):
            for k in range(wall_thickness, resolution-wall_thickness):
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
        if is_cube_shell_intact(pattern, wall_thickness) and is_connected(pattern) and is_solid_connected(pattern):
            score = evaluate_solution(pattern, target_probabilities, wall_thickness)
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
                    if wall_thickness <= i < resolution-wall_thickness and wall_thickness <= j < resolution-wall_thickness and wall_thickness <= k < resolution-wall_thickness:
                        simple_pattern[i, j, k] = False
        
        # Create a path from center to one side
        side = max_prob_idx // 2  # Convert to axis (0=z, 1=y, 2=x)
        direction = max_prob_idx % 2  # 0=negative, 1=positive
        
        path_length = max(1, resolution // 3 - wall_thickness)
        coords = [center, center, center]
        
        # Determine step direction (-1 or 1)
        step = 1 if direction else -1
        
        # Create path
        for offset in range(path_length):
            coords[side] = center + step * offset
            if wall_thickness <= coords[0] < resolution-wall_thickness and wall_thickness <= coords[1] < resolution-wall_thickness and wall_thickness <= coords[2] < resolution-wall_thickness:
                simple_pattern[coords[0], coords[1], coords[2]] = False
        
        voxels = simple_pattern
        current_score = evaluate_solution(voxels, target_probabilities, wall_thickness)
    else:
        # Sort by score and select the best
        pattern_scores.sort()
        best_pattern_idx = pattern_scores[0][1]
        
        print(f"Selected initial pattern {best_pattern_idx+1} with score {pattern_scores[0][0]:.6f}")
        voxels = patterns[best_pattern_idx].copy()
        current_score = pattern_scores[0][0]
    best_voxels = voxels.copy()
    best_score = current_score
    
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

        # Report progress if callback provided
        if progress_callback is not None and iteration % 20 == 0:
            progress_percent = min(100, int((iteration / max_iterations) * 100))
            try:
                progress_callback(progress_percent / 100, current_score, voxels.copy())
            except TypeError:
                # if the callback function does not accept all parameters, only pass the progress value
                try:
                    progress_callback(progress_percent / 100)
                except Exception as e:
                    print(f"Warning: Progress callback error: {e}")
            
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
            if not is_cube_shell_intact(temp_voxels, wall_thickness):
                temp_voxels[x, y, z] = not temp_voxels[x, y, z]  # Revert
                continue
        
        # check if the modified cube satisfies all constraints
        if not is_cube_shell_intact(temp_voxels, wall_thickness) or not is_connected(temp_voxels) or not is_solid_connected(temp_voxels):
            continue  # not satisfied, skip this modification
        
        valid_modification = True
        
        if valid_modification:
            new_score = evaluate_solution(temp_voxels, target_probabilities, wall_thickness)
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