import numpy as np

try:
    # When running from project root
    from src.connectivity import is_connected, is_solid_connected, ensure_connectivity
    from src.probability_models import calculate_probabilities
    from src.overhang import remove_overhang
except ImportError:
    # When running from src directory
    from connectivity import is_connected, is_solid_connected, ensure_connectivity
    from probability_models import calculate_probabilities
    from overhang import remove_overhang

# ===========================================
# Helper Functions
# ===========================================

def create_outer_layer_mask(resolution):
    mask = np.zeros((resolution, resolution, resolution), dtype=bool)
    mask[0, :, :] = mask[-1, :, :] = True
    mask[:, 0, :] = mask[:, -1, :] = True
    mask[:, :, 0] = mask[:, :, -1] = True
    return mask

def is_cube_shell_intact(voxels):
    return np.all(voxels[0, :, :] == 1) and np.all(voxels[-1, :, :] == 1) and \
           np.all(voxels[:, 0, :] == 1) and np.all(voxels[:, -1, :] == 1) and \
           np.all(voxels[:, :, 0] == 1) and np.all(voxels[:, :, -1] == 1)

def evaluate_solution(voxels, target_probabilities):
    if not (is_cube_shell_intact(voxels) and is_connected(voxels) and is_solid_connected(voxels)):
        return float('inf')
    probs = calculate_probabilities(voxels)
    return np.mean((probs - target_probabilities) ** 2)

def select_random_internal_voxel(voxels, outer_layer_mask):
    internal_voxels = np.logical_and(voxels, ~outer_layer_mask)
    indices = np.where(internal_voxels)
    if len(indices[0]) == 0:
        return None
    idx = np.random.randint(len(indices[0]))
    return indices[0][idx], indices[1][idx], indices[2][idx]

def select_weighted_internal_voxel(voxels, outer_mask, center, bias_dirs):
    interior = np.array(np.where((voxels == True) & (~outer_mask))).T
    if interior.shape[0] == 0:
        return None

    rel = interior - center  # shape (N,3)
    # Sum projections along all bias directions
    proj = np.sum([np.dot(bias_dir, rel.T) for bias_dir in bias_dirs], axis=0)
    
    weights = np.clip(-proj, 0, None).astype(np.float64)

    # Fallback to uniform if all weights are zero
    if weights.sum() < 1e-9:
        weights[:] = 1.0

    weights /= weights.sum()
    idx = np.random.choice(len(weights), p=weights)
    return tuple(interior[idx])

def is_valid_pattern(voxels):
    return is_cube_shell_intact(voxels) and is_connected(voxels) and is_solid_connected(voxels)


# ===========================================
# Main Optimization
# ===========================================

def optimize_biased_dice(target_probabilities,
                         resolution=20,
                         max_iterations=15000,
                         T_initial=0.05,
                         T_min=1e-7,
                         alpha=0.997):
   
    """
    Create cavity patterns for a biased die based on target probabilities.
    Returns a list of initial patterns to be used for optimization.
    """
    # Initial full cube
    voxels = np.ones((resolution, resolution, resolution), dtype=bool)
    outer_mask = create_outer_layer_mask(resolution)
    center = np.array([resolution//2, resolution//2, resolution//2])

    # Determine bias direction based on highest probability face
    probs_list = list(target_probabilities)
    max_value = max(probs_list) - 0.25
    max_indices = [i for i, value in enumerate(probs_list) if value >= max_value]
    directions = [
        (0,0,-1), (0,0,1),   # Z-axis (faces 0,1)
        (0,-1,0), (0,1,0),   # Y-axis (faces 2,3)
        (-1,0,0), (1,0,0)    # X-axis (faces 4,5)
    ]
    bias_dirs = [np.array(directions[i]) for i in max_indices]
    
    patterns = []
    
    # === Pattern 1: Medium Central Cavity ===
    pattern1 = voxels.copy()
    cavity_size = resolution // 3
    
    for i in range(center[0]-cavity_size, center[0]+cavity_size+1):
        for j in range(center[1]-cavity_size, center[1]+cavity_size+1):
            for k in range(center[2]-cavity_size, center[2]+cavity_size+1):
                if 1 <= i < resolution-1 and 1 <= j < resolution-1 and 1 <= k < resolution-1:
                    pattern1[i,j,k] = False

    if is_valid_pattern(pattern1):
        patterns.append(pattern1)

    # === Pattern 2: Extended Cavity with Tapering ===
    pattern2 = voxels.copy()
    threshold = resolution // 2
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                rel_pos = np.array([i - center[0], j - center[1], k - center[2]])
                manhattan_dist = np.sum(np.abs(rel_pos))

                for bias_dir in bias_dirs:
                    projection = np.dot(bias_dir, rel_pos)
                    
                    if manhattan_dist < threshold and projection < 0:
                        carve_prob = min(1.0, abs(projection) / (threshold/2))
                        if np.random.random() < carve_prob:
                            test = pattern2.copy()
                            test[i, j, k] = False
                            
                            if is_valid_pattern(test):
                                pattern2[i, j, k] = False

    if is_valid_pattern(pattern2):
        patterns.append(pattern2)
    
    # === Pattern 3: Tube-Like Cavity ===
    pattern3 = voxels.copy()
    axis = np.argmax(np.abs(bias_dirs[0]))  # Assuming bias_dirs are normalized
    tube_radius = resolution // 4
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                coords = np.array([i, j, k])
                axis_dist = coords[axis] - center[axis]
                
                for bias_dir in bias_dirs:
                    axis_projection = axis_dist * bias_dir[axis]
                    coords_2d = np.delete(coords, axis)
                    center_2d = np.delete(center, axis)
                    perp_dist = np.linalg.norm(coords_2d - center_2d)
                    
                    if axis_projection < 0 and perp_dist < tube_radius:
                        test = pattern3.copy()
                        test[i, j, k] = False
                        if is_valid_pattern(test):
                            pattern3[i, j, k] = False
                            
    if is_valid_pattern(pattern3):
        patterns.append(pattern3)
    
    # === Pattern 4: Wedge Cavity ===
    pattern4 = voxels.copy()
    opposite_dir = -bias_dirs[0]  # Using the first bias_dir as the primary direction
    angle_threshold = np.cos(np.pi/6)
    wedge_threshold = resolution // 5
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                rel_pos = np.array([i, j, k]) - center
                if np.linalg.norm(rel_pos) == 0:
                    continue
                    
                projection = np.dot(rel_pos, opposite_dir)
                if projection > wedge_threshold:
                    alignment = projection / np.linalg.norm(rel_pos)
                    if alignment > angle_threshold:
                        test = pattern4.copy()
                        test[i, j, k] = False
                        if is_valid_pattern(test):
                            pattern4[i, j, k] = False
                            
    if is_valid_pattern(pattern4):
        patterns.append(pattern4)
    
    # === Pattern 5: Face-Centered Dome ===
    pattern5 = voxels.copy()

    # Calculate face centers in index space
    face_centers = [
        np.array([center[0], center[1], 0]),             # bottom (Z-)
        np.array([center[0], center[1], resolution-1]),  # top (Z+)
        np.array([center[0], 0, center[2]]),             # back (Y-)
        np.array([center[0], resolution-1, center[2]]),  # front (Y+)
        np.array([0, center[1], center[2]]),             # left (X-)
        np.array([resolution-1, center[1], center[2]])   # right (X+)
    ]

    # Calculate scaled radii based on probability weights
    probs_array = np.array(probs_list)
    scaled_probs = probs_array / np.sum(probs_array)  # Normalize probabilities
    base_radius = resolution // 2
    max_radius_adjustment = resolution // 4
    radii = {}

    # Calculate radius for each face based on its probability
    for idx, prob in enumerate(probs_list):
        # Higher probability = smaller cavity = larger radius
        # Scale inversely proportional to the probability
        weight_factor = 1 - (scaled_probs[idx] / max(scaled_probs))
        radius_reduction = weight_factor * max_radius_adjustment
        radii[idx] = base_radius - radius_reduction

    # Create a mask to track which voxels are protected by any dome
    protected_voxels = np.zeros((resolution, resolution, resolution), dtype=bool)

    # First pass: Mark all voxels that should be protected by any dome
    for max_idx in max_indices:
        # Find the opposite face index
        opposite_idx = max_idx + 1 if max_idx % 2 == 0 else max_idx - 1
        opposite_face = face_centers[opposite_idx]
        bias_dir = bias_dirs[max_indices.index(max_idx)]
        
        # Get probability-adjusted radius for this face
        sphere_radius = radii[max_idx]
        
        # Place sphere center at the opposite face center
        sphere_center = opposite_face
        
        # Mark voxels protected by this dome
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                for k in range(1, resolution-1):
                    pos = np.array([i, j, k])
                    # Check if voxel is within this dome
                    if np.linalg.norm(pos - sphere_center) <= sphere_radius:
                        protected_voxels[i, j, k] = True

    # Second pass: Carve voxels that are not protected by any dome
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                if not protected_voxels[i, j, k]:
                    # Only carve if maintaining pattern validity
                    test = pattern5.copy()
                    test[i, j, k] = False
                    if is_valid_pattern(test):
                        pattern5[i, j, k] = False

    # After processing all domes
    if is_valid_pattern(pattern5):
        patterns.append(pattern5)

    # === Pattern 6: Combined Biases Cavity ===
    pattern6 = voxels.copy()
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                rel_pos = np.array([i - center[0], j - center[1], k - center[2]])
                combined_projection = 0

                for bias_dir in bias_dirs:
                    combined_projection += np.dot(bias_dir, rel_pos)
                
                if combined_projection < 0:  # Carve away from the center in the combined bias direction
                    test = pattern6.copy()
                    test[i, j, k] = False
                    if is_valid_pattern(test):
                        pattern6[i, j, k] = False
                        
    if is_valid_pattern(pattern6):
        patterns.append(pattern6)

    # Report pattern statistics
    for i, pattern in enumerate(patterns):
        cavity_size = np.sum(pattern == False)
        print(f"Pattern {i+1} created with {cavity_size} voxels removed")

    # Select the best pattern and perform optimization
    candidates = []
    for idx, pat in enumerate(patterns, start=1):
        if not np.any(pat == 0):
            continue

        fixed = ensure_connectivity(pat)
        valid = is_cube_shell_intact(fixed) and is_connected(fixed) and is_solid_connected(fixed)
        if valid:
            score = evaluate_solution(fixed, target_probabilities)
            candidates.append((score, idx, fixed))

    # Choose best starting pattern
    candidates.sort(key=lambda x: x[0])
    best_score, best_pattern_idx, voxels = candidates[0]
    current_score = best_score
    best_voxels = voxels.copy()
    print(f"Selected initial pattern {best_pattern_idx} with score {best_score:.6f}")

    # Annealing setup with spherical bias
    T = T_initial
    phase1_iters = max_iterations // 3
    iteration_history, score_history, temp_history = [], [], []
    accepted = 0
    proposed = 0

    # Define sphere growth parameters
    R_min = 0.2 * resolution
    R_max = 0.5 * resolution
    rim_width = resolution / 8
    
    # Get the first max index to determine opposite face
    primary_max_idx = max_indices[0]
    opposite_idx = primary_max_idx + 1 if primary_max_idx % 2 == 0 else primary_max_idx - 1
    opposite_face_center = face_centers[opposite_idx]
    primary_bias_dir = bias_dirs[0]  # Use the first bias direction for optimization

    # Main loop
    for it in range(max_iterations):
        if T <= T_min:
            break
        if it == phase1_iters:
            T = max(T, 0.01)

        # grow sphere radius over time
        t = it / max_iterations
        sphere_radius = R_min + (R_max - R_min) * np.sqrt(t)

        K = 5
        best_cand = None
        best_dE = float('inf')

        for _ in range(K):
            # 70%: sample voxels near current spherical boundary
            if np.random.random() < 0.7:
                # --- spherical‐rim sampling using rim_width ---
                rim_candidates = []
                for _ in range(10):
                    sel = select_random_internal_voxel(voxels, outer_mask)
                    if sel is None:
                        continue
                    pos = np.array(sel)
                    d = abs(np.linalg.norm(pos - opposite_face_center) - sphere_radius)
                    # only keep those within rim_width of the ideal sphere
                    if d <= rim_width:
                        rim_candidates.append((sel, d))
                if rim_candidates:
                    # pick one at random from the rim set
                    sel = rim_candidates[np.random.randint(len(rim_candidates))][0]
                else:
                    # fallback to weighted‐COM sampling
                    sel = select_weighted_internal_voxel(voxels, outer_mask, center, bias_dirs)
            else:
                # 30%: weighted‐COM selection
                sel = select_weighted_internal_voxel(voxels, outer_mask, center, bias_dirs)

            if sel is None:
                continue

            x,y,z = sel
            temp = voxels.copy()
            temp[x,y,z] = not temp[x,y,z]
            if not is_valid_pattern(temp):
                continue

            std_dE = evaluate_solution(temp, target_probabilities) - current_score

            # spherical‐bias nudges: 
            # if current state aligns with the dome, penalize flipping; else encourage it
            pos = np.array([x, y, z])
            inside = np.linalg.norm(pos - opposite_face_center) <= sphere_radius
            if (voxels[x, y, z] and inside) or (not voxels[x, y, z] and not inside):
                sph_bias =  0.2 * (1 - t)
            else:
                sph_bias = -0.1 * (1 - t)

            dE = std_dE + sph_bias
            if dE < best_dE:
                best_dE = dE
                best_cand = (x, y, z)

        if best_cand is None:
            continue

        # apply and accept/reject based on the true energy
        x,y,z = best_cand
        temp = voxels.copy()
        temp[x,y,z] = not temp[x,y,z]
        true_dE = evaluate_solution(temp, target_probabilities) - current_score

        accept = False
        if true_dE < 0 or np.random.random() < np.exp(-true_dE/T):
            voxels = temp
            current_score = evaluate_solution(voxels, target_probabilities)
            accept = True
            if current_score < best_score:
                best_voxels = voxels.copy()
                best_score = current_score

        # adaptive reheating
        proposed += 1
        if accept:
            accepted += 1
        if it > 0 and it % 1000 == 0:
            rate = accepted / proposed
            if rate < 0.1:
                T *= 1.5
            accepted = 0
            proposed = 0

        # cooling
        if it < phase1_iters:
            T *= alpha
        else:
            T *= alpha**1.2

        # record history occasionally
        if it % 100 == 0:
            iteration_history.append(it)
            score_history.append(current_score)
            temp_history.append(T)
    
    # return best_voxels
    
    # 1) Orient so the most weighted face is on the build plate (z=0)
    face_idx = int(np.argmax(target_probabilities))
    voxels_clean = remove_overhang(best_voxels, face_idx)

    # Return the best found configuration
    return voxels_clean