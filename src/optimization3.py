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

def create_outer_layer_mask(resolution):
    mask = np.zeros((resolution, resolution, resolution), dtype=bool)
    mask[0, :, :] = mask[-1, :, :] = True
    mask[:, 0, :] = mask[:, -1, :] = True
    mask[:, :, 0] = mask[:, :, -1] = True
    return mask

def is_cube_shell_intact(voxels):
    return (
        np.all(voxels[0, :, :] == 1) and
        np.all(voxels[-1, :, :] == 1) and
        np.all(voxels[:, 0, :] == 1) and
        np.all(voxels[:, -1, :] == 1) and
        np.all(voxels[:, :, 0] == 1) and
        np.all(voxels[:, :, -1] == 1)
    )

def evaluate_solution(voxels, target_probabilities):
    if not (is_cube_shell_intact(voxels) and is_connected(voxels) and is_solid_connected(voxels)):
        return float('inf')
    probs = calculate_probabilities(voxels)
    return np.mean((probs - target_probabilities)**2)

def select_random_internal_voxel(voxels, outer_layer_mask):
    """Select a random internal voxel"""
    internal_voxels = np.logical_and(voxels, ~outer_layer_mask)
    indices = np.where(internal_voxels)
    if len(indices[0]) == 0:
        return None
    idx = np.random.randint(len(indices[0]))
    return indices[0][idx], indices[1][idx], indices[2][idx]


def select_weighted_internal_voxel(voxels, outer_mask, center, bias_dir):
    """Choose an internal voxel whose removal likely shifts COM toward bias_dir."""
    # Only consider interior voxels that are currently filled (True)
    interior = np.array(np.where((voxels == True) & (~outer_mask))).T
    if interior.shape[0] == 0:
        return None

    # Compute relative vectors from the center
    rel = interior - center  # shape (N,3)

    # Projection onto bias direction: removal shifts COM toward bias_dir when proj < 0
    proj = bias_dir @ rel.T  # shape (N,)
    # We want positive weights for proj < 0, so:
    weights = np.clip(-proj, 0, None).astype(np.float64)

    # Fallback to uniform if all weights are zero
    if weights.sum() < 1e-9:
        weights[:] = 1.0

    # Normalize to get a probability distribution
    weights /= weights.sum()

    # Sample one index according to those weights
    idx = np.random.choice(len(weights), p=weights)
    return tuple(interior[idx])

def is_valid_pattern(voxels):
    """Check if a voxel pattern maintains all required constraints."""
    return (is_cube_shell_intact(voxels) and 
            is_connected(voxels) and 
            is_solid_connected(voxels))

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
    max_idx = probs_list.index(max(probs_list))
    directions = [
        (0,0,-1), (0,0,1),   # Z-axis (faces 0,1)
        (0,-1,0), (0,1,0),   # Y-axis (faces 2,3)
        (-1,0,0), (1,0,0)    # X-axis (faces 4,5)
    ]
    bias_dir = np.array(directions[max_idx])
    
    patterns = []
    
    # === Pattern 1: Medium Central Cavity ===
    pattern1 = voxels.copy()
    cavity_size = resolution // 3
    
    for i in range(center[0]-cavity_size, center[0]+cavity_size+1):
        for j in range(center[1]-cavity_size, center[1]+cavity_size+1):
            for k in range(center[2]-cavity_size, center[2]+cavity_size+1):
                # Ensure we stay within interior voxels
                if 1 <= i < resolution-1 and 1 <= j < resolution-1 and 1 <= k < resolution-1:
                    pattern1[i,j,k] = False
    
    # Verify pattern maintains cube integrity
    if is_valid_pattern(pattern1):
        patterns.append(pattern1)
    
    # === Pattern 2: Extended Cavity with Tapering ===
    pattern2 = voxels.copy()
    threshold = resolution // 2
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                # Calculate position relative to center
                rel_pos = np.array([i - center[0], j - center[1], k - center[2]])
                manhattan_dist = np.sum(np.abs(rel_pos))
                # Projection along bias direction
                projection = np.dot(bias_dir, rel_pos)
                
                # Only carve in direction opposite to highest probability face
                if manhattan_dist < threshold and projection < 0:
                    # Probability increases with distance from center along bias
                    carve_prob = min(1.0, abs(projection) / (threshold/2))
                    if np.random.random() < carve_prob:
                        # Try removing the voxel
                        test = pattern2.copy()
                        test[i, j, k] = False
                        
                        # Only accept if constraints are maintained
                        if is_valid_pattern(test):
                            pattern2[i, j, k] = False
    
    if is_valid_pattern(pattern2):
        patterns.append(pattern2)
    
    # === Pattern 3: Tube-Like Cavity ===
    pattern3 = voxels.copy()
    axis = np.argmax(np.abs(bias_dir))
    tube_radius = resolution // 4
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                coords = np.array([i, j, k])
                # Distance along primary axis
                axis_dist = coords[axis] - center[axis]
                # Make sure we're carving on side opposite to highest probability
                axis_projection = axis_dist * bias_dir[axis]
                
                # Calculate perpendicular distance from axis
                coords_2d = np.delete(coords, axis)
                center_2d = np.delete(center, axis)
                perp_dist = np.linalg.norm(coords_2d - center_2d)
                
                # Carve tube in direction away from highest probability face
                if axis_projection < 0 and perp_dist < tube_radius:
                    test = pattern3.copy()
                    test[i, j, k] = False
                    if is_valid_pattern(test):
                        pattern3[i, j, k] = False
    
    if is_valid_pattern(pattern3):
        patterns.append(pattern3)
    
    # === Pattern 4: Wedge Cavity ===
    pattern4 = voxels.copy()
    opposite_dir = -bias_dir
    # 30-degree angle threshold (cos(π/6))
    angle_threshold = np.cos(np.pi/6)  
    wedge_threshold = resolution // 5
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                rel_pos = np.array([i, j, k]) - center
                # Skip center voxel
                if np.linalg.norm(rel_pos) == 0:
                    continue
                    
                # Projection along opposite direction to bias
                projection = np.dot(rel_pos, opposite_dir)
                
                # Only carve if projection exceeds threshold and alignment is good
                if projection > wedge_threshold:
                    # Cosine of angle between vector and opposite_dir
                    alignment = projection / np.linalg.norm(rel_pos)
                    if alignment > angle_threshold:
                        test = pattern4.copy()
                        test[i, j, k] = False
                        if is_valid_pattern(test):
                            pattern4[i, j, k] = False
    
    if is_valid_pattern(pattern4):
        patterns.append(pattern4)
    
    # === Pattern 5: Ellipsoidal Cavity ===
    pattern5 = voxels.copy()
    offset = resolution // 8
    
    # Place cavity center offset from cube center away from highest probability face
    cavity_center = center - bias_dir * offset
    
    # Elongate ellipsoid along bias direction
    radius_along = resolution // 4
    radius_perp = resolution // 6
    
    # Set radii based on bias direction
    radii = np.full(3, radius_perp, dtype=np.float64)
    axis = np.nonzero(bias_dir)[0][0]  # Find non-zero axis of bias_dir
    radii[axis] = radius_along
    
    # Carve ellipsoid
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                pos = np.array([i, j, k], dtype=np.float64)
                rel_pos = pos - cavity_center
                
                # Check if point is inside ellipsoid: (x/a)² + (y/b)² + (z/c)² ≤ 1
                if np.sum((rel_pos / radii)**2) <= 1.0:
                    test = pattern5.copy()
                    test[i, j, k] = False
                    if is_valid_pattern(test):
                        pattern5[i, j, k] = False
    
    if is_valid_pattern(pattern5):
        patterns.append(pattern5)
    
    # === Pattern 6: Spherical-Cap (Dome) Cavity === 
    pattern6 = voxels.copy()
    p = target_probabilities[max_idx]
    radius = resolution * 0.3
    
    # Calculate cap height offset based on probability
    # Higher probability = smaller cavity on opposite side
    cap_depth = np.clip(radius * (1 - 2*p), 0.1, radius - 0.1)
    cap_center = center - bias_dir * cap_depth
    
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                pos = np.array([i, j, k])
                # Only carve on opposite side to highest probability face
                if np.dot(pos - center, bias_dir) < 0:
                    if np.linalg.norm(pos - cap_center) <= radius:
                        test = pattern6.copy()
                        test[i, j, k] = False
                        if is_valid_pattern(test):
                            pattern6[i, j, k] = False
    
    if is_valid_pattern(pattern6):
        patterns.append(pattern6)
    
    # === Pattern 7: Face-Centered Dome ===
    pattern7 = voxels.copy()
    # Set radius to half the cube width
    sphere_radius = resolution // 2

    # Calculate face centers in index space
    face_centers = [
        np.array([center[0], center[1], 0]),             # bottom (Z-)
        np.array([center[0], center[1], resolution-1]),  # top (Z+)
        np.array([center[0], 0, center[2]]),             # back (Y-)
        np.array([center[0], resolution-1, center[2]]),  # front (Y+)
        np.array([0, center[1], center[2]]),             # left (X-)
        np.array([resolution-1, center[1], center[2]])   # right (X+)
    ]

    # Find the opposite face index (faces are in pairs: 0-1, 2-3, 4-5)
    opposite_idx = max_idx + 1 if max_idx % 2 == 0 else max_idx - 1
    opposite_face = face_centers[opposite_idx]

    # Place sphere center at the opposite face center
    sphere_center = opposite_face

    # Carve everything OUTSIDE the sphere but INSIDE the cube
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                pos = np.array([i, j, k])
                # Only modify voxels on interior side of opposite face
                if np.dot(pos - opposite_face, -bias_dir) < 0:
                    # Keep sphere, remove everything else
                    if np.linalg.norm(pos - sphere_center) <= sphere_radius:
                        test = pattern7.copy()
                        test[i, j, k] = False
                        if is_valid_pattern(test):
                            pattern7[i, j, k] = False
    
    if is_valid_pattern(pattern7):
        patterns.append(pattern7)

    # Report pattern statistics
    for i, pattern in enumerate(patterns):
        cavity_size = np.sum(pattern == False)
        print(f"Pattern {i+1} created with {cavity_size} voxels removed")


    candidates = []
    print("Initial pattern performance:")
    for idx, pat in enumerate(patterns, start=1):
        if not np.any(pat == 0):
            # no cavity at all
            print(f"  Pattern {idx}: no cavity (skipped)")
            continue

        fixed = ensure_connectivity(pat)
        valid = (
            is_cube_shell_intact(fixed)
            and is_connected(fixed)
            and is_solid_connected(fixed)
        )
        if not valid:
            print(f"  Pattern {idx}: invalid (connectivity or shell failed)")
            continue

        score = evaluate_solution(fixed, target_probabilities)
        print(f"  Pattern {idx}: score = {score:.6f}")
        candidates.append((score, idx, fixed))

    # Fallback if none valid
    if not candidates:
        # small central cavity
        fallback = voxels.copy()
        for d in range(3):
            for delta in (-1,0,1):
                idx = center.copy()
                idx[d] += delta
                fallback[tuple(idx)] = False
        score = evaluate_solution(fallback, target_probabilities)
        print(f"  Fallback: score = {score:.6f}")
        candidates = [(score, 0, fallback)]

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
    opposite_idx = max_idx + 1 if max_idx % 2 == 0 else max_idx - 1
    opposite_face_center = face_centers[opposite_idx]

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
                    sel = select_weighted_internal_voxel(voxels, outer_mask, center, bias_dir)
            else:
                # 30%: your original weighted‐COM selection
                sel = select_weighted_internal_voxel(voxels, outer_mask, center, bias_dir)

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

    # Return the best found configuration
    return best_voxels
