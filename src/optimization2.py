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

# ===========================================
# Main Optimization
# ===========================================

def optimize_biased_dice(target_probabilities,
                         resolution=20,
                         max_iterations=15000,
                         T_initial=0.05,
                         T_min=1e-7,
                         alpha=0.997):
    # Initial full cube
    voxels = np.ones((resolution, resolution, resolution), dtype=bool)
    outer_mask = create_outer_layer_mask(resolution)
    center = np.array([resolution//2]*3)

    # Determine bias direction
    probs_list = list(target_probabilities)
    max_idx = probs_list.index(max(probs_list))
    directions = [
        (0,0,-1), (0,0,1),   # Z
        (0,-1,0), (0,1,0),   # Y
        (-1,0,0), (1,0,0)    # X
    ]
    bias_dir = np.array(directions[max_idx])

    # === Pattern 1: Medium Central Cavity ===
    pattern1 = voxels.copy()
    cav = resolution//3
    for i in range(center[0]-cav, center[0]+cav+1):
        for j in range(center[1]-cav, center[1]+cav+1):
            for k in range(center[2]-cav, center[2]+cav+1):
                if 1 <= i < resolution-1:
                    pattern1[i,j,k] = False

    # === Pattern 2: Extended Cavity with Tapering ===
    pattern2 = voxels.copy()
    thr = resolution // 2

    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                di, dj, dk = i - center[0], j - center[1], k - center[2]
                manh = abs(di) + abs(dj) + abs(dk)
                proj = bias_dir @ np.array([di, dj, dk])
                
                if manh < thr and proj < 0:
                    p = min(1.0, abs(proj) / (thr/2))
                    if np.random.random() < p:
                        # Tentatively remove
                        test = pattern2.copy()
                        test[i, j, k] = False
                        
                        # Only accept if shell + connectivity remain valid
                        if (is_cube_shell_intact(test)
                            and is_connected(test)
                            and is_solid_connected(test)):
                            pattern2[i, j, k] = False

    # === Pattern 3: Tube‑Like Cavity ===
    pattern3 = voxels.copy()
    axis = np.argmax(np.abs(bias_dir))
    tube_r = resolution//4
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                coords = np.array([i,j,k])
                axis_dist = coords[axis] - center[axis]
                proj_side = axis_dist * bias_dir[axis]
                perp = np.delete(coords, axis) - np.delete(center, axis)
                if proj_side < 0 and np.linalg.norm(perp) < tube_r:
                    pattern3[i,j,k] = False

    # === Pattern 4: Diagonal/Wedge Cavity ===
    pattern4 = voxels.copy()
    opp = -bias_dir
    ang_thr = np.cos(np.pi/6)
    wed_thr = resolution//5
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                r = np.array([i,j,k]) - center
                proj = r @ opp
                if proj > wed_thr and np.linalg.norm(r)>0:
                    align = proj/np.linalg.norm(r)
                    if align > ang_thr:
                        pattern4[i,j,k] = False

    # === Pattern 5: Ellipsoidal Cavity ===
    pattern5 = voxels.copy()
    offset = resolution//8
    cav_ctr = center - bias_dir*offset
    r_ax = resolution//4
    r_oth = resolution//6
    if bias_dir[0]!=0:
        radii = np.array([r_ax, r_oth, r_oth])
    elif bias_dir[1]!=0:
        radii = np.array([r_oth, r_ax, r_oth])
    else:
        radii = np.array([r_oth, r_oth, r_ax])
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                rel = np.array([i,j,k]) - cav_ctr
                if np.sum((rel/radii)**2) <= 1:
                    pattern5[i,j,k] = False

    # === Pattern 6: Spherical‑cap (dome) cavity === 
    pattern6 = voxels.copy()
    p = target_probabilities[max_idx]
    # choose a radius in voxels
    R = resolution * 0.3
    # compute cap‐height offset d = R (1 – 2p), but clamp 0 < d < R
    d = np.clip(R * (1 - 2*p), 0.1, R - 0.1)
    cap_center = center - bias_dir * d

    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            for k in range(1, resolution-1):
                pos = np.array([i, j, k])
                # only carve on the opposite side of bias
                if np.dot(pos - center, bias_dir) < 0:
                    if np.linalg.norm(pos - cap_center) <= R:
                        pattern6[i, j, k] = False

    # Debug: print how many voxels were removed
    removed = np.sum(pattern6 == False)
    print(f"Pattern 6 (dome) removed {removed} voxels.")

    # Fix connectivity and collect valid patterns
    raw_patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6]

    candidates = []
    print("Initial pattern performance:")
    for idx, pat in enumerate(raw_patterns, start=1):
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

    # Annealing setup
    T = T_initial
    phase1_iters = max_iterations // 3
    iteration_history, score_history, temp_history = [], [], []
    accepted = 0
    proposed = 0

    # Annealing loop
    for it in range(max_iterations):
        if T <= T_min:
            break

        if it == phase1_iters:
            T = max(T, 0.01)

        # 3) Candidate‑List & Greedy ΔE Selection
        K = 5
        best_cand = None
        best_dE = float('inf')
        for _ in range(K):
            sel = select_weighted_internal_voxel(voxels, outer_mask, center, bias_dir)
            if sel is None:
                continue
            x,y,z = sel
            temp = voxels.copy()
            temp[x,y,z] = not temp[x,y,z]
            if not (is_cube_shell_intact(temp) and is_connected(temp) and is_solid_connected(temp)):
                continue
            dE = evaluate_solution(temp, target_probabilities) - current_score
            if dE < best_dE:
                best_dE = dE
                best_cand = (x,y,z)
        if best_cand is None:
            continue

        x,y,z = best_cand
        temp_voxels = voxels.copy()
        temp_voxels[x,y,z] = not temp_voxels[x,y,z]

        # Evaluate & accept
        delta_E = best_dE
        accept = False
        if delta_E < 0 or np.random.random() < np.exp(-delta_E/T):
            voxels = temp_voxels
            current_score = evaluate_solution(voxels, target_probabilities)
            accept = True
            if current_score < best_score:
                best_voxels = voxels.copy()
                best_score = current_score

        # 4) Adaptive Cooling & Reheating
        proposed += 1
        if accept:
            accepted += 1
        if it>0 and it % 1000 == 0:
            rate = accepted / proposed
            if rate < 0.1:
                T *= 1.5
            accepted = 0
            proposed = 0

        # Cool down
        if it < phase1_iters:
            T *= alpha
        else:
            T *= alpha**1.2

        # History
        if it % 100 == 0:
            iteration_history.append(it)
            score_history.append(current_score)
            temp_history.append(T)

    # Return the best found configuration
    return best_voxels
