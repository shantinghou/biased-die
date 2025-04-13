import numpy as np
import random
import trimesh

# ============================
# Evaluation Functions
# ============================

def is_cube_shell_intact(voxel_cube: np.ndarray) -> bool:
    return (
        np.all(voxel_cube[ 0, :, :] == 1) and
        np.all(voxel_cube[-1, :, :] == 1) and
        np.all(voxel_cube[:,  0, :] == 1) and
        np.all(voxel_cube[:, -1, :] == 1) and
        np.all(voxel_cube[:, :,  0] == 1) and
        np.all(voxel_cube[:, :, -1] == 1)
    )

def voxel_to_face_probabilities(voxel_cube: np.ndarray) -> dict:
    if np.sum(voxel_cube) == 0:
        return {face: 1/6 for face in ['px', 'nx', 'py', 'ny', 'pz', 'nz']}
    voxel_positions = np.argwhere(voxel_cube > 0)
    centroid = voxel_positions.mean(axis=0)
    center = np.array(voxel_cube.shape) / 2.0
    r = center - centroid
    face_normals = {
        'px': np.array([ 1,  0,  0]),
        'nx': np.array([-1,  0,  0]),
        'py': np.array([ 0,  1,  0]),
        'ny': np.array([ 0, -1,  0]),
        'pz': np.array([ 0,  0,  1]),
        'nz': np.array([ 0,  0, -1])
    }
    scores = {face: np.dot(r, normal) for face, normal in face_normals.items()}
    scores = {k: max(0, v) for k, v in scores.items()}
    total = sum(scores.values())
    if total == 0:
        return {face: 1/6 for face in face_normals}
    return {k: v / total for k, v in scores.items()}

def face_bias_score(prob_mapping: dict, face: str) -> float:
    target_prob = prob_mapping.get(face, 0)
    other_probs = [v for k, v in prob_mapping.items() if k != face]
    return target_prob - sum(other_probs) / len(other_probs)

def evaluate_voxel_bias(voxel_cube: np.ndarray, target_face: str) -> float:
    if not is_cube_shell_intact(voxel_cube):
        return -np.inf
    probs = voxel_to_face_probabilities(voxel_cube)
    return face_bias_score(probs, target_face)


# ============================
# Optimization Loop
# ============================

def optimize_voxel_cube(size=5, target_face='nz', steps=5000, temperature=1.0, cooling=0.995):
    cube = np.ones((size, size, size), dtype=int)
    inner_indices = [
        (i, j, k)
        for i in range(1, size - 1)
        for j in range(1, size - 1)
        for k in range(1, size - 1)
    ]
    for idx in inner_indices:
        cube[idx] = random.choice([0, 1])

    best_cube = cube.copy()
    best_score = evaluate_voxel_bias(best_cube, target_face)

    for step in range(steps):
        i, j, k = random.choice(inner_indices)
        cube[i, j, k] = 1 - cube[i, j, k]
        score = evaluate_voxel_bias(cube, target_face)
        delta = score - best_score

        if delta > 0 or np.exp(delta / temperature) > np.random.rand():
            if score > best_score:
                best_cube = cube.copy()
                best_score = score
        else:
            cube[i, j, k] = 1 - cube[i, j, k]

        temperature *= cooling

        if step % 100 == 0:
            print(f"Step {step}, Temp {temperature:.4f}, Best Score: {best_score:.4f}")

    return best_cube, best_score


# ============================
# STL Export Function (True Blocky Cubes)
# ============================

def export_voxel_to_stl(voxel_cube: np.ndarray, filename: str = "output.stl"):
    from trimesh.creation import box
    from trimesh.scene import Scene

    scene = Scene()
    size = 1.0  # Each voxel is a unit cube
    filled = np.argwhere(voxel_cube > 0)
    for (i, j, k) in filled:
        b = box(extents=(size, size, size))
        b.apply_translation([i * size, j * size, k * size])
        scene.add_geometry(b)

    combined = trimesh.util.concatenate(scene.dump())
    combined.export(filename)
    print(f"STL saved to: {filename}")


# ============================
# Run Optimization and Export
# ============================

if __name__ == "__main__":
    best_voxel, score = optimize_voxel_cube(size=5, target_face='nz')
    print(f"Final bias score toward nz: {score:.4f}")
    export_voxel_to_stl(best_voxel, "biased_cube_nz.stl")
