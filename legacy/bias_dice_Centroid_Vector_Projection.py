import numpy as np
import random
import trimesh

# ============================
# Evaluation Functions
# ============================

def is_cube_shell_intact(voxel_cube: np.ndarray) -> bool:
    """check if the shell of the cube is intact"""
    return (
        np.all(voxel_cube[ 0, :, :] == 1) and
        np.all(voxel_cube[-1, :, :] == 1) and
        np.all(voxel_cube[:,  0, :] == 1) and
        np.all(voxel_cube[:, -1, :] == 1) and
        np.all(voxel_cube[:, :,  0] == 1) and
        np.all(voxel_cube[:, :, -1] == 1)
    )

def voxel_to_face_probabilities(voxel_cube: np.ndarray) -> dict:
    """calculate the probability distribution of the faces of the cube"""
    # get the coordinates of all solid voxels
    voxel_positions = np.argwhere(voxel_cube > 0)
    
    # if there are no voxels, return the uniform distribution
    if len(voxel_positions) == 0:
        return {face: 1/6 for face in ['px', 'nx', 'py', 'ny', 'pz', 'nz']}
    
    # calculate the centroid (the average position of all solid voxels)
    centroid = voxel_positions.mean(axis=0)
    
    # the geometric center of the cube
    center = (np.array(voxel_cube.shape) - 1) / 2.0
    
    # the vector from the centroid to the center (note the direction, it is from the centroid to the center, not the center to the centroid)
    r = centroid - center
    
    # define the normal vectors of the six faces (outward)
    face_normals = {
        'px': np.array([ 1,  0,  0]),
        'nx': np.array([-1,  0,  0]),
        'py': np.array([ 0,  1,  0]),
        'ny': np.array([ 0, -1,  0]),
        'pz': np.array([ 0,  0,  1]),
        'nz': np.array([ 0,  0, -1])
    }
    
    # calculate the projection of the centroid offset on each direction (dot product)
    raw_scores = {face: np.dot(r, normal) for face, normal in face_normals.items()}
    
    # use the softmax function to convert the raw scores to a probability distribution
    # this is more reasonable than simply truncating negative values and normalizing
    scores_list = list(raw_scores.values())
    max_score = max(scores_list)
    exp_scores = [np.exp(s - max_score) for s in scores_list]  # subtract the maximum value to avoid numerical overflow
    total = sum(exp_scores)
    
    # return the probability dictionary
    return {face: score/total for face, score in zip(raw_scores.keys(), exp_scores)}

def face_bias_score(prob_mapping: dict, target_face: str) -> float:
    """calculate the score of the target face"""
    return prob_mapping.get(target_face, 0)

def is_connected(voxel_cube: np.ndarray) -> bool:
    """check if the empty space (voxels with value 0) is connected"""
    from scipy.ndimage import label
    structure = np.ones((3, 3, 3), dtype=int)
    # create a mask, representing the empty space
    empty_space = (voxel_cube == 0)
    # if there is no empty space, then it is connected
    if not np.any(empty_space):
        return True
    # label the connected components
    labeled, num_features = label(empty_space, structure=structure)
    return num_features == 1

def evaluate_voxel_bias(voxel_cube: np.ndarray, target_face: str) -> float:
    """evaluate the bias score of the voxel cube"""
    # check the constraint conditions
    if not is_cube_shell_intact(voxel_cube):
        return -np.inf
    if not is_connected(voxel_cube):
        return -np.inf
    
    # calculate the probability distribution of the faces
    probs = voxel_to_face_probabilities(voxel_cube)
    
    # return the probability of the target face as the score
    score = face_bias_score(probs, target_face)
    
    # add additional evaluation metrics: the lower the probability distribution of the other faces, the better
    # this encourages the algorithm to find a more stable solution
    entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs.values())
    # we want the entropy to be lower (the distribution to be more concentrated), but still prioritize the target face probability
    combined_score = score - 0.1 * entropy
    
    return combined_score

# ============================
# Debug Function
# ============================

def print_face_probabilities(voxel_cube):
    """print the probability distribution of the faces, for debugging"""
    probs = voxel_to_face_probabilities(voxel_cube)
    print("Face probabilities:")
    for face, prob in probs.items():
        print(f"  {face}: {prob:.4f}")

# ============================
# Optimization Loop
# ============================

def optimize_voxel_cube(size=5, target_face='nz', steps=5000, temperature=5.0, cooling=0.999):
    """optimize the voxel cube to be biased toward the target face"""
    # initialize a solid cube
    cube = np.ones((size, size, size), dtype=int)
    
    # get the indices of all inner voxels (not including the shell)
    inner_indices = [
        (i, j, k)
        for i in range(1, size - 1)
        for j in range(1, size - 1)
        for k in range(1, size - 1)
    ]
    
    # initial setup: keep the bottom voxels, remove the top voxels (to make the centroid biased toward the bottom)
    for i, j, k in inner_indices:
        # 根据目标面选择初始化策略
        if target_face == 'nz':
            # if the target face is the bottom face, then keep the bottom voxels
            cube[i, j, k] = 1 if k < size // 2 else 0
        elif target_face == 'pz':
            # if the target face is the top face, then keep the top voxels
            cube[i, j, k] = 1 if k >= size // 2 else 0
        elif target_face == 'nx':
            # if the target face is the left face, then keep the left voxels
            cube[i, j, k] = 1 if i < size // 2 else 0
        elif target_face == 'px':
            # if the target face is the right face, then keep the right voxels
            cube[i, j, k] = 1 if i >= size // 2 else 0
        elif target_face == 'ny':
            # if the target face is the front face, then keep the front voxels
            cube[i, j, k] = 1 if j < size // 2 else 0
        elif target_face == 'py':
            # if the target face is the back face, then keep the back voxels
            cube[i, j, k] = 1 if j >= size // 2 else 0
    
    # ensure the initial state of the empty space is connected
    retries = 0
    max_retries = 100
    
    while not is_connected(cube) and retries < max_retries:
        # randomly remove a voxel to try to make the empty space connected
        filled_inner = [(i, j, k) for i, j, k in inner_indices if cube[i, j, k] == 1]
        if not filled_inner:
            break
        i, j, k = random.choice(filled_inner)
        cube[i, j, k] = 0
        retries += 1
    
    # if the empty space is still not connected, use a simpler initialization strategy
    if not is_connected(cube):
        print("Warning: Could not create connected initial state with directional bias.")
        # reinitialize to a solid cube
        cube = np.ones((size, size, size), dtype=int)
        # remove the center voxel to create a simple cavity
        mid = size // 2
        cube[mid, mid, mid] = 0
    
    # evaluate the initial state
    current_score = evaluate_voxel_bias(cube, target_face)
    best_cube = cube.copy()
    best_score = current_score
    
    # print the initial state
    print("\nInitial state:")
    print_face_probabilities(cube)
    print(f"Initial score: {current_score:.4f}")
    
    # record the optimization history
    score_history = []
    
    # start optimization
    for step in range(steps):
        # randomly select an inner voxel
        i, j, k = random.choice(inner_indices)
        original_value = cube[i, j, k]
        
        # try to reverse the state of this voxel
        cube[i, j, k] = 1 - original_value
        
        # check if the modified state is still valid (shell intact and empty space connected)
        valid_state = is_cube_shell_intact(cube) and is_connected(cube)
        
        if valid_state:
            # calculate the new score
            new_score = evaluate_voxel_bias(cube, target_face)
            delta = new_score - current_score
            
            # according to the simulated annealing algorithm, decide whether to accept this modification
            if delta > 0 or (temperature > 1e-8 and np.exp(delta / temperature) > np.random.rand()):
                current_score = new_score
                if new_score > best_score:
                    best_cube = cube.copy()
                    best_score = new_score
            else:
                # if not accepted, restore the original state
                cube[i, j, k] = original_value
        else:
            # if the modification leads to an invalid state, restore the original state
            cube[i, j, k] = original_value
        
        # reduce the temperature
        temperature *= cooling
        
        # print the state every 100 steps
        if step % 100 == 0:
            print(f"Step {step}, Temp {temperature:.4f}, Current Score: {current_score:.4f}, Best Score: {best_score:.4f}")
            score_history.append(best_score)
            
            # 每1000步详细输出一次当前最佳模型的面概率
            if step % 1000 == 0:
                print("\nCurrent best model face probabilities:")
                print_face_probabilities(best_cube)
    
    # 输出最终结果
    print("\nFinal best model:")
    print_face_probabilities(best_cube)
    print(f"Final score: {best_score:.4f}")
    
    # 输出每100步的最佳分数历史
    print("\nScore history (every 100 steps):")
    print(score_history)
    
    # 返回最佳立方体和对应分数
    return best_cube, best_score


# ============================
# STL Export Function
# ============================

def export_voxel_to_stl(voxel_cube: np.ndarray, filename: str = "output.stl", voxel_scale: float = 1.0):
    """将体素立方体导出为STL文件"""
    from trimesh.creation import box
    from trimesh.scene import Scene

    scene = Scene()
    filled = np.argwhere(voxel_cube > 0)
    for (i, j, k) in filled:
        b = box(extents=(voxel_scale, voxel_scale, voxel_scale))
        b.apply_translation([i * voxel_scale, j * voxel_scale, k * voxel_scale])
        scene.add_geometry(b)

    combined = trimesh.util.concatenate(scene.dump())
    combined.export(filename)
    print(f"STL saved to: {filename} (voxel scale = {voxel_scale})")


# ============================
# Visualization Function
# ============================

def visualize_voxel_cube(voxel_cube: np.ndarray):
    """可视化体素立方体并显示面概率"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制体素
        x, y, z = np.indices(voxel_cube.shape)
        ax.voxels(voxel_cube, facecolors='lightblue', edgecolor='gray', alpha=0.7)
        
        # 设置轴标签和限制
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 显示计算出的面概率
        probs = voxel_to_face_probabilities(voxel_cube)
        prob_text = "\n".join([f"{face}: {prob:.3f}" for face, prob in probs.items()])
        plt.figtext(0.02, 0.02, f"Face probabilities:\n{prob_text}", fontsize=10)
        
        # 绘制重心
        voxel_positions = np.argwhere(voxel_cube > 0)
        centroid = voxel_positions.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], centroid[2], color='red', s=100, label='Center of Mass')
        
        # 绘制几何中心
        center = (np.array(voxel_cube.shape) - 1) / 2.0
        ax.scatter(center[0], center[1], center[2], color='green', s=100, label='Geometric Center')
        
        # 绘制从重心到中心的向量
        ax.quiver(centroid[0], centroid[1], centroid[2], 
                 center[0]-centroid[0], center[1]-centroid[1], center[2]-centroid[2], 
                 color='purple', label='CM to Center')
        
        # 添加图例
        ax.legend()
        
        # 调整视角
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib is not available for visualization")


# ============================
# Run Optimization and Export
# ============================

if __name__ == "__main__":
    resolution = int(input("Enter cube resolution (e.g., 10): "))
    target_face = input("Enter target face (px, nx, py, ny, pz, nz) [default=nz]: ") or 'nz'
    steps = int(input("Enter optimization steps [default=5000]: ") or 5000)
    
    voxel_scale = 1.0 / resolution
    best_voxel, score = optimize_voxel_cube(
        size=resolution, 
        target_face=target_face,
        steps=steps
    )
    
    print(f"\nFinal bias score toward {target_face}: {score:.4f}")
    output_filename = f"biased_cube_{target_face}.stl"
    export_voxel_to_stl(best_voxel, output_filename, voxel_scale=voxel_scale)
    
    # 尝试可视化最终结果
    try:
        visualize_voxel_cube(best_voxel)
    except Exception as e:
        print(f"Visualization failed: {e}")