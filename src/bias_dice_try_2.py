import numpy as np
import random
import trimesh

# ============================
# Evaluation Functions
# ============================

def is_cube_shell_intact(voxel_cube: np.ndarray) -> bool:
    """检查立方体的外壳是否完整"""
    return (
        np.all(voxel_cube[ 0, :, :] == 1) and
        np.all(voxel_cube[-1, :, :] == 1) and
        np.all(voxel_cube[:,  0, :] == 1) and
        np.all(voxel_cube[:, -1, :] == 1) and
        np.all(voxel_cube[:, :,  0] == 1) and
        np.all(voxel_cube[:, :, -1] == 1)
    )

def voxel_to_face_probabilities(voxel_cube: np.ndarray) -> dict:
    """计算立方体各面的倾倒概率分布"""
    # 获取所有实心体素的坐标
    voxel_positions = np.argwhere(voxel_cube > 0)
    
    # 如果没有体素，返回均匀分布
    if len(voxel_positions) == 0:
        return {face: 1/6 for face in ['px', 'nx', 'py', 'ny', 'pz', 'nz']}
    
    # 计算重心（所有实心体素的平均位置）
    centroid = voxel_positions.mean(axis=0)
    
    # 立方体几何中心
    center = (np.array(voxel_cube.shape) - 1) / 2.0
    
    # 从重心到中心的向量（注意方向，是重心到中心，不是中心到重心）
    r = centroid - center
    
    # 定义六个面的法向量（朝外）
    face_normals = {
        'px': np.array([ 1,  0,  0]),
        'nx': np.array([-1,  0,  0]),
        'py': np.array([ 0,  1,  0]),
        'ny': np.array([ 0, -1,  0]),
        'pz': np.array([ 0,  0,  1]),
        'nz': np.array([ 0,  0, -1])
    }
    
    # 计算重心偏移在各个方向的投影（点积）
    raw_scores = {face: np.dot(r, normal) for face, normal in face_normals.items()}
    
    # 使用softmax函数将原始分数转换为概率分布
    # 这比简单地截断负值并归一化更合理
    scores_list = list(raw_scores.values())
    max_score = max(scores_list)
    exp_scores = [np.exp(s - max_score) for s in scores_list]  # 减去最大值以避免数值溢出
    total = sum(exp_scores)
    
    # 返回概率字典
    return {face: score/total for face, score in zip(raw_scores.keys(), exp_scores)}

def face_bias_score(prob_mapping: dict, target_face: str) -> float:
    """计算目标面的偏向分数"""
    return prob_mapping.get(target_face, 0)

def is_connected(voxel_cube: np.ndarray) -> bool:
    """检查空洞（值为0的体素）是否连通"""
    from scipy.ndimage import label
    structure = np.ones((3, 3, 3), dtype=int)
    # 创建一个掩码，表示空洞
    empty_space = (voxel_cube == 0)
    # 如果没有空洞，则认为是连通的
    if not np.any(empty_space):
        return True
    # 标记连接的组件
    labeled, num_features = label(empty_space, structure=structure)
    return num_features == 1

def evaluate_voxel_bias(voxel_cube: np.ndarray, target_face: str) -> float:
    """评估体素立方体的偏向性分数"""
    # 检查约束条件
    if not is_cube_shell_intact(voxel_cube):
        return -np.inf
    if not is_connected(voxel_cube):
        return -np.inf
    
    # 计算各面的概率分布
    probs = voxel_to_face_probabilities(voxel_cube)
    
    # 返回目标面的概率作为分数
    score = face_bias_score(probs, target_face)
    
    # 添加额外的评估指标：其他面的概率分布越低越好
    # 这促使算法寻找更稳定的解决方案
    entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs.values())
    # 我们希望熵更低（分布更集中），但仍以目标面概率为主要指标
    combined_score = score - 0.1 * entropy
    
    return combined_score

# ============================
# Debug Function
# ============================

def print_face_probabilities(voxel_cube):
    """输出面概率分布，用于调试"""
    probs = voxel_to_face_probabilities(voxel_cube)
    print("Face probabilities:")
    for face, prob in probs.items():
        print(f"  {face}: {prob:.4f}")

# ============================
# Optimization Loop
# ============================

def optimize_voxel_cube(size=5, target_face='nz', steps=5000, temperature=5.0, cooling=0.999):
    """优化体素立方体使其偏向于目标面"""
    # 初始化一个实心立方体
    cube = np.ones((size, size, size), dtype=int)
    
    # 获取所有内部体素的索引（不包括外壳）
    inner_indices = [
        (i, j, k)
        for i in range(1, size - 1)
        for j in range(1, size - 1)
        for k in range(1, size - 1)
    ]
    
    # 初始设置：保留底部体素，移除顶部体素（使重心偏向底部）
    for i, j, k in inner_indices:
        # 根据目标面选择初始化策略
        if target_face == 'nz':
            # 如果目标是底面朝下，则保留底部体素
            cube[i, j, k] = 1 if k < size // 2 else 0
        elif target_face == 'pz':
            # 如果目标是顶面朝上，则保留顶部体素
            cube[i, j, k] = 1 if k >= size // 2 else 0
        elif target_face == 'nx':
            # 如果目标是左面朝左，则保留左侧体素
            cube[i, j, k] = 1 if i < size // 2 else 0
        elif target_face == 'px':
            # 如果目标是右面朝右，则保留右侧体素
            cube[i, j, k] = 1 if i >= size // 2 else 0
        elif target_face == 'ny':
            # 如果目标是前面朝前，则保留前侧体素
            cube[i, j, k] = 1 if j < size // 2 else 0
        elif target_face == 'py':
            # 如果目标是后面朝后，则保留后侧体素
            cube[i, j, k] = 1 if j >= size // 2 else 0
    
    # 确保初始状态空洞是连通的
    retries = 0
    max_retries = 100
    
    while not is_connected(cube) and retries < max_retries:
        # 随机移除一个体素以尝试使空洞连通
        filled_inner = [(i, j, k) for i, j, k in inner_indices if cube[i, j, k] == 1]
        if not filled_inner:
            break
        i, j, k = random.choice(filled_inner)
        cube[i, j, k] = 0
        retries += 1
    
    # 如果仍然不连通，使用更简单的初始化策略
    if not is_connected(cube):
        print("Warning: Could not create connected initial state with directional bias.")
        # 重新初始化为实心立方体
        cube = np.ones((size, size, size), dtype=int)
        # 移除中心体素以创建简单的空腔
        mid = size // 2
        cube[mid, mid, mid] = 0
    
    # 评估初始状态
    current_score = evaluate_voxel_bias(cube, target_face)
    best_cube = cube.copy()
    best_score = current_score
    
    # 输出初始面概率
    print("\nInitial state:")
    print_face_probabilities(cube)
    print(f"Initial score: {current_score:.4f}")
    
    # 记录优化历程
    score_history = []
    
    # 开始优化
    for step in range(steps):
        # 随机选择一个内部体素
        i, j, k = random.choice(inner_indices)
        original_value = cube[i, j, k]
        
        # 尝试反转这个体素的状态
        cube[i, j, k] = 1 - original_value
        
        # 检查修改后是否仍然有效（壳完整且空洞连通）
        valid_state = is_cube_shell_intact(cube) and is_connected(cube)
        
        if valid_state:
            # 计算新的分数
            new_score = evaluate_voxel_bias(cube, target_face)
            delta = new_score - current_score
            
            # 按照模拟退火算法决定是否接受这一修改
            if delta > 0 or (temperature > 1e-8 and np.exp(delta / temperature) > np.random.rand()):
                current_score = new_score
                if new_score > best_score:
                    best_cube = cube.copy()
                    best_score = new_score
            else:
                # 如果不接受，恢复原状态
                cube[i, j, k] = original_value
        else:
            # 如果修改导致无效状态，恢复原状态
            cube[i, j, k] = original_value
        
        # 降低温度
        temperature *= cooling
        
        # 每隔一定步数输出状态
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