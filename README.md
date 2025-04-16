# 偏置骰子设计工具 (Biased Dice Designer)

这个工具允许你设计和优化具有特定概率分布的骰子模型。可以通过调整骰子内部的质量分布来偏置骰子，使某些面比其他面更容易朝上。

## 文件结构

项目的核心文件组织如下：

### 启动脚本
- `start_app.py` - 启动图形用户界面版本
- `start_cli.py` - 启动命令行界面版本

### 核心模块
- `src/dice_designer_gui.py` - 图形用户界面实现
- `src/dice_designer_cli.py` - 命令行界面实现
- `src/dice_visualizer.py` - 骰子可视化工具
- `src/probability_models.py` - 概率计算模型
- `src/mesh_generation.py` - 网格生成工具
- `src/optimization.py` - 优化算法

## 使用方法

### 图形界面版本

执行以下命令启动GUI版本：

```
python start_app.py
```

GUI提供以下功能：
- 设置每个面的目标概率
- 调整优化参数（体素分辨率、迭代次数等）
- 运行优化算法
- 可视化生成的骰子模型
- 保存可视化图像和导出STL模型文件

### 命令行版本

执行以下命令启动命令行版本：

```
python start_cli.py
```

CLI版本会引导你：
- 选择概率分布
- 设置优化参数
- 运行优化过程
- 保存结果和可视化图像

## 命名规范说明

项目文件采用了以下命名规范：
- `dice_designer_*`: 骰子设计器的不同界面实现
- `dice_visualizer.py`: 骰子可视化功能
- `start_*.py`: 应用程序启动脚本

## 骰子面编号说明

骰子面的标准编号遵循传统骰子规则：对面之和为7。即：
- 1号面在底部，对应6号面在顶部
- 2号面在左侧，对应5号面在右侧
- 3号面在后方，对应4号面在前方

## Project Structure

```
src/
├── main.py - Main entry point and user interface
├── optimization.py - Simulated annealing algorithm and voxel optimization
├── probability_models.py - Probability calculation based on solid angles based on Centroid Solid Angle Model
├── mesh_generation.py - 3D mesh generation from voxel models
├── connectivity.py - Connectivity validation for dice structure
└── visualization.py - Visualization and STL export
```

## Features

- Voxel model optimization with simulated annealing
- Multiple initialization patterns for different bias profiles
- Connectivity validation to ensure structural integrity
- Probability calculation using solid angle method
- 3D visualization of dice with probability distribution
- STL export for 3D printing

## Dependencies

```
numpy
scipy
meshio
open3d
scikit-image
matplotlib
```

## Usage

Run the main interface:

```bash
python -m src.main
```

The interactive design interface will guide you to:
1. Select a target probability distribution
2. Set voxel resolution and iteration count
3. Generate and visualize the biased dice
4. Export STL files for 3D printing

## Algorithm

The tool creates biased dice through:

1. Starting with a solid cube and iteratively removing internal voxels
2. Optimizing voxel distribution via simulated annealing
3. Maintaining outer shell integrity and internal connectivity
4. Calculating face probabilities using solid angle method
5. Generating a 3D mesh for visualization and printing

graph TD
    A[main.py] --> B[visualization.py]
    A --> C[probability_models.py]
    A --> D[mesh_generation.py]
    A --> E[optimization.py]
    E --> F[connectivity.py]
    E --> C
    F --> G[is_connected/is_solid_connected]

