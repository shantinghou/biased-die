# Biased Dice Design Tool

A tool for designing biased dice with customizable probability distributions through voxel model optimization.

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

