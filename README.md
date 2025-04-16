# Biased Dice Designer

A tool for designing and optimizing dice models with specific probability distributions. You can create biased dice by adjusting the internal mass distribution, making certain faces more likely to land face up than others.

## Project Structure

```
src/
├── dice_designer_gui.py - Graphical user interface implementation
├── dice_designer_cli.py - Command line interface implementation
├── visualizer.py - 3D visualization tools
├── probability_models.py - Probability calculation using solid angle method
├── mesh_generation.py - 3D mesh generation from voxel models
├── optimization.py - Simulated annealing optimization algorithm
└── connectivity.py - Connectivity validation for dice structure
```

## Features

- Voxel model optimization with simulated annealing
- Multiple initialization patterns for different bias profiles
- Connectivity validation to ensure structural integrity
- Probability calculation using solid angle method
- 3D visualization of dice with probability distribution
- STL export for 3D printing

## Usage

### Graphical Interface Version

The GUI provides an intuitive interface with the following features:
- Set target probabilities for each face
- Adjust optimization parameters (voxel resolution, iteration count, etc.)
- Run optimization algorithm
- Visualize the generated dice model
- Save visualization images and export STL model files

### Command Line Interface Version

The CLI version guides you through:
- Selecting a probability distribution
- Setting optimization parameters
- Running the optimization process
- Saving results and visualization images

## Dice Face Numbering

The dice face numbering follows standard dice conventions, where opposite faces sum to 7:
- Face 1 at the bottom, corresponding to face 6 at the top
- Face 2 on the left, corresponding to face 5 on the right
- Face 3 at the back, corresponding to face 4 at the front

## Algorithm

The tool creates biased dice through:

1. Starting with a solid cube and iteratively removing internal voxels
2. Optimizing voxel distribution via simulated annealing
3. Maintaining outer shell integrity and internal connectivity
4. Calculating face probabilities using the centroid solid angle model
5. Generating a 3D mesh for visualization and 3D printing

## Dependencies

```
numpy
scipy
matplotlib
PyQt5 (for GUI)
meshio
trimesh
```

