# Biased Die Project

This project uses LibIGL to design and optimize an octahedral die with biased internal density distribution, making it more likely to land on a target face.

## Project Structure

- `src/` - Source code
- `include/` - Header files and external libraries
- `assets/` - 3D models and output files
- `CMakeLists.txt` - Build configuration

## Requirements

- C++17 compiler
- CMake 3.16+
- LibIGL (automatically fetched by CMake)
- Eigen3

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Running the Program

From the build directory:

```bash
./biased_die
```

## How It Works

1. The program loads or generates an octahedral die model
2. It creates a tetrahedral mesh of the interior
3. It optimizes the internal density distribution to bias the die toward a target face
4. A visual representation shows the shifted center of mass
5. Optional: Export the model for 3D printing with variable infill

## Physics Behind the Bias

The program shifts the center of mass toward the opposite face of the target face. When the die is rolled:
- The heaviest side tends to end up at the bottom
- This makes the opposite side (our target) more likely to end up on top

## Customization

Edit `main.cpp` to change:
- Target face index
- Bias strength
- Density range

## 3D Printing Notes

The current implementation exports a density map that can be used with 3D printing software that supports variable infill patterns. Future versions may include direct export to printer-specific formats.