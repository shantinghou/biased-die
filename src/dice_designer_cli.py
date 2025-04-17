#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
biased dice designer - command line interface

This module contains the CLI implementation of the biased dice designer, providing a command line interface for designing, optimizing, and exporting dice models with a biased probability distribution.
"""

import numpy as np

# Import
try:
    # When running from project root
    from src.visualizer import visualize_dice, save_to_stl
    from src.probability_models import calculate_solid_angles
    from src.mesh_generation import create_blocky_mesh_from_voxels
    from src.optimization import optimize_biased_dice
except ImportError:
    # When running from src directory
    from visualizer import visualize_dice, save_to_stl
    from probability_models import calculate_solid_angles
    from mesh_generation import create_blocky_mesh_from_voxels
    from optimization import optimize_biased_dice

# ===========================================
# Main Function
# ===========================================

def design_biased_dice(target_probabilities, resolution=20, max_iterations=15000, wall_thickness=1, output_filename='biased_dice', progress_callback=None):
    """Main function to execute optimization, mesh generation, validation and return results"""
    if not isinstance(target_probabilities, np.ndarray):
        target_probabilities = np.array(target_probabilities)

    if len(target_probabilities) != 6:
        raise ValueError("Target probabilities must be a list or array of length 6.")
    if not np.isclose(np.sum(target_probabilities), 1.0):
         print(f"Warning: Target probabilities sum to {np.sum(target_probabilities)}, normalizing.")
         target_probabilities = target_probabilities / np.sum(target_probabilities)
    
    # Validate wall thickness
    max_wall_thickness = (resolution - 2) // 2
    if wall_thickness > max_wall_thickness:
        raise ValueError(f"Wall thickness ({wall_thickness}) is too large for resolution ({resolution}). Maximum allowed is {max_wall_thickness}.")
    
    # optimize parameters
    T_initial = 0.01
    T_min = 1e-6
    alpha = 0.995
    
    print(f"Optimization parameters: T_initial={T_initial}, T_min={T_min}, alpha={alpha}")
    print(f"Wall thickness: {wall_thickness} voxels")
    print("Using blocky mesh representation")

    print("Starting voxel optimization...")
    try:
        optimized_voxels = optimize_biased_dice(
            target_probabilities, 
            resolution=resolution, 
            max_iterations=max_iterations,
            T_initial=T_initial,
            T_min=T_min,
            alpha=alpha
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None, None, None, None

    print("\nGenerating blocky mesh using Trimesh...")
    vertices, faces = create_blocky_mesh_from_voxels(optimized_voxels, voxel_size=1.0)

    if len(vertices) == 0 or len(faces) == 0:
        print("Warning: Mesh generation failed, returning empty mesh.")
        vertices, faces = np.array([]), np.array([])

    # Calculate final probabilities
    print("\nCalculating final probabilities...")
    final_solid_angles = calculate_solid_angles(optimized_voxels)
    total_solid_angle = np.sum(final_solid_angles)
    if total_solid_angle > 1e-6:
        final_probabilities = final_solid_angles / total_solid_angle
    else:
        final_probabilities = np.ones(6) / 6 # Fallback if no solid angle
        print("Warning: Total solid angle is near zero, using uniform probability.")

    final_error = np.mean((final_probabilities - target_probabilities)**2)

    print("\n--- Results ---")
    print(f"Target Probabilities: {np.round(target_probabilities, 4)}")
    print(f"Achieved Probabilities: {np.round(final_probabilities, 4)}")
    print(f"Mean Squared Error: {final_error:.6f}")
    print(f"Generated Mesh: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Wall Thickness: {wall_thickness} voxels")
    print("---------------")

    return vertices, faces, optimized_voxels, final_probabilities

# ===========================================
# Main Execution
# ===========================================

if __name__ == "__main__":
    try:
        print("--- Biased Dice Design Tool ---")
        
        # Parameter setup
        print("\nSelect probability distribution:")
        print("1. Uniform [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]")
        print("2. High six [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]")
        print("3. High five and six [0, 0.5, 0, 0, 0, 0.5]")
        print("4. Linear increase [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]")
        print("5. Custom probabilities")
        
        try:
            choice = int(input("Select option (1-5): ") or "2")
            if choice == 1:
                target_probs = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
            elif choice == 2:
                target_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
            elif choice == 3:
                target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3])
            elif choice == 4:
                target_probs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
            elif choice == 5:
                print("Enter 6 probability values separated by spaces (sum will be normalized):")
                probs_input = input("Probability values: ")
                try:
                    probs_list = [float(p) for p in probs_input.split()]
                    if len(probs_list) != 6:
                        print("Must enter 6 values, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
                        target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
                    else:
                        target_probs = np.array(probs_list)
                except:
                    print("Invalid input, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
                    target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
            else:
                print("Invalid selection, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
                target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
        except:
            print("Input error, using default [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]")
            target_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
            
        # Normalize target probabilities
        target_probs = target_probs / np.sum(target_probs)
        print(f"\nTarget probability distribution: {np.round(target_probs, 4)}")
            
        # Set voxel resolution
        try:
            voxel_resolution = int(input("\nVoxel resolution (5-30, recommended 20): ") or "20")
            if voxel_resolution < 5 or voxel_resolution > 50:
                print("Resolution out of range, using default 20")
                voxel_resolution = 20
        except:
            print("Invalid input, using default 20")
            voxel_resolution = 20
            
        print(f"Using voxel resolution: {voxel_resolution}x{voxel_resolution}x{voxel_resolution}")
        
        # Set wall thickness
        max_wall_thickness = (voxel_resolution - 2) // 2
        try:
            wall_thickness = int(input(f"\nWall thickness in voxels (1-{max_wall_thickness}, recommended 1): ") or "1")
            if wall_thickness < 1 or wall_thickness > max_wall_thickness:
                print(f"Wall thickness out of range, using default 1")
                wall_thickness = 1
        except:
            print("Invalid input, using default 1")
            wall_thickness = 1
            
        print(f"Using wall thickness: {wall_thickness} voxels")
        
        # Set iteration count
        try:
            max_iter = int(input("\nMaximum iterations (1000-10000, recommended 5000): ") or "5000")
            if max_iter < 100 or max_iter > 20000:
                print("Iteration count out of range, using default 5000")
                max_iter = 5000
        except:
            print("Invalid input, using default 5000")
            max_iter = 5000
            
        print(f"Maximum iterations: {max_iter}")
        
        # Set output filename
        try:
            output_file = input("\nOutput STL filename (default biased_dice.stl): ") or "assets/biased_dice.stl"
            if not output_file.lower().endswith('.stl'):
                output_file += '.stl'
        except:
            print("Invalid input, using default filename")
            output_file = "assets/biased_dice.stl"
            
        print(f"Output file: {output_file}")
        
        print("\n--- Starting Design Process ---")
            
        # Run design process
        final_vertices, final_faces, final_voxels, achieved_probs = design_biased_dice(
            target_probs,
            resolution=voxel_resolution,
            wall_thickness=wall_thickness,
            max_iterations=max_iter
        )

        # Save if mesh generation was successful
        if final_vertices is not None and final_faces is not None and len(final_vertices) > 0 and len(final_faces) > 0:
            save_to_stl(final_vertices, final_faces, filename=output_file)
            
            # Also generate a final visualization with custom filename
            print("\nGenerating final visualization...")
            visualize_dice(final_voxels, achieved_probs, filename=f"{output_file.split('.')[0]}_viz.png")
        else:
            print("\nNo valid mesh was generated.")

        print("\n--- Design Process Complete ---")

    except ImportError as e:
        print(f"\nError: Missing required libraries. {e}")
        print("Please install all required libraries:")
        print("pip install numpy scipy meshio open3d igl trimesh matplotlib")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
