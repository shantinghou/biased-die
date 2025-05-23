#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Biased Dice Visualizer

This module provides functions for visualizing dice models, including 3D rendering, label display, and STL file export.
"""

import numpy as np
import matplotlib.pyplot as plt
import meshio

def save_to_stl(vertices, faces, filename="assets/biased_dice_blocky.stl"):
    """Save mesh to STL file"""
    if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
        print("Error: Cannot save empty mesh.")
        return

    print(f"Saving mesh to {filename}: {len(vertices)} vertices, {len(faces)} faces")

    try:
        mesh_to_save = meshio.Mesh(
            points=np.array(vertices, dtype=np.float64),
            cells=[("triangle", np.array(faces, dtype=np.int32))]
        )
        mesh_to_save.write(filename)
        print(f"Successfully saved mesh to {filename}")

    except Exception as e:
        print(f"Error saving STL file: {e}")
        print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")

def visualize_dice(voxels, probabilities, filename="biased_dice_viz.png", target_ax=None, show=True, show_walls=True, wall_thickness=1, preserve_view=False):
    """
    Visualize the biased dice in 3D with labeled faces and probability information.
    
    Args:
        voxels: 3D numpy array of the voxel model
        probabilities: Array of 6 probabilities for each face
        filename: Output filename for the visualization
        target_ax: Target matplotlib axis to draw on (if None, create a new figure)
        show: Whether to display the figure with plt.show()
        show_walls: Whether to display the dice walls or only the internal structure
        wall_thickness: Thickness of the dice walls in voxels
        preserve_view: Whether to preserve the current view angle (only used with target_ax)
    
    Returns:
        fig, ax: The matplotlib figure and axis objects
    """
    if target_ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        preserve_view = False  # Cannot preserve view for new figure
    else:
        ax = target_ax
        fig = ax.figure
        
        # Store current view angle before clearing
        if preserve_view and hasattr(ax, 'elev') and hasattr(ax, 'azim'):
            current_elev = ax.elev
            current_azim = ax.azim
        else:
            current_elev = 30
            current_azim = -60
    
    resolution = voxels.shape[0]
    
    # Standard dice face numbering (opposite faces sum to 7)
    face_numbers = {
        0: 1,  # Bottom face (-z) = 1
        1: 6,  # Top face (+z) = 6
        2: 3,  # Back face (-y) = 3
        3: 4,  # Front face (+y) = 4
        4: 2,  # Left face (-x) = 2
        5: 5   # Right face (+x) = 5
    }
    
    # Directions for each face
    directions = [
        (0, 0, -1),  # Bottom (-z)
        (0, 0, 1),   # Top (+z)
        (0, -1, 0),  # Back (-y)
        (0, 1, 0),   # Front (+y)
        (-1, 0, 0),  # Left (-x)
        (1, 0, 0)    # Right (+x)
    ]
    
    # Positions for each face label
    face_positions = [
        (resolution/2, resolution/2, 0),             # Bottom
        (resolution/2, resolution/2, resolution),    # Top
        (resolution/2, 0, resolution/2),             # Back
        (resolution/2, resolution, resolution/2),    # Front
        (0, resolution/2, resolution/2),             # Left
        (resolution, resolution/2, resolution/2)     # Right
    ]
    
    # Create a modified voxel array for visualization
    display_voxels = voxels.copy()
    
    # If not showing walls, create a mask for the wall voxels based on wall_thickness
    if not show_walls:
        # Create wall mask - identifying which voxels are part of the wall
        wall_mask = np.zeros_like(voxels, dtype=bool)
        
        # Mark each layer of walls according to the wall_thickness
        for layer in range(wall_thickness):
            wall_mask[layer, :, :] = True
            wall_mask[resolution-1-layer, :, :] = True
            wall_mask[:, layer, :] = True
            wall_mask[:, resolution-1-layer, :] = True
            wall_mask[:, :, layer] = True
            wall_mask[:, :, resolution-1-layer] = True
        
        # Create a version with internal structure only by keeping non-wall voxels
        internal_voxels = voxels & ~wall_mask
        
        # Use the internal voxels for display
        display_voxels = internal_voxels
    
    # Visualize the voxel model consistently
    voxel_colors = np.empty(display_voxels.shape, dtype=object)
    voxel_colors[display_voxels] = 'lightgray'
    
    # Display the voxels with consistent style
    ax.voxels(display_voxels, facecolors=voxel_colors, edgecolors='gray', alpha=0.7)
    
    # Set consistent axis limits regardless of wall visibility
    ax.set_xlim(0, resolution)
    ax.set_ylim(0, resolution)
    ax.set_zlim(0, resolution)
    
    # Restore original view angle if we're using an existing axis
    if target_ax is not None and preserve_view:
        ax.view_init(elev=current_elev, azim=current_azim)
    else:
        # Default view angle for new figures
        ax.view_init(elev=30, azim=-60)
    
    # Add labels for each face with size proportional to probability
    for i, (pos, prob) in enumerate(zip(face_positions, probabilities)):
        face_num = face_numbers[i]
        direction = directions[i]
        
        # Position text slightly outside each face
        text_pos = (
            pos[0] + direction[0] * 2,
            pos[1] + direction[1] * 2,
            pos[2] + direction[2] * 2
        )
        
        # Text size proportional to probability
        text_size = 10 + 20 * prob
        
        # Add the face number and probability as text
        ax.text(
            text_pos[0], text_pos[1], text_pos[2],
            f"{face_num}\n({prob:.3f})",
            color='black', fontsize=text_size,
            horizontalalignment='center',
            verticalalignment='center'
        )
    
    # Calculate the center of mass
    total_mass = np.sum(voxels)  # Always use the full voxel model for CoM calculation
    if total_mass > 0:
        x = y = z = np.linspace(0.5 / resolution, 1 - 0.5 / resolution, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        center_of_mass_x = np.sum(X * voxels) / total_mass
        center_of_mass_y = np.sum(Y * voxels) / total_mass
        center_of_mass_z = np.sum(Z * voxels) / total_mass
        center_of_mass = np.array([center_of_mass_x, center_of_mass_y, center_of_mass_z])
        center_of_mass = center_of_mass * resolution
        ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], 
                  color='red', s=100, label='center of mass')
    
    # Highlight the geometric center
    geo_center = np.array(voxels.shape) / 2
    ax.scatter(geo_center[0], geo_center[1], geo_center[2], 
              color='blue', s=100, label='geometric center')
    
    # Draw a line between centers to visualize bias
    if total_mass > 0:
        ax.plot([center_of_mass[0], geo_center[0]], 
                [center_of_mass[1], geo_center[1]], 
                [center_of_mass[2], geo_center[2]], 
                'purple', linewidth=2, label='bias vector')
    
    # Add title and legend
    display_mode = "Full Dice" if show_walls else "Internal Structure Only"
    ax.set_title(f'Biased Dice Visualization - {display_mode}', fontsize=16)
    ax.legend()
    
    # Ensure consistent aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Force the aspect ratio to be equal and fix it
    # This ensures the cube looks like a cube and not stretched
    ax.set_aspect('equal')
    
    # Add probability information (only if creating a new figure)
    if target_ax is None:
        info_text = "face probabilities:\n"
        for i, prob in enumerate(probabilities):
            info_text += f"face {face_numbers[i]}: {prob:.3f}\n"
        
        # Calculate bias magnitude
        if total_mass > 0:
            bias_vector = center_of_mass - geo_center
            bias_magnitude = np.linalg.norm(bias_vector)
            info_text += f"\nbias magnitude: {bias_magnitude:.3f}"
        
        # Add text box with probability information
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7))

    # Save the figure if filename is provided
    if filename:
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
        except Exception as e:
            print(f"Error saving visualization: {e}")

    # Show the figure and keep window open if requested
    if show and target_ax is None:
        plt.show()
    
    return fig, ax 