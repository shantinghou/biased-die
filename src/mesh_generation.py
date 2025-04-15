import numpy as np
import trimesh

def create_blocky_mesh_from_voxels(voxels, voxel_size=1.0):
    """Create a blocky mesh preserving voxel cube geometry using Trimesh"""
    try:
        from trimesh.creation import box
        from trimesh.scene import Scene
        resolution = voxels.shape[0]
        scale_factor = voxel_size / resolution
        scene = Scene()
        filled_voxels = np.argwhere(voxels)
        if len(filled_voxels) == 0:
            print("Error: No filled voxels found!")
            return np.array([]), np.array([])
            
        # generate blocky mesh
        boxes = []
        for idx, (i, j, k) in enumerate(filled_voxels):
            position = np.array([i, j, k]) * scale_factor
            box_mesh = box(extents=[scale_factor, scale_factor, scale_factor], transform=np.array([
                [1, 0, 0, position[0]],
                [0, 1, 0, position[1]],
                [0, 0, 1, position[2]],
                [0, 0, 0, 1]
            ]))
            boxes.append(box_mesh)
            
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx}/{len(filled_voxels)} voxels...")
        
        if not boxes:
            print("Error: Failed to create any boxes!")
            return np.array([]), np.array([])
            
        # merge all the boxes
        try:
            combined_mesh = trimesh.util.concatenate(boxes)
            vertices = np.array(combined_mesh.vertices)
            faces = np.array(combined_mesh.faces)
            print(f"Blocky mesh generated: {len(vertices)} vertices, {len(faces)} faces")
            return vertices, faces
        except Exception as merge_err:
            print(f"Error merging boxes: {merge_err}")
    except Exception as e:
        print(f"Error generating blocky mesh: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]) 