import igl
import numpy as np
import pyvista as pv

def visualize_with_pyvista(vertices, faces, colors):
    # turn faces to pyvista format
    faces_pv = np.hstack([[3, *face] for face in faces]).astype(np.int32)

    mesh = pv.PolyData(vertices, faces_pv)
    mesh.point_data['colors'] = colors

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=True)
    plotter.show()

# create the vertices of the cube
def create_cube_vertices():
    """create a cube with side length 1"""
    vertices = np.array([
        [0, 0, 0],  # vertex 0
        [1, 0, 0],  # vertex 1
        [1, 1, 0],  # vertex 2
        [0, 1, 0],  # vertex 3
        [0, 0, 1],  # vertex 4
        [1, 0, 1],  # vertex 5
        [1, 1, 1],  # vertex 6
        [0, 1, 1],  # vertex 7
    ], dtype=np.float64)
    return vertices

# create the faces of the cube, each quadrilateral face is composed of two triangles
def create_cube_faces():
    faces = np.array([
        [0, 1, 2],  # bottom triangle 1
        [0, 2, 3],  # bottom triangle 2
        [4, 5, 6],  # top triangle 1
        [4, 6, 7],  # top triangle 2
        [0, 1, 5],  # front triangle 1
        [0, 5, 4],  # front triangle 2
        [2, 3, 7],  # back triangle 1
        [2, 7, 6],  # back triangle 2
        [0, 3, 7],  # left triangle 1
        [0, 7, 4],  # left triangle 2
        [1, 2, 6],  # right triangle 1
        [1, 6, 5],  # right triangle 2
    ], dtype=np.int32)
    return faces

def main():
    # create the cube
    vertices = create_cube_vertices()
    faces = create_cube_faces()
    
    # calculate the vertex normals
    normals = igl.per_vertex_normals(vertices, faces)
    
    # create the vertex colors, each vertex has a RGB color
    colors = np.zeros((vertices.shape[0], 3))
    # set some colors - color the cube by the vertex positions
    colors[:, 0] = vertices[:, 0]  # x coordinate -> red channel
    colors[:, 1] = vertices[:, 1]  # y coordinate -> green channel
    colors[:, 2] = vertices[:, 2]  # z coordinate -> blue channel
    
    # save the cube as an OBJ file using igl
    igl.write_obj("assets/cube.obj", vertices, faces)
    

    try:
        igl.write_triangle_mesh("assets/cube.ply", vertices, faces)
        print("successfully saved as PLY format")
    except AttributeError:
        print("current igl version does not support the write_triangle_mesh function")
    
    visualize_with_pyvista(vertices, faces, colors)

if __name__ == "__main__":
    main()
