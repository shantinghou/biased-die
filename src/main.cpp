#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <iostream>
#include "octahedron.h"
#include "density_optimizer.h"

int main(int argc, char *argv[])
{
    // Mesh matrices
    Eigen::MatrixXd V; // Vertices
    Eigen::MatrixXi F; // Faces

    std::cout << "No octahedron model found. Generating regular octahedron..." << std::endl;
    create_regular_octahedron(V, F);

    // Verify the generated octahedron
    std::cout << "Generated octahedron with " << V.rows() << " vertices and " 
                << F.rows() << " faces." << std::endl;
                
    // Save the generated octahedron for future use
    bool written = igl::writeOBJ("assets/models/octahedron.obj", V, F);
    if (!written) {
        std::cerr << "Warning: Failed to write octahedron.obj" << std::endl;
    }
    
    // Make sure we have a valid octahedron before proceeding
    if (V.rows() == 0 || F.rows() == 0) {
        std::cerr << "Failed to create or load a valid octahedron!" << std::endl;
        return 1;
    }
    
    // Initialize our density optimizer
    DensityOptimizer optimizer(V, F);
    
    // Set target face (0-7 for an octahedron, assuming standard indexing)
    int targetFace = 3;
    
    // Create a tetrahedral mesh for the interior
    bool tetrahedralized = optimizer.tetrahedralize();
    
    // Only continue with optimization if tetrahedralization was successful
    if (tetrahedralized) {
        // Run the optimization to bias the die
        optimizer.optimizeDensity(targetFace);
        
        // Get the optimized center of mass for visualization
        Eigen::Vector3d COM = optimizer.getCenterOfMass();
        
        // Visualize the mesh and center of mass
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V, F);
        viewer.data().set_face_based(true);
        
        // Add point for center of mass
        viewer.data().add_points(COM.transpose(), Eigen::RowVector3d(1, 0, 0));
        
        // Draw a line from geometric center to COM
        Eigen::Vector3d center = V.colwise().mean();
        Eigen::MatrixXd line(2, 3);
        line.row(0) = center.transpose();
        line.row(1) = COM.transpose();
        viewer.data().add_edges(line.row(0), line.row(1), Eigen::RowVector3d(0, 1, 0));
        
        std::cout << "Green line shows the shift in center of mass" << std::endl;
        std::cout << "Red point is the new center of mass" << std::endl;
        viewer.launch();
    } else {
        // If tetrahedralization failed, just show the octahedron
        std::cout << "Could not optimize density since tetrahedralization failed." << std::endl;
        std::cout << "Displaying the octahedron model only..." << std::endl;
        
        igl::opengl::glfw::Viewer viewer;
        viewer.data().set_mesh(V, F);
        viewer.data().set_face_based(true);
        viewer.launch();
    }
    
    return 0;
}