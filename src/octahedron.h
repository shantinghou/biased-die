#ifndef OCTAHEDRON_H
#define OCTAHEDRON_H

#include <Eigen/Core>
#include <iostream>

// Creates a regular octahedron with vertices at unit distance from origin
void create_regular_octahedron(Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    // Octahedron has 6 vertices and 8 triangular faces
    V.resize(6, 3);
    F.resize(8, 3);
    
    // Vertices
    V << 1.0, 0.0, 0.0,  // +x  (0)
         -1.0, 0.0, 0.0,  // -x  (1)
         0.0, 1.0, 0.0,  // +y  (2)
         0.0, -1.0, 0.0,  // -y  (3)
         0.0, 0.0, 1.0,  // +z  (4)
         0.0, 0.0, -1.0;  // -z  (5)
    
    // Faces (CCW orientation)
    F << 0, 2, 4,  // +x, +y, +z
         0, 4, 3,  // +x, +z, -y
         0, 3, 5,  // +x, -y, -z
         0, 5, 2,  // +x, -z, +y
         1, 4, 2,  // -x, +z, +y
         1, 3, 4,  // -x, -y, +z
         1, 5, 3,  // -x, -z, -y
         1, 2, 5;  // -x, +y, -z
         
    // Debug output
    std::cout << "Created octahedron with vertices:\n" << V << std::endl;
    std::cout << "and faces:\n" << F << std::endl;
}

#endif // OCTAHEDRON_H