#ifndef DENSITY_OPTIMIZER_H
#define DENSITY_OPTIMIZER_H

#include <Eigen/Core>
#include <vector>

class DensityOptimizer {
public:
    DensityOptimizer(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);
    
    // Generate tetrahedral mesh from the input surface mesh
    bool tetrahedralize();
    
    // Optimize internal density to bias toward a specific face
    bool optimizeDensity(int targetFace);
    
    // Get the center of mass of the current density distribution
    Eigen::Vector3d getCenterOfMass() const;
    
    // Export the optimized density distribution
    bool exportDensityMap(const std::string& filename) const;
    
    // Generate a 3D printable model with variable infill
    bool exportPrintableModel(const std::string& filename) const;
    
private:
    // Original surface mesh
    Eigen::MatrixXd mVertices;
    Eigen::MatrixXi mFaces;
    
    // Tetrahedral mesh
    Eigen::MatrixXd mTetVertices;
    Eigen::MatrixXi mTetElements;
    
    // Density values for each tetrahedron
    Eigen::VectorXd mDensities;
    
    // Compute the bias target point (opposite to the desired up face)
    Eigen::Vector3d computeBiasTarget(int targetFace) const;
    
    // Compute center of mass given density distribution
    Eigen::Vector3d computeCenterOfMass(const Eigen::VectorXd& densities) const;
    
    // Get the face normal
    Eigen::Vector3d getFaceNormal(int faceIndex) const;
    
    // Get the face center
    Eigen::Vector3d getFaceCenter(int faceIndex) const;
};

#endif // DENSITY_OPTIMIZER_H