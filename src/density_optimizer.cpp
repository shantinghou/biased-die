#include "density_optimizer.h"
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/writeMESH.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>
#include <Eigen/Dense>
#include <iostream>

// Constructor
DensityOptimizer::DensityOptimizer(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
    : mVertices(vertices), mFaces(faces)
{
    // Initialize with uniform density
    mDensities.resize(0); // Will be properly sized after tetrahedralization
    
    // Make sure the vertices and faces are valid
    std::cout << "Initializing optimizer with " << mVertices.rows() << " vertices and " 
              << mFaces.rows() << " faces." << std::endl;
              
    // Debug: save the input mesh
    igl::writeOBJ("assets/output/input_mesh.obj", mVertices, mFaces);
}

// Generate tetrahedral mesh
bool DensityOptimizer::tetrahedralize()
{
    // Check if the input mesh is valid
    if (mVertices.rows() < 4 || mFaces.rows() < 4) {
        std::cerr << "Input mesh is invalid for tetrahedralization. Need at least 4 vertices and 4 faces." << std::endl;
        return false;
    }
    
    // Debug vertices and faces
    std::cout << "Vertices before tetrahedralization:\n" << mVertices.topRows(5) << std::endl;
    std::cout << "Faces before tetrahedralization:\n" << mFaces.topRows(5) << std::endl;
    
    // Parameters for tetrahedralization - use more conservative settings
    std::string tetgenFlags = "pq1.414a0.1"; // Less aggressive quality constraints
    
    try {
        // Generate tetrahedral mesh using TetGen
        Eigen::MatrixXi TF;
        igl::copyleft::tetgen::tetrahedralize(mVertices, mFaces, tetgenFlags, 
                                             mTetVertices, mTetElements, TF);
        
        std::cout << "Tetrahedralization successful. Generated " 
                  << mTetElements.rows() << " tetrahedra." << std::endl;
                  
        // Check if tetrahedralization produced any elements
        if (mTetElements.rows() == 0) {
            std::cerr << "Tetrahedralization did not produce any elements!" << std::endl;
            return false;
        }
        
        // Initialize densities with uniform values
        mDensities = Eigen::VectorXd::Ones(mTetElements.rows());
        
        // Save the tetrahedral mesh for debugging/visualization
        igl::writeMESH("assets/output/tetmesh.mesh", mTetVertices, mTetElements, TF);
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Tetrahedralization failed: " << e.what() << std::endl;
        return false;
    }
}

// Compute bias target point
Eigen::Vector3d DensityOptimizer::computeBiasTarget(int targetFace) const
{
    // Get the normal of the target face
    Eigen::Vector3d targetNormal = getFaceNormal(targetFace);
    
    // Get the center of the target face
    Eigen::Vector3d targetCenter = getFaceCenter(targetFace);
    
    // Find the opposite face (approximately)
    // In an octahedron, it would be the face with most opposite normal
    int oppositeIdx = -1;
    double minDot = 1.0;
    
    for (int i = 0; i < mFaces.rows(); ++i) {
        if (i == targetFace) continue;
        
        Eigen::Vector3d normal = getFaceNormal(i);
        double dot = normal.dot(targetNormal);
        
        // The most negative dot product indicates most opposite direction
        if (dot < minDot) {
            minDot = dot;
            oppositeIdx = i;
        }
    }
    
    if (oppositeIdx == -1) {
        std::cerr << "Could not find opposite face!" << std::endl;
        return Eigen::Vector3d::Zero();
    }
    
    // Get center and normal of opposite face
    Eigen::Vector3d oppositeCenter = getFaceCenter(oppositeIdx);
    Eigen::Vector3d oppositeNormal = getFaceNormal(oppositeIdx);
    
    // Create bias point below the opposite face
    // This is where we want to shift the center of mass to
    double biasStrength = 0.3; // Adjust this to control bias strength
    Eigen::Vector3d biasTarget = oppositeCenter - biasStrength * oppositeNormal;
    
    std::cout << "Target face: " << targetFace << ", Opposite face: " << oppositeIdx << std::endl;
    std::cout << "Bias target: " << biasTarget.transpose() << std::endl;
    
    return biasTarget;
}

// Get face normal
Eigen::Vector3d DensityOptimizer::getFaceNormal(int faceIndex) const
{
    Eigen::MatrixXd faceNormals;
    igl::per_face_normals(mVertices, mFaces, faceNormals);
    return faceNormals.row(faceIndex).normalized();
}

// Get face center
Eigen::Vector3d DensityOptimizer::getFaceCenter(int faceIndex) const
{
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i) {
        center += mVertices.row(mFaces(faceIndex, i)).transpose();
    }
    return center / 3.0;
}

// Compute center of mass
Eigen::Vector3d DensityOptimizer::computeCenterOfMass(const Eigen::VectorXd& densities) const
{
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    double totalMass = 0.0;
    
    // For each tetrahedron
    for (int i = 0; i < mTetElements.rows(); ++i) {
        // Get vertices of the tetrahedron
        Eigen::Vector3d v0 = mTetVertices.row(mTetElements(i, 0)).transpose();
        Eigen::Vector3d v1 = mTetVertices.row(mTetElements(i, 1)).transpose();
        Eigen::Vector3d v2 = mTetVertices.row(mTetElements(i, 2)).transpose();
        Eigen::Vector3d v3 = mTetVertices.row(mTetElements(i, 3)).transpose();
        
        // Calculate volume of the tetrahedron
        double volume = std::abs((v1 - v0).cross(v2 - v0).dot(v3 - v0)) / 6.0;
        
        // Calculate mass of the tetrahedron
        double mass = volume * densities(i);
        
        // Calculate center of the tetrahedron
        Eigen::Vector3d center = (v0 + v1 + v2 + v3) / 4.0;
        
        // Add weighted contribution to COM
        com += mass * center;
        totalMass += mass;
    }
    
    // Normalize by total mass
    if (totalMass > 0) {
        com /= totalMass;
    }
    
    return com;
}

// Optimize density distribution
bool DensityOptimizer::optimizeDensity(int targetFace)
{
    // Check if tetrahedralization has been performed
    if (mTetElements.rows() == 0) {
        std::cerr << "No tetrahedral mesh available. Run tetrahedralize() first." << std::endl;
        return false;
    }
    
    // Define density bounds
    double minDensity = 0.2;  // Minimum density (e.g., for infill)
    double maxDensity = 1.0;  // Maximum density (solid material)
    
    // Compute the bias target (where we want to shift the COM)
    Eigen::Vector3d biasTarget = computeBiasTarget(targetFace);
    
    // For this simplified example, we'll use a simple heuristic:
    // Assign higher density to tetrahedra closer to the bias target
    
    // Get geometric center of the die
    Eigen::Vector3d center = mVertices.colwise().mean();
    
    // Direction from center to bias target
    Eigen::Vector3d biasDir = (biasTarget - center).normalized();
    
    // For each tetrahedron
    for (int i = 0; i < mTetElements.rows(); ++i) {
        // Get center of tetrahedron
        Eigen::Vector3d v0 = mTetVertices.row(mTetElements(i, 0)).transpose();
        Eigen::Vector3d v1 = mTetVertices.row(mTetElements(i, 1)).transpose();
        Eigen::Vector3d v2 = mTetVertices.row(mTetElements(i, 2)).transpose();
        Eigen::Vector3d v3 = mTetVertices.row(mTetElements(i, 3)).transpose();
        Eigen::Vector3d tetCenter = (v0 + v1 + v2 + v3) / 4.0;
        
        // Project the tetrahedron center onto the bias direction
        double projection = (tetCenter - center).dot(biasDir);
        
        // Normalize to [0,1] based on distance from center
        double normalizedDist = (projection + 1.0) / 2.0;  // Assuming [-1,1] range
        
        // Assign density based on distance (closer to bias target = higher density)
        mDensities(i) = minDensity + normalizedDist * (maxDensity - minDensity);
    }
    
    std::cout << "Density optimization complete." << std::endl;
    std::cout << "New center of mass: " << getCenterOfMass().transpose() << std::endl;
    
    return true;
}

// Get the center of mass
Eigen::Vector3d DensityOptimizer::getCenterOfMass() const
{
    return computeCenterOfMass(mDensities);
}

// Export density map
bool DensityOptimizer::exportDensityMap(const std::string& filename) const
{
    // This is a simplified implementation - would need format-specific code
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write header
    outFile << "# Density map for biased die\n";
    outFile << "# Format: tet_idx density\n";
    
    // Write densities
    for (int i = 0; i < mDensities.size(); ++i) {
        outFile << i << " " << mDensities(i) << "\n";
    }
    
    outFile.close();
    return true;
}

// Export 3D printable model
bool DensityOptimizer::exportPrintableModel(const std::string& filename) const
{
    // This would require integration with a 3D printing slicer or custom format
    // For now, just export the tetrahedral mesh with densities
    std::cout << "Note: Full 3D printing export functionality not implemented." << std::endl;
    std::cout << "Exporting density map instead." << std::endl;
    
    return exportDensityMap(filename + ".densities");
}