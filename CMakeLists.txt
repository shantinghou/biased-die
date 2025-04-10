cmake_minimum_required(VERSION 3.1)
project(BiasedDie)

# Set dependencies directory - adjust this path to match your setup
set(DEPENDENCIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../P0-setup/cpp_setup/dependencies/")
message("DEPENDENCIES_DIR: ${DEPENDENCIES_DIR}")

########################################
######## Configure DEPENDENCIES ########
########################################

# libigl settings
option(LIBIGL_USE_STATIC_LIBRARY    "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR      "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO           "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK             "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE           "Use Embree"         OFF)
option(LIBIGL_WITH_LIM              "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB           "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK            "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"         ON)
option(LIBIGL_WITH_PNG              "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         ON)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML              "Use XML"            OFF)

# Include libigl
set(LIBIGL_DIR "${DEPENDENCIES_DIR}/libigl/")
set(LIBIGL_INCLUDE_DIR "${LIBIGL_DIR}/include/")
list(APPEND CMAKE_MODULE_PATH "${LIBIGL_INCLUDE_DIR}/../cmake")
include(libigl)

###################################
######## Configure PROJECT ########
###################################

# set c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wuninitialized -pedantic-errors -Wno-unused-parameter)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wc++20-extensions)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

###############################
######## Build PROJECT ########
###############################

# Set source root path
set(SRC_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/")
include_directories(${SRC_ROOT_PATH})

# Add source files
set(SOURCES
    ${SRC_ROOT_PATH}/main.cpp
    ${SRC_ROOT_PATH}/density_optimizer.cpp
)

# Create output directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/assets/models)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/assets/output)

# Add executable
add_executable(biased_die ${SOURCES})

# Link libraries
target_link_libraries(biased_die igl::core igl::opengl igl::opengl_glfw igl::opengl_glfw_imgui igl::tetgen)

# Windows-specific settings
if(WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
    message("  use -D_USE_MATH_DEFINES on windows")
endif()

# Copy assets to build directory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/assets)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()