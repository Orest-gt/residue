cmake_minimum_required(VERSION 3.15)

project(residue VERSION 2.1.0)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Create the residue_v2 module
pybind11_add_module(residue_v2
    src/residue/bindings.cpp
    src/residue/core.cpp
)

# Set compile options
target_compile_definitions(residue_v2 PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_features(residue_v2 PRIVATE cxx_std_17)

# Include directories
target_include_directories(residue_v2 PRIVATE
    src
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(residue_v2 PRIVATE
    ${Python_LIBRARIES}
)

# Set output directory
set_target_properties(residue_v2 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/residue_v2
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/residue_v2
)
