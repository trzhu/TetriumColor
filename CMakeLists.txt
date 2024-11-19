cmake_minimum_required(VERSION 3.22)

# Define the project name and version
project(TetriumColor VERSION 1.0 LANGUAGES CXX)

# python
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python3.13)

if (UNIX)
    if(APPLE)
        set(PYTHON_LIBRARY $ENV{CONDA_PREFIX}/lib/libpython3.13.dylib)
    else()
        set(PYTHON_LIBRARY $ENV{CONDA_PREFIX}/lib/libpython3.13.so)
    endif()
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source files
set(SOURCES
    cpp/src/PseudoIsochromaticPlateGenerator.cpp
)

# Create the library
add_library(TetriumColor STATIC ${SOURCES})

# Add the include directory
target_include_directories(TetriumColor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Find Python libraries
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

target_include_directories(TetriumColor PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(TetriumColor ${Python3_LIBRARIES})

target_precompile_headers(${PROJECT_NAME} PRIVATE cpp/PCH.h)
