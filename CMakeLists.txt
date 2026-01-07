cmake_minimum_required(VERSION 3.15...3.30)
project(sauerkraut VERSION 0.1
        DESCRIPTION "Sauerkraut: Serialization for Python's Control State" 
        LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type and compiler flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall")

# Find Python (ensure the version matches your Python version)
# Allow PYTHON_EXECUTABLE to be passed from setup.py
if(DEFINED PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()
find_package(Python3 3.13 COMPONENTS Interpreter Development REQUIRED)

# Set consistent output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Handle Flatbuffers
if(DEFINED FLATBUFFERS_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${FLATBUFFERS_ROOT}")
endif()

find_package(flatbuffers QUIET)

if(NOT flatbuffers_FOUND)
    message(STATUS "System Flatbuffers not found, will build from source")
    # Set the install path for the built Flatbuffers
    set(FLATBUFFERS_INSTALL_DIR "${CMAKE_BINARY_DIR}/flatbuffers-install")
    list(APPEND CMAKE_PREFIX_PATH "${FLATBUFFERS_INSTALL_DIR}")
endif()

# Get Flatbuffers include path
if(flatbuffers_FOUND)
    get_target_property(FLATBUFFERS_INCLUDE_DIR flatbuffers::flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
else()
    set(FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_INSTALL_DIR}/include")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sauerkraut/include)
include_directories(${FLATBUFFERS_INCLUDE_DIR})

add_subdirectory(sauerkraut)

