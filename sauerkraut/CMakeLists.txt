# Source files
set(SOURCES
    sauerkraut.C
)

# Create the Python module
# On macOS, Python extensions must be bundles (MODULE), not shared libraries (SHARED)
if(APPLE)
    add_library(sauerkraut MODULE ${SOURCES})
else()
    add_library(sauerkraut SHARED ${SOURCES})
endif()
# Set the output properties to match Python module naming conventions
set_target_properties(sauerkraut PROPERTIES
    OUTPUT_NAME "_sauerkraut"
    PREFIX ""
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

# Set CMAKE_PREFIX_PATH to include Flatbuffers install directory
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/flatbuffers-install")

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "Flatbuffers include directory: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Flatbuffers include path: ${FlatBuffersIncludePath}")
# Include directories for the target
target_include_directories(sauerkraut PUBLIC
    ${FLATBUFFERS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated headers
    ${CMAKE_CURRENT_BINARY_DIR}/buffer  # For flatbuffers generated headers
)

add_subdirectory(buffer)
add_subdirectory(serdes)
add_subdirectory(greenlet_compat)
add_dependencies(sauerkraut generate_flatbuffers serdes greenlet_compat)

# Link necessary libraries
if(APPLE)
    # On macOS, don't link directly to Python library - use dynamic_lookup instead
    # This ensures symbols are resolved from the running Python interpreter
    target_link_libraries(sauerkraut
        PRIVATE
        serdes
        greenlet_compat
    )
    target_link_options(sauerkraut PRIVATE -undefined dynamic_lookup)
else()
    target_link_libraries(sauerkraut
        PRIVATE
        Python::Python
        serdes
        greenlet_compat
        pthread
        m
        util
        dl
    )
endif()

# shared library should go to the build directory
set_target_properties(sauerkraut PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
