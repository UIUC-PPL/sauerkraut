set(SOURCES
    serdes.C
    include/serdes.h
)

add_library(serdes SHARED ${SOURCES})

set_target_properties(serdes PROPERTIES
    OUTPUT_NAME "serdes"
    PREFIX ""
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

add_dependencies(serdes generate_flatbuffers)
target_include_directories(serdes PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/sauerkraut/buffer  # For generated flatbuffers headers
)
