set(SOURCES
    greenlet_compat.C
    include/greenlet_compat.h
)

add_library(greenlet_compat SHARED ${SOURCES})

set_target_properties(greenlet_compat PROPERTIES
    OUTPUT_NAME "greenlet_compat"
    PREFIX ""
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

target_include_directories(greenlet_compat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/sauerkraut/include
)

if(APPLE)
    target_link_options(greenlet_compat PRIVATE -undefined dynamic_lookup)
endif()
