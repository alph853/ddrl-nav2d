add_library(ddrl_collision)
add_library(ddrl::collision ALIAS ddrl_collision)

target_sources(ddrl_collision
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/collision/bvh.hpp
            include/collision/broad_phase.hpp
            include/collision/ray_casting.hpp
            include/collision/narrow_phase.hpp
    PRIVATE
        # Collision detection implementations
        src/bvh.cpp
        src/broad_phase.cpp
        src/ray_casting.cpp
        src/sat.cpp
)

target_include_directories(ddrl_collision
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Public link dependencies can be added here later if needed
target_link_libraries(ddrl_collision
    PUBLIC
        ddrl::core
        ddrl::threading
        ddrl::logging
        ddrl::config
)

# (Install handled at top level)
