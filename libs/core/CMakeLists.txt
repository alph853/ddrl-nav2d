add_library(ddrl_core)
add_library(ddrl::core ALIAS ddrl_core)

file(GLOB CORE_BASE_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/core/base/*.hpp"
)

target_sources(ddrl_core
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            ${CORE_BASE_HEADERS}

            include/core/math/primitives.hpp
            include/core/math/numeric.hpp
            include/core/math/pose.hpp
            include/core/math/twist.hpp
            include/core/math/quaternion.hpp

            include/core/geom/aabb.hpp
            include/core/geom/shape.hpp

            include/core/kinematics/ackermann.hpp

            include/core/transform/tf.hpp

            include/core/sim/world_state.hpp
            include/core/sim/collision.hpp

            include/core/rl/rl.hpp
            include/core/rl/observation_encoding.hpp
            include/core/rl/reward.hpp
    PRIVATE
        # Math implementations
        src/math/pose.cpp
        src/math/quaternion.cpp

        # Geometry implementations
        src/geom/shape.cpp

        # Transform implementations
        src/transform/tf.cpp

        # Sim models implementations
        src/sim/world_state.cpp

        # RL implementations
        src/rl/observation_encoding.cpp
        src/rl/reward.cpp
        src/rl/reward_deprecated.cpp
)

target_include_directories(ddrl_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Public link dependencies can be added here later if needed
# Keep third-party headers private unless exposed in public headers
target_link_libraries(ddrl_core
    PUBLIC

    PRIVATE
        ddrl::logging
        moodycamel::readerwriterqueue
)

# (Install handled at top level)
