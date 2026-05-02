add_library(ddrl_config)
add_library(ddrl::config ALIAS ddrl_config)

target_sources(ddrl_config
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/config/simulator.hpp
            include/config/world.hpp
            include/config/visualizer.hpp
            include/config/dynamic_profile.hpp
            include/config/robot_profile.hpp
            include/config/model_profile.hpp
            include/config/map.hpp
            include/config/parsing.hpp
            include/config/rl.hpp
    PRIVATE
        src/parsing.cpp
)

target_include_directories(ddrl_config
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(ddrl_config PUBLIC cxx_std_23)

find_package(yaml-cpp REQUIRED)

target_link_libraries(ddrl_config
    PUBLIC
        ddrl::core
        ddrl::logging
    PRIVATE
        yaml-cpp::yaml-cpp
)

# (Install handled at top level)
