add_library(ddrl_sim)
add_library(ddrl::sim ALIAS ddrl_sim)

target_sources(ddrl_sim
    PRIVATE
        src/simulator.cpp
        src/map_manager.cpp
        src/world.cpp
        src/object.cpp
        src/robot.cpp
        src/spawn_manager.cpp
        src/sensors/depth_camera.cpp
        src/sensors/factory.cpp
        src/sensors/lidar_2d.cpp
        src/sensors/lidar_3d.cpp
        src/sensors/sensor_base.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/sim/simulator.hpp
            include/sim/map_manager.hpp
            include/sim/world.hpp
            include/sim/object.hpp
            include/sim/robot.hpp
            include/sim/spawn_manager.hpp
            include/sim/sensors.hpp
)

target_include_directories(ddrl_sim
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(ddrl_sim PUBLIC cxx_std_23)

# Link dependencies
target_link_libraries(ddrl_sim
    PUBLIC
        ddrl::core
        ddrl::threading
        ddrl::collision
        ddrl::control
        ddrl::kinematics
        ddrl::config
        ddrl::events
)
