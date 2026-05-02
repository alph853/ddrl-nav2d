add_library(ddrl_control)
add_library(ddrl::control ALIAS ddrl_control)

target_sources(ddrl_control
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/control/pid_controller.hpp
            include/control/pure_pursuit.hpp
    PRIVATE
        src/pid_controller.cpp
        src/pure_pursuit.cpp
)

target_include_directories(ddrl_control
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(ddrl_control PUBLIC cxx_std_23)

target_link_libraries(ddrl_control
    PRIVATE
        yaml-cpp
        ddrl::core
        ddrl::logging
        ddrl::config
)

# (Install handled at top level)
