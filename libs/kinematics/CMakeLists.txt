add_library(ddrl_kinematics)
add_library(ddrl::kinematics ALIAS ddrl_kinematics)

target_sources(ddrl_kinematics
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/kinematics/ackermann.hpp
    PRIVATE
        src/ackermann.cpp
)

target_include_directories(ddrl_kinematics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(ddrl_kinematics
    PUBLIC
        ddrl::core
        ddrl::config
)

