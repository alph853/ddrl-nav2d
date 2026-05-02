cmake_minimum_required(VERSION 3.25)
project(DDRL_Collision_Avoidance
    VERSION 1.0.0
    DESCRIPTION "Distributed Deep Reinforcement Learning for 2.5D Robot Navigation"
    LANGUAGES C CXX
)

# ───────────────────────────
# Build & Compiler Settings
# ───────────────────────────

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install")

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(DDRL_ENABLE_LTO "Enable Link Time Optimization" OFF)
option(DDRL_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(DDRL_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(DDRL_ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(DDRL_BUILD_TESTS "Build unit tests" OFF)
option(DDRL_BUILD_DOCS "Build documentation" OFF)
option(DDRL_BUILD_EXAMPLES "Build example projects" OFF)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

add_compile_definitions(
    $<$<CONFIG:Release>:DDRL_ENABLE_LOGGING=0>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wcast-align -Wcast-qual -Wconversion
        -Wdouble-promotion -Wformat=2 -Wnon-virtual-dtor
        -Wold-style-cast -Woverloaded-virtual -Wshadow
        -Wsign-conversion -Wunused
        -fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=
        -fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=
    )

    if(DDRL_ENABLE_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()

# Sanitizers
if(DDRL_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

if(DDRL_ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

if(DDRL_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT lto_supported OUTPUT error)
    if(lto_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "LTO not supported: ${error}")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ───────────────────────────
# Find Dependencies
# ───────────────────────────

find_package(PkgConfig REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/ddrl/config.hpp
    @ONLY
)

# ───────────────────────────
# Testing setup
# ───────────────────────────

if(DDRL_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9fd7188146816e6da900d21.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# ───────────────────────────
# Submodules
# ───────────────────────────

# Third-party header-only libraries
add_subdirectory(third_party)

add_subdirectory(libs/core)
add_subdirectory(libs/logging)
add_subdirectory(libs/config)
add_subdirectory(libs/threading)
add_subdirectory(libs/collision)
add_subdirectory(libs/kinematics)
add_subdirectory(libs/control)
add_subdirectory(libs/sim)
add_subdirectory(libs/events)
add_subdirectory(libs/visualizer)

add_subdirectory(apps/worker)

# ───────────────────────────
# Examples
# ───────────────────────────

if(DDRL_BUILD_EXAMPLES)
    # add_subdirectory(examples/cpp_sim_python_app)
    # add_subdirectory(examples/vtk_viz)
    # add_subdirectory(examples/visualizer)
endif()

# ───────────────────────────
# Install Rules
# ───────────────────────────

install(TARGETS
    ddrl_core ddrl_logging ddrl_config ddrl_threading
    ddrl_collision ddrl_kinematics ddrl_control
    ddrl_sim ddrl_events ddrl_visualizer
    ddrl_worker
    EXPORT ddrlTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
    FILE_SET HEADERS DESTINATION include
)

install(EXPORT ddrlTargets
    FILE ddrlTargets.cmake
    NAMESPACE ddrl::
    DESTINATION lib/cmake/ddrl
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ddrlConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ddrlConfig.cmake
    INSTALL_DESTINATION lib/cmake/ddrl
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ddrlConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ddrlConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ddrlConfigVersion.cmake
    DESTINATION lib/cmake/ddrl
)

install(DIRECTORY config/
    DESTINATION share/ddrl/config
    FILES_MATCHING PATTERN "*.yaml"
)

# ───────────────────────────
# Finalize
# ───────────────────────────

message(STATUS "DDRL Collision Avoidance Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  LTO: ${DDRL_ENABLE_LTO}")
message(STATUS "  ASAN: ${DDRL_ENABLE_ASAN}")
message(STATUS "  UBSAN: ${DDRL_ENABLE_UBSAN}")
message(STATUS "  Tests: ${DDRL_BUILD_TESTS}")
