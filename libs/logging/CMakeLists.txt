add_library(ddrl_logging)
add_library(ddrl::logging ALIAS ddrl_logging)

find_package(spdlog REQUIRED)

target_sources(ddrl_logging
    PRIVATE
        src/logging.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/logging/logging.hpp
)

target_include_directories(ddrl_logging
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(ddrl_logging
    PUBLIC
        spdlog::spdlog
)

# Install handled by main CMakeLists.txt
