add_library(ddrl_threading)
add_library(ddrl::threading ALIAS ddrl_threading)

target_sources(ddrl_threading
    PRIVATE
        src/thread_pool.cpp

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/threading/thread_pool.hpp
)

target_include_directories(ddrl_threading
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(ddrl_threading
    PUBLIC
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
