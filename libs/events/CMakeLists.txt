add_library(ddrl_events)
add_library(ddrl::events ALIAS ddrl_events)

target_sources(ddrl_events
    PRIVATE
        src/event_bus.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/events/event_bus.hpp
            include/events/base_event.hpp
            include/events/sim_models.hpp
)

target_include_directories(ddrl_events
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(ddrl_events PUBLIC cxx_std_23)

# Link dependencies
target_link_libraries(ddrl_events
    PUBLIC
        ddrl::core
        ddrl::logging
        moodycamel_readerwriterqueue
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
