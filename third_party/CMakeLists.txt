# Header-only interface target for moodycamel ReaderWriterQueue
add_library(moodycamel_readerwriterqueue INTERFACE)
add_library(moodycamel::readerwriterqueue ALIAS moodycamel_readerwriterqueue)

target_include_directories(moodycamel_readerwriterqueue INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/moodycamel/readerwriterqueue>
    $<INSTALL_INTERFACE:include/third_party/moodycamel/readerwriterqueue>
)

# Export this target in the same export set as the rest of the project
install(TARGETS moodycamel_readerwriterqueue
    EXPORT ddrlTargets
)

# Install headers (and keep benchmarks/tests out)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/moodycamel/readerwriterqueue/
    DESTINATION include/third_party/moodycamel/readerwriterqueue
    FILES_MATCHING
        PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "benchmarks" EXCLUDE
    PATTERN "tests" EXCLUDE
    PATTERN "msvc*" EXCLUDE
)

# License compliance
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/moodycamel/readerwriterqueue/LICENSE.md)
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/moodycamel/readerwriterqueue/LICENSE.md
        DESTINATION share/licenses/moodycamel)
endif()
