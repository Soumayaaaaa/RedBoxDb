add_executable(RedBoxDbTests
    test_main.cpp
    test_hnsw.cpp
)

target_link_libraries(RedBoxDbTests
    PRIVATE
        gtest
        gtest_main
        RedBoxDbLib
        spdlog::spdlog
)

set_project_warnings(RedBoxDbTests)


# Register tests with CTest (Standard CMake testing tool)
include(GoogleTest)
gtest_discover_tests(RedBoxDbTests)