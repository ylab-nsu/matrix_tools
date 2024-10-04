cmake_minimum_required(VERSION 3.5)
project(habr_article_project)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O3 -march=native")

include(FetchContent)
FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG        v1.15.2
		FIND_PACKAGE_ARGS NAMES GTest
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
        habr_article_project
		test_habr_gemm.cpp
		habr_dgemm.cpp
		common.h
		common.cpp
)
target_link_libraries(
		habr_article_project
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(
		habr_article_project
)
