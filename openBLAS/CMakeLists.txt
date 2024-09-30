cmake_minimum_required(VERSION 3.5)
project(gemm_project)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

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
        gemm_project
#		test_cblas_dgemm.cpp
		test_slow_gemm.cpp
		slow_dgemm.cpp
		common.h
		common.cpp
)
target_link_libraries(
        gemm_project
#        openblas
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(
        gemm_project
)
