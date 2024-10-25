cmake_minimum_required(VERSION 3.5)
project(habr_article_project)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O3 -march=native ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O3 -march=native ${OpenMP_CXX_FLAGS}")

if (WIN32)
	set(INC_OPENBLAS_DIR C:\\1\\OpenBlas\\include)
	set(LINK_OPENBLAS_DIR C:\\1\\OpenBlas\\lib)

	include_directories(${INC_OPENBLAS_DIR})
	link_directories(${LINK_OPENBLAS_DIR})
endif (WIN32)

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

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(
        habr_article_project
		test/test_habr_gemm.cpp
		src/habr_dgemm.cpp
		src/cpu_info.cpp
		include/cpu_info.h
		include/common.h
		src/common.cpp
)

target_link_libraries(
		habr_article_project
        GTest::gtest_main
		openblas.a
		OpenMP::OpenMP_CXX
)

include(GoogleTest)
gtest_discover_tests(
		habr_article_project
)
