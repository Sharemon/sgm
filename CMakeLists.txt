cmake_minimum_required(VERSION 3.0.0)
project(sgm VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

include_directories(src)
file(GLOB_RECURSE cpu_srcs src/sgm_cpu/*.cpp)
file(GLOB_RECURSE gpu_srcs src/sgm_gpu/*.cu)
file(GLOB_RECURSE common_srcs src/common/*.cpp)

add_executable(sgm_cpu ${cpu_srcs} ${common_srcs} example/sgm_cpu_main.cpp)
add_executable(sgm_gpu ${gpu_srcs} ${common_srcs} example/sgm_gpu_main.cpp)

