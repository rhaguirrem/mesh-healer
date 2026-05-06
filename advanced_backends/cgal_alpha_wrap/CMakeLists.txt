cmake_minimum_required(VERSION 3.22)

project(mesh_heal_cgal_alpha_wrap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL REQUIRED)

add_executable(mesh_heal_cgal_alpha_wrap
    src/main.cpp
)

target_link_libraries(mesh_heal_cgal_alpha_wrap PRIVATE CGAL::CGAL)
