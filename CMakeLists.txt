cmake_minimum_required(VERSION 3.20)

project(Raytracer LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(raylib GIT_REPOSITORY https://github.com/raysan5/raylib.git GIT_TAG 5.5)
FetchContent_MakeAvailable(raylib)

add_executable(Raytracer src/main.c)

target_link_libraries(Raytracer raylib)

