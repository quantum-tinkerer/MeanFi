cmake_minimum_required(VERSION 3.18...3.30)
project(meanfi LANGUAGES CXX)

if(NOT SKBUILD)
  message(
    WARNING
      "This CMake file is intended to be driven by scikit-build-core. "
      "Use 'pip install -ve .' or the existing pixi environment to build the extension."
  )
endif()

add_subdirectory(cpp)
