cmake_minimum_required(VERSION 3.18...3.30)
project(meanfi LANGUAGES CXX)

if(NOT SKBUILD)
  message(
    WARNING
      "This CMake file is intended to be driven by scikit-build-core. "
      "Use 'pip install -ve .' or the existing pixi environment to build the extension."
  )
endif()

option(
  MEANFI_NATIVE_PROFILE
  "Build the native zero-temp extension with profiling-friendly symbols and frame pointers."
  OFF
)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(
  _meanfi_zero_temp_sources
  src/zero_temp/native_types.cpp
  src/zero_temp/native_tight_binding.cpp
  src/zero_temp/native_spectral_cache.cpp
  src/zero_temp/native_geometry.cpp
  src/zero_temp/native_frontier.cpp
  src/zero_temp/native_simplex_cache.cpp
  src/zero_temp/native_charge.cpp
  src/zero_temp/native_density.cpp
  src/zero_temp/bindings.cpp
)

if(MEANFI_NATIVE_PROFILE)
  nanobind_add_module(
    _zero_temp_native
    NOSTRIP
    NOMINSIZE
    ${_meanfi_zero_temp_sources}
  )
  target_compile_options(
    _zero_temp_native PRIVATE -fno-omit-frame-pointer -gline-tables-only
  )
else()
  nanobind_add_module(
    _zero_temp_native
    ${_meanfi_zero_temp_sources}
  )
endif()

target_compile_features(_zero_temp_native PRIVATE cxx_std_17)
target_link_libraries(
  _zero_temp_native PRIVATE Eigen3::Eigen ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
)
set_target_properties(
  _zero_temp_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/meanfi"
)

install(TARGETS _zero_temp_native LIBRARY DESTINATION meanfi)
