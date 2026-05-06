option(
  MEANFI_ZERO_TEMP_PROFILE
  "Build the compiled zero-temp extension with profiling-friendly symbols and frame pointers."
  OFF
)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 3.4 CONFIG REQUIRED NO_MODULE)

set(
  _meanfi_zero_temp_sources
  zero_temp/src/types.cpp
  zero_temp/src/tight_binding.cpp
  zero_temp/src/vertex_cache.cpp
  zero_temp/src/geometry.cpp
  zero_temp/src/simplex_rules.cpp
  zero_temp/src/adaptive_integrator.cpp
  zero_temp/src/bindings.cpp
)

if(MEANFI_ZERO_TEMP_PROFILE)
  nanobind_add_module(
    _zero_temp_ext
    NOSTRIP
    NOMINSIZE
    ${_meanfi_zero_temp_sources}
  )
  target_compile_options(
    _zero_temp_ext PRIVATE -fno-omit-frame-pointer -gline-tables-only
  )
else()
  nanobind_add_module(
    _zero_temp_ext
    ${_meanfi_zero_temp_sources}
  )
endif()

target_compile_features(_zero_temp_ext PRIVATE cxx_std_17)
target_include_directories(
  _zero_temp_ext PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/zero_temp/include
)
target_link_libraries(_zero_temp_ext PRIVATE Eigen3::Eigen)

install(TARGETS _zero_temp_ext LIBRARY DESTINATION meanfi)
