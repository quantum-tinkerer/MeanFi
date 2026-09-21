#!/usr/bin/env bash
# Build the companion FermiSimplex checkout into the selected Python environment.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${FERMISIMPLEX_SOURCE_DIR:-$project_root/build/deps/fermisimplex}"
adaptive_source="${ADAPTIVESIMPLEX_SOURCE_DIR:-$project_root/build/deps/adaptivesimplex}"
python_bin="${PYTHON:-$project_root/.pixi/envs/latest/bin/python}"
if [[ ! -e "$source_dir" ]]; then
    git clone https://gitlab.kwant-project.org/qt/lineartetrahedron.git "$source_dir"
    git -C "$source_dir" switch -c codex/density-p-cubature 7921df11518f45505bd09ff32497f50d75ba4bce
    git -C "$source_dir" apply --unidiff-zero "$project_root/performance/reports/density_p_cubature/fermisimplex.patch"
fi
if [[ ! -f "$source_dir/cpp/src/integration/density_p.cpp" ]]; then
    echo "Set FERMISIMPLEX_SOURCE_DIR to the companion density-p-cubature checkout" >&2
    exit 1
fi
build_args="${CMAKE_ARGS:-}"
if [[ -f "$adaptive_source/CMakeLists.txt" ]]; then
    build_args="$build_args -DFETCHCONTENT_SOURCE_DIR_ADAPTIVESIMPLEX=$adaptive_source"
fi
# The selected environment supplies CMake, Ninja, nanobind, LAPACK and a compiler.
python_prefix="$(dirname "$(dirname "$python_bin")")"
export PATH="$python_prefix/bin:$PATH"
export CMAKE_PREFIX_PATH="$python_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CMAKE_ARGS="$build_args"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
"$python_bin" -m pip install --no-deps --no-build-isolation --editable "$source_dir"
