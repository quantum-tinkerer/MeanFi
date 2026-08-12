#!/usr/bin/env bash
set -euo pipefail

repo_url="${ADAPTIVESIMPLEX_REPOSITORY:-https://gitlab.kwant-project.org/qt/adaptivesimplex.git}"
revision="${ADAPTIVESIMPLEX_REVISION:-57df88574cf9726c97b01fc29688ed4d8896e827}"
project_root="${PIXI_PROJECT_ROOT:-$(pwd)}"
source_dir="${ADAPTIVESIMPLEX_SOURCE_DIR:-$project_root/.pixi/adaptivesimplex-src}"
build_dir="${ADAPTIVESIMPLEX_BUILD_DIR:-$project_root/.pixi/adaptivesimplex-build}"
prefix="${ADAPTIVESIMPLEX_PREFIX:-$project_root/.pixi/adaptivesimplex-prefix}"

if [[ -e "$source_dir" && ! -e "$source_dir/.git" ]]; then
    echo "$source_dir exists but is not a git checkout" >&2
    exit 1
fi

if [[ ! -e "$source_dir/.git" ]]; then
    git clone --filter=blob:none "$repo_url" "$source_dir"
fi

git -C "$source_dir" fetch --filter=blob:none origin "$revision"
git -C "$source_dir" checkout --detach "$revision"

cmake \
    -S "$source_dir" \
    -B "$build_dir" \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DADAPTIVESIMPLEX_BUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF
cmake --build "$build_dir"
cmake --install "$build_dir"
