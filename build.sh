#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
BUILD_TYPE="Release"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

usage() {
    cat <<EOF
Usage: ./build.sh [command]

Commands:
  configure   CMake configure (only needed once or after CMakeLists changes)
  build       Build all targets
  test        Run unit tests
  server      Build and run the server
  bench       Build and run benchmarks
  recall      Build and run recall test
  clean       Remove build directory
  rebuild     Clean + configure + build
  (no args)   Build all targets

Examples:
  ./build.sh            # build
  ./build.sh configure  # cmake configure
  ./build.sh test       # build + run tests
  ./build.sh server     # build + run server
  ./build.sh bench      # build + run benchmarks
  ./build.sh recall     # build + run recall test
  ./build.sh rebuild    # full clean rebuild
EOF
}

do_configure() {
    echo ">> Configuring (${BUILD_TYPE})..."
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -G Ninja 2>/dev/null \
        || cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
}

do_build() {
    if [ ! -f "$BUILD_DIR/build.ninja" ] && [ ! -f "$BUILD_DIR/Makefile" ]; then
        do_configure
    fi
    echo ">> Building (${JOBS} jobs)..."
    cmake --build "$BUILD_DIR" -j "$JOBS"
}

do_test() {
    do_build
    echo ""
    echo ">> Running tests..."
    "$BUILD_DIR/tests/RedBoxDbTests" "$@"
}

do_server() {
    do_build
    echo ""
    echo ">> Starting server..."
    "$BUILD_DIR/src/RedBoxServer"
}

do_bench() {
    do_build
    echo ""
    echo ">> Running benchmarks..."
    "$BUILD_DIR/benchmark/InsertMicro"
}

do_recall() {
    do_build
    echo ""
    echo ">> Running recall test..."
    python3 benchmark/recall_test.py "$@"
}

do_clean() {
    echo ">> Removing ${BUILD_DIR}/..."
    rm -rf "$BUILD_DIR"
}

# --- main ---
case "${1:-build}" in
    configure) do_configure ;;
    build)     do_build ;;
    test)      do_test "${@:2}" ;;
    server)    do_server ;;
    bench)     do_bench ;;
    recall)    do_recall "${@:2}" ;;
    clean)     do_clean ;;
    rebuild)   do_clean && do_configure && do_build ;;
    -h|--help) usage ;;
    *)         echo "Unknown command: $1"; usage; exit 1 ;;
esac
