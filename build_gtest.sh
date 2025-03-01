#!/bin/bash

# Default action
ACTION="build_and_run"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --build)
            ACTION="build"
            shift
            ;;
        --run)
            ACTION="run"
            shift
            ;;
        --clean)
            ACTION="clean"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --build      Only build the tests"
            echo "  --run        Only run the tests"
            echo "  --clean      Clean build artifacts"
            echo "  --help       Show this help message"
            echo "Default: Build and run tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set compiler flags
CXX=${CXX:-clang++}
CXXFLAGS="-std=c++17 -Wall -I. -g"

# Detect system and add appropriate flags
if [[ "$(uname -s)" == "Darwin" ]]; then
    # macOS needs an extra flag for <filesystem>
    CXXFLAGS="$CXXFLAGS -stdlib=libc++"
    # Recent macOS doesn't need -lc++fs
    LDFLAGS=""
    # On macOS, gtest might be in Homebrew or system paths
    if [ -d "/opt/homebrew/include/gtest" ]; then
        GTEST_DIR="/opt/homebrew/include"
        GTEST_LIB="/opt/homebrew/lib"
    elif [ -d "/usr/local/include/gtest" ]; then
        GTEST_DIR="/usr/local/include"
        GTEST_LIB="/usr/local/lib"
    else
        echo "Google Test not found in expected locations. Please install with Homebrew: brew install googletest"
        exit 1
    fi
else
    # Linux might need -lstdc++fs for older gcc versions
    LDFLAGS="-lstdc++fs"
    # On Linux, gtest is usually in /usr/include
    GTEST_DIR="/usr/include"
    GTEST_LIB="/usr/lib"
fi

# Add Google Test include paths and libraries
CXXFLAGS="$CXXFLAGS -I$GTEST_DIR"
LDFLAGS="$LDFLAGS -L$GTEST_LIB -lgtest -lgtest_main -lgmock -pthread"

# Define test executables
TESTS=(
    "MLWorkloadAnalyticsTest:tests/monitor_collector/MLWorkloadAnalyticsTest.cc:src/monitor_collector/service/MLWorkloadAnalytics.cc"
)

# Function to build a test
build_test() {
    local test_info=$1
    IFS=':' read -r test_name test_file test_src <<< "$test_info"
    
    echo "Using Google Test from: $GTEST_DIR"
    echo "Building $test_name..."
    $CXX $CXXFLAGS -o $test_name $test_file $test_src $LDFLAGS
    
    if [ $? -eq 0 ]; then
        echo "Build successful!"
        return 0
    else
        echo "Build failed!"
        return 1
    fi
}

# Function to run a test
run_test() {
    local test_name=$1
    
    if [ -x "./$test_name" ]; then
        echo "Running $test_name..."
        ./$test_name
        return $?
    else
        echo "Test executable $test_name not found or not executable."
        return 1
    fi
}

# Function to clean up
clean() {
    echo "Cleaning up..."
    for test_info in "${TESTS[@]}"; do
        IFS=':' read -r test_name test_file test_src <<< "$test_info"
        rm -f $test_name
    done
    echo "Clean complete."
}

# Main execution logic
case $ACTION in
    "build")
        for test_info in "${TESTS[@]}"; do
            build_test "$test_info" || exit 1
        done
        ;;
    "run")
        for test_info in "${TESTS[@]}"; do
            IFS=':' read -r test_name test_file test_src <<< "$test_info"
            run_test "$test_name" || exit 1
        done
        ;;
    "clean")
        clean
        ;;
    "build_and_run")
        for test_info in "${TESTS[@]}"; do
            if build_test "$test_info"; then
                IFS=':' read -r test_name test_file test_src <<< "$test_info"
                run_test "$test_name" || exit 1
            else
                exit 1
            fi
        done
        ;;
esac

exit 0 