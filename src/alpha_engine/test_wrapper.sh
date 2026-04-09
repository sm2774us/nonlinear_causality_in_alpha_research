#!/bin/bash
# Find the test binary within the runfiles directory
TEST_BIN=$(find . -name "alpha_engine_test" -type f -executable | head -n 1)

if [ -z "$TEST_BIN" ]; then
    echo "Error: alpha_engine_test binary not found in runfiles."
    exit 1
fi

echo "Executing: $TEST_BIN"
exec "$TEST_BIN"