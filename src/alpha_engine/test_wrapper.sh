#!/bin/bash
# Move to the execution directory
cd "$(dirname "$0")"

# Find the binary in the sandboxed runfiles
TARGET_BIN=$(find . -name "alpha_engine_test" -type f -executable | head -n 1)

if [ -z "$TARGET_BIN" ]; then
    echo "CRITICAL ERROR: alpha_engine_test not found."
    exit 1
fi

echo "Running Alpha Engine Test: $TARGET_BIN"
exec "$TARGET_BIN"