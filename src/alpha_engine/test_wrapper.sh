#!/bin/bash
# Find and execute the binary inside the sandboxed runfiles tree.
BIN_PATH=$(find . -name "alpha_engine_test" -type f -executable | head -n 1)
if [ -z "$BIN_PATH" ]; then
    echo "ERROR: alpha_engine_test not found."
    exit 1
fi
exec "$BIN_PATH"