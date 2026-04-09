#!/bin/bash
# Find the binary in the sandboxed runfiles and execute it.
BIN_PATH=$(find . -name "alpha_engine_test" -type f -executable | head -n 1)
exec "$BIN_PATH"