# Dockerfile — AlphaPod v2 (nonlinear, regime-aware)
# Copyright 2026 AlphaPod Contributors. All Rights Reserved.
# Multi-stage: builder installs toolchain; runtime image is minimal.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    clang-19 libc++-19-dev libc++abi-19-dev \
    python3.13 python3.13-dev python3.13-venv \
    bazel-7.0.0 \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Build C++26 extension (v2 regime-aware SIMD kernel)
RUN bazel build //src/alpha_engine:alpha_engine_cpp.so --config=linux
RUN cp bazel-bin/src/alpha_engine/alpha_engine_cpp.so src/research/

# Install Python 3.13 dependencies (v2 adds scipy for KSG TE)
RUN python3.13 -m venv /opt/venv
RUN /opt/venv/bin/pip install --quiet --upgrade pip
RUN /opt/venv/bin/pip install --quiet polars numpy scipy pytest pytest-cov

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3.13 libc++1-19 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app/src/research

# Default: run v2 nonlinear pipeline
ENTRYPOINT ["python3.13", "alpha_pipeline.py"]
