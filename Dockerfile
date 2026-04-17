# Dockerfile — public, self-contained build for picoDiffusion.
# No local mirrors or pre-built base images needed.  Just Docker.
#
# Multi-stage build:
#   Stage 1 (builder) — installs PyTorch and all Python dependencies.
#   Stage 2 (runtime) — copies only the installed packages into a clean
#                        slim image, keeping the final size down.

# ── stage 1: builder ─────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install PyTorch with CUDA 12.6 support from the official PyTorch index.
# This is a large download (~2GB) but only happens on the first build.
# Docker caches this layer, so rebuilds are fast.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# Install the remaining Python dependencies.
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── stage 2: runtime ─────────────────────────────────────────────────────
# Start fresh from a clean slim image.  Only the installed Python packages
# are copied across, keeping the final image much smaller than if we built
# everything in a single stage.
FROM python:3.12-slim

# Install minimal runtime dependencies:
#   dumb-init — a simple init system that handles signals correctly (so
#               "docker stop" works cleanly instead of waiting for a timeout)
#   wget     — used by the Docker healthcheck to probe the /health endpoint
RUN apt-get update && \
    apt-get install -y --no-install-recommends dumb-init wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user.  The container runs as this user by default.
# The docker-compose file can override the UID/GID to match the host user.
RUN groupadd -g 1000 appgroup && useradd -u 1000 -g appgroup -m appuser

# Tell HuggingFace to store its download cache in a known location.
# This directory is bind-mounted from the host so downloads survive
# container restarts and rebuilds.
ENV HF_HOME=/app/cache/huggingface

# Copy only the installed Python packages and scripts from the builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create the directories the app expects to find.  When running with
# docker-compose these are overlaid by bind mounts, but they need to
# exist for the app to start even without mounts.
RUN mkdir -p /app/cache/models/checkpoints /app/cache/models/vae /app/cache/models/loras /app/cache/huggingface

WORKDIR /app

# Copy the application code (FastAPI server, pipeline, templates, static files).
COPY app/ .

# The FastAPI server listens on this port.
EXPOSE 8004

# Use dumb-init as PID 1 so signals (like SIGTERM from "docker stop")
# are forwarded to the Python process correctly.
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Start the FastAPI server with Uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
