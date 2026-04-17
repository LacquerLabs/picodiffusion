# Makefile for picoDiffusion
#
# Targets:
#   make up    — build and start the container (detects GPU automatically)
#   make down  — stop the container
#   make build — build the container without starting it
#   make logs  — tail the container logs (Ctrl+C to stop)
#
# GPU detection:
#   If nvidia-smi is found, the GPU override compose file is layered in
#   automatically.  No manual flags needed.

.PHONY: build up down logs scan

# Detect NVIDIA GPU and layer the GPU override if available.
HAS_NVIDIA := $(shell nvidia-smi > /dev/null 2>&1 && echo yes || echo no)
COMPOSE := docker compose -f docker-compose.yml
ifeq ($(HAS_NVIDIA),yes)
COMPOSE += -f docker-compose.gpu.yml
endif

# Build the container image.
build:
	$(COMPOSE) build

# Build and start the container in detached mode.
up: build
	$(COMPOSE) up -d

# Stop and remove the container.
down:
	$(COMPOSE) down

# Tail container logs.  Press Ctrl+C to stop watching.
logs:
	$(COMPOSE) logs -f

# Scan source and dependencies for HIGH/CRITICAL CVEs with available fixes.
scan:
	docker run --rm \
		-v $(PWD):/workspace \
		-v /DATA/persistent/trivy:/root/.cache/trivy \
		aquasec/trivy:0.69.3 \
		fs \
		--severity HIGH,CRITICAL \
		--ignore-unfixed \
		--exit-code 1 \
		--no-progress \
		/workspace
