# SD1.5 Image Generation API — Specification

## Overview

A Dockerised image generation service running Stable Diffusion 1.5 inference
using HuggingFace Diffusers. Provides a browser UI and REST API. Accepts a
prompt, returns a base64-encoded PNG. The model is not loaded at startup — it
is loaded lazily on the first `/generate` request, and reloaded if the
checkpoint or VAE selection changes.

Designed for the Counterfeit-V3.0 checkpoint on a Quadro P2000 (4GB VRAM,
sm_61), but works on any CUDA GPU, Apple Silicon (MPS), or CPU.

---

## Runtime

| Item | Value |
|---|---|
| Python | 3.12 |
| Port | 8004 |
| Framework | FastAPI + Uvicorn |
| Inference | HuggingFace Diffusers |
| PyTorch | cu126 (Pascal/sm_61 compatible) |
| Precision | fp16 (CUDA), fp32 (MPS/CPU) |
| Output size | 512×512 or 256×256 PNG |

---

## Project Structure

```
picogen/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
├── Makefile
├── src_picogen/         # Bind-mounted model files (not committed)
│   ├── checkpoints/
│   ├── vae/
│   ├── output/
│   └── hf-cache/
└── app/
    ├── main.py          # FastAPI app and endpoints
    ├── pipeline.py      # Model loading, device detection, inference
    ├── config.py        # Paths and inference defaults
    ├── log.py           # Structured JSON logging
    ├── pl_stub.py       # pytorch_lightning stub for old checkpoints
    ├── requirements.txt
    └── templates/
        └── index.html   # Browser UI
```

---

## Docker

### Dockerfile

- Multi-stage build from `python:local` base image
- Builder stage installs PyTorch from a local mirror and pip requirements
- Runtime stage copies installed packages from the builder
- Model files are mounted as a volume — not baked into the image
- Uses `dumb-init` as PID 1
- Runs Uvicorn on port `8004`

### docker-compose.yml

- Networks: `network_proxy` (external) and `echolok_network` (external)
- Model volume: `./src_picogen/checkpoints` → `/models/checkpoints` (read-only)
- VAE volume: `./src_picogen/vae` → `/models/vae` (read-only)
- Output volume: `./src_picogen/output` → `/output`
- HuggingFace cache: `./src_picogen/hf-cache` → `/root/.cache/huggingface`
- Port: `8004:8004`
- Restart: `unless-stopped`
- Healthcheck: `wget -qO- http://127.0.0.1:8004/health` every 30s

### docker-compose.gpu.yml

- NVIDIA GPU passthrough via `nvidia` driver reservation
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- The Makefile auto-detects NVIDIA and layers this override automatically

---

## Endpoints

### `GET /`

Serves the browser UI for interactive image generation.

### `GET /health`

Returns current model state.

Status values:
- `idle` — no model loaded, ready to accept a generate request
- `loading` — a model is currently being loaded
- `ready` — model is loaded and ready for inference

```json
{
  "status": "ready",
  "model_loaded": true,
  "loading": false,
  "checkpoint": "counterfeitV30_v30.safetensors",
  "vae": "kl-f8-anime2.vae.pt",
  "device": "cuda"
}
```

### `GET /models`

Lists available checkpoint and VAE files on disk.

```json
{
  "checkpoints": ["counterfeitV30_v30.safetensors"],
  "vaes": ["kl-f8-anime2.vae.pt"]
}
```

### `GET /progress`

Returns the current inference step and total steps. Returns `0 / 0` when idle.

```json
{
  "step": 12,
  "total": 25
}
```

### `POST /generate`

Loads the requested model if needed, then generates an image.

#### Request

Content-Type: `application/json`

```json
{
  "checkpoint": "counterfeitV30_v30.safetensors",
  "vae": "kl-f8-anime2.vae.pt",
  "prompt": "string (optional)",
  "negative_prompt": "string (optional)",
  "steps": 25,
  "cfg": 10.0,
  "sampler": "DPM++ 2M SDE Karras",
  "seed": 12345,
  "size": "large"
}
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `checkpoint` | Yes | — | Filename from `/models` |
| `vae` | No | `null` | Omit or `null` to use the VAE embedded in the checkpoint |
| `prompt` | No | corgi prompt | See `config.py` |
| `negative_prompt` | No | quality filter | See `config.py` |
| `steps` | No | `25` | Range: 1–60 |
| `cfg` | No | `10.0` | Range: 1.0–20.0 |
| `sampler` | No | `"DPM++ 2M SDE Karras"` | See `config.SAMPLER_NAMES` |
| `seed` | No | random | 0–4294967295 |
| `size` | No | `"large"` | `"small"` (256×256) or `"large"` (512×512) |

#### Response — success `200`

```json
{
  "image": "<base64 encoded PNG string>",
  "seed": 12345,
  "prompt": "prompt that was used",
  "negative_prompt": "negative prompt that was used",
  "steps": 25,
  "cfg": 10.0,
  "checkpoint": "counterfeitV30_v30.safetensors",
  "vae": "kl-f8-anime2.vae.pt",
  "sampler": "DPM++ 2M SDE Karras",
  "elapsed_seconds": 4.2
}
```

#### Response — error `200` (with error field)

```json
{
  "error": "description of what went wrong"
}
```

Note: errors are returned with HTTP 200 and an `error` field in the JSON body,
not with HTTP 4xx/5xx status codes.

---

## Model Configuration

| Item | Value |
|---|---|
| Checkpoint | Selected at generate time from UI |
| VAE | Selected at generate time from UI (optional) |
| Clip skip | 2 |
| Scheduler (default) | DPM++ 2M SDE Karras (`DPMSolverMultistepScheduler`) |
| Available samplers | DPM++ 2M SDE Karras, DPM++ 2M Karras, Euler a, UniPC |
| Precision | fp16 (CUDA), fp32 (MPS/CPU) |
| Memory | Attention slicing enabled on all devices |

The model is loaded lazily on the first `/generate` request and kept in memory.
If a different checkpoint or VAE is requested, the current model is unloaded
and the new one loaded before inference runs.

Old `.pt` VAE files (like `kl-f8-anime2.vae.pt`) that contain pickled
`pytorch_lightning` metadata are handled by `pl_stub.py`, which registers a
fake `pytorch_lightning` module so pickle can resolve the reference without
installing the full package.

---

## Logging

All log output goes to stdout as newline-delimited JSON.

Every log line includes at minimum:

```json
{
  "timestamp": "2026-04-12T21:00:00.000Z",
  "level": "INFO",
  "logger": "picogen.server",
  "message": "human readable message"
}
```

Request log line (emitted on every POST):

```json
{
  "timestamp": "2026-04-12T21:00:00.000Z",
  "level": "INFO",
  "logger": "picogen.server",
  "message": "generate request received",
  "checkpoint": "counterfeitV30_v30.safetensors",
  "vae": "kl-f8-anime2.vae.pt",
  "seed": 12345,
  "steps": 25,
  "cfg": 10.0,
  "sampler": "DPM++ 2M SDE Karras",
  "size": "large"
}
```

Completion log line:

```json
{
  "timestamp": "2026-04-12T21:00:00.000Z",
  "level": "INFO",
  "logger": "picogen.server",
  "message": "generate request complete",
  "seed": 12345,
  "steps": 25,
  "cfg": 10.0,
  "sampler": "DPM++ 2M SDE Karras",
  "elapsed_seconds": 4.2
}
```

Error log line:

```json
{
  "timestamp": "2026-04-12T21:00:00.000Z",
  "level": "ERROR",
  "logger": "picogen.pipeline",
  "message": "inference failed",
  "traceback": "..."
}
```

Model loaded log line:

```json
{
  "timestamp": "2026-04-12T21:00:00.000Z",
  "level": "INFO",
  "logger": "picogen.pipeline",
  "message": "model ready",
  "checkpoint": "counterfeitV30_v30.safetensors",
  "vae": "kl-f8-anime2.vae.pt"
}
```

---

## Notes

- The model is not loaded at startup. It loads on the first `/generate` request.
- Requests made while a model is loading will block until the load completes (serialised by a threading lock).
- Generation time on a P2000 at 25 steps is approximately 30–60 seconds.
- The API is single-threaded for inference. Concurrent requests queue behind the generate lock.
- Inference runs in a background thread (`asyncio.to_thread`) so the FastAPI event loop stays free for `/health` and `/progress` polling during generation.
