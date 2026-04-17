"""Paths, defaults, and constants for the SD1.5 API."""

import os
from pathlib import Path

# Server
HOST: str = os.environ.get("PICO_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("PICO_PORT", "8004"))

# Output
OUTPUT_WIDTH: int = 512
OUTPUT_HEIGHT: int = 512
OUTPUT_MIN: int = 256
OUTPUT_MAX: int = 1024
OUTPUT_STEP: int = 32

# Model locations.  These are relative paths for native running.
# In Docker, bind mounts overlay these with the host directories.
CHECKPOINT_DIR: Path = Path("cache/models/checkpoints")
VAE_DIR: Path = Path("cache/models/vae")
LORA_DIR: Path = Path("cache/models/loras")

# Inference defaults
DEFAULT_PROMPT: str = (
    "welsh pembroke corgi, sitting on wooden floor, "
    "small laptop, warm lighting, soft shadows, cute, stylized"
)
DEFAULT_NEGATIVE_PROMPT: str = (
    "bad anatomy, low quality, blurry, watermark, deformed, extra limbs, "
    "disfigured, poorly drawn face, poorly drawn hands, duplicate, signature"
)
DEFAULT_STEPS: int = 24
DEFAULT_CFG: float = 10.0
DEFAULT_SAMPLER: str = "DPM++ 2M SDE Karras"
DEFAULT_LORA_WEIGHT: float = 0.7

SAMPLER_NAMES: list[str] = [
    "DPM++ 2M SDE Karras",
    "DPM++ 2M Karras",
    "Euler a",
    "Euler",
    "DDIM",
    "UniPC",
]
