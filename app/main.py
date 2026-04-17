"""FastAPI service for Stable Diffusion 1.5 image generation.

Endpoints:
    GET  /         — browser UI
    GET  /health   — readiness and current model state
    GET  /models   — lists available checkpoints and VAEs on disk
    GET  /progress — current inference step progress
    POST /generate — load model if needed, then generate an image

The model is not loaded at startup. It is loaded on the first /generate
request (or when the checkpoint/VAE selection changes).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import config
import pipeline
from log import get_logger, setup_uvicorn_logging

setup_uvicorn_logging()
log = get_logger("picodiffusion.server")


def _log(level: str, message: str, **fields: Any) -> None:
    """Emit a structured JSON log line."""
    record = log.makeRecord(
        log.name,
        getattr(logging, level),
        "",
        0,
        message,
        (),
        None,
    )
    record.fields = fields  # type: ignore[attr-defined]
    log.handle(record)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="picodiffusion",
    description="Stable Diffusion 1.5 image generation.",
    version="1.0.0",
    debug=os.getenv("DEBUG", "").lower() in ("1", "true", "yes"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

_log("INFO", "picodiffusion starting", device=pipeline.get_device())

# Load and generate are mutually exclusive — only one runs at a time.
_generate_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    checkpoint: str
    vae: str | None = None
    lora: str | None = None
    lora_weight: float = Field(default=config.DEFAULT_LORA_WEIGHT, ge=0.0, le=2.0)
    prompt: str = config.DEFAULT_PROMPT
    negative_prompt: str = config.DEFAULT_NEGATIVE_PROMPT
    steps: int = Field(default=config.DEFAULT_STEPS, ge=1, le=60)
    cfg: float = Field(default=config.DEFAULT_CFG, ge=1.0, le=20.0)
    sampler: str = config.DEFAULT_SAMPLER
    seed: int | None = None
    width: int = Field(default=config.OUTPUT_WIDTH, ge=config.OUTPUT_MIN, le=config.OUTPUT_MAX)
    height: int = Field(default=config.OUTPUT_HEIGHT, ge=config.OUTPUT_MIN, le=config.OUTPUT_MAX)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the browser UI."""
    _log("DEBUG", "serving index page")
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "defaults": {
                "prompt": config.DEFAULT_PROMPT,
                "negative_prompt": config.DEFAULT_NEGATIVE_PROMPT,
                "steps": config.DEFAULT_STEPS,
                "cfg": config.DEFAULT_CFG,
                "sampler": config.DEFAULT_SAMPLER,
                "samplers": config.SAMPLER_NAMES,
                "lora_weight": config.DEFAULT_LORA_WEIGHT,
                "width": config.OUTPUT_WIDTH,
                "height": config.OUTPUT_HEIGHT,
                "output_min": config.OUTPUT_MIN,
                "output_max": config.OUTPUT_MAX,
                "output_step": config.OUTPUT_STEP,
            },
        },
    )


@app.get("/health")
async def health() -> dict:
    """Return current model state.

    Status values:
        idle    — no model loaded, ready to accept a generate request
        loading — a model is currently being loaded
        ready   — model is loaded and ready for inference
    """
    loaded = pipeline.is_loaded()
    loading = pipeline.is_loading()

    if loading:
        status = "loading"
    elif loaded:
        status = "ready"
    else:
        status = "idle"

    body = {
        "status": status,
        "model_loaded": loaded,
        "loading": loading,
        "checkpoint": pipeline.get_loaded_checkpoint(),
        "vae": pipeline.get_loaded_vae(),
        "device": pipeline.get_device(),
    }
    _log("DEBUG", "health check", status=status)
    return body


@app.get("/models")
async def models() -> dict:
    """Return lists of available checkpoint and VAE files on disk."""
    return {
        "checkpoints": pipeline.list_checkpoints(),
        "vaes": pipeline.list_vaes(),
        "loras": pipeline.list_loras(),
    }


@app.get("/progress")
async def progress() -> dict:
    """Return the current inference step and total steps."""
    step, total = pipeline.get_progress()
    return {"step": step, "total": total}


@app.post("/generate")
async def generate(body: GenerateRequest) -> dict:
    """Load the requested model if needed, then generate an image."""
    available_checkpoints = pipeline.list_checkpoints()
    if body.checkpoint not in available_checkpoints:
        return {"error": f"checkpoint not found: {body.checkpoint}"}

    vae = body.vae or None
    if vae is not None and vae not in pipeline.list_vaes():
        return {"error": f"vae not found: {vae}"}

    lora = body.lora or None
    if lora is not None and lora not in pipeline.list_loras():
        return {"error": f"lora not found: {lora}"}

    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

    _log(
        "INFO", "generate request received",
        checkpoint=body.checkpoint, vae=vae or "none",
        lora=lora or "none", lora_weight=body.lora_weight,
        seed=seed, steps=body.steps, cfg=body.cfg,
        sampler=body.sampler, width=body.width, height=body.height,
    )

    def _run_generation() -> dict:
        with _generate_lock:
            # Load (or reload) if the requested model differs from what is loaded.
            if not pipeline.is_loaded_with(body.checkpoint, vae):
                _log("INFO", "loading model", checkpoint=body.checkpoint, vae=vae or "none")
                try:
                    pipeline.load(body.checkpoint, vae)
                except Exception:
                    log.exception("model load failed")
                    return {"error": "model failed to load — check logs"}

            start = time.monotonic()
            try:
                image = pipeline.generate(
                    prompt=body.prompt,
                    negative_prompt=body.negative_prompt,
                    steps=body.steps,
                    cfg=body.cfg,
                    seed=seed,
                    sampler=body.sampler,
                    width=body.width,
                    height=body.height,
                    lora_file=lora,
                    lora_weight=body.lora_weight,
                )
            except Exception:
                log.exception("inference failed")
                return {"error": "inference failed — check logs"}

            elapsed = round(time.monotonic() - start, 2)

        from PIL.PngImagePlugin import PngInfo
        png_meta = PngInfo()
        png_meta.add_text("prompt", body.prompt)
        png_meta.add_text("negative_prompt", body.negative_prompt)
        png_meta.add_text("seed", str(seed))
        png_meta.add_text("steps", str(body.steps))
        png_meta.add_text("cfg", str(body.cfg))
        png_meta.add_text("sampler", body.sampler)
        png_meta.add_text("checkpoint", body.checkpoint)
        png_meta.add_text("vae", vae or "")
        png_meta.add_text("lora", lora or "")
        png_meta.add_text("lora_weight", str(body.lora_weight))
        png_meta.add_text("width", str(body.width))
        png_meta.add_text("height", str(body.height))
        png_meta.add_text("generator", "picodiffusion")

        buf = io.BytesIO()
        image.save(buf, format="PNG", pnginfo=png_meta)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        _log(
            "INFO", "generate request complete",
            seed=seed, steps=body.steps, cfg=body.cfg,
            sampler=body.sampler, elapsed_seconds=elapsed,
        )

        return {
            "image": encoded,
            "seed": seed,
            "prompt": body.prompt,
            "negative_prompt": body.negative_prompt,
            "steps": body.steps,
            "cfg": body.cfg,
            "checkpoint": body.checkpoint,
            "vae": vae,
            "lora": lora,
            "lora_weight": body.lora_weight,
            "sampler": body.sampler,
            "elapsed_seconds": elapsed,
        }

    # Run the blocking torch inference in a thread so the event loop stays free
    # for health/progress polling during generation.
    return await asyncio.to_thread(_run_generation)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
