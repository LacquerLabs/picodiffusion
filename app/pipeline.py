"""Model loading and inference for Stable Diffusion 1.5.

The pipeline is loaded on first generate request, not at startup.
All inference runs on the same loaded pipeline instance until a different
checkpoint or VAE is requested, at which point the old pipeline is released
and the new one loaded before inference runs.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
import time
from collections.abc import Generator

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

import config
import pl_stub  # noqa: F401 — register pytorch_lightning stub for old .pt checkpoints
from log import get_logger

log = get_logger("picodiffusion.pipeline")

_pipeline: StableDiffusionPipeline | None = None
_model_loaded: bool = False
_model_loading: bool = False
_loaded_checkpoint: str | None = None
_loaded_vae: str | None = None
_lock = threading.Lock()

# Inference progress — written by the callback, read by the /progress endpoint.
_progress_step: int = 0
_progress_total: int = 0


def _select_device() -> str:
    """Return the best available torch device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    """Return the preferred dtype for the given device.

    CUDA handles float16 natively and efficiently.
    MPS and CPU use float32 — MPS float16 support in diffusers has edge cases.
    """
    return torch.float16 if device == "cuda" else torch.float32


def _make_generator(device: str, seed: int) -> torch.Generator:
    """Return a seeded Generator appropriate for the device.

    MPS has known issues with torch.Generator(device='mps') inside diffusers.
    Using a CPU generator works correctly even when the pipeline is on MPS.
    """
    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


_device: str = _select_device()


def _log(level: str, message: str, **fields: object) -> None:
    """Emit a structured log line with optional extra fields."""
    record = log.makeRecord(
        log.name, getattr(logging, level), "", 0, message, (), None,
    )
    record.fields = fields  # type: ignore[attr-defined]
    log.handle(record)


@contextlib.contextmanager
def _allow_unsafe_load() -> Generator[None, None, None]:
    """Temporarily patch torch.load to use weights_only=False.

    Some older .pt checkpoints (like kl-f8-anime2.vae.pt) contain pickled
    pytorch_lightning globals.  PyTorch >= 2.6 rejects these by default.
    Rather than pulling in the entire pytorch-lightning package just to
    allowlist the class, we patch torch.load for the duration of the load
    call.  This is scoped and restored immediately afterwards.

    Only use this for model files you trust.
    """
    original = torch.load

    @functools.wraps(original)
    def _patched(*args: object, **kwargs: object) -> object:
        kwargs["weights_only"] = False  # type: ignore[arg-type]
        return original(*args, **kwargs)

    torch.load = _patched  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


def _apply_sampler(pipe: StableDiffusionPipeline, sampler_name: str) -> None:
    """Configure the pipeline's scheduler to match the given sampler name."""
    base_config = pipe.scheduler.config

    if sampler_name == "DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
    elif sampler_name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(base_config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(base_config)
    elif sampler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(base_config)
    elif sampler_name == "UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(base_config)
    else:
        _log("WARNING", "unknown sampler, falling back to DPM++ 2M SDE Karras",
             sampler=sampler_name)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )

    _log("DEBUG", "sampler configured", sampler=sampler_name)


# ---------------------------------------------------------------------------
# Public state accessors
# ---------------------------------------------------------------------------

def get_progress() -> tuple[int, int]:
    """Return (current_step, total_steps) for the active inference."""
    return _progress_step, _progress_total


def is_loaded() -> bool:
    """Return True if a model is fully loaded and ready."""
    return _model_loaded


def is_loading() -> bool:
    """Return True if a model is currently being loaded."""
    return _model_loading


def get_device() -> str:
    """Return the device string the pipeline is running on."""
    return _device


def get_loaded_checkpoint() -> str | None:
    """Return the filename of the currently loaded checkpoint, or None."""
    return _loaded_checkpoint


def get_loaded_vae() -> str | None:
    """Return the filename of the currently loaded VAE, or None."""
    return _loaded_vae


def is_loaded_with(checkpoint_file: str, vae_file: str | None) -> bool:
    """Return True if the pipeline is already loaded with these exact params."""
    return _model_loaded and _loaded_checkpoint == checkpoint_file and _loaded_vae == vae_file


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

_CHECKPOINT_EXTS = {".safetensors", ".ckpt", ".pt"}
_VAE_EXTS = {".safetensors", ".pt", ".bin"}
_LORA_EXTS = {".safetensors", ".pt", ".bin"}


def list_checkpoints() -> list[str]:
    """Return sorted list of checkpoint filenames available in CHECKPOINT_DIR."""
    if not config.CHECKPOINT_DIR.exists():
        return []
    return sorted(
        f.name for f in config.CHECKPOINT_DIR.iterdir()
        if f.is_file() and f.suffix in _CHECKPOINT_EXTS
    )


def list_vaes() -> list[str]:
    """Return sorted list of VAE filenames available in VAE_DIR."""
    if not config.VAE_DIR.exists():
        return []
    return sorted(
        f.name for f in config.VAE_DIR.iterdir()
        if f.is_file() and f.suffix in _VAE_EXTS
    )


def list_loras() -> list[str]:
    """Return sorted list of LoRA filenames available in LORA_DIR."""
    if not config.LORA_DIR.exists():
        return []
    return sorted(
        f.name for f in config.LORA_DIR.iterdir()
        if f.is_file() and f.suffix in _LORA_EXTS
    )


# ---------------------------------------------------------------------------
# Load / unload
# ---------------------------------------------------------------------------

def _unload() -> None:
    """Release the current pipeline and free device memory."""
    global _pipeline, _model_loaded, _loaded_checkpoint, _loaded_vae
    if _pipeline is not None:
        _log("INFO", "unloading current pipeline")
        del _pipeline
        _pipeline = None
    if _device == "cuda":
        torch.cuda.empty_cache()
    elif _device == "mps":
        torch.mps.empty_cache()
    _model_loaded = False
    _loaded_checkpoint = None
    _loaded_vae = None


def load(checkpoint_file: str, vae_file: str | None = None) -> None:
    """Load a checkpoint (and optionally a VAE) into memory.

    If the requested checkpoint and VAE are already loaded, this is a no-op.
    If a different model is loaded, it is released before loading the new one.

    Args:
        checkpoint_file: Filename inside CHECKPOINT_DIR.
        vae_file: Filename inside VAE_DIR, or None to use the VAE embedded
                  in the checkpoint.
    """
    global _pipeline, _model_loaded, _model_loading, _loaded_checkpoint, _loaded_vae

    if is_loaded_with(checkpoint_file, vae_file):
        _log("DEBUG", "requested model already loaded, skipping", checkpoint=checkpoint_file)
        return

    _model_loading = True
    _model_loaded = False
    _unload()

    checkpoint_path = str(config.CHECKPOINT_DIR / checkpoint_file)
    dtype = _dtype_for_device(_device)

    # -- VAE ----------------------------------------------------------------
    vae: AutoencoderKL | None = None
    if vae_file is not None:
        vae_path = str(config.VAE_DIR / vae_file)
        _log("INFO", "loading vae", path=vae_path)
        vae_start = time.monotonic()
        with _allow_unsafe_load():
            vae = AutoencoderKL.from_single_file(
                vae_path,
                config="stabilityai/sd-vae-ft-mse",
                torch_dtype=dtype,
            )
        _log("INFO", "vae loaded", elapsed_seconds=round(time.monotonic() - vae_start, 2))
    else:
        _log("INFO", "no vae specified, using checkpoint embedded vae")

    # -- Checkpoint ---------------------------------------------------------
    _log("INFO", "loading checkpoint", path=checkpoint_path)
    ckpt_start = time.monotonic()

    pipe_kwargs: dict[str, object] = dict(
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if vae is not None:
        pipe_kwargs["vae"] = vae

    with _allow_unsafe_load():
        pipe = StableDiffusionPipeline.from_single_file(checkpoint_path, **pipe_kwargs)

    _log("INFO", "checkpoint loaded", elapsed_seconds=round(time.monotonic() - ckpt_start, 2))

    # -- Device placement ---------------------------------------------------
    _log("INFO", "moving pipeline to device", device=_device)
    device_start = time.monotonic()
    pipe = pipe.to(_device)
    _log("INFO", "pipeline on device", device=_device, elapsed_seconds=round(time.monotonic() - device_start, 2))

    # -- Memory optimisations -----------------------------------------------
    # Attention slicing processes attention heads one at a time instead of all
    # at once. Mandatory on CPU (reduces peak RAM by ~40%) and helpful on MPS.
    # On CUDA it trades a small speed cost for lower VRAM usage.
    pipe.enable_attention_slicing()

    with _lock:
        _pipeline = pipe
        _loaded_checkpoint = checkpoint_file
        _loaded_vae = vae_file
        _model_loaded = True
        _model_loading = False

    _log("INFO", "model ready", checkpoint=checkpoint_file, vae=vae_file or "none")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int,
    sampler: str = config.DEFAULT_SAMPLER,
    width: int = config.OUTPUT_WIDTH,
    height: int = config.OUTPUT_HEIGHT,
    lora_file: str | None = None,
    lora_weight: float = config.DEFAULT_LORA_WEIGHT,
) -> Image.Image:
    """Run one inference pass and return a PIL Image.

    Args:
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of denoising steps.
        cfg: Classifier-free guidance scale.
        seed: RNG seed for reproducibility.
        sampler: Sampler/scheduler name (see config.SAMPLER_NAMES).
        width: Output image width in pixels.
        height: Output image height in pixels.
        lora_file: LoRA filename inside LORA_DIR, or None for no LoRA.
        lora_weight: LoRA influence strength, 0.0 to 1.0.

    Returns:
        A PIL Image of the requested size.
    """
    global _progress_step, _progress_total

    assert _pipeline is not None, "Pipeline not loaded — call load() first"

    _apply_sampler(_pipeline, sampler)

    # -- LoRA ---------------------------------------------------------------
    lora_applied = False
    if lora_file is not None:
        lora_path = str(config.LORA_DIR / lora_file)
        _log("INFO", "loading lora", path=lora_path, weight=lora_weight)
        _pipeline.load_lora_weights(
            config.LORA_DIR,
            weight_name=lora_file,
        )
        _pipeline.fuse_lora(lora_scale=lora_weight)
        lora_applied = True
        _log("INFO", "lora fused", lora=lora_file, weight=lora_weight)

    _progress_step = 0
    _progress_total = steps

    _log(
        "DEBUG", "inference starting",
        seed=seed, steps=steps, cfg=cfg, sampler=sampler,
        width=width, height=height,
        lora=lora_file or "none", lora_weight=lora_weight,
    )

    def _on_step(pipe: object, step: int, timestep: object, kwargs: dict[str, object]) -> dict[str, object]:
        global _progress_step
        _progress_step = step + 1
        return kwargs

    generator = _make_generator(_device, seed)
    start = time.monotonic()

    try:
        result = _pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            width=width,
            height=height,
            clip_skip=2,
            callback_on_step_end=_on_step,
        )
    finally:
        # Always clean up LoRA so the next generate starts from the base model.
        if lora_applied:
            _pipeline.unfuse_lora()
            _pipeline.unload_lora_weights()
            _log("DEBUG", "lora unloaded", lora=lora_file)

    _progress_step = 0
    _progress_total = 0

    elapsed = round(time.monotonic() - start, 2)
    _log("INFO", "inference complete", seed=seed, steps=steps, elapsed_seconds=elapsed)

    return result.images[0]
