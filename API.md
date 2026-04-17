# picoDiffusion API

The REST API is available at `http://localhost:8004`.  There is no authentication.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Current model state |
| `GET` | `/models` | Lists available checkpoints, VAEs, and LoRAs |
| `GET` | `/progress` | Current inference step progress |
| `POST` | `/generate` | Generate an image |

## GET /health

Returns the current state of the model.

```json
{
  "status": "ready",
  "model_loaded": true,
  "loading": false,
  "checkpoint": "dreamshaper_8.safetensors",
  "vae": "vae-ft-mse-840000-ema-pruned.safetensors",
  "device": "cuda"
}
```

Status values:
- `idle` — no model loaded
- `loading` — model is being loaded
- `ready` — model is loaded and ready to generate

## GET /models

Lists available files on disk.

```json
{
  "checkpoints": ["dreamshaper_8.safetensors"],
  "vaes": ["vae-ft-mse-840000-ema-pruned.safetensors"],
  "loras": ["50sPanavisionMovieSD1.safetensors"]
}
```

## GET /progress

Returns the current inference step.  Returns `0 / 0` when idle.

```json
{
  "step": 12,
  "total": 24
}
```

## POST /generate

Generate an image.  Returns base64-encoded PNG.

### Request

```json
{
  "checkpoint": "dreamshaper_8.safetensors",
  "vae": "vae-ft-mse-840000-ema-pruned.safetensors",
  "lora": "50sPanavisionMovieSD1.safetensors",
  "lora_weight": 0.8,
  "prompt": "welsh pembroke corgi, sitting on wooden floor",
  "negative_prompt": "bad anatomy, low quality, blurry",
  "steps": 24,
  "cfg": 7.0,
  "sampler": "DPM++ 2M SDE Karras",
  "seed": 12345,
  "width": 512,
  "height": 512
}
```

| Field | Required | Default | Notes |
|---|---|---|---|
| `checkpoint` | Yes | — | Filename from `/models` |
| `vae` | No | `null` | Omit or `null` for checkpoint's built-in VAE |
| `lora` | No | `null` | Omit or `null` for no LoRA |
| `lora_weight` | No | `0.7` | 0.0 to 2.0 |
| `prompt` | No | corgi prompt | See `config.py` |
| `negative_prompt` | No | quality filter | See `config.py` |
| `steps` | No | `24` | 1 to 60 |
| `cfg` | No | `10.0` | 1.0 to 20.0 |
| `sampler` | No | `DPM++ 2M SDE Karras` | See below |
| `seed` | No | random | 0 to 4294967295 |
| `width` | No | `512` | 256 to 1024, steps of 32 |
| `height` | No | `512` | 256 to 1024, steps of 32 |

### Available samplers

- `DPM++ 2M SDE Karras`
- `DPM++ 2M Karras`
- `Euler a`
- `Euler`
- `DDIM`
- `UniPC`

### Response — success

```json
{
  "image": "<base64 encoded PNG>",
  "seed": 12345,
  "prompt": "welsh pembroke corgi, sitting on wooden floor",
  "negative_prompt": "bad anatomy, low quality, blurry",
  "steps": 24,
  "cfg": 7.0,
  "checkpoint": "dreamshaper_8.safetensors",
  "vae": "vae-ft-mse-840000-ema-pruned.safetensors",
  "lora": "50sPanavisionMovieSD1.safetensors",
  "lora_weight": 0.8,
  "sampler": "DPM++ 2M SDE Karras",
  "elapsed_seconds": 22.5
}
```

### Response — error

```json
{
  "error": "checkpoint not found: missing.safetensors"
}
```

Errors return HTTP 200 with an `error` field, not HTTP 4xx/5xx.
