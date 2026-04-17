# picoDiffusion

A minimal sd1.5 image generator.  Built for learning, not production.

## What it is

A single-page web UI for generating images with Stable Diffusion 1.5.  Runs in Docker on Linux with an NVIDIA GPU.  Every control has built-in help that explains what it does.

## Quick start

```bash
git clone https://github.com/lacquerlabs/picodiffusion.git
cd picodiffusion
mkdir -p cache/models/checkpoints cache/models/vae cache/models/loras cache/huggingface
make up
```

Download an sd1.5 checkpoint from [civitai.com](https://civitai.com) and drop it in `cache/models/checkpoints/`.  Open `http://localhost:8004`.

## What you need

- A computer running Linux with Docker
- An NVIDIA GPU with at least 4GB VRAM (we built this on a 2017 Quadro P2000)
- A Stable Diffusion 1.5 checkpoint file

## Features

- Checkpoints, VAEs, and LoRAs with adjustable weight
- 6 samplers (DPM++ 2M SDE Karras, DPM++ 2M Karras, Euler a, Euler, DDIM, UniPC)
- Width/height sliders from 256 to 1024
- Seed lock for reproducible results
- PNG metadata embedded in every generated image
- Built-in help system for every control
- Save/download generated images

## API

There is a REST API.  See [API.md](API.md) for endpoints and examples.

## Running without Docker

There is a [native install guide](https://picodiffusion.com/native-install.html) for running directly with Python.  It is not the recommended path.

## Documentation

Full docs at [picodiffusion.com](https://picodiffusion.com):

- [Getting Started](https://picodiffusion.com/getting-started.html)
- [Guide](https://picodiffusion.com/guide.html)
- [FAQ](https://picodiffusion.com/faq.html)
- [CUDA / NVIDIA Setup](https://picodiffusion.com/cuda-nvidia-setup.html)
- [Native Install](https://picodiffusion.com/native-install.html)

## Local use only

This tool has no authentication, no access controls, and an open API.  Do not expose it to the internet.

## Licence

[CC BY-NC-SA 4.0](https://picodiffusion.com/licence.html).  Free to use, share, and modify for non-commercial purposes with attribution.

## lacquerlabs

lacquerlabs makes games, tools, toys, websites, and bad jokes.

🐾
