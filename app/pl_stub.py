"""Minimal stub so torch.load can unpickle old .pt checkpoints.

Some older VAE .pt files (like kl-f8-anime2.vae.pt) were saved with
pytorch_lightning ModelCheckpoint metadata baked in.  When unpickling
with weights_only=False, Python needs the module to exist even though
we never use the object.  This stub registers a fake module so pickle
can resolve the reference without installing the full pytorch-lightning
package (~500 MB of dependencies we do not need).

This module self-registers on import — just `import pl_stub` once
before loading such checkpoints.
"""

import sys
import types


class _ModelCheckpoint:
    """No-op stand-in for pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


# Build the fake module tree:
#   pytorch_lightning
#   pytorch_lightning.callbacks
#   pytorch_lightning.callbacks.model_checkpoint
_pl = types.ModuleType("pytorch_lightning")
_cb = types.ModuleType("pytorch_lightning.callbacks")
_mc = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

_mc.ModelCheckpoint = _ModelCheckpoint  # type: ignore[attr-defined]
_cb.model_checkpoint = _mc  # type: ignore[attr-defined]
_pl.callbacks = _cb  # type: ignore[attr-defined]

sys.modules["pytorch_lightning"] = _pl
sys.modules["pytorch_lightning.callbacks"] = _cb
sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = _mc
