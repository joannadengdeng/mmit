"""mmit.training — pluggable training framework for multimodal instruction tuning.

Quick start
-----------
>>> from mmit.training import TrainingMethod

Register a custom training method::

    from mmit import registry
    from mmit.training import TrainingMethod

    class MyMethod(TrainingMethod):
        name = "my-method"
        display_name = "My Method"
        def default_config(self): ...
        def ui_params(self): ...
        def prepare_model(self, model, processor, config): ...
        def compute_loss(self, model, batch, outputs): ...
        def get_trainable_params(self, model): ...
        def save_checkpoint(self, model, processor, path, metadata): ...
        def load_for_inference(self, path, base_model_id, **kwargs): ...

    registry.register("training_method", "my-method", MyMethod)
"""
from mmit.training.methods.base import TrainingMethod
from mmit.training.methods import (
    QLoRAMethod, LoRAMethod, DoRAMethod, FullFTMethod,
    FreezeTuningMethod, L2TMethod, MoReSMethod,
)

__all__ = [
    "TrainingMethod",
    "QLoRAMethod",
    "LoRAMethod",
    "DoRAMethod",
    "FullFTMethod",
    "FreezeTuningMethod",
    "L2TMethod",
    "MoReSMethod",
]
