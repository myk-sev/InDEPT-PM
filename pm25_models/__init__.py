from .dual_encoder_patch_transformer import (
    DualEncoderPatchTransformer,
    PatchTransformerConfig,
    PatchEmbedding,
    _missing_aware_history,
)
from .dual_encoder_transformer import (
    DualEncoderTransformer,
    TimestepEmbedding,
    TimestepTransformerConfig,
)
from .registry import (
    DEFAULT_MODEL,
    MODEL_SPECS,
    ModelSpec,
    build_config,
    build_model,
    model_names,
    register_model,
)

ModelConfig = PatchTransformerConfig | TimestepTransformerConfig

register_model(
    DEFAULT_MODEL, DualEncoderPatchTransformer, PatchTransformerConfig
)
register_model(
    "transformer-no-patches",
    DualEncoderTransformer,
    TimestepTransformerConfig,
)

__all__ = [
    "DEFAULT_MODEL",
    "MODEL_SPECS",
    "DualEncoderPatchTransformer",
    "DualEncoderTransformer",
    "ModelConfig",
    "ModelSpec",
    "PatchTransformerConfig",
    "PatchEmbedding",
    "TimestepEmbedding",
    "TimestepTransformerConfig",
    "_missing_aware_history",
    "build_config",
    "build_model",
    "model_names",
    "register_model",
]
