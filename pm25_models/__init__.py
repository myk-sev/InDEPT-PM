from .cyclical_transformers import (
    CyclicalDualEncoderPatchTransformer,
    CyclicalDualEncoderTransformer,
    CyclicalPatchTransformerConfig,
    CyclicalTimestepTransformerConfig,
)
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
from .patchtst import (
    ChannelIndependentPatchEmbedding,
    PatchTST,
    PatchTSTConfig,
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

ModelConfig = (
    PatchTransformerConfig
    | TimestepTransformerConfig
    | CyclicalPatchTransformerConfig
    | CyclicalTimestepTransformerConfig
    | PatchTSTConfig
)

register_model(
    DEFAULT_MODEL, DualEncoderPatchTransformer, PatchTransformerConfig
)
register_model(
    "transformer-no-patches",
    DualEncoderTransformer,
    TimestepTransformerConfig,
)
register_model(
    "transformer-cyclical",
    CyclicalDualEncoderPatchTransformer,
    CyclicalPatchTransformerConfig,
)
register_model(
    "transformer-no-patches-cyclical",
    CyclicalDualEncoderTransformer,
    CyclicalTimestepTransformerConfig,
)
register_model("patchtst", PatchTST, PatchTSTConfig)

__all__ = [
    "DEFAULT_MODEL",
    "MODEL_SPECS",
    "ChannelIndependentPatchEmbedding",
    "CyclicalDualEncoderPatchTransformer",
    "CyclicalDualEncoderTransformer",
    "CyclicalPatchTransformerConfig",
    "CyclicalTimestepTransformerConfig",
    "DualEncoderPatchTransformer",
    "DualEncoderTransformer",
    "ModelConfig",
    "ModelSpec",
    "PatchTransformerConfig",
    "PatchEmbedding",
    "PatchTST",
    "PatchTSTConfig",
    "TimestepEmbedding",
    "TimestepTransformerConfig",
    "_missing_aware_history",
    "build_config",
    "build_model",
    "model_names",
    "register_model",
]
