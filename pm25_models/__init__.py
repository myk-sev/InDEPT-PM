from .cyclical_transformers import (
    CyclicalDualEncoderPatchTransformer,
    CyclicalDualEncoderTransformer,
    CyclicalPatchTransformerConfig,
    CyclicalTimestepTransformerConfig,
)
from .delta_models import (
    AbsoluteDeltaCyclicalDualEncoderPatchTransformer,
    AbsoluteDeltaCyclicalDualEncoderTransformer,
    AbsoluteDeltaCyclicalPatchTST,
    AbsoluteDeltaPatchTST,
    DeltaCyclicalDualEncoderPatchTransformer,
    DeltaCyclicalDualEncoderTransformer,
    DeltaCyclicalPatchTST,
    DeltaPatchTST,
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
    CyclicalPatchTST,
    CyclicalPatchTSTConfig,
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
from .separate_stream_fusion import (
    SeparateStreamCrossFusionTransformer,
    SeparateStreamSelfFusionTransformer,
)

ModelConfig = (
    PatchTransformerConfig
    | TimestepTransformerConfig
    | CyclicalPatchTransformerConfig
    | CyclicalTimestepTransformerConfig
    | CyclicalPatchTSTConfig
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
    "separate-stream-cross-fusion",
    SeparateStreamCrossFusionTransformer,
    TimestepTransformerConfig,
)
register_model(
    "separate-stream-self-fusion",
    SeparateStreamSelfFusionTransformer,
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
register_model("patchtst-cyclical", CyclicalPatchTST, CyclicalPatchTSTConfig)
register_model(
    "transformer-cyclical-delta",
    DeltaCyclicalDualEncoderPatchTransformer,
    CyclicalPatchTransformerConfig,
)
register_model(
    "transformer-no-patches-cyclical-delta",
    DeltaCyclicalDualEncoderTransformer,
    CyclicalTimestepTransformerConfig,
)
register_model("patchtst-delta", DeltaPatchTST, PatchTSTConfig)
register_model(
    "patchtst-cyclical-delta",
    DeltaCyclicalPatchTST,
    CyclicalPatchTSTConfig,
)
register_model(
    "transformer-cyclical-absolute-delta",
    AbsoluteDeltaCyclicalDualEncoderPatchTransformer,
    CyclicalPatchTransformerConfig,
)
register_model(
    "transformer-no-patches-cyclical-absolute-delta",
    AbsoluteDeltaCyclicalDualEncoderTransformer,
    CyclicalTimestepTransformerConfig,
)
register_model("patchtst-absolute-delta", AbsoluteDeltaPatchTST, PatchTSTConfig)
register_model(
    "patchtst-cyclical-absolute-delta",
    AbsoluteDeltaCyclicalPatchTST,
    CyclicalPatchTSTConfig,
)

__all__ = [
    "AbsoluteDeltaCyclicalDualEncoderPatchTransformer",
    "AbsoluteDeltaCyclicalDualEncoderTransformer",
    "AbsoluteDeltaCyclicalPatchTST",
    "AbsoluteDeltaPatchTST",
    "DEFAULT_MODEL",
    "MODEL_SPECS",
    "ChannelIndependentPatchEmbedding",
    "CyclicalDualEncoderPatchTransformer",
    "CyclicalDualEncoderTransformer",
    "CyclicalPatchTST",
    "CyclicalPatchTSTConfig",
    "CyclicalPatchTransformerConfig",
    "CyclicalTimestepTransformerConfig",
    "DeltaCyclicalDualEncoderPatchTransformer",
    "DeltaCyclicalDualEncoderTransformer",
    "DeltaCyclicalPatchTST",
    "DeltaPatchTST",
    "DualEncoderPatchTransformer",
    "DualEncoderTransformer",
    "ModelConfig",
    "ModelSpec",
    "PatchTransformerConfig",
    "PatchEmbedding",
    "PatchTST",
    "PatchTSTConfig",
    "SeparateStreamCrossFusionTransformer",
    "SeparateStreamSelfFusionTransformer",
    "TimestepEmbedding",
    "TimestepTransformerConfig",
    "_missing_aware_history",
    "build_config",
    "build_model",
    "model_names",
    "register_model",
]
