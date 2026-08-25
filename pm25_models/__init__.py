from .bridge_forecaster import (
    BridgeForecastConfig,
    BridgeForecaster,
    bridge_config_values,
    bridge_forecast_name,
    bridge_forecast_names,
    bridge_history_model_name,
    load_bridge_checkpoint,
    validate_bridge_checkpoint,
)
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
    BridgeForecastConfig
    |
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
for _bridge_name in bridge_forecast_names():
    _history_name = bridge_history_model_name(_bridge_name)
    register_model(
        _bridge_name,
        lambda config, name=_history_name: BridgeForecaster(name, config),
        BridgeForecastConfig,
    )
del _bridge_name, _history_name

__all__ = [
    "AbsoluteDeltaCyclicalDualEncoderPatchTransformer",
    "AbsoluteDeltaCyclicalDualEncoderTransformer",
    "AbsoluteDeltaCyclicalPatchTST",
    "AbsoluteDeltaPatchTST",
    "BridgeForecastConfig",
    "BridgeForecaster",
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
    "bridge_config_values",
    "bridge_forecast_name",
    "bridge_forecast_names",
    "bridge_history_model_name",
    "build_config",
    "build_model",
    "model_names",
    "load_bridge_checkpoint",
    "register_model",
    "validate_bridge_checkpoint",
]
