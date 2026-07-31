from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from torch import nn


DEFAULT_MODEL = "transformer"
ModelBuilder = Callable[[Any], nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    builder: ModelBuilder
    config_type: type


MODEL_SPECS: dict[str, ModelSpec] = {}


def register_model(name: str, builder: ModelBuilder, config_type: type) -> None:
    if not name or name in MODEL_SPECS:
        raise ValueError(f"invalid or duplicate model name: {name}")
    MODEL_SPECS[name] = ModelSpec(builder, config_type)


def build_config(name: str, values: Mapping[str, Any]) -> Any:
    try:
        config_type = MODEL_SPECS[name].config_type
    except KeyError as error:
        raise ValueError(f"unknown model: {name}") from error
    return config_type(
        **{
            field.name: values[field.name]
            for field in fields(config_type)
            if field.name in values
        }
    )


def build_model(name: str, config: Any) -> nn.Module:
    try:
        return MODEL_SPECS[name].builder(config)
    except KeyError as error:
        raise ValueError(f"unknown model: {name}") from error


def model_names() -> tuple[str, ...]:
    return tuple(MODEL_SPECS)
