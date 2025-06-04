import importlib
from collections.abc import Callable

DENOISER_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(cls):
        module = cls.__module__
        qualname = cls.__qualname__
        DENOISER_REGISTRY[name] = f"{module}.{qualname}"
        return cls

    return decorator


def get_model(cfg):
    name = cfg.get("type", None)
    if name is None:
        raise ValueError("[get_model] Model type is not specified.")
    
    if name not in DENOISER_REGISTRY:
        raise ValueError(f"[get_model] Unknown model '{name}'. Available: {list(DENOISER_REGISTRY.keys())}")

    module_path, class_name = DENOISER_REGISTRY[name].rsplit(".", 1)
    cfg.pop("type")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**cfg)
    except Exception as e:
        raise ImportError(f"[get_model] Failed to import {DENOISER_REGISTRY[name]}: {e}") from e


def list_model() -> dict[str, Callable]:
    return DENOISER_REGISTRY.copy()
