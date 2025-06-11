import importlib
from collections.abc import Callable

DATASET_REGISTRY: dict[str, Callable] = {}


def register_dataset(name: str):
    def decorator(cls):
        module = cls.__module__
        qualname = cls.__qualname__
        DATASET_REGISTRY[name] = f"{module}.{qualname}"
        return cls

    return decorator


def get_dataset(cfg):
    name = cfg.get("type", None)
    if name is None:
        raise ValueError("[get_dataset] Dataset type is not specified.")
    
    if name not in DATASET_REGISTRY:
        raise ValueError(f"[get_dataset] Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")

    module_path, class_name = DATASET_REGISTRY[name].rsplit(".", 1)
    cfg.pop("type")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**cfg)
    except Exception as e:
        raise ImportError(f"[get_dataset] Failed to import {DATASET_REGISTRY[name]}: {e}") from e


def list_dataset() -> dict[str, Callable]:
    return DATASET_REGISTRY.copy()
