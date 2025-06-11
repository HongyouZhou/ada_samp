import importlib
from collections.abc import Callable

TRAINER_REGISTRY = {}


def register_trainer(name: str):
    def decorator(cls):
        module = cls.__module__
        qualname = cls.__qualname__
        TRAINER_REGISTRY[name] = f"{module}.{qualname}"
        return cls

    return decorator

def get_trainer(cfg):
    method = cfg.trainer.type
    if method is None:
        raise ValueError("[get_trainer] Trainer type is not specified.")
    
    if method not in TRAINER_REGISTRY:
        raise ValueError(f"[get_trainer] Unknown trainer '{method}'. Available: {list(TRAINER_REGISTRY.keys())}")

    module_path, class_name = TRAINER_REGISTRY[method].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except Exception as e:
        raise ImportError(f"[get_trainer] Failed to import {TRAINER_REGISTRY[method]}: {e}") from e


def list_trainer() -> dict[str, Callable]:
    return TRAINER_REGISTRY.copy()

