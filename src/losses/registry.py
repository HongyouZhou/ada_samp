import importlib
from collections.abc import Callable

LOSS_REGISTRY: dict[str, Callable] = {}


def register_loss(name: str):
    def decorator(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' already registered.")
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def build_loss(cfg) -> Callable:
    name = cfg.get("type", None)
    if name is None:
        raise ValueError("[build_loss] Loss type is not specified.")

    if name not in LOSS_REGISTRY:
        raise ValueError(f"[build_loss] Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}")

    cfg.pop("type")
    try:
        cls = LOSS_REGISTRY[name]
        return cls(**cfg)
    except Exception as e:
        raise ValueError(f"[build_loss] Failed to instantiate loss '{name}': {e}") from e


def list_losses() -> dict[str, Callable]:
    return LOSS_REGISTRY.copy()
