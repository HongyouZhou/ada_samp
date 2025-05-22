from collections.abc import Callable

LOSS_REGISTRY: dict[str, Callable] = {}


def register_loss(name: str):
    def decorator(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' already registered.")
        LOSS_REGISTRY[name] = cls
        return cls

    return decorator


def build_loss(name: str) -> Callable:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Loss '{name}' not found in LOSS_REGISTRY.")
    return LOSS_REGISTRY[name]


def list_losses() -> dict[str, Callable]:
    return LOSS_REGISTRY.copy()
