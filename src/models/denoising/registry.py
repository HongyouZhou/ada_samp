from collections.abc import Callable

DENOISER_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(cls):
        if name in DENOISER_REGISTRY:
            raise ValueError(f"Model '{name}' already registered.")
        DENOISER_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str) -> Callable:
    if name not in DENOISER_REGISTRY:
        raise KeyError(f"Model '{name}' not found in DENOISER_REGISTRY.")
    return DENOISER_REGISTRY[name]


def list_models() -> dict[str, Callable]:
    return DENOISER_REGISTRY.copy()
