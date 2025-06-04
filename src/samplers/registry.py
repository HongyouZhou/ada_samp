import importlib

TIMESTEP_SAMPLER_REGISTRY = {}
NOISE_SAMPLER_REGISTRY = {}

def register_timestep_sampler(name):
    def decorator(cls):
        module = cls.__module__
        qualname = cls.__qualname__
        TIMESTEP_SAMPLER_REGISTRY[name] = f"{module}.{qualname}"
        return cls
    return decorator

def register_noise_sampler(name):
    def decorator(cls):
        module = cls.__module__
        qualname = cls.__qualname__
        NOISE_SAMPLER_REGISTRY[name] = f"{module}.{qualname}"
        return cls
    return decorator

def get_timestep_sampler(cfg):
    name = cfg.get("type", None)
    if name is None:
        raise ValueError("[get_timestep_sampler] Timestep sampler type is not specified.")

    if name not in TIMESTEP_SAMPLER_REGISTRY:
        raise ValueError(f"[get_timestep_sampler] Unknown timestep sampler '{name}'. Available: {list(TIMESTEP_SAMPLER_REGISTRY.keys())}")

    module_path, class_name = TIMESTEP_SAMPLER_REGISTRY[name].rsplit(".", 1)
    cfg.pop("type")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**cfg)
    except Exception as e:
        raise ImportError(f"[get_timestep_sampler] Failed to import {TIMESTEP_SAMPLER_REGISTRY[name]}: {e}") from e

def get_noise_sampler(cfg):
    name = cfg.get("type", None)
    if name is None:
        raise ValueError("[get_noise_sampler] Noise sampler type is not specified.")

    if name not in NOISE_SAMPLER_REGISTRY:
        raise ValueError(f"[get_noise_sampler] Unknown noise sampler '{name}'. Available: {list(NOISE_SAMPLER_REGISTRY.keys())}")

    module_path, class_name = NOISE_SAMPLER_REGISTRY[name].rsplit(".", 1)
    cfg.pop("type")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**cfg)
    except Exception as e:
        raise ImportError(f"[get_noise_sampler] Failed to import {NOISE_SAMPLER_REGISTRY[name]}: {e}") from e