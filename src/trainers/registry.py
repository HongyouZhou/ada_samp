import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trainers.base_trainer import BaseTrainer

TRAINER_REGISTRY = {
    "gmflow": "src.trainers.gmflow_trainer.GMFlowTrainer",
    "ddpm": "src.trainers.ddpm_trainer.DDPMTrainer",
    "flow_matching": "src.trainers.flowmatch_trainer.FlowMatchingTrainer",
    "checkboard": "src.trainers.toymodel_trainer.CheckboardTrainer",
}


def get_trainer(name: str) -> "type[BaseTrainer]":
    name = name.lower()
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"[get_trainer] Unknown trainer '{name}'. Available: {list(TRAINER_REGISTRY.keys())}")

    module_path, class_name = TRAINER_REGISTRY[name].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except Exception as e:
        raise ImportError(f"[get_trainer] Failed to import {TRAINER_REGISTRY[name]}: {e}") from e
