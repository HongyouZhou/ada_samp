import importlib

from .base_trainer import BaseTrainer

TRAINER_REGISTRY = {
    "gmflow": "ada_samp.src.trainers.gmflow_trainer.GMFlowTrainer",
    "ddpm": "ada_samp.src.trainers.ddpm_trainer.DDPMTrainer",
    "flow_matching": "ada_samp.src.trainers.flowmatch_trainer.FlowMatchingTrainer",
}


def get_trainer(name: str) -> BaseTrainer:
    path = TRAINER_REGISTRY.get(name)
    if path is None:
        raise ValueError(f"[get_trainer] Trainer '{name}' not found.")
    module_path, class_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"[get_trainer] Failed to import '{path}': {e}") from e
