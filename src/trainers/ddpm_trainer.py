from .base_trainer import BaseTrainer


class DDPMTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)

    def train(self):
        pass
