from .base_trainer import BaseTrainer


class GMFlowTrainer(BaseTrainer):
    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)

    def train(self):
        pass
