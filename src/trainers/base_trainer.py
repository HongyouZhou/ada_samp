
class BaseTrainer:
    def __init__(self, cfg, accelerator):
        self.cfg = cfg
        self.accelerator = accelerator

    def train(self):
        pass