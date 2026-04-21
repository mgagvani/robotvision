import pytorch_lightning as pl

class GradientDebugCallback(pl.Callback):
    def __init__(self, log_every=10):
        self.log_every = log_every
