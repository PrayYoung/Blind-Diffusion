import copy

import torch


class ModelEMA:
    """Checkpointable EMA; copy integer and non-parameter schedule buffers exactly."""

    def __init__(self, model, decay=0.995):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for target, source in zip(self.model.parameters(), model.parameters()):
            target.lerp_(source, 1 - decay)
        for target, source in zip(self.model.buffers(), model.buffers()):
            target.copy_(source)
