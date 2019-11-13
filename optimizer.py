class MFNptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = 0.1
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        if self.step_num in [36000, 52000, 58000]:
            self.lr = self.lr / 10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
