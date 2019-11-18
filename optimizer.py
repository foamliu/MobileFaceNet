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
        if self.step_num in [36000, 52000, 78000, 100000]:
            self.lr = self.lr / 10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def clip_gradient(self, grad_clip):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.
        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
