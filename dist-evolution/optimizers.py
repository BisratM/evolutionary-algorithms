"""Optimization algorithms for evolutionary strategies."""

import numpy as np


class Optimizer:
    """Base class for parameter optimization algorithms."""

    def __init__(self, pi):
        """Initialize the optimizer.

        Args:
            pi: Policy object with trainable parameters.

        """
        self.pi = pi
        self.dim = pi.num_params
        self.t = 0

    def update(self, globalg):
        """Update parameters using the computed gradient.

        Args:
            globalg: Global gradient for parameter update.

        Returns:
            Ratio of step norm to parameter norm.

        """
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.get_trainable_flat()
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        self.pi.set_trainable_flat(theta + step)
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum."""

    def __init__(self, pi, stepsize, momentum=0.9):
        """Initialize SGD optimizer.

        Args:
            pi: Policy object with trainable parameters.
            stepsize: Learning rate for parameter updates.
            momentum: Momentum coefficient for velocity (default: 0.9).

        """
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates."""

    def __init__(self, pi, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        """Initialize Adam optimizer.

        Args:
            pi: Policy object with trainable parameters.
            stepsize: Base learning rate for parameter updates.
            beta1: Exponential decay rate for first moment (default: 0.9).
            beta2: Exponential decay rate for second moment (default: 0.999).
            epsilon: Small constant for numerical stability (default: 1e-08).

        """
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
