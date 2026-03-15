import torch
import torch.nn as nn


class TVLoss(nn.Module):
    """Isotropic total variation with smoothing parameter eps."""

    def __init__(self, eps=1e-2):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[..., 1:] - x[..., :-1]
        tv = torch.sqrt(dx[..., :-1] ** 2 + dy[:, :, :-1, :] ** 2 + self.eps)
        return tv.sum(dim=(1, 2, 3))

    def grad(self, x):
        x = x.clone().requires_grad_(True)
        loss = self.forward(x).sum()
        return torch.autograd.grad(loss, x)[0]


class KL(nn.Module):
    """Generalized KL divergence (I-divergence) for Poisson data fidelity.

    Computes D(y || A(x) + eps) = sum_i [y_i log(y_i / (A(x)_i + eps))
                                          + A(x)_i + eps - y_i]
    with the convention that the term is A(x)_i when y_i = 0.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y, physics, alpha=1):
        ax = physics.A(x * alpha)
        val = ax + self.eps
        kl = torch.where(y > 0, y * torch.log(y / val) + val - y, ax)
        return kl.sum(dim=(1, 2, 3))

    def grad(self, x, y, physics, alpha=1):
        ax = physics.A(x)
        return alpha * physics.A_adjoint(1.0 - y / (alpha * ax + self.eps))
