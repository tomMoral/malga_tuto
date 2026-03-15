from benchopt import BaseSolver

import torch
from benchmark_utils.losses import TVLoss, KL


class Solver(BaseSolver):
    """Projected Gradient Descent with KL divergence + TV regularization.

    Step size is tau = step_factor / L where L = ||A||^2_op / back is a
    Lipschitz upper bound for the KL gradient, derived from the bound
    y / (Ax + back)^2 <= 1 / back (valid under mild Poisson noise).
    """

    name = "PGD"

    parameters = {
        'lam': [1e-2],
        'eps_tv': [1e-2],
        'step_factor': [1],
    }

    sampling_strategy = 'callback'

    def set_objective(self, y, physics, back):
        self.y = y
        self.physics = physics
        self.device = y.device

        self.kl = KL(eps=back)
        self.tv = TVLoss(eps=self.eps_tv)

        # Lipschitz upper bound for the KL gradient.
        # The Hessian of D_KL(y || Ax+eps) is A^T diag(y/(Ax+eps)^2) A.
        # Bounding y/(Ax+eps)^2 <= 1/eps = 1/back (valid when y <= Ax+back,
        # i.e. Poisson noise is not far above its mean) gives:
        #   L <= ||A||^2_op / back
        # This is tighter than the naive bound norm_y/back^2 by a factor of
        # norm_y/back (≈ flux/back), and explains the empirical *1e-2=*back
        # used in the notebook.
        x_ones = torch.ones_like(y)
        norm_H = (
            torch.max(physics(x_ones)) * torch.max(physics.A_adjoint(x_ones))
        )
        lip = norm_H / back + 8 * self.lam / self.eps_tv
        self.tau = self.step_factor / lip

        self.x = physics.A_adjoint(y).detach()

    def run(self, callback):
        while callback():
            g = self.kl.grad(self.x, self.y, self.physics)
            g = g + self.lam * self.tv.grad(self.x)
            self.x = torch.clamp(self.x - self.tau * g, min=0.0).detach()

    def get_result(self):
        return dict(x_hat=self.x)
