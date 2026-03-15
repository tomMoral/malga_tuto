from benchopt import BaseSolver

from benchmark_utils.losses import TVLoss, KL


class Solver(BaseSolver):
    """Multiplicative Mirror Descent with KL divergence + TV regularization.

    Uses h(x) = -sum log(x_i) as Bregman generator, yielding the closed-form
    update  x_{k+1} = x_k / (1 + tau * x_k * grad_f(x_k))  which naturally
    enforces non-negativity.

    Stability requires tau * x * g = O(1). Bounding |x * g_KL| by
    ||A||^2 * x_max * y_max / back (with normalised kernel ||A|| ≈ 1 and
    x_max ≈ y_max) gives lip = y.max() / back, which is dimension-independent
    (unlike y.sum() which grows with image resolution and flux level).
    """

    name = "MirrorDescent"

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

        # Dimension-independent Lipschitz bound: y.max() / back
        # (analogous to norm_H / back for PGD, with normalised norm_H ≈ 1)
        lip = y.max() / back + 8 * self.lam / self.eps_tv
        self.tau = 2.0 * self.step_factor / lip

        self.x = physics.A_adjoint(y).detach()

    def run(self, callback):
        while callback():
            g = self.kl.grad(self.x, self.y, self.physics)
            g = g + self.lam * self.tv.grad(self.x)
            self.x = (self.x / (1.0 + self.tau * self.x * g)).detach()

    def get_result(self):
        return dict(x_hat=self.x)
