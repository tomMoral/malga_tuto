from benchopt import BaseSolver

import torch
import torch.nn.functional as F
import deepinv as dinv
from deepinv.optim import TVPrior
from deepinv.optim.data_fidelity import L2


class Solver(BaseSolver):
    """Blind deblurring via bilevel optimization.

    Inner loop: PGD minimizing TV-regularized data fidelity for a fixed kernel.
    Outer loop: Adam step on the observed data-fit loss w.r.t. the kernel.
    """

    name = "Bilevel-PGD"

    parameters = {
        'outer_lr': [0.2],
        'inner_iters': [50],
        'inner_stepsize': [1.0],
        'reg_param': [0.02],
    }

    sampling_strategy = 'callback'

    def set_objective(self, y, kernel_size):
        self.y = y
        self.kernel_size = kernel_size
        self.device = y.device

        # Initialise kernel parameter and optimizer
        self.kernel_param = torch.randn(
            1, 1, self.kernel_size, self.kernel_size,
            device=self.device, requires_grad=True,
        )
        self.outer_opt = torch.optim.Adam(
            [self.kernel_param], lr=self.outer_lr
        )

        self.data_fidelity = L2()
        self.inner_opt = dinv.optim.PGD(
                prior=TVPrior(n_it_max=20),
                data_fidelity=self.data_fidelity,
                stepsize=self.inner_stepsize,
                lambda_reg=self.reg_param,
                max_iter=self.inner_iters,
                early_stop=True,
                verbose=False,
            )

        # Default estimates (identity kernel, initial solution)
        self.x_hat = self._solver_inner()
        k_delta = torch.zeros_like(self.kernel_param.detach())
        k_delta[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        self.k_hat = k_delta

    def _solver_inner(self):
        # Inner optimization (detached kernel)
        k_inner = F.softplus(self.kernel_param.detach())
        k_inner = k_inner / k_inner.sum()
        physics_inner = dinv.physics.BlurFFT(
            filter=k_inner, device=self.device,
            img_size=self.y.shape[-3:],
        )
        x_init = (
            physics_inner.A_adjoint(self.y).detach().requires_grad_(True)
        )
        x_star = self.inner_opt(
            y=self.y, physics=physics_inner, init=x_init
        ).detach()
        return x_star

    def run(self, callback):
        while callback():
            x_star = self._solver_inner()

            # ---- Outer optimization (differentiable kernel) ----
            k_outer = F.softplus(self.kernel_param)
            k_outer = k_outer / k_outer.sum()
            physics_outer = dinv.physics.BlurFFT(
                filter=k_outer, device=self.device,
                img_size=self.y.shape[-3:],
            )
            self.outer_opt.zero_grad()
            y_pred = physics_outer(x_star)
            outer_loss = self.data_fidelity(y_pred, self.y, physics_outer)
            outer_loss.backward()
            self.outer_opt.step()

            # Store latest estimates for get_result()
            self.x_hat = x_star
            k_final = F.softplus(self.kernel_param).detach()
            self.k_hat = k_final / k_final.sum()

    def get_result(self):
        return dict(x_hat=self.x_hat, k_hat=self.k_hat)
