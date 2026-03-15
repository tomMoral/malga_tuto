from benchopt import BaseObjective
from benchopt.stopping_criterion import SufficientProgressCriterion

import torch
import deepinv as dinv


class Objective(BaseObjective):

    name = "Poisson Deblurring"
    url = "https://github.com/tommoral/malga/poisson_deblur"
    requirements = ["pip::deepinv"]
    min_benchopt_version = "1.8"

    stopping_criterion = SufficientProgressCriterion(
        strategy='callback', patience=10,
        key_to_monitor='psnr', minimize=False,
    )

    def set_data(self, x_true, y, physics, back):
        self.x_true = x_true
        self.y = y
        self.physics = physics
        self.back = back

    def get_objective(self):
        self.x_prev = None  # for monitoring relative change in iterates
        return dict(
            y=self.y,
            physics=self.physics,
            back=self.back,
        )

    def evaluate_result(self, x_hat):
        device = x_hat.device
        x_true = self.x_true.to(device)

        # Normalize to [0, 1] range as done in the notebook
        x_true_norm = x_true / (x_true.max() + 1e-8)
        x_hat_norm = x_hat / (x_hat.max() + 1e-8)

        try:
            dinv.metric.SSIM()
        except Exception:
            pass

        psnr = dinv.metric.PSNR()(x_hat_norm, x_true_norm).item()
        ssim = dinv.metric.SSIM()(x_hat_norm, x_true_norm).item()

        if self.x_prev is not None:
            iter_grad = (
                torch.norm(self.x_prev - x_hat, 'fro')
                / (torch.norm(self.x_prev, 'fro') + 1e-10)
            ).item()
        else:
            iter_grad = 1
        self.x_prev = x_hat.detach()

        return dict(
            psnr=psnr,
            ssim=ssim,
            iter_grad=iter_grad
        )

    def get_one_result(self):
        return dict(x_hat=self.y)
