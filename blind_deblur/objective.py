from benchopt import BaseObjective
from benchopt.stopping_criterion import SufficientProgressCriterion

import torch
import deepinv as dinv


class Objective(BaseObjective):

    name = "Blind Deblurring"
    url = "https://github.com/tommoral/malga/blind_deblur"
    requirements = ["pip::torch", "pip::deepinv"]
    min_benchopt_version = "1.8"

    stopping_criterion = SufficientProgressCriterion(
        strategy='callback', patience=10,
        key_to_monitor='psnr', minimize=False,
    )

    def set_data(self, x_true, y, physics, kernel_size):
        self.x_true = x_true
        self.physics = physics
        self.y = y
        self.kernel_size = kernel_size

    def get_objective(self):
        return dict(
            y=self.y,
            kernel_size=self.kernel_size,
        )

    def save_last_result(self, x_hat, k_hat):
        return dict(x_hat=k_hat.cpu())

    def evaluate_result(self, x_hat, k_hat):
        device = x_hat.device
        x_true = self.x_true.to(device)
        k_hat = k_hat.to(device)

        psnr = dinv.metric.PSNR()(x_hat, x_true).item()

        k_true = self.physics.filter.to(device)
        k_true_norm = k_true / (k_true.sum() + 1e-8)
        k_hat_norm = k_hat / (k_hat.sum() + 1e-8)
        kernel_mse = torch.mean((k_hat_norm - k_true_norm) ** 2).item()

        return dict(
            psnr=psnr,
            kernel_mse=kernel_mse,
        )

    def get_one_result(self):
        k_delta = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        k_delta[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        return dict(x_hat=self.y.cpu(), k_hat=k_delta)
