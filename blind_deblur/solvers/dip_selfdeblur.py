from benchopt import BaseSolver

import torch
import deepinv as dinv
import torch.nn.functional as F
from benchmark_utils.networks import DIPSkipNet, KernelNet


class Solver(BaseSolver):
    """Blind deblurring via Deep Image Prior / SelfDeblur.

    Both the image and the blur kernel are parameterized by neural networks
    whose inputs are fixed random noise codes.
    """

    name = "DIP-SelfDeblur"

    parameters = {
        'lr': [1e-3],
        'z_dim': [32],
        'add_noise': [0.03],
    }

    def set_objective(self, y, kernel_size):
        self.y = y
        self.kernel_size = kernel_size
        self.device = y.device

        _, _, H, W = y.shape
        torch.manual_seed(1)
        self.z_x = torch.randn(1, self.z_dim, H, W, device=self.device)
        self.z_k = torch.randn(1, self.z_dim, device=self.device)

        channels = y.shape[-3]
        self.x_net = DIPSkipNet(
            in_channels=self.z_dim, out_channels=channels
        ).to(self.device)
        self.k_net = KernelNet(
            z_dim=self.z_dim, kernel_size=self.kernel_size
        ).to(self.device)

        self.opt = torch.optim.Adam(
            list(self.x_net.parameters()) + list(self.k_net.parameters()),
            lr=self.lr,
        )

        # Initial estimates: network output and delta kernel
        with torch.no_grad():
            self.x_hat = self.x_net(self.z_x).detach()
        k_delta = torch.zeros(
            1, 1, self.kernel_size, self.kernel_size, device=self.device
        )
        k_delta[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        self.k_hat = k_delta

    def run(self, callback):
        while callback():
            self.opt.zero_grad()

            x_hat = self.x_net(
                self.z_x + self.add_noise * torch.randn_like(self.z_x)
            )
            k_hat = self.k_net(self.z_k)

            physics_opt = dinv.physics.BlurFFT(
                filter=k_hat, device=self.device, img_size=x_hat.shape[-3:],
            )
            loss = F.mse_loss(physics_opt(x_hat), self.y)
            loss.backward()
            self.opt.step()

            self.x_hat = x_hat.detach()
            self.k_hat = k_hat.detach()

    def get_result(self):
        return dict(x_hat=self.x_hat, k_hat=self.k_hat)
