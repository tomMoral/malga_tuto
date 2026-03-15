from pathlib import Path

from benchopt import BaseDataset

import torch
from deepinv.physics import Denoising, PoissonNoise
import deepinv as dinv

# tub_level.pth leaving in the same repository for simplicity,
# but could be download easily
DATA_FILE = Path(__file__).parent / "tub_level.pth"


class Dataset(BaseDataset):

    name = "TubLevel"

    parameters = {
        'flux': [30],
        'back': [0.01],
        'sigma_blur': [3.0],
    }

    def get_data(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        image = torch.load(DATA_FILE, map_location=device)
        # ensure shape (1, 1, H, W)
        while image.dim() < 4:
            image = image.unsqueeze(0)

        ground_truth = image * self.flux

        blur_filter = dinv.physics.blur.gaussian_blur(
            sigma=(self.sigma_blur, self.sigma_blur), angle=0
        ).to(device)
        physics = dinv.physics.BlurFFT(
            img_size=ground_truth.shape,
            filter=blur_filter,
            device=device,
        )

        physics_noise = Denoising()
        physics_noise.noise_model = PoissonNoise(gain=1.0)

        with torch.no_grad():
            clean_image = physics(ground_truth) + self.back
            y = physics_noise(clean_image)

        return dict(
            x_true=ground_truth,
            y=y,
            physics=physics,
            back=self.back,
        )
