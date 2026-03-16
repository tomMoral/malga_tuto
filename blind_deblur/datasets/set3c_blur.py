from benchopt import BaseDataset
from benchopt.config import get_data_path

import torch
from torchvision import transforms
from deepinv.utils import load_dataset
from deepinv.physics.blur import Blur, gaussian_blur
from deepinv.physics import GaussianNoise


class Dataset(BaseDataset):

    name = "Set3C"

    parameters = {
        'sigma_blur': [2.0],
        'sigma_noise': [0.01],
        'img_size': [128],
        'img_idx': [0],
    }

    requirements = ["pip::torchvision"]

    def get_data(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])
        data_dir = get_data_path("set3c")
        dataset = load_dataset(
            "set3c", data_dir=data_dir, transform=transform
        )
        x_true = dataset[self.img_idx].to(device)[None, ...]

        blur_kernel = gaussian_blur(sigma=self.sigma_blur, device=device)
        kernel_size = blur_kernel.shape[-1]

        physics = Blur(
            filter=blur_kernel,
            padding="circular",
            device=device,
            noise_model=GaussianNoise(sigma=self.sigma_noise),
        )
        with torch.no_grad():
            y = physics(x_true)

        return dict(
            x_true=x_true,
            y=y,
            physics=physics,
            kernel_size=kernel_size,
        )
