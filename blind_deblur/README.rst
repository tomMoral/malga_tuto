Blind Deblurring Benchmark
==========================

A `benchopt <https://benchopt.github.io>`_ benchmark for **blind image deblurring**:
given a blurry and noisy observation ``y = k * x + noise``, jointly recover
the clean image ``x`` and the blur kernel ``k``.

Install
--------

This benchmark can be run using the following commands::

    $ pip install benchopt
    $ benchopt run .

Use ``benchopt run . --help`` to see all options, or refer to the
`benchopt documentation <https://benchopt.github.io>`_.

Datasets
--------

- **Set3C-BlindBlur** — a single image from the Set3C dataset degraded with
  a Gaussian blur kernel of controllable width and additive Gaussian noise.

Solvers
-------

- **Bilevel-PGD** — Bilevel optimization: PGD inner loop (TV regularization)
  for image reconstruction with a fixed kernel, followed by an Adam outer step
  on the data-fit loss to update the kernel parameterization.

- **DIP-SelfDeblur** — Deep Image Prior / SelfDeblur: both the image and
  the kernel are parameterized by small neural networks driven by fixed
  random noise codes, trained end-to-end with a reconstruction loss.

Objective
---------

The primary metric is **PSNR** (higher is better; stored as ``-PSNR`` so
benchopt's default minimization convention applies).  Additional metrics:

- ``ssim`` — structural similarity index
- ``kernel_mse`` — mean squared error between the estimated and true
  normalized kernels
