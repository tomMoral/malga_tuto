Poisson Deblurring Benchmark
=============================

A `benchopt <https://benchopt.github.io>`_ benchmark for **Poisson image
deblurring**: given an observation ``y ~ Poisson(k * x + back)``, recover
the clean image ``x``.

Install
--------

::

    $ pip install benchopt
    $ benchopt run .

Dataset
-------

- **TubLevel** — a fluorescence microscopy image (``tub_level.pth``, expected
  at the ``malga/`` workspace root) convolved with a Gaussian blur kernel and
  corrupted by Poisson noise with a constant background offset.

Solvers
-------

Both solvers minimize the same objective
``D_KL(y || A(x) + back) + lam * TV(x)``
with non-negativity constraints.

- **PGD** — Projected Gradient Descent with a Lipschitz step-size estimate.
- **MirrorDescent** — Multiplicative mirror descent; the KL Bregman update
  ``x_{k+1} = x_k / (1 + tau * x_k * grad_f(x_k))`` naturally preserves
  non-negativity without a projection step.

Shared utilities
----------------

``benchmark_utils/losses.py`` provides ``TVLoss`` and ``KL`` used by both
solvers.

Objective
---------

Metrics are computed on max-normalized images (matching the notebook):

- ``psnr`` — peak signal-to-noise ratio (primary, higher is better)
- ``ssim`` — structural similarity index
