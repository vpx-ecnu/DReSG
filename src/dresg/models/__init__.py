"""DReSG model domains.

The public package keeps diffusion-guidance components and Gaussian
Splatting components under the same model namespace:

- :mod:`dresg.models.diffusion` exposes one run-specific guidance facade over
  its internal frozen backbone, attention losses, latent state, and scales.
- :mod:`dresg.models.gs` owns the Gaussian scene, fitting losses,
  rasterization boundary, and direct-RGB PLY serialization.
"""
