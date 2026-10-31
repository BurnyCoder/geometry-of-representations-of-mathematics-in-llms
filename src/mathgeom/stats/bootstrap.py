"""Bias-corrected and accelerated (BCa) bootstrap intervals.

Global context: the field-map ρ is computed from field centroids, and a concept bootstrap duplicates
concepts, which makes the centroids noisier and pulls every resampled ρ toward 0. The percentile
interval then sits below the point estimate (Qwen3.5-2B: ρ = 0.387, bootstrap mean 0.341). BCa corrects
for that median bias (z0) and for skew (acceleration a, from a leave-one-out jackknife).
Efron 1987, "Better bootstrap confidence intervals", JASA 82:171 (doi:10.1080/01621459.1987.10478410);
the same formulas as scipy.stats.bootstrap(method="BCa")
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), which cannot be used
directly here because our resampling is stratified by field and paired across models.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bca_interval(theta_hat: float, boots: np.ndarray, jack: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """BCa interval from bootstrap replicates `boots` and leave-one-out jackknife values `jack`."""
    boots, jack = np.asarray(boots, float), np.asarray(jack, float)
    z0 = norm.ppf((np.sum(boots < theta_hat) + 0.5 * np.sum(boots == theta_hat)) / len(boots))  # median bias
    d = jack.mean() - jack
    a = np.sum(d**3) / (6.0 * np.sum(d**2) ** 1.5) if np.any(d) else 0.0  # acceleration (skew of the jackknife)
    zs = norm.ppf([(1 - level) / 2, (1 + level) / 2])
    q = norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))  # adjusted percentile levels
    return float(np.percentile(boots, 100 * q[0])), float(np.percentile(boots, 100 * q[1]))
