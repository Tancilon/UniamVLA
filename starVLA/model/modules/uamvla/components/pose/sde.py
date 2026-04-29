import functools
from collections.abc import Callable

import torch


def ve_marginal_prob(
    x: torch.Tensor | None, t: torch.Tensor,
    sigma_min: float = 0.01, sigma_max: float = 90.0,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std


def ve_sde(
    t: torch.Tensor, sigma_min: float = 0.01, sigma_max: float = 90.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.zeros_like(t)
    log_ratio = torch.log(torch.tensor(sigma_max / sigma_min, device=t.device, dtype=t.dtype))
    diffusion_coeff = sigma * torch.sqrt(2.0 * log_ratio)
    return drift_coeff, diffusion_coeff


def ve_prior(
    shape: tuple[int, ...], sigma_min: float = 0.01, sigma_max: float = 90.0, T: float = 1.0,
) -> torch.Tensor:
    _, sigma_max_prior = ve_marginal_prob(None, torch.tensor(T), sigma_min=sigma_min, sigma_max=sigma_max)
    return torch.randn(*shape) * sigma_max_prior


def vp_marginal_prob(
    x: torch.Tensor, t: torch.Tensor, beta_0: float = 0.1, beta_1: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_mean_coeff = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    return mean, std


def vp_sde(
    t: torch.Tensor, beta_0: float = 0.1, beta_1: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    diffusion_coeff = torch.sqrt(beta_t)
    return drift_coeff, diffusion_coeff


def vp_prior(
    shape: tuple[int, ...], beta_0: float = 0.1, beta_1: float = 20.0,
) -> torch.Tensor:
    del beta_0, beta_1
    return torch.randn(*shape)


def init_sde(
    sde_mode: str,
) -> tuple[
    Callable[..., torch.Tensor],
    Callable[..., tuple[torch.Tensor | None, torch.Tensor]],
    Callable[..., tuple[torch.Tensor, torch.Tensor]],
    float,
    float,
]:
    """Initialize SDE components for a selected mode."""
    if sde_mode == "ve":
        sigma_min = 0.01
        sigma_max = 3.0
        eps = 1e-5
        marginal_prob_fn = functools.partial(ve_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)
        sde_fn = functools.partial(ve_sde, sigma_min=sigma_min, sigma_max=sigma_max)
        terminal_t = 1.0
        prior_fn = functools.partial(ve_prior, sigma_min=sigma_min, sigma_max=sigma_max)
    elif sde_mode == "vp":
        beta_0 = 0.1
        beta_1 = 20.0
        eps = 1e-3
        prior_fn = functools.partial(vp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(vp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(vp_sde, beta_0=beta_0, beta_1=beta_1)
        terminal_t = 1.0
    else:
        raise NotImplementedError(f"SDE mode '{sde_mode}' not supported.")
    return prior_fn, marginal_prob_fn, sde_fn, eps, terminal_t
