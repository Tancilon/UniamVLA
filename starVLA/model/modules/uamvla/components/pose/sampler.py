import math
from collections.abc import Callable

import torch

from .pose_utils import get_pose_dim, normalize_rotation


def cond_pc_sampler(
    score_model: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    data: dict[str, torch.Tensor],
    prior: Callable[..., torch.Tensor],
    sde_coeff: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    num_steps: int = 500,
    snr: float = 0.16,
    device: str | torch.device = "cuda",
    eps: float = 1e-5,
    pose_mode: str = "quat_wxyz",
    init_x: torch.Tensor | None = None,
) -> torch.Tensor:
    sample_device = torch.device(device)
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data["pts_feat"].shape[0]
    # Run the Langevin loop in the score network's dtype (bf16 under
    # DeepSpeed, float32 on CPU). pts_feat was already cast by the caller,
    # so it is the source of truth for score_dtype.
    score_dtype = data["pts_feat"].dtype

    if init_x is None:
        x = prior((batch_size, pose_dim)).to(sample_device).to(score_dtype)
    else:
        x = init_x.to(sample_device).to(score_dtype)

    time_steps = torch.linspace(1.0, eps, num_steps, device=sample_device, dtype=score_dtype)
    step_size = (
        time_steps[0] - time_steps[1]
        if num_steps > 1
        else torch.tensor(1.0 - eps, device=sample_device, dtype=score_dtype)
    )
    noise_norm = math.sqrt(float(pose_dim))
    mean_x = x

    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(
                (batch_size, 1), device=sample_device, dtype=score_dtype,
            ) * time_step
            data["sampled_pose"] = x
            data["t"] = batch_time_step
            grad = score_model(data)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean().clamp_min(1.0)
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2 * grad
            # Anderson reverse SDE with positive step_size: x(t-h) = x(t) -
            # [f - g^2 s] * h + g * sqrt(h) * xi. Upstream GenPose2 wrote
            # `mean_x = x + drift * step_size`, which inverts the sign and
            # causes the predictor to push x AWAY from the data manifold,
            # compounding as the score becomes accurate.
            mean_x = x - drift * step_size
            x = mean_x + diffusion * torch.sqrt(step_size) * torch.randn_like(x)
            x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)

    # Cast to float32 before world-frame de-centering. pts_center carries
    # meter-scale world coordinates; keeping this step in float32 avoids
    # ~1cm rounding error that bf16 would introduce for absolute positions.
    mean_x = mean_x.float()
    pts_center = data["pts_center"].to(sample_device).float()
    mean_x[:, -3:] += pts_center
    mean_x[:, :-3] = normalize_rotation(mean_x[:, :-3], pose_mode)
    return mean_x
