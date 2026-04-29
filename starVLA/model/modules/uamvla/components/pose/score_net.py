import torch
import torch.nn as nn

from .pose_utils import get_pose_dim



class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * (2 * torch.pi)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PoseScoreNet(nn.Module):
    def __init__(self, marginal_prob_func, semantic_dim: int,
                 pose_mode: str = "rot_matrix", regression_head: str = "RT") -> None:
        super().__init__()
        if regression_head != "RT":
            raise NotImplementedError("Only regression_head='RT' is supported.")
        self.regression_head = regression_head
        self.semantic_dim = semantic_dim
        self.act = nn.ReLU(True)
        pose_dim = get_pose_dim(pose_mode)

        self.pose_encoder = nn.Sequential(nn.Linear(pose_dim, 256), self.act, nn.Linear(256, 256), self.act)
        self.t_encoder = nn.Sequential(GaussianFourierProjection(embed_dim=128), nn.Linear(128, 128), self.act)
        fusion_in_dim = 128 + 256 + 1024 + semantic_dim
        self.fusion_tail = nn.Sequential(
            nn.Linear(fusion_in_dim, 512),
            self.act,
            nn.Linear(512, pose_dim),
        )
        self.marginal_prob_func = marginal_prob_func

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        pts_feat = data["pts_feat"]
        sampled_pose = data["sampled_pose"]
        t = data["t"]
        t_feat = self.t_encoder(t.squeeze(1))
        pose_feat = self.pose_encoder(sampled_pose)
        if self.semantic_dim > 0:
            semantic_feat = data["semantic_feat"]
            total_feat = torch.cat([pts_feat, t_feat, pose_feat, semantic_feat], dim=-1)
        else:
            total_feat = torch.cat([pts_feat, t_feat, pose_feat], dim=-1)
        _, std = self.marginal_prob_func(total_feat, t)
        if std.ndim == 1:
            std = std.unsqueeze(-1)
        return self.fusion_tail(total_feat) / (std + 1e-7)
