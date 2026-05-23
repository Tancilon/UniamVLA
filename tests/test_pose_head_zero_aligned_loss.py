import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.pose_head import PoseHead


class _FakeQueryReader(nn.Module):
    def __init__(self, num_queries: int) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.scale = nn.Parameter(torch.ones(()))
        self.last_shape = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(hidden_states.shape)
        return hidden_states[:, : self.num_queries, :] * self.scale


class _FakePtsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 1024)
        self.last_shape = None

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(point_cloud.shape)
        return self.proj(point_cloud.mean(dim=1))


class _FakeScoreNet(nn.Module):
    def __init__(self, pose_dim: int, semantic_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1024 + semantic_dim + pose_dim + 1, pose_dim)
        self.calls = 0

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        self.calls += 1
        x = torch.cat(
            [
                data["pts_feat"],
                data["semantic_feat"],
                data["sampled_pose"],
                data["t"],
            ],
            dim=-1,
        )
        return self.proj(x)


def _make_pose_head() -> PoseHead:
    head = PoseHead.__new__(PoseHead)
    nn.Module.__init__(head)
    head.loss_weight = 0.25
    head.repeat_num = 2
    head.sampling_steps = 1
    head.pose_mode = "rot_matrix"
    head.pose_dim = 9
    head.sde_eps = 1e-5
    head.sde_T = 1.0
    head.marginal_prob_fn = lambda pose, t: (
        pose,
        torch.ones(pose.shape[0], 1, device=pose.device, dtype=pose.dtype),
    )
    head.query_reader = _FakeQueryReader(num_queries=2)
    head.semantic_proj = nn.Linear(8, 5)
    head.pts_encoder = _FakePtsEncoder()
    head.score_net = _FakeScoreNet(pose_dim=9, semantic_dim=5)
    return head


def test_pose_empty_mask_runs_zero_aligned_score_forward():
    head = _make_pose_head()
    hidden = torch.randn(2, 4, 4)
    batch = {"input_ids": torch.ones(2, 4, dtype=torch.long)}
    mask = torch.tensor([False, False])

    out = head.compute_loss(hidden, batch, mask=mask)

    assert out.loss.item() == 0.0
    assert out.metrics["pose_loss"] == 0.0
    assert out.metrics["pose_translation_residual_mean"] == 0.0
    assert head.query_reader.last_shape == (2, 4, 4)
    assert head.pts_encoder.last_shape == (2, 1024, 3)
    assert head.score_net.calls == 1
