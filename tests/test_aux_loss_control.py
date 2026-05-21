import torch
import torch.nn as nn

from starVLA.model.modules.uamvla.aux_heads.base import HeadOutput
from starVLA.model.modules.uamvla.aux_loss_control import AuxDenoisingSuite


class _Head(nn.Module):
    def __init__(self, loss_value: float, valid_ratio: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.loss_value = loss_value
        self.valid_ratio = valid_ratio

    def compute_loss(self, hidden_states, batch, mask):
        raw = torch.tensor(self.loss_value, device=hidden_states.device)
        loss = raw * self.weight
        return HeadOutput(
            loss=loss,
            metrics={"loss_raw": float(self.loss_value), "valid_ratio": self.valid_ratio},
            predictions=None,
        )


def test_aux_suite_applies_static_aux_budget_only():
    heads = nn.ModuleDict({"depth": _Head(4.0, 0.5), "grounding": _Head(2.0, 1.0)})
    suite = AuxDenoisingSuite(
        heads=heads,
        aux_budget=0.25,
    )
    hidden = torch.zeros(2, 5, 3)
    batch = {}
    masks = {
        "depth": torch.tensor([True, False]),
        "grounding": torch.tensor([True, True]),
    }

    action_loss = torch.tensor(2.0)
    aux_loss, metrics = suite(
        action_loss=action_loss,
        hidden_states=hidden,
        batch=batch,
        masks=masks,
        global_step=5,
    )

    # raw aux = 6.0, final aux = aux_budget * raw aux = 1.5
    assert torch.allclose(aux_loss, torch.tensor(1.5))
    assert metrics["aux_total_pre_budget"].item() == 6.0
    assert torch.allclose(metrics["aux_total_post_budget"], torch.tensor(1.5))
    assert "aux_scale_budget" not in metrics
    assert "aux_budget_warmup" not in metrics
    assert "action_loss_ema" not in metrics
    assert "aux_cap" not in metrics
    assert torch.allclose(metrics["depth_loss_contribution_post_budget"], torch.tensor(1.0))
    assert torch.allclose(metrics["grounding_loss_contribution_post_budget"], torch.tensor(0.5))
    assert metrics["depth_valid_ratio"] == 0.5
    assert metrics["grounding_valid_ratio"] == 1.0


def test_aux_suite_disabled_returns_zero_loss():
    suite = AuxDenoisingSuite(heads=nn.ModuleDict({}), enabled=False)
    hidden = torch.zeros(1, 5, 3)
    aux_loss, metrics = suite(
        action_loss=torch.tensor(1.0),
        hidden_states=hidden,
        batch={},
        masks={},
        global_step=100,
    )
    assert aux_loss.item() == 0.0
    assert metrics == {}
