"""Test that setup_optimizer_and_scheduler dispatches to model.get_lr_groups when present."""
from unittest.mock import MagicMock

import pytest
import torch.nn as nn


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def get_lr_groups(self, lr_cfg):
        return [{"name": "linear", "params": list(self.linear.parameters()), "lr": 1e-3}]


def test_setup_optimizer_uses_get_lr_groups_when_present():
    pytest.importorskip("wandb", reason="wandb not installed; runs on remote")
    pytest.importorskip("accelerate", reason="accelerate not installed; runs on remote")

    from starVLA.training.train_starvla import setup_optimizer_and_scheduler

    cfg = MagicMock()
    cfg.trainer.learning_rate.base = 1e-4
    cfg.trainer.optimizer.betas = [0.9, 0.95]
    cfg.trainer.optimizer.eps = 1e-8
    cfg.trainer.optimizer.weight_decay = 1e-8
    cfg.trainer.lr_scheduler_type = "cosine"
    cfg.trainer.num_warmup_steps = 0
    cfg.trainer.max_train_steps = 100
    cfg.trainer.scheduler_specific_kwargs = {}

    model = _FakeModel()
    spy = MagicMock(side_effect=model.get_lr_groups)
    model.get_lr_groups = spy
    opt, _ = setup_optimizer_and_scheduler(model, cfg)
    spy.assert_called_once()
