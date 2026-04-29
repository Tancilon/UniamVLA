import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.uamvla.aux_heads.base import AuxHead, HeadOutput


class ActionHead(AuxHead):
    """Action prediction head using the backbone's lm_head.

    Computes cross-entropy loss on action token positions only.
    """

    def __init__(
        self,
        hidden_size: int,
        lm_head: nn.Linear,
        action_token_begin_id: int,
        loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.lm_head = lm_head
        self.action_token_begin_id = action_token_begin_id
        self.loss_weight = loss_weight

    def compute_loss(
        self, hidden_states: torch.Tensor, batch: dict, mask: torch.Tensor,
    ) -> HeadOutput:
        if not mask.any():
            return HeadOutput(loss=self.get_dummy_loss(), metrics={}, predictions=None)

        labels = batch["labels"]
        # Shift: position t's hidden state predicts token t+1.
        h_shift = hidden_states[:, :-1, :]        # [B, L-1, H]
        target = labels[:, 1:]                    # [B, L-1]

        # Gather only positions with a real action label before running lm_head.
        # Running lm_head on the full [B, L-1] would allocate
        # [B*(L-1), vocab] ≈ 1.94 GiB on Qwen3-8B (vocab ≈152k), plus another
        # same-shape buffer inside cross_entropy's log_softmax. Masking first
        # collapses it to [N_valid, vocab], which is orders of magnitude
        # smaller since only the action-token positions contribute. Math is
        # identical to the previous ignore_index=-100 path because
        # F.cross_entropy(reduction="mean") averages over unmasked positions,
        # and we are passing exactly those positions.
        valid = target != -100                    # [B, L-1]
        if not valid.any():
            return HeadOutput(loss=self.get_dummy_loss(), metrics={}, predictions=None)

        h_valid = h_shift[valid]                  # [N_valid, H]
        target_valid = target[valid]              # [N_valid]
        logits = self.lm_head(h_valid)            # [N_valid, vocab]
        loss = F.cross_entropy(logits, target_valid)

        if not torch.isfinite(loss):
            return HeadOutput(
                loss=self.get_dummy_loss(),
                metrics={"action_loss": float("nan")},
                predictions=None,
            )

        return HeadOutput(
            loss=self.loss_weight * loss,
            metrics={"action_loss": loss.detach().item()},
            predictions=None,
        )

    def predict(
        self, hidden_states: torch.Tensor, batch: dict,
    ) -> HeadOutput:
        logits = self.lm_head(hidden_states)
        pred_ids = logits.argmax(dim=-1)
        return HeadOutput(
            loss=None,
            metrics={},
            predictions={"token_ids": pred_ids},
        )
