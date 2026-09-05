"""Two-phase (freeze -> unfreeze) training for the timm classifier trainers.

## The bug this exists to prevent

Every classifier trainer trained the whole model — a pretrained backbone plus a
freshly, randomly initialized N-way head — at a single lr=1e-3 from epoch 1.
The head's huge initial gradients flow straight back through the pretrained
backbone at that same high lr, overwriting the ImageNet features that made the
backbone useful in the first place ("unfrozen head shock"). Combined with how
visually similar the classes usually are here (e.g. two cats), the model then
overfits within 10-15 epochs and early-stopping cuts it off with mediocre
accuracy — the backbone never got a chance to fine-tune, it got clobbered.

The fix is the standard two-phase recipe: freeze everything except the head for
a short warmup so the head can settle without disturbing the backbone, then
unfreeze everything and fine-tune the whole network at a much lower lr with
cosine annealing.

Freezing is done via ``model.get_classifier()`` rather than guessing an
attribute name (``classifier``/``fc``/``head`` differ per architecture family —
see ``_backbone.py``'s docstring for the bug that caused), so this works
uniformly across every architecture ``create_classifier_backbone`` can build.
"""

from __future__ import annotations

from typing import Any, Optional


def phase_for_epoch(epoch: int, freeze_epochs: int) -> str:
    """ "frozen" for the warmup epochs, "unfrozen" afterward.

    ``freeze_epochs <= 0`` disables warmup entirely — every epoch is
    "unfrozen", preserving the old single-phase behaviour exactly (see
    ``build_phase_optimizer_and_scheduler``).
    """
    return "frozen" if freeze_epochs > 0 and epoch <= freeze_epochs else "unfrozen"


def set_backbone_trainable(model: Any, trainable: bool) -> None:
    """Freeze/unfreeze every parameter except the classification head."""
    head_param_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        if id(p) not in head_param_ids:
            p.requires_grad = trainable


def build_phase_optimizer_and_scheduler(
    model: Any,
    *,
    phase: str,
    max_epochs: int,
    freeze_epochs: int,
    head_lr: float,
    finetune_lr: float,
    weight_decay: float = 1e-4,
) -> tuple[Any, Optional[Any]]:
    """A fresh (optimizer, scheduler) for `phase`.

    "frozen": only the head's parameters are passed to the optimizer, at
    ``head_lr``. No scheduler — this phase is a short, fixed-lr warmup.

    "unfrozen": every parameter is trainable, with a CosineAnnealingLR over
    exactly the epochs left in the budget (``max_epochs - freeze_epochs``).
    Its lr is ``finetune_lr`` when there was a warmup phase to fine-tune down
    from, or ``head_lr`` when ``freeze_epochs<=0`` disabled warmup entirely —
    that case anneals ``head_lr`` over the full ``max_epochs``, identical to
    the single-phase behaviour this replaces, so ``freeze_epochs=0`` is a
    genuine opt-out rather than an accidental 30x lr drop.
    """
    import torch

    set_backbone_trainable(model, trainable=(phase == "unfrozen"))

    if phase == "frozen":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=head_lr, weight_decay=weight_decay)
        return optimizer, None

    unfrozen_lr = finetune_lr if freeze_epochs > 0 else head_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=unfrozen_lr, weight_decay=weight_decay)
    remaining_epochs = max(1, max_epochs - freeze_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    return optimizer, scheduler
