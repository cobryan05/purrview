"""Shared classifier-backbone construction for the timm-based trainers.

One function, because the obvious hand-rolled alternative is wrong on most of the
architectures this app offers and wrong *quietly* on some of them.

## The bug this exists to prevent

Every classifier trainer used to build its head like this::

    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.classifier = nn.Linear(model.num_features, n)

That reads as "take the backbone, bolt on my own head", and it works for exactly
one of the three families in ``MODEL_TYPE_ARCHITECTURES``. Measured:

===========================  ==========================================
efficientnet_b0 / b2 / b4    correct — head really is called ``classifier``
resnet_18 / resnet_50        head is ``fc``. The assignment lands on an unused
                             attribute and ``forward`` returns 512/2048 pooled
                             features. **Silent**: cross-entropy accepts it,
                             because labels 0/1 are valid indices into a
                             512-way output, so training completes, exports an
                             ONNX with the wrong output width, and the sidecar
                             claims a class count the graph does not have.
mobilenet_v3_small / large   ``num_features`` (576) is not the head's input
                             width (1024), so ``forward`` raises a shape error.
===========================  ==========================================

``num_classes=n`` hands the problem to timm, which knows each family's head by
construction. There is no architecture-specific branching to keep in sync here,
and adding one would be reintroducing the same class of bug.
"""

from __future__ import annotations

from typing import Any


def create_classifier_backbone(timm_model: str, num_classes: int, pretrained: bool = True) -> Any:
    """A pretrained timm backbone with an ``num_classes``-wide head.

    Args:
        timm_model:  timm's own model name (already mapped from the DB architecture).
        num_classes: Output units — one per class the trainer will predict.
        pretrained:  False in tests, to skip the weight download.

    The returned model's forward output is ``[batch, num_classes]`` for every
    architecture this app offers; ``tests_training/test_backbone.py`` asserts that
    across all of them, which is the check that would have caught the original bug.
    """
    import timm

    if num_classes < 1:
        raise ValueError(f"num_classes must be at least 1, got {num_classes}")
    return timm.create_model(timm_model, pretrained=pretrained, num_classes=num_classes)
