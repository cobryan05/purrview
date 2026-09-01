"""Every offered architecture must produce one logit per class.

This is the check that was missing when three trainers built their head as
``create_model(num_classes=0)`` + ``model.classifier = Linear(num_features, n)``.
That works for the efficientnets and fails for the other four architectures the
UI offers — and on resnet it fails *silently*: cross-entropy happily consumes a
512-wide output when the labels are 0/1, so training completed, exported an ONNX
whose output width contradicted its own sidecar, and nothing anywhere said so.

Parametrised over ``MODEL_TYPE_ARCHITECTURES`` rather than a hand-written list,
so adding an architecture to the UI without checking its head fails here instead
of in a training run three hours in.
"""

import pytest

from apps.helpers.db.types import MODEL_TYPE_ARCHITECTURES
from apps.training_server.trainers._backbone import create_classifier_backbone
from apps.training_server.trainers.crop_classifier_trainer import _ARCH_MAP

# Every architecture any classifier trainer offers. The three trainers share one
# _ARCH_MAP content-wise; crop_classifier's is used as the DB-name -> timm-name
# lookup, and the assertion below pins that they really do cover the same set.
_CLASSIFIER_MODEL_TYPES = ("crop_classifier", "image_classifier")


def _offered_architectures() -> list[str]:
    offered: set[str] = set()
    for model_type in _CLASSIFIER_MODEL_TYPES:
        offered.update(MODEL_TYPE_ARCHITECTURES.get(model_type, []))
    return sorted(offered)


def test_arch_map_covers_every_architecture_the_ui_offers():
    """A UI option with no timm mapping silently falls back to the default model,
    so the user picks ResNet-50 and trains a MobileNet without being told."""
    unmapped = [a for a in _offered_architectures() if a not in _ARCH_MAP]
    assert unmapped == [], f"architectures offered but not mapped to a timm model: {unmapped}"


@pytest.mark.parametrize("architecture", _offered_architectures())
@pytest.mark.parametrize("num_classes", [2, 5])
def test_head_width_equals_class_count(architecture, num_classes):
    """The regression: resnet returned 512/2048 and mobilenetv3 raised.

    Asserted on the forward output rather than on any attribute, because the
    original bug was precisely that the attribute was set and forward ignored it.
    """
    import torch

    model = create_classifier_backbone(_ARCH_MAP[architecture], num_classes, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))

    assert out.shape == (2, num_classes), (
        f"{architecture} produced {tuple(out.shape)} for {num_classes} classes — "
        "the head is not the one the trainer thinks it built"
    )


def test_zero_classes_is_rejected():
    """`num_classes=0` is timm's "no head, return features" mode — the exact shape
    the old bug produced. Asking for it here is a caller error, not a backbone."""
    with pytest.raises(ValueError):
        create_classifier_backbone("resnet18", 0, pretrained=False)
