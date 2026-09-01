"""Tests for the shared crop geometry used by crop classifiers.

``crop_region`` is called from two places that must agree exactly: the trainer
(building the dataset) and every caller that runs the resulting model. If they
frame the subject differently the model still runs and still returns confident
answers — it is just being shown something other than what it learned on. There
is no error to catch, so these assertions are the only thing standing between a
padding change and a quiet accuracy drop.
"""

import numpy as np

from apps.helpers.classifier_utils import DEFAULT_CROP_PADDING, crop_region, preprocess_crop


def _image(h: int = 100, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestCropRegion:
    def test_padding_grows_the_region_beyond_the_box(self):
        """A padded crop must be strictly larger than the bare box — the padding is
        the context the model is trained to use."""
        img = _image()
        bare = crop_region(img, 0.4, 0.4, 0.2, 0.2, padding=0.0)
        padded = crop_region(img, 0.4, 0.4, 0.2, 0.2, padding=0.25)
        assert padded.shape[0] > bare.shape[0]
        assert padded.shape[1] > bare.shape[1]

    def test_padding_is_proportional_to_the_box_not_the_image(self):
        """Two boxes of different sizes get different amounts of padding. Padding a
        fixed number of pixels instead would over-pad small boxes into mostly
        background and under-pad large ones."""
        img = _image(400, 400)
        small = crop_region(img, 0.1, 0.1, 0.1, 0.1, padding=0.5)
        large = crop_region(img, 0.1, 0.1, 0.4, 0.4, padding=0.5)
        small_pad = small.shape[0] - 0.1 * 400
        large_pad = large.shape[0] - 0.4 * 400
        assert large_pad > small_pad

    def test_region_is_clamped_to_the_image(self):
        """A box at the edge pads into nothing rather than wrapping or overflowing."""
        img = _image(100, 200)
        crop = crop_region(img, 0.0, 0.0, 0.2, 0.2, padding=0.5)
        assert crop.shape[0] <= 100 and crop.shape[1] <= 200
        # Top-left box: padding can only extend right/down, so the result is
        # smaller than the same box padded on all four sides in the middle.
        middle = crop_region(img, 0.4, 0.4, 0.2, 0.2, padding=0.5)
        assert crop.shape[0] < middle.shape[0]

    def test_degenerate_box_still_yields_a_usable_region(self):
        """A zero-extent or fully out-of-frame box must not produce an empty array:
        it would reach cv2.resize inside a DataLoader worker and kill the epoch."""
        img = _image()
        for box in [(0.5, 0.5, 0.0, 0.0), (1.0, 1.0, 0.1, 0.1), (0.999, 0.999, 0.5, 0.5)]:
            crop = crop_region(img, *box, padding=0.0)
            assert crop.size > 0, f"empty crop for {box}"
            assert crop.shape[0] >= 1 and crop.shape[1] >= 1

    def test_preprocess_crop_returns_a_network_ready_tensor(self):
        """The whole point of the helper: any box, however awkward, comes back as the
        fixed CHW shape the ONNX session wants."""
        tensor = preprocess_crop(_image(), 0.9, 0.9, 0.5, 0.5, (224, 224), padding=DEFAULT_CROP_PADDING)
        assert tensor.shape == (3, 224, 224)
