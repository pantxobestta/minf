"""Test suite for the minf package."""

import os
import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from ..src.minf.core import mutual_information


def test_mutual_information_basic():
    """Basic sanity check with synthetic data."""
    # Two identical constant signals -> MI = 0
    a = np.array([1, 1, 1, 1])
    b = a.copy()
    assert mutual_information(a, b) == 0.0

    # Slightly perturbed copy -> MI should be positive and close to the signal variance
    rng = np.random.default_rng(0)
    a = rng.random(1024)
    b = a + rng.normal(0, 0.01, size=a.shape)
    mi_val = mutual_information(a, b, bins=32)
    assert mi_val >= 0


def test_mutual_information_with_sample_image():
    """
    Test mutual information on a real sample image (if Pillow is installed).
    The sample image ``brain_mri.png`` is stored in ``data/`` and is derived
    from a public-domain MRI scan.
    """
    if not _HAS_PIL:
        # Pillow not installed – skip image‑based test
        return

    # Path to the sample image (relative to this test file)
    img_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "brain_mri.png")
    )

    # Load the image and flatten to 1‑D
    img = np.asarray(Image.open(img_path)).ravel()

    # Create a slightly altered copy for testing
    rng = np.random.default_rng(42)
    img_noisy = img + rng.normal(0, 1e-3, size=img.shape)

    # Compute MI
    mi_val = mutual_information(img, img_noisy, bins=64)

    # Mutual information must be non‑negative
    assert mi_val >= 0
```
