# minf – Fast Mutual Information in Python

`minf` provides a lightweight, pure‑Python implementation of **mutual information** for 1‑D signals and 2‑D images, following the algorithm used in the MATLAB File Exchange submission *"Fast mutual information of two images or signals"*.

## Features

- Computes mutual information in **nats** using a joint‑histogram approach.
- Works with **signals** (1‑D arrays) and **images** (2‑D arrays).
- Simple API: `mutual_information(a, b, bins=256)`.
- Includes a small test that evaluates MI on a public‑domain **MRI sample image** (`data/brain_mri.png`), if Pillow is available.

## Installation

```bash
# Clone the repo
git clone https://github.com/pantxobestta/minf.git
cd minf

# Install in editable mode (requires Python ≥3.8)
python -m pip install -e .
```

The package has a single dependency: **NumPy**.

```bash
python -c "import numpy; print('NumPy version:', numpy.__version__)"
```

## Quick Start

```python
import numpy as np
from minf.core import mutual_information

# Example 1: identical signals
a = np.array([1, 2, 3, 4])
b = a.copy()
mi = mutual_information(a, b)
print('MI identical signals:', mi)   # → 0.0

# Example 2: two images (NumPy arrays)
# Here we just reuse the same array for demo purposes
img = np.random.rand(64, 64).astype(np.float32)
mi_img = mutual_information(img, img)
print('MI identical images:', mi_img)   # → 0.0
```

If Pillow is installed (`pip install pillow`), the test suite will also load a real MRI sample image located at `data/Brain_MRI.jpg`.

## Running the Tests

```bash
# Install the test extra (optional)
pip install pytest  # or any test runner you prefer
pytest                # discovers and runs the tests
```

Or simply:

```bash
python -m unittest discover -v
```

## Citation

If you use this implementation in research, please cite the original MATLAB File Exchange submission:

> Jose Delpiano (2026). *Fast mutual information of two images or signals* (https://la.mathworks.com/matlabcentral/fileexchange/13289-fast-mutual-information-of-two-images-or-signals), MATLAB Central File Exchange. Retrieved 22 March, 2026.

---

## How the sample image is used

The repository ships a small brain MRI image (`data/Brain_MRI.jpg`). The test `test_mutual_information_with_sample_image` loads this image, adds a tiny amount of noise, and verifies that mutual information is non‑negative. If the `Pillow` library is not available, the test gracefully skips the image‑based part.
