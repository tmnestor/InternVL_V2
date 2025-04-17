# Utilities

This directory contains utility functions used across the InternVL_V2 project.

## Available Utilities

- `device.py`: Functions for device management (CPU/GPU)
- `focal_loss.py`: Implementation of focal loss for classification
- `logging.py`: Logging utilities
- `metrics.py`: Evaluation metrics
- `reproducibility.py`: Functions for ensuring reproducible results

## Usage

Utilities can be imported directly. For example:

```python
from utils.device import get_device
from utils.focal_loss import FocalLoss

device = get_device()
loss_fn = FocalLoss(gamma=2.0)
```
