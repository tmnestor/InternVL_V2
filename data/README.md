# Data

This directory contains all data processing modules and dataset definitions.

## Directory Structure

- `datasets/`: Dataset class definitions
  - `classification/`: Datasets for classification tasks
  - `multimodal/`: Datasets for vision-language tasks
- `generators/`: Data generation modules for synthetic data

## Usage

The datasets can be used with PyTorch DataLoader. For example:

```python
from data.datasets.classification.balanced_question_dataset import BalancedQuestionDataset

dataset = BalancedQuestionDataset(data_dir="data/balanced_question_data")
loader = torch.utils.data.DataLoader(dataset, batch_size=16)
```

For generating synthetic data:

```python
from data.generators.receipt_generator import ReceiptGenerator

generator = ReceiptGenerator()
receipt = generator.generate_receipt()
```
