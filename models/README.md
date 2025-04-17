# Models

This directory contains all model implementations used in the InternVL_V2 project.

## Directory Structure

- `classification/`: Models for question classification, including question type identification
- `components/`: Shared model components used across different models
- `vision_language/`: Vision-language model implementations, including InternVL

## Usage

Each model is implemented as a PyTorch module and can be imported directly. For example:

```python
from models.classification.question_classifier import QuestionClassifier
from models.vision_language.internvl2 import InternVLModel
```
