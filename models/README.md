# Models

This directory contains all model implementations used in the InternVL_V2 project.

## Directory Structure

- `classification/`: Models for question classification, including question type identification
- `components/`: Shared model components used across different models
  - `projection_head.py`: Classification head, cross-attention, and response generation components
  - `template_system.py`: Templates for structured response generation
  - `legacy/`: Deprecated components that are no longer actively used
- `vision_language/`: Vision-language model implementations, including InternVL2
  - `internvl2.py`: Core implementation of the InternVL2 multimodal model

## Usage

Each model is implemented as a PyTorch module and can be imported directly. For example:

```python
from models.classification.question_classifier import QuestionClassifier
from models.vision_language.internvl2 import InternVL2MultimodalModel

# Components can be imported directly
from models.components import ClassificationHead, CrossAttention, ResponseGenerator, TemplateSelector
```
