# Configuration

This directory contains configuration files for various components of the InternVL_V2 project.

## Directory Structure

- `classifier/`: Configuration for question classifier
- `data_generation/`: Configuration for data generation
- `model/`: Configuration for model training

## Usage

Configuration files are in YAML format and can be loaded using the YAML module:

```python
import yaml

with open('config/model/multimodal_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```
