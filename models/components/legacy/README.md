# Legacy Model Components

This directory contains model components that are no longer actively used in the main workflow of the InternVL_V2 project.

## Deprecated Components

### `detail_extractor.py`
- This component was designed to extract specific details from receipts and tax documents
- It has been replaced by direct feature extraction in the main InternVL2 model
- The functionality is now integrated into the vision-language processing pipeline

### `question_classifier.py`
- This is a duplicated implementation of the question classifier
- The active implementation is now in `models/classification/question_classifier.py`
- This version was an earlier iteration that is no longer referenced in the codebase

## Recommended Alternatives

1. For detail extraction functionality:
   - Use the InternVL2 model's built-in vision processing via `models/vision_language/internvl2.py`

2. For question classification:
   - Use the active implementation in `models/classification/question_classifier.py`