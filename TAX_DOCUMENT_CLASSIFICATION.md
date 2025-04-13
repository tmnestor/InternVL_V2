# Tax Document Classification

## Purpose and Background

This project implements a specialized document classification system designed for the Australian Taxation Office (ATO) to distinguish between tax documents and taxpayer substantiations (receipts).

### Problem Statement

Tax officers at the ATO process large volumes of documents, including:

1. **Official Tax Documents**: Forms, notices, returns, and other official documents
2. **Substantiation Documents**: Receipts and invoices provided by taxpayers to support deduction claims

Manual sorting of these documents is time-consuming and error-prone. An automated system that can accurately classify documents would significantly improve processing efficiency.

### Solution Approach

We've implemented a specialized classifier using the InternVL2 vision-language model that:

1. Classifies images based on receipt count (0, 1, or 2+ receipts)
2. Specifically treats "0 receipts" as tax documents (not just generic non-receipt images)
3. Is trained to understand the visual differences between tax forms and receipts

## Technical Implementation

### Dataset Design and Class 0 Images

A key innovation in our approach is the specific treatment of the "0 receipts" class:

- **Class 0 = Tax Documents**: All images labeled with `receipt_count=0` in the metadata represent Australian Taxation Office (ATO) documents
- **File Pattern**: These tax documents follow the naming convention `receipt_collage_XXXXX.png` (same as receipt images)
- **Examples**: `receipt_collage_00004.png`, `receipt_collage_00005.png`, `receipt_collage_00006.png`, etc.
- **Location**: Stored alongside receipt images in the `datasets/synthetic_receipts/images/` directory
- **Purpose**: This structure creates a more realistic classification task mimicking real-world scenarios

> **Enhanced Tax Document Generation**: The class 0 images are now generated with distinct visual characteristics of official tax forms:
> - Higher whitespace ratio for cleaner appearance
> - Lower edge density with structured layouts
> - Light blue backgrounds common in ATO documents
> - Official-looking headers and footers
> - Formal tables and structured content areas
> - Australian Taxation Office watermarks
> 
> These improvements ensure that tax documents are visually distinct from receipts, allowing the model to better learn the difference between document types.

### Model Architecture

- Uses InternVL2 5-1B vision-language model
- Processes high-resolution images (448x448)
- Features a custom classification head
- Implements three-stage training for optimal performance

### Evaluation Metrics

The model is specifically evaluated on:

- Standard classification metrics (accuracy, F1)
- Confusion matrix with special attention to:
  - False positives (tax documents classified as receipts)
  - False negatives (receipts classified as tax documents)
- Per-class metrics to ensure balanced performance

## Usage for Tax Officers

Tax officers can use this system to:

1. Automatically sort incoming document scans
2. Prioritize processing of substantiations
3. Route documents to appropriate departments
4. Identify potential misclassifications in document batches

## Dataset Organization and Splits

The dataset has been properly split into train/validation/test sets:

- **Training Set**: `datasets/synthetic_receipts/metadata_train.csv` (70% of data)
- **Validation Set**: `datasets/synthetic_receipts/metadata_val.csv` (15% of data)
- **Test Set**: `datasets/synthetic_receipts/metadata_test.csv` (15% of data)

Each split maintains the proper distribution of document classes (0, 1, 2+ receipts), ensuring balanced representation of tax documents and receipts in all evaluation phases.

## Verification and Quality Control

We've implemented verification scripts to assess the dataset:

1. **`inspect_class0_images.py`**: Creates visual samples of class 0 images for manual inspection
2. **`basic_image_check.py`**: Performs basic visual analysis comparing class 0 and receipt images
3. **`verify_tax_documents.py`**: Conducts more comprehensive verification (requires additional dependencies)

Our analysis confirms:
- Class 0 images have fewer distinct colors than receipts (typical of tax documents)
- They may differ in some visual characteristics from stereotypical tax forms
- All expected tax document images are present in the designated paths

## Future Improvements

- Expand class 0 with more varied tax document examples
- Add support for multi-page documents
- Integrate OCR for text-based classification features
- Implement active learning for continuous improvement

## References

- Original concept: [github.com/tmnestor/internvl-receipt-counter](https://github.com/tmnestor/internvl-receipt-counter)
- Australian Taxation Office: [ato.gov.au](https://www.ato.gov.au)