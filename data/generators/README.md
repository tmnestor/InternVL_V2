# Ab Initio Data Generation for InternVL2 Receipt Counter

This module provides high-quality, first-principles-based (ab initio) implementations for generating synthetic receipts and tax documents for the InternVL2 Receipt Counter project.

## Approach

Instead of using conversion-based approaches, these generators create realistic documents from scratch, with complete control over styling, content, and variation. This approach ensures:

1. Higher visual fidelity and realism
2. Greater control over document characteristics
3. Better visual distinction between receipts and tax documents
4. More realistic variations in layout, format, and content

## Components

### Receipt Generator

The `receipt_generator.py` module creates realistic Australian receipts with multiple styles:

- **Standard receipts**: Traditional format with store header, items, and totals
- **Detailed receipts**: More comprehensive layout with tax invoice format
- **Fancy receipts**: Stylized receipts with decorative elements
- **Minimal receipts**: Clean, simple layout focused on essential information

All receipts feature:
- Realistic Australian store names and locations
- Proper Australian address and phone number formatting
- Authentic tax and pricing (including 10% GST)
- Varied payment methods (EFTPOS, credit cards, etc.)
- Multiple layout variations and visual styles

### Tax Document Generator

The `tax_document_generator.py` module creates Australian government documents that are visually distinct from receipts, including:

- **ATO documents**: Tax assessments, payment summaries, etc.
- **Services Australia documents**: Medicare, Centrelink, etc.
- **Other government forms**: Various official Australian government documents

Key features:
- Official ATO styling with blue headers and color schemes
- Proper document structure with headers, reference numbers
- Realistic financial data and tax calculations
- Government watermarks and official styling
- Multiple layout variations and document types

## Usage

```python
# Generate a receipt
from data.data_generators_new.receipt_generator import create_receipt
receipt_image = create_receipt(image_size=2048)

# Generate a tax document
from data.data_generators_new.tax_document_generator import create_tax_document
tax_document = create_tax_document(image_size=2048)

# Save images
receipt_image.save("receipt_example.png")
tax_document.save("tax_document_example.png")
```

## Data Generation

For full dataset generation including image collages with multiple receipts, use the script:

```bash
PYTHONPATH=. python scripts/data_generation/generate_data.py --output_dir datasets/synthetic_receipts --num_collages 300
```

For generating multimodal datasets with question-answer pairs:

```bash
PYTHONPATH=. python scripts/data_generation/generate_multimodal_data.py --output_dir datasets/multimodal --num_samples 300
```

For unified datasets that combine receipt counting and QA tasks:

```bash
PYTHONPATH=. python scripts/data_generation/generate_unified_data.py --output_dir data/unified_dataset --num_samples 300
```

## Comparing Implementations (Legacy)

To visually compare the original and ab initio implementations (legacy tool):

```bash
PYTHONPATH=. python scripts/data_generation/legacy/visual_compare.py --output comparison.png --samples 2
```

This will generate a side-by-side comparison showing the differences between implementations.

## Key Advantages

1. **Visual Distinction**: Better separation between receipt and tax document classes
2. **Realism**: More authentic representations of Australian documents
3. **Variety**: Greater variation in document styles and formats
4. **Control**: Complete control over all document parameters
5. **Australian Context**: Properly localized for the Australian context