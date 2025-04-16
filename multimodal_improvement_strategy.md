# InternVL_V2 Multimodal Generation Improvement Strategy

This document outlines the strategy to enhance the multimodal text generation capabilities of the InternVL_V2 model, particularly focusing on improving the quality and relevance of generated responses to questions about receipts and tax documents.

## 1. Fine-tune the Language Generation Component

### Current Limitations
- The language generator currently produces placeholder content ("...")
- Text is being generated through template-based responses rather than actual generation
- No context-specific details from the documents are being incorporated

### Implementation Plan

#### 1.1 Architecture Improvements
- **Replace ResponseGenerator**: Implement a decoder-only architecture based on a smaller LLM (e.g., OPT-350M or Llama-2-1B)
- **Cross-Modal Conditioning**: Enhance cross-attention mechanism to better integrate visual features with text generation
- **Multi-Task Heads**: Create separate prediction heads for different response types (classification, counting, extraction)

#### 1.2 Training Strategy
- **Two-Stage Training**:
  - Stage 1: Train vision encoder and classification head
  - Stage 2: Freeze vision components and train text generation with teacher forcing
- **Mixed Precision Training**: Use tf32/bfloat16 for stable training with lower memory requirements
- **Gradient Accumulation**: Use smaller batch sizes with gradient accumulation to fit in memory

#### 1.3 Loss Function Improvements
- **Weighted Loss Components**:
  ```python
  # Example implementation
  class MultimodalLoss(nn.Module):
      def __init__(self, classification_weight=1.0, generation_weight=1.0):
          super().__init__()
          self.classification_weight = classification_weight
          self.generation_weight = generation_weight
          self.classification_loss = nn.CrossEntropyLoss()
          self.generation_loss = nn.CrossEntropyLoss(ignore_index=-100)
          
      def forward(self, outputs, targets):
          class_loss = self.classification_loss(outputs["logits"], targets["classification_labels"])
          gen_loss = self.generation_loss(
              outputs["response_logits"].view(-1, outputs["response_logits"].size(-1)), 
              targets["labels"].view(-1)
          )
          return self.classification_weight * class_loss + self.generation_weight * gen_loss
  ```

- **Dynamic Loss Weighting**: Adjust weights based on training progress
- **Contrastive Learning**: Add contrastive component to improve separation between different answer types

## 2. Expand the Template System

### Current Limitations
- Templates are too simplistic (only distinguishing tax documents vs. receipts)
- No support for detailed question answering about document content
- No handling of question variations

### Implementation Plan

#### 2.1 Template Categories
Create a hierarchical template system with the following major categories:

| Question Type | Template Category | Example Templates |
|---------------|-------------------|-------------------|
| Document Type | `TYPE` | "This is a {document_type} from {issuer}." |
| Counting | `COUNT` | "There are {count} receipts in this image." |
| Existence | `EXISTENCE` | "Yes, there {is/are} {count} receipt(s) visible." |
| Detail Extraction | `DETAIL` | "The receipt shows a purchase from {store_name} on {date}." |
| Payment Method | `PAYMENT` | "Payment was made using {payment_method}." |
| Total Amount | `AMOUNT` | "The total amount on the receipt is ${amount}." |

#### 2.2 Template Selection Logic
```python
def select_template(question_type, document_class, extracted_details):
    """
    Select the appropriate template based on question type and document details.
    
    Args:
        question_type: Classified question type (TYPE, COUNT, etc.)
        document_class: Classified document class (0=tax_doc, 1-5=receipts)
        extracted_details: Dictionary of extracted document details
        
    Returns:
        Filled template string
    """
    templates = TEMPLATE_REGISTRY[question_type]
    
    # Select appropriate template based on document class
    if question_type == "TYPE":
        if document_class == 0:
            return templates["tax_document"].format(
                issuer="Australian Taxation Office"
            )
        else:
            return templates["receipt"].format(
                count=document_class.item()
            )
    
    # Handle other question types with extracted details
    elif question_type == "DETAIL":
        if document_class == 0:
            return templates["tax_document_detail"].format(**extracted_details)
        else:
            return templates["receipt_detail"].format(**extracted_details)
    
    # Add more conditional logic for other template types
    
    # Fallback template
    return templates["default"].format(
        document_type="tax document" if document_class == 0 else "receipt",
        count=document_class.item() if document_class > 0 else 0
    )
```

#### 2.3 Template Dataset Expansion
Create an expanded dataset with:
- More diverse question phrasings
- Multiple reference answers for the same question
- Varied detail extraction targets
- Domain-specific terminology for tax documents and receipts

## 3. Implement Question Classifier

### Current Limitations
- No question understanding capability
- Same response regardless of question type
- Can't distinguish between different information needs

### Implementation Plan

#### 3.1 Question Taxonomy
Define a comprehensive taxonomy of question types:

1. **Document Classification**
   - "Is this a receipt or tax document?"
   - "What type of document is shown?"

2. **Receipt Counting**
   - "How many receipts are in this image?"
   - "Count the number of receipts visible."

3. **Content Extraction**
   - "What store is this receipt from?"
   - "What is the date on this receipt?"
   - "What items were purchased?"
   - "What was the total amount?"

4. **Payment Information**
   - "How was this purchase paid for?"
   - "What payment method was used?"

5. **Tax Document Information**
   - "What tax form is this?"
   - "What tax year does this document cover?"
   - "What is the ABN on this document?"

#### 3.2 Question Classifier Implementation

```python
class QuestionClassifier(nn.Module):
    def __init__(self, tokenizer, hidden_size=768, num_classes=5):
        super().__init__()
        # Use a pre-trained encoder
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//2, num_classes)
        )
        self.tokenizer = tokenizer
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(pooled_output)
        return logits
    
    def predict_question_type(self, question):
        inputs = self.tokenizer(
            question, 
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self(inputs.input_ids, inputs.attention_mask)
            pred = torch.argmax(logits, dim=1).item()
        return QUESTION_CLASSES[pred]
```

#### 3.3 Training Data for Question Classifier
Create a labeled dataset of questions with their types:

```python
QUESTION_DATASET = [
    {"question": "Is this a receipt?", "type": "DOCUMENT_TYPE"},
    {"question": "What kind of document is this?", "type": "DOCUMENT_TYPE"},
    {"question": "How many receipts are in this image?", "type": "COUNTING"},
    {"question": "Count the number of receipts.", "type": "COUNTING"},
    {"question": "What store is this receipt from?", "type": "DETAIL_EXTRACTION"},
    {"question": "What is the date on this receipt?", "type": "DETAIL_EXTRACTION"},
    {"question": "How was this purchase paid for?", "type": "PAYMENT_INFO"},
    {"question": "What payment method was used?", "type": "PAYMENT_INFO"},
    {"question": "What tax form is this?", "type": "TAX_INFO"},
    {"question": "What is the ABN on this document?", "type": "TAX_INFO"},
    # Add more examples covering variations in phrasing
]
```

#### 3.4 Integration with Response Generation
Update the `generate_response` method:

```python
def generate_response(self, pixel_values, text_input_ids, attention_mask=None):
    # Get question text
    question = self.tokenizer.decode(text_input_ids[0], skip_special_tokens=True)
    
    # Classify question type
    question_type = self.question_classifier.predict_question_type(question)
    
    # Generate embeddings and get document class
    outputs = self.forward(pixel_values, text_input_ids, attention_mask)
    _, predicted_class = outputs["logits"].max(1)
    
    # Extract document details based on question type
    if question_type in ["DETAIL_EXTRACTION", "PAYMENT_INFO", "TAX_INFO"]:
        extracted_details = self.detail_extractor(pixel_values, question_type)
    else:
        extracted_details = {}
    
    # Select and fill template
    response = self.template_selector.select_template(
        question_type, 
        predicted_class, 
        extracted_details
    )
    
    return response
```

## 4. Implementation Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Implement question classifier | 1 week | - |
| 1 | Expand template system | 1 week | - |
| 2 | Create labeled question dataset | 2 weeks | - |
| 2 | Train question classifier | 1 week | Phases 1 & 2 |
| 3 | Architecture improvements | 2 weeks | - |
| 3 | Implement improved loss functions | 1 week | Phase 3 start |
| 4 | Training data preparation | 2 weeks | - |
| 4 | Two-stage training | 3 weeks | Phases 1-3 |
| 5 | Evaluation and refinement | 2 weeks | Phase 4 |

## 5. Evaluation Metrics

The improved system will be evaluated on:

1. **Classification Accuracy**: Document type and receipt count accuracy
2. **Response Quality**:
   - BLEU score
   - Token overlap
   - Exact match percentage
   - ROUGE-L score
3. **Question Understanding**:
   - Question classification accuracy
   - Answer relevance rating
4. **Detail Extraction**:
   - Named entity recognition accuracy
   - Value extraction accuracy (dates, amounts, ABNs, etc.)

## 6. Next Steps

1. Create prototype question classifier
2. Expand template database with more variations
3. Collect and label additional training data for question types
4. Implement improved response generator architecture
5. Set up evaluation pipeline for detailed metrics