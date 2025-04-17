# Question Classification Improvement Plan

## Current Status

The question classifier currently shows the following performance:

- Overall test accuracy: 37.89%
- Class-wise accuracy:
  - DOCUMENT_TYPE (Class 0): 0%
  - COUNTING (Class 1): 65.38%
  - DETAIL_EXTRACTION (Class 2): 34.88%
  - PAYMENT_INFO (Class 3): 96.15%
  - TAX_INFO (Class 4): 0%

The model is heavily biased toward predicting PAYMENT_INFO, which is causing poor performance on other classes. While we successfully expanded the DETAIL_EXTRACTION examples, creating a more comprehensive and balanced dataset is necessary.

## Proposed Improvements

### 1. Dataset Balancing

- Create equal representation of all 5 classes in the dataset
- Add more diverse and challenging examples for each class to ensure better generalization
- Include adversarial examples that look similar across classes but have subtle differences
- Use data augmentation techniques like word substitution and sentence restructuring

### 2. Model Architecture Enhancements

- Use a more sophisticated pre-trained language model (RoBERTa or BART)
- Add attention mechanisms specifically designed for question classification
- Implement a hierarchical classification approach for better performance
- Experiment with sentence-pair encoding for better context understanding

### 3. Training Strategy Adjustments

- Implement stronger class weighting (adjust current weights from [1.15, 1.15, 0.39, 1.15, 1.15] to [2.0, 1.0, 0.5, 0.2, 2.0])
- Use a curriculum learning approach (start with easy examples, then gradually introduce harder ones)
- Implement mixup or other advanced regularization techniques
- Use a multi-stage training process with initial pretraining on a balanced subset

### 4. Loss Function Improvements

- Use focal loss instead of weighted cross-entropy to focus on hard examples
- Experiment with hierarchical softmax for better class separation
- Implement confidence calibration to prevent overconfident predictions
- Add contrastive learning components to better separate embeddings of different classes

### 5. Evaluation and Monitoring

- Track per-class F1 scores in addition to accuracy
- Perform error analysis on misclassified examples to identify patterns
- Use confusion matrices to visualize class transitions
- Implement early stopping based on worst-class performance rather than overall metrics

## Implementation Timeline

1. Dataset Enhancements (1-2 days)
   - Expand all classes to have equal representation
   - Add more challenging examples for DOCUMENT_TYPE and TAX_INFO classes
   - Implement data augmentation techniques

2. Model Architecture Updates (2-3 days)
   - Experiment with alternative pre-trained models
   - Implement attention mechanism enhancements
   - Add regularization techniques

3. Training Improvements (1-2 days)
   - Implement focal loss
   - Set up curriculum learning
   - Tune hyperparameters for better convergence

4. Evaluation and Refinement (1-2 days)
   - Perform thorough error analysis
   - Implement comprehensive evaluation metrics
   - Refine model based on findings

## Expected Outcome

With these improvements, we expect to achieve:

- Overall classification accuracy of at least 80%
- Minimum per-class accuracy of 70%
- Better generalization to unseen and complex questions
- More balanced performance across all question types

This enhanced question classifier will significantly improve the template selection and detail extraction stages of the multimodal system, leading to more accurate and context-appropriate responses.