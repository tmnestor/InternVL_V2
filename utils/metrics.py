"""
Evaluation metrics for multimodal vision-language tasks.

Includes metrics for:
- Classification (accuracy, precision, recall, F1)
- Natural language generation (BLEU, ROUGE, perplexity)
"""
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import NLG metrics if available
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge import Rouge
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def compute_classification_metrics(
    predictions: Union[np.ndarray, List, torch.Tensor],
    targets: Union[np.ndarray, List, torch.Tensor],
    average: str = "macro"
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average, zero_division=0)
    recall = recall_score(targets, predictions, average=average, zero_division=0)
    f1 = f1_score(targets, predictions, average=average, zero_division=0)
    
    return {
        "accuracy": accuracy * 100,  # Convert to percentage
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100
    }


def compute_nlg_metrics(
    predictions: List[str],
    references: List[str],
    max_order: int = 4
) -> Dict[str, float]:
    """
    Compute natural language generation metrics.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
        max_order: Maximum n-gram order for BLEU
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Check if we have the required libraries
    if not NLTK_AVAILABLE:
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge import Rouge
            NLTK_AVAILABLE = True
        except ImportError:
            return {"error": "NLTK or Rouge not available. Install with 'pip install nltk rouge'."}
    
    # BLEU score calculation
    if NLTK_AVAILABLE:
        # Initialize smoothing function
        smooth = SmoothingFunction().method1
        
        # Calculate BLEU scores for different n-gram orders
        bleu_scores = []
        weights_list = [
            (1.0,),  # BLEU-1
            (0.5, 0.5),  # BLEU-2
            (0.33, 0.33, 0.33),  # BLEU-3
            (0.25, 0.25, 0.25, 0.25)  # BLEU-4
        ]
        
        # Tokenize texts
        tokenized_predictions = [prediction.split() for prediction in predictions]
        tokenized_references = [reference.split() for reference in references]
        
        # Calculate BLEU scores
        for i, weights in enumerate(weights_list[:max_order]):
            bleu_i = 0.0
            count = 0
            
            # Calculate BLEU for each example
            for pred_tokens, ref_tokens in zip(tokenized_predictions, tokenized_references):
                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    try:
                        # Calculate BLEU with smoothing
                        bleu_i += sentence_bleu([ref_tokens], pred_tokens, 
                                               weights=weights, 
                                               smoothing_function=smooth)
                        count += 1
                    except Exception:
                        # Skip errors in BLEU calculation
                        continue
            
            # Average BLEU score
            if count > 0:
                bleu_i /= count
                metrics[f"bleu{i+1}"] = bleu_i * 100  # Convert to percentage
            else:
                metrics[f"bleu{i+1}"] = 0.0
        
        # Also include BLEU-4 as the default BLEU metric
        metrics["bleu"] = metrics.get("bleu4", 0.0)
        
        # ROUGE scores
        try:
            rouge = Rouge()
            # Filter out empty predictions/references
            valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                           if len(p) > 0 and len(r) > 0]
            
            if valid_pairs:
                valid_preds, valid_refs = zip(*valid_pairs)
                rouge_scores = rouge.get_scores(valid_preds, valid_refs, avg=True)
                
                # Extract and store ROUGE scores
                metrics["rouge1_f"] = rouge_scores["rouge-1"]["f"] * 100
                metrics["rouge2_f"] = rouge_scores["rouge-2"]["f"] * 100
                metrics["rougeL_f"] = rouge_scores["rouge-l"]["f"] * 100
            else:
                # No valid pairs
                metrics["rouge1_f"] = 0.0
                metrics["rouge2_f"] = 0.0
                metrics["rougeL_f"] = 0.0
        except Exception as e:
            metrics["rouge_error"] = str(e)
    
    return metrics


def perplexity_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate perplexity from language model logits.
    
    Args:
        logits: Model logits [batch_size, sequence_length, vocab_size]
        labels: Target token IDs [batch_size, sequence_length]
        ignore_index: Token value to ignore in calculation
        
    Returns:
        Perplexity score
    """
    # Move tensors to CPU for calculation
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    
    # Reshape logits for loss calculation
    batch_size, seq_len, vocab_size = logits.size()
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Calculate cross-entropy loss
    loss = loss_fn(logits_flat, labels_flat)
    
    # Perplexity is exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity