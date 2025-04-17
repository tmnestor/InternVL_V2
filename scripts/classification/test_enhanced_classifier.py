"""
Test script for evaluating the enhanced question classifier.

This script loads the enhanced classifier model and evaluates it on custom
test questions, providing detailed analysis of its performance.
"""
import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from models.classification.question_classifier import QuestionClassifier
from utils.device import get_device


def create_test_questions() -> List[Dict[str, Any]]:
    """
    Create a comprehensive set of test questions covering all categories.
    
    Returns:
        List of question dictionaries with text and expected type
    """
    test_questions = [
        # Document type questions
        {"text": "Is this a receipt?", "type": "DOCUMENT_TYPE"},
        {"text": "What kind of document is this?", "type": "DOCUMENT_TYPE"},
        {"text": "Is this a tax document?", "type": "DOCUMENT_TYPE"},
        {"text": "Can you identify this document?", "type": "DOCUMENT_TYPE"},
        {"text": "Does this look like a receipt to you?", "type": "DOCUMENT_TYPE"},
        {"text": "Is this document from the ATO?", "type": "DOCUMENT_TYPE"},
        {"text": "What category of document is shown?", "type": "DOCUMENT_TYPE"},
        {"text": "Quickly identify whether this is a receipt or tax document.", "type": "DOCUMENT_TYPE"},
        {"text": "Is this a docket from a shop?", "type": "DOCUMENT_TYPE"},
        {"text": "What document shows information about my payment?", "type": "DOCUMENT_TYPE"},
        
        # Counting questions
        {"text": "How many receipts are in this image?", "type": "COUNTING"},
        {"text": "Count the number of receipts.", "type": "COUNTING"},
        {"text": "Are there multiple receipts here?", "type": "COUNTING"},
        {"text": "How many receipts do you see?", "type": "COUNTING"},
        {"text": "Tell me how many receipts are shown.", "type": "COUNTING"},
        {"text": "Is there more than one receipt?", "type": "COUNTING"},
        {"text": "Could you count these receipts for me?", "type": "COUNTING"},
        {"text": "Carefully count how many separate receipts appear in this image.", "type": "COUNTING"},
        {"text": "How many tax documents are in this image?", "type": "COUNTING"},
        {"text": "Count the number of documents with payment info.", "type": "COUNTING"},
        
        # Detail extraction questions
        {"text": "What store is this receipt from?", "type": "DETAIL_EXTRACTION"},
        {"text": "What is the date on this receipt?", "type": "DETAIL_EXTRACTION"},
        {"text": "What items were purchased?", "type": "DETAIL_EXTRACTION"},
        {"text": "When was this purchase made?", "type": "DETAIL_EXTRACTION"},
        {"text": "What time was this purchase made?", "type": "DETAIL_EXTRACTION"},
        {"text": "What's the GST amount on this receipt?", "type": "DETAIL_EXTRACTION"},
        {"text": "What's the subtotal before tax?", "type": "DETAIL_EXTRACTION"},
        {"text": "What's the cashier's name on the receipt?", "type": "DETAIL_EXTRACTION"},
        {"text": "What's the store's address on the receipt?", "type": "DETAIL_EXTRACTION"},
        {"text": "Which items in the docket have the highest price?", "type": "DETAIL_EXTRACTION"},
        
        # Payment information questions
        {"text": "How was this purchase paid for?", "type": "PAYMENT_INFO"},
        {"text": "What payment method was used?", "type": "PAYMENT_INFO"},
        {"text": "Was this paid by credit card?", "type": "PAYMENT_INFO"},
        {"text": "Did they pay with cash or card?", "type": "PAYMENT_INFO"},
        {"text": "What form of payment was used?", "type": "PAYMENT_INFO"},
        {"text": "Was this paid in cash?", "type": "PAYMENT_INFO"},
        {"text": "Please check what payment method was used.", "type": "PAYMENT_INFO"},
        {"text": "Briefly explain how this purchase was paid for based on the receipt.", "type": "PAYMENT_INFO"},
        {"text": "Was the payment processed successfully?", "type": "PAYMENT_INFO"},
        {"text": "Did they tap or insert their card to pay?", "type": "PAYMENT_INFO"},
        
        # Tax information questions
        {"text": "What tax form is this?", "type": "TAX_INFO"},
        {"text": "What is the ABN on this document?", "type": "TAX_INFO"},
        {"text": "What tax year does this document cover?", "type": "TAX_INFO"},
        {"text": "Is this an official ATO document?", "type": "TAX_INFO"},
        {"text": "What is the tax file number?", "type": "TAX_INFO"},
        {"text": "What's the TFN shown on this document?", "type": "TAX_INFO"},
        {"text": "Precisely identify which tax year this ATO document pertains to.", "type": "TAX_INFO"},
        {"text": "What's my taxable income according to this document?", "type": "TAX_INFO"},
        {"text": "Is this a HECS/HELP statement?", "type": "TAX_INFO"},
        {"text": "What tax details can you extract from this?", "type": "TAX_INFO"},
        
        # Challenging examples (potentially confusing between categories)
        {"text": "What does this tax document say about payment methods?", "type": "TAX_INFO"},
        {"text": "How many items were purchased according to this receipt?", "type": "DETAIL_EXTRACTION"},
        {"text": "Can you identify the payment on this tax form?", "type": "TAX_INFO"},
        {"text": "Count the receipts that show payment methods.", "type": "COUNTING"},
        {"text": "Is this a document that lists items purchased?", "type": "DOCUMENT_TYPE"},
        {"text": "Can you tell me what kind of receipt shows the payment method?", "type": "DOCUMENT_TYPE"},
        {"text": "How many payment methods are listed on this receipt?", "type": "COUNTING"},
        {"text": "What details about tax are shown on this receipt?", "type": "DETAIL_EXTRACTION"},
        {"text": "Count the number of tax items in this document.", "type": "COUNTING"},
        {"text": "Extract all payment-related information from this document.", "type": "PAYMENT_INFO"},
    ]
    
    return test_questions


def test_question_classifier(
    model_path: str,
    test_questions: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "outputs/enhanced_classifier_test",
    device: Optional[torch.device] = None,
    tokenizer_name: str = "distilbert-base-uncased"
) -> Dict[str, Any]:
    """
    Test the enhanced question classifier on custom examples.
    
    Args:
        model_path: Path to the trained model
        test_questions: List of test questions with expected types
        output_dir: Directory to save test results
        device: Device to run inference on
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        Dictionary with test results
    """
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "test_results.log"))
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Get device
    if device is None:
        device = get_device()
    logger.info(f"Using device: {device}")
    
    # Use provided test questions or create default ones
    if test_questions is None:
        test_questions = create_test_questions()
        logger.info(f"Created {len(test_questions)} default test questions")
    else:
        logger.info(f"Using {len(test_questions)} provided test questions")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        logger.info("Falling back to distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Define question type mapping
    question_types = {
        "DOCUMENT_TYPE": 0,
        "COUNTING": 1,
        "DETAIL_EXTRACTION": 2,
        "PAYMENT_INFO": 3,
        "TAX_INFO": 4,
    }
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    try:
        # Try to load the entire saved dictionary
        saved_dict = torch.load(model_path, map_location=device)
        
        # Check if it's a state dict directly or part of a larger dict
        if isinstance(saved_dict, dict) and "model_state_dict" in saved_dict:
            model_state_dict = saved_dict["model_state_dict"]
            
            # Load question types if available
            if "question_types" in saved_dict:
                question_types = saved_dict["question_types"]
                logger.info(f"Loaded question types from model: {question_types}")
        else:
            model_state_dict = saved_dict
    except Exception as e:
        logger.error(f"Error loading saved model dict: {e}")
        return {"error": str(e)}
    
    # Create model
    model = QuestionClassifier(
        model_name=tokenizer_name,
        hidden_size=768,
        num_classes=len(question_types),
        device=device,
        use_custom_path=False
    )
    
    # Load state dict
    try:
        model.load_state_dict(model_state_dict)
        logger.info("Successfully loaded model state")
    except Exception as e:
        logger.error(f"Error loading model state dict: {e}")
        return {"error": str(e)}
    
    # Set model to evaluation mode
    model.eval()
    
    # Reverse the question types mapping for output
    id_to_type = {v: k for k, v in question_types.items()}
    
    # Run inference on test questions
    results = []
    predictions = []
    true_labels = []
    
    logger.info("Running inference on test questions...")
    print("\n" + "="*100)
    print(f"{'QUESTION':<50} {'PREDICTED':<15} {'EXPECTED':<15} {'CORRECT':<10}")
    print("="*100)
    
    with torch.no_grad():
        for question in test_questions:
            # Get question text and expected type
            question_text = question["text"]
            expected_type = question["type"]
            expected_id = question_types[expected_type]
            
            # Tokenize the question
            inputs = tokenizer(
                question_text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            # Get model prediction
            logits = model(inputs.input_ids, inputs.attention_mask)
            probabilities = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(logits, dim=1).item()
            pred_type = id_to_type[pred_id]
            
            # Check if prediction is correct
            is_correct = (pred_type == expected_type)
            
            # Store results
            results.append({
                "question": question_text,
                "expected_type": expected_type,
                "expected_id": expected_id,
                "predicted_type": pred_type,
                "predicted_id": pred_id,
                "probabilities": probabilities.cpu().numpy().tolist(),
                "is_correct": is_correct
            })
            
            # Track predictions and true labels for metrics
            predictions.append(pred_id)
            true_labels.append(expected_id)
            
            # Print result with success indicator
            status = "✓" if is_correct else "✗"
            print(f"{question_text[:47] + '...' if len(question_text) > 50 else question_text:<50} {pred_type:<15} {expected_type:<15} {status:<10}")
    
    print("="*100)
    
    # Calculate overall accuracy
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    logger.info(f"Overall accuracy: {accuracy:.4f} ({sum(1 for r in results if r['is_correct'])}/{len(results)})")
    
    # Calculate per-class metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Log per-class metrics
    logger.info("\nPer-class metrics:")
    print("\nPer-class metrics:")
    print(f"{'CLASS':<20} {'PRECISION':<10} {'RECALL':<10} {'F1':<10} {'SUPPORT':<10}")
    print("-"*60)
    
    for i, class_name in id_to_type.items():
        logger.info(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        print(f"{class_name:<20} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # Calculate aggregate metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    logger.info(f"\nMacro metrics: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}")
    logger.info(f"Weighted metrics: Precision={weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1={weighted_f1:.4f}")
    
    print("\nAggregate metrics:")
    print(f"Macro: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}")
    print(f"Weighted: Precision={weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1={weighted_f1:.4f}")
    print(f"Overall accuracy: {accuracy:.4f}\n")
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = [id_to_type[i] for i in range(len(id_to_type))]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save the confusion matrix plot
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")
    plt.close()
    
    # Analyze errors
    errors = [r for r in results if not r["is_correct"]]
    
    if errors:
        logger.info("\nError analysis:")
        logger.info(f"Total errors: {len(errors)}")
        
        # Count errors by true class
        error_by_true_class = Counter([e["expected_type"] for e in errors])
        logger.info(f"Errors by true class: {dict(error_by_true_class)}")
        
        # Count errors by predicted class
        error_by_pred_class = Counter([e["predicted_type"] for e in errors])
        logger.info(f"Errors by predicted class: {dict(error_by_pred_class)}")
        
        # Log most common error patterns
        error_patterns = Counter([(e["expected_type"], e["predicted_type"]) for e in errors])
        logger.info("Most common error patterns:")
        for (true_type, pred_type), count in error_patterns.most_common():
            logger.info(f"  {true_type} classified as {pred_type}: {count} times")
        
        # Save the most challenging examples
        logger.info("\nMost challenging examples:")
        for e in errors[:5]:
            logger.info(f"Question: '{e['question']}'")
            logger.info(f"  Expected: {e['expected_type']}, Predicted: {e['predicted_type']}")
            probs = {id_to_type[i]: e['probabilities'][i] for i in range(len(id_to_type))}
            logger.info(f"  Probabilities: {probs}")
            logger.info("")
    
    # Save results to JSON
    results_path = os.path.join(output_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "class_metrics": {
                id_to_type[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i])
                } for i in range(len(id_to_type))
            },
            "macro_metrics": {
                "precision": float(macro_precision),
                "recall": float(macro_recall),
                "f1": float(macro_f1)
            },
            "weighted_metrics": {
                "precision": float(weighted_precision),
                "recall": float(weighted_recall),
                "f1": float(weighted_f1)
            },
            "confusion_matrix": cm.tolist(),
            "results": results,
            "errors": errors
        }, f, indent=2)
    
    logger.info(f"Saved detailed results to {results_path}")
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "results": results,
        "errors": errors
    }


def main():
    """Main function for testing the enhanced classifier."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test enhanced question classifier")
    parser.add_argument("--model-path", type=str, default="models/enhanced_classifier/best_model.pt", 
                      help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default="outputs/enhanced_classifier_test",
                      help="Output directory for test results")
    parser.add_argument("--tokenizer-name", type=str, default="distilbert-base-uncased",
                      help="Name of tokenizer to use")
    args = parser.parse_args()
    
    # Run test
    test_question_classifier(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name
    )


if __name__ == "__main__":
    main()