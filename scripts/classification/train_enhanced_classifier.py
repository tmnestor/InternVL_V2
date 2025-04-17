"""
Enhanced training script for question classifier with improved techniques.

This script implements the improvements recommended in the classification
improvement plan, including balanced datasets, focal loss, and enhanced
evaluation metrics.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from data.datasets.classification.balanced_question_dataset import create_balanced_question_dataloaders
from models.classification.question_classifier import QuestionClassifier
from utils.device import get_device
from utils.reproducibility import set_seed
from utils.focal_loss import FocalLoss, get_class_weights


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = None,
    epoch: int = 0,
    gradient_clip_val: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch with enhanced monitoring and diagnostics.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fn: Loss function (e.g., FocalLoss)
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        gradient_clip_val: Maximum gradient norm for clipping
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    logger = logging.getLogger(__name__)
    
    # Track per-class accuracy during training
    class_correct = Counter()
    class_total = Counter()
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Get inputs
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        
        # Update weights
        optimizer.step()
        
        # Step the learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update per-class accuracy
        for i, (pred, label) in enumerate(zip(preds, labels)):
            if pred == label:
                class_correct[label.item()] += 1
            class_total[label.item()] += 1
        
        # Store all predictions and labels for metrics calculation
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress
        total_loss += loss.item()
        batch_acc = (preds == labels).float().mean().item() * 100
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar with more detailed metrics
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "batch_acc": f"{batch_acc:.1f}%",
            "lr": f"{current_lr:.8f}"
        })
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    epoch_acc = 100 * correct / total if total > 0 else 0
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for class_idx in sorted(class_total.keys()):
        if class_total[class_idx] > 0:
            per_class_acc[class_idx] = class_correct[class_idx] / class_total[class_idx] * 100
        else:
            per_class_acc[class_idx] = 0.0
    
    # Calculate F1 scores
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Log epoch results with enhanced metrics
    logger.info(f"TRAIN EPOCH {epoch}:")
    logger.info(f"  Average loss = {avg_loss:.4f}")
    logger.info(f"  Accuracy = {epoch_acc:.2f}%")
    logger.info(f"  F1 score (micro) = {f1_micro:.4f}")
    logger.info(f"  F1 score (macro) = {f1_macro:.4f}")
    logger.info(f"  F1 score (weighted) = {f1_weighted:.4f}")
    logger.info(f"  Per-class accuracy: {per_class_acc}")
    
    # Return all metrics
    return {
        "loss": avg_loss,
        "accuracy": epoch_acc / 100,  # Convert back to 0-1 range
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_accuracy": per_class_acc
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    question_classes: Dict[int, str] = None,
    split: str = "val"
) -> Dict[str, Union[float, Dict]]:
    """
    Evaluate model with comprehensive metrics and analysis.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        loss_fn: Loss function (e.g., FocalLoss)
        device: Device to evaluate on
        question_classes: Mapping from class indices to class names
        split: Dataset split name ('val' or 'test')
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_questions = []
    all_question_types = []
    logger = logging.getLogger(__name__)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}")):
            # Get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Save questions and types for analysis
            all_questions.extend(batch["question"])
            all_question_types.extend(batch["question_type"])
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays for metric calculation
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    
    # Calculate main metrics
    accuracy = (all_preds_np == all_labels_np).mean()
    f1_micro = f1_score(all_labels_np, all_preds_np, average='micro')
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')
    f1_weighted = f1_score(all_labels_np, all_preds_np, average='weighted')
    
    # Calculate per-class metrics
    per_class_acc = {}
    per_class_f1 = {}
    classes = sorted(set(all_labels))
    
    for c in classes:
        class_mask = (all_labels_np == c)
        if class_mask.sum() > 0:
            class_acc = (all_preds_np[class_mask] == c).mean()
            per_class_acc[c] = class_acc
            
            # Calculate per-class F1 scores
            class_f1 = f1_score(
                all_labels_np == c, 
                all_preds_np == c, 
                average='binary'
            )
            per_class_f1[c] = class_f1
    
    # Get confusion matrix
    cm = confusion_matrix(all_labels_np, all_preds_np)
    
    # Get full classification report
    if question_classes:
        target_names = [question_classes[i] for i in range(len(question_classes))]
        report = classification_report(
            all_labels_np, 
            all_preds_np, 
            target_names=target_names,
            output_dict=True
        )
    else:
        report = classification_report(
            all_labels_np, 
            all_preds_np, 
            output_dict=True
        )
    
    # Analyze worst performing examples
    error_examples = []
    worst_class_id = min(per_class_acc, key=per_class_acc.get)
    worst_class_examples = []
    
    for i, (label, pred, question, qtype) in enumerate(zip(
        all_labels, all_preds, all_questions, all_question_types
    )):
        if label != pred:
            error_examples.append((question, qtype, int(label), int(pred)))
            
            # Collect examples from worst performing class
            if label == worst_class_id:
                worst_class_examples.append((question, qtype, int(label), int(pred)))
    
    # Log confusion matrix
    logger.info(f"\nConfusion Matrix ({split}):")
    if question_classes:
        class_names = [question_classes[i] for i in range(len(question_classes))]
        logger.info(f"Classes: {class_names}")
    logger.info(f"{cm}")
    
    # Log per-class metrics
    logger.info(f"\nPer-class accuracy ({split}):")
    for c in sorted(per_class_acc.keys()):
        class_name = question_classes[c] if question_classes else f"Class {c}"
        logger.info(f"  {class_name}: {per_class_acc[c]*100:.2f}%")
    
    logger.info(f"\nPer-class F1 score ({split}):")
    for c in sorted(per_class_f1.keys()):
        class_name = question_classes[c] if question_classes else f"Class {c}"
        logger.info(f"  {class_name}: {per_class_f1[c]:.4f}")
    
    # Log examples of errors for worst performing class
    logger.info(f"\nWorst performing class: {question_classes[worst_class_id] if question_classes else worst_class_id}")
    logger.info(f"Examples of errors for this class:")
    for i, (question, qtype, label, pred) in enumerate(worst_class_examples[:5]):
        pred_class = question_classes[pred] if question_classes else f"Class {pred}"
        logger.info(f"  Example {i+1}: '{question}' (True: {qtype}, Predicted: {pred_class})")
    
    # Save confusion matrix as image
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {split}')
    plt.colorbar()
    
    if question_classes:
        classes = [question_classes[i] for i in range(len(question_classes))]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the figure
    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrix_{split}.png')
    plt.close()
    
    # Return all metrics
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_accuracy": per_class_acc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "error_examples": error_examples[:10],  # Limit to first 10 examples
        "worst_class_examples": worst_class_examples[:10]  # Limit to first 10 examples
    }


def main():
    """Main training function with enhanced features."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train enhanced question classifier")
    parser.add_argument("--config", type=str, default="config/classifier/question_classifier_config.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="models/enhanced_classifier",
                      help="Output directory for model checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/balanced_question_data",
                      help="Directory for question datasets")
    parser.add_argument("--model-name", type=str, default=None,
                      help="Base model for question classifier (overrides config value if provided)")
    parser.add_argument("--batch-size", type=int, default=16,
                      help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=20,
                      help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                      help="Weight decay for optimizer")
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                      help="Proportion of training to perform LR warmup")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                      help="Gamma parameter for focal loss")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                      help="Label smoothing factor")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                      help="Gradient clipping value")
    parser.add_argument("--weight-strategy", type=str, default="manual",
                      choices=["inverse_frequency", "effective_samples", "manual"],
                      help="Strategy for calculating class weights")
    parser.add_argument("--early-stopping", type=int, default=5,
                      help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('archive/logging/logs', exist_ok=True)
    log_filename = args.model_name.replace('/', '_') if args.model_name else "question_classifier"
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"archive/logging/logs/training_{log_filename}.log")
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for results
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load configuration from YAML file
    import yaml
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get model parameters from config
        model_name = args.model_name if args.model_name is not None else config["model"]["name"]
        custom_path = config["model"].get("custom_path", "")
        use_custom_path = config["model"].get("use_custom_path", True)  # Default to True for production
        
        # Verify model path exists if using custom path
        if use_custom_path:
            if Path(custom_path).exists():
                logger.info(f"Using model from local path: {custom_path}")
            else:
                logger.error(f"Model path does not exist: {custom_path}")
                raise FileNotFoundError(f"Model path not found: {custom_path}")
        else:
            logger.error("use_custom_path must be true - we only load from local paths")
            raise ValueError("use_custom_path must be true - we only load from local paths")
            
        # Get other hyperparameters from config
        max_length = config["training"].get("max_length", 128)
    else:
        # Config file not found - this is a fatal error, we need the config
        error_msg = f"Config file not found at {config_path}. Configuration is required."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Create dataloaders with balanced dataset
    logger.info("Creating balanced dataloaders...")
    
    # Use the same model path for tokenizer
    tokenizer_path = custom_path if use_custom_path else None
    
    if tokenizer_path is None or not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer path not found: {tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer path not found or not specified: {tokenizer_path}")
        
    logger.info(f"Using tokenizer from path: {tokenizer_path}")
    
    dataloaders = create_balanced_question_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        tokenizer_name=tokenizer_path,  # Pass the path to the local tokenizer
        max_length=max_length,
        num_workers=0
    )
    
    # Check class distribution in the training set
    train_dataset = dataloaders["train"].dataset
    
    # Count classes
    class_counts = Counter()
    for item in train_dataset.questions:
        class_type = item["type"]
        class_idx = train_dataset.question_types[class_type]
        class_counts[class_idx] += 1
    
    logger.info(f"Class distribution in training set: {class_counts}")
    
    # Calculate class weights
    class_weights = get_class_weights(
        class_counts,
        num_classes=len(train_dataset.question_types),
        weight_strategy=args.weight_strategy
    )
    logger.info(f"Using {args.weight_strategy} class weights: {class_weights}")
    
    # Create model using configuration
    logger.info(f"Creating model with {model_name}...")
    
    # Get model parameters from config
    use_custom_path = config["model"].get("use_custom_path", False)
    custom_path = config["model"].get("custom_path", "")
    hidden_size = config["model"].get("hidden_size", 768)
    
    try:
        # IMPORTANT: We need to pass the custom path as the model_name parameter
        # This ensures we're loading from the local path and not trying HuggingFace
        model = QuestionClassifier(
            model_name=custom_path,  # Use the local path from config
            hidden_size=hidden_size,
            num_classes=len(train_dataset.question_types),
            device=device,
            use_custom_path=use_custom_path  # Use the value from config
        )
        logger.info(f"Successfully created model with model from: {custom_path}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Failed to initialize model: {e}")
    
    # Initialize focal loss
    loss_fn = FocalLoss(
        alpha=class_weights.to(device),
        gamma=args.focal_gamma,
        reduction='mean',
        label_smoothing=args.label_smoothing
    )
    logger.info(f"Using Focal Loss with gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler with warmup
    num_training_steps = len(dataloaders["train"]) * args.num_epochs
    num_warmup_steps = int(args.warmup_proportion * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Using cosine schedule with warmup: {num_warmup_steps} warmup steps, {num_training_steps} total steps")
    
    # Define question type mapping (for logging and visualization)
    question_classes = {v: k for k, v in train_dataset.question_types.items()}
    
    # Train model with enhanced logging and early stopping
    logger.info("Starting enhanced training...")
    best_val_metric = 0  # Track best validation metric (using F1 macro)
    best_epoch = 0
    patience_counter = 0
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"EPOCH {epoch}/{args.num_epochs}: Beginning training")
        
        # Train for one epoch with enhanced metrics
        train_metrics = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            gradient_clip_val=args.gradient_clip
        )
        train_metrics_history.append(train_metrics)
        
        # Evaluate on validation set
        logger.info(f"EPOCH {epoch}/{args.num_epochs}: Evaluating on validation set")
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            loss_fn=loss_fn,
            device=device,
            question_classes=question_classes,
            split="val"
        )
        val_metrics_history.append(val_metrics)
        
        # Log validation metrics
        logger.info(f"Validation metrics:")
        logger.info(f"  Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {val_metrics['accuracy']*100:.2f}%")
        logger.info(f"  F1 score (micro): {val_metrics['f1_micro']:.4f}")
        logger.info(f"  F1 score (macro): {val_metrics['f1_macro']:.4f}")
        logger.info(f"  F1 score (weighted): {val_metrics['f1_weighted']:.4f}")
        
        # Track best validation metric (using F1 macro for more balanced evaluation)
        current_val_metric = val_metrics["f1_macro"]
        
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt"
            )
            logger.info(f"New best model saved with F1 macro: {best_val_metric:.4f}")
            
            # Track worst performing class for targeted improvements
            worst_class = min(
                val_metrics["per_class_accuracy"], 
                key=val_metrics["per_class_accuracy"].get
            )
            worst_class_name = question_classes[worst_class]
            worst_class_acc = val_metrics["per_class_accuracy"][worst_class] * 100
            logger.info(f"Worst performing class: {worst_class_name} with accuracy {worst_class_acc:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation F1. Patience: {patience_counter}/{args.early_stopping}")
            
            if patience_counter >= args.early_stopping:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Plot training history
    epochs = list(range(1, len(train_metrics_history) + 1))
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m["loss"] for m in train_metrics_history], label="Train Loss")
    plt.plot(epochs, [m["loss"] for m in val_metrics_history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(results_dir / "loss_curves.png")
    plt.close()
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m["accuracy"] for m in train_metrics_history], label="Train Accuracy")
    plt.plot(epochs, [m["accuracy"] for m in val_metrics_history], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(results_dir / "accuracy_curves.png")
    plt.close()
    
    # Plot F1 score curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m["f1_macro"] for m in train_metrics_history], label="Train F1 (macro)")
    plt.plot(epochs, [m["f1_macro"] for m in val_metrics_history], label="Val F1 (macro)")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score (Macro)")
    plt.legend()
    plt.savefig(results_dir / "f1_curves.png")
    plt.close()
    
    # Load best model for final evaluation
    logger.info(f"Loading best model from epoch {best_epoch} for final evaluation")
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    
    # Evaluate on test set
    logger.info("Evaluating best model on test set...")
    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        loss_fn=loss_fn,
        device=device,
        question_classes=question_classes,
        split="test"
    )
    
    # Log final test metrics
    logger.info(f"Final test metrics:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"  F1 score (micro): {test_metrics['f1_micro']:.4f}")
    logger.info(f"  F1 score (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 score (weighted): {test_metrics['f1_weighted']:.4f}")
    
    # Save final model with metadata
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "final_metrics": {
                "train": train_metrics_history[-1],
                "val": val_metrics_history[-1],
                "test": test_metrics,
                "best_epoch": best_epoch,
                "best_val_metric": best_val_metric,
            },
            "question_types": train_dataset.question_types
        },
        output_dir / "final_model.pt"
    )
    
    logger.info(f"Training completed. Final model saved to {output_dir / 'final_model.pt'}")
    logger.info(f"Best validation F1 (macro): {best_val_metric:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()