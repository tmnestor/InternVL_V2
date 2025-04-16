"""
Training script for question classifier component.

This script trains the question classifier component for the multimodal system,
which is used to understand different types of questions about receipts and tax documents.
"""
import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from data.question_dataset import create_question_dataloaders
from models.components.question_classifier import QuestionClassifier
from utils.device import get_device
from utils.reproducibility import set_seed


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    class_weights: Optional[torch.Tensor] = None
) -> float:
    """
    Train for one epoch with enhanced logging and diagnostics.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        class_weights: Optional tensor of weights for each class to address class imbalance
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    # Use weighted cross entropy with label smoothing if class weights are provided
    label_smoothing = 0.1  # Add label smoothing to prevent overconfidence
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing
        )
        logger.info(f"Using weighted loss with class weights: {class_weights} and label_smoothing={label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
        
    logger = logging.getLogger(__name__)
    
    # Detailed logging for first batch of first epoch
    first_batch_detailed = (epoch == 1)
    
    # Track accuracy during training
    correct = 0
    total = 0
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Get inputs
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Detailed logging for first batch
        if first_batch_detailed and batch_idx == 0:
            logger.info(f"TRAIN BATCH CHECK: Input IDs shape: {input_ids.shape}")
            logger.info(f"TRAIN BATCH CHECK: Attention mask shape: {attention_mask.shape}")
            logger.info(f"TRAIN BATCH CHECK: Labels: {labels}")
            
            # Check for data issues
            if len(set(labels.cpu().numpy())) < 2:
                logger.warning("TRAIN BATCH CHECK: Batch contains only one class label. This may cause training issues.")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Detailed logits logging for first batch
        if first_batch_detailed and batch_idx == 0:
            logger.info(f"TRAIN BATCH CHECK: Logits shape: {logits.shape}")
            logger.info(f"TRAIN BATCH CHECK: Logits sample: {logits[0]}")
            logger.info(f"TRAIN BATCH CHECK: Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        if first_batch_detailed and batch_idx == 0:
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
                    
                    if name == "encoder.encoder.layer.0.attention.self.query.weight" or \
                       name == "classifier.0.weight":
                        logger.info(f"GRAD CHECK: {name} grad norm = {param_norm:.6f}")
            
            total_grad_norm = total_grad_norm ** 0.5
            logger.info(f"GRAD CHECK: Total grad norm = {total_grad_norm:.6f}")
            
            # Warn about exploding/vanishing gradients
            if total_grad_norm > 10.0:
                logger.warning(f"GRAD CHECK: Gradient norm is high ({total_grad_norm:.2f}). Consider gradient clipping.")
            elif total_grad_norm < 0.01:
                logger.warning(f"GRAD CHECK: Gradient norm is very low ({total_grad_norm:.6f}). Learning may be slow.")
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress
        total_loss += loss.item()
        batch_acc = (preds == labels).float().mean().item() * 100
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "batch_acc": f"{batch_acc:.1f}%"
        })
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    epoch_acc = 100 * correct / total if total > 0 else 0
    
    # Log epoch results
    logger.info(f"TRAIN EPOCH {epoch}: Average loss = {avg_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
    
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    Evaluate model with detailed diagnostics.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_questions = []
    all_question_types = []
    criterion = nn.CrossEntropyLoss()
    logger = logging.getLogger(__name__)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Save questions and types for analysis
            all_questions.extend(batch["question"])
            all_question_types.extend(batch["question_type"])
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log first batch details for debugging
            if batch_idx == 0:
                logger.info(f"EVAL BATCH CHECK: Input IDs shape: {input_ids.shape}")
                logger.info(f"EVAL BATCH CHECK: Attention mask shape: {attention_mask.shape}")
                logger.info(f"EVAL BATCH CHECK: Labels: {labels}")
                logger.info(f"EVAL BATCH CHECK: Predictions: {preds}")
                logger.info(f"EVAL BATCH CHECK: Logits samples: {logits[:2]}")
    
    # Calculate accuracy
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    accuracy = (all_preds_np == all_labels_np).mean()
    
    # Calculate per-class metrics
    classes = sorted(set(all_labels))
    per_class_acc = {}
    for c in classes:
        class_mask = (all_labels_np == c)
        if class_mask.sum() > 0:
            class_acc = (all_preds_np[class_mask] == c).mean()
            per_class_acc[c] = class_acc
    
    # Log confusion matrix statistics
    class_counts = {}
    for label, pred in zip(all_labels, all_preds):
        key = f"{label}->{pred}"
        class_counts[key] = class_counts.get(key, 0) + 1
    
    # Log success and error examples
    success_examples = []
    error_examples = []
    for i, (label, pred, question, qtype) in enumerate(zip(all_labels, all_preds, all_questions, all_question_types)):
        if label == pred:
            if len(success_examples) < 3:  # Limit to 3 examples
                success_examples.append((question, qtype, label))
        else:
            if len(error_examples) < 3:  # Limit to 3 examples
                error_examples.append((question, qtype, label, pred))
    
    # Log detailed evaluation statistics
    logger.info(f"EVAL CHECK: Accuracy by class: {per_class_acc}")
    logger.info(f"EVAL CHECK: Class transition counts: {class_counts}")
    logger.info(f"EVAL CHECK: Success examples: {success_examples}")
    logger.info(f"EVAL CHECK: Error examples: {error_examples}")
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc
    }


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train question classifier")
    parser.add_argument("--config", type=str, default="config/question_classifier_config.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="models/question_classifier",
                      help="Output directory for model checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/question_data",
                      help="Directory for question datasets")
    parser.add_argument("--model-name", type=str, default="ModernBert-base",
                      help="Base model for question classifier")
    parser.add_argument("--batch-size", type=int, default=16,
                      help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                      help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load configuration file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.warning(f"Error loading config file: {e}. Using default settings.")
        config = {
            "model": {
                "name": args.model_name,
                "custom_path": "/home/jovyan/nfs_share/models/huggingface/hub/ModernBERT-base",
                "use_custom_path": True,
                "hidden_size": 768
            }
        }
    
    # Get model configuration
    model_config = config.get("model", {})
    use_custom_path = model_config.get("use_custom_path", True)
    custom_path = model_config.get("custom_path", "/home/jovyan/nfs_share/models/huggingface/hub/ModernBERT-base")
    
    # Check if custom path should be used and exists with detailed diagnostics
    if use_custom_path:
        logger.info(f"CONFIG CHECK: Using custom path option from config: {use_custom_path}")
        logger.info(f"CONFIG CHECK: Custom path from config: {custom_path}")
        
        if os.path.exists(custom_path):
            logger.info(f"PATH CHECK: Custom path exists: {custom_path}")
            # List contents for debugging
            contents = os.listdir(custom_path)
            logger.info(f"PATH CHECK: Directory contents: {contents}")
            
            # Verify if it has the expected HuggingFace model files
            logger.info("PATH CHECK: Checking for essential model files:")
            expected_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
            for file in expected_files:
                file_path = os.path.join(custom_path, file)
                if os.path.exists(file_path):
                    logger.info(f"PATH CHECK: Found {file}")
                    # For config.json, print a snippet
                    if file == "config.json":
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                logger.info(f"PATH CHECK: config.json snippet: {content[:200]}...")
                        except Exception as e:
                            logger.warning(f"PATH CHECK: Error reading config.json: {e}")
                else:
                    logger.warning(f"PATH CHECK: Missing expected file: {file}")
            
            # Register the model path with HuggingFace
            cache_dir = os.path.dirname(custom_path)
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            logger.info(f"ENV CHECK: Set TRANSFORMERS_CACHE to {cache_dir}")
            logger.info(f"ENV CHECK: All environment variables: {[f'{k}={v}' for k, v in os.environ.items() if 'TRANSFORM' in k or 'HF_' in k]}")
        else:
            logger.warning(f"PATH CHECK: Custom path does not exist: {custom_path}")
            logger.warning(f"PATH CHECK: Current working directory: {os.getcwd()}")
            logger.warning(f"PATH CHECK: Parent directory exists? {os.path.exists(os.path.dirname(custom_path))}")
            logger.warning(f"PATH CHECK: Will try to use from HuggingFace Hub.")
    
    # Create dataloaders with appropriate tokenizer path
    tokenizer_path = custom_path if use_custom_path else args.model_name
    max_length = config.get("data", {}).get("max_length", 128)
    batch_size = config.get("training", {}).get("batch_size", args.batch_size)
    
    try:
        # Try to load the tokenizer from the configured path
        if use_custom_path:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=True
            )
            logger.info("Successfully loaded tokenizer from custom path")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info("Successfully loaded tokenizer from model name")
        
        dataloaders = create_question_dataloaders(
            data_dir=args.data_dir,
            batch_size=batch_size,
            tokenizer_name=tokenizer_path,
            max_length=max_length,
            num_workers=0,
            use_custom_path=use_custom_path
        )
    except Exception as e:
        logger.warning(f"Error loading tokenizer: {e}")
        logger.info("Falling back to default tokenizer")
        dataloaders = create_question_dataloaders(
            data_dir=args.data_dir,
            batch_size=batch_size,
            tokenizer_name="distilbert-base-uncased",
            max_length=max_length,
            num_workers=0,
            use_custom_path=False  # Don't use custom path for fallback
        )
    
    # Create model with appropriate configuration
    model_name = tokenizer_path  # Use same path that worked for tokenizer
    hidden_size = model_config.get("hidden_size", 768)
    num_classes = model_config.get("num_classes", 5)
    
    try:
        # Pass custom config to QuestionClassifier
        model = QuestionClassifier(
            model_name=model_name,
            hidden_size=hidden_size,
            num_classes=num_classes,
            device=device,
            use_custom_path=use_custom_path
        )
        logger.info(f"Successfully loaded model from {model_name}")
    except Exception as e:
        logger.warning(f"Error loading model: {e}")
        logger.info("Falling back to default model")
        model = QuestionClassifier(
            model_name="distilbert-base-uncased",
            hidden_size=hidden_size,
            num_classes=num_classes,
            device=device,
            use_custom_path=False
        )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.1
    )
    
    # Check model before training
    logger.info("MODEL CHECK: Examining model before training")
    logger.info(f"MODEL CHECK: Model type: {type(model).__name__}")
    logger.info(f"MODEL CHECK: Encoder type: {type(model.encoder).__name__}")
    logger.info(f"MODEL CHECK: Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"MODEL CHECK: Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Check data loaders and compute class weights for balanced loss
    logger.info("DATA CHECK: Examining dataloaders before training")
    
    # Initialize counters for class balancing
    class_counts = {}
    total_samples = 0
    
    for split, loader in dataloaders.items():
        logger.info(f"DATA CHECK: {split} - Number of batches: {len(loader)}")
        logger.info(f"DATA CHECK: {split} - Batch size: {loader.batch_size}")
        logger.info(f"DATA CHECK: {split} - Dataset size: {len(loader.dataset)}")
        
        # Check first batch and collect class statistics for the training set
        try:
            sample_batch = next(iter(loader))
            logger.info(f"DATA CHECK: {split} - Sample batch keys: {list(sample_batch.keys())}")
            logger.info(f"DATA CHECK: {split} - Sample input_ids shape: {sample_batch['input_ids'].shape}")
            logger.info(f"DATA CHECK: {split} - Sample labels: {sample_batch['label']}")
            logger.info(f"DATA CHECK: {split} - Sample questions: {sample_batch['question'][:2]}")  # Show first 2 questions
            logger.info(f"DATA CHECK: {split} - Sample question types: {sample_batch['question_type'][:2]}")
            
            # Count class instances for training set
            if split == "train":
                for label in sample_batch['label'].cpu().numpy():
                    class_counts[int(label)] = class_counts.get(int(label), 0) + 1
                    total_samples += 1
        except Exception as e:
            logger.warning(f"DATA CHECK: Error examining sample batch: {e}")
    
    # Get full statistics from the training dataset
    if "train" in dataloaders:
        class_counts = {}
        for batch in dataloaders["train"]:
            for label in batch['label'].cpu().numpy():
                label_int = int(label)
                class_counts[label_int] = class_counts.get(label_int, 0) + 1
        
        total_samples = sum(class_counts.values())
        logger.info(f"DATA CHECK: Class distribution in training set: {class_counts}")
        
        # Calculate inverse frequency class weights
        if class_counts:
            num_classes = max(class_counts.keys()) + 1
            # Initialize weights for all possible classes
            class_weights = torch.ones(num_classes)
            
            # Calculate inverse frequency weight for each class
            for label, count in class_counts.items():
                # Use inverse frequency with smoothing to avoid extreme weights
                class_weights[label] = total_samples / (count * num_classes)
            
            # Normalize weights to have mean of 1
            class_weights = class_weights * (num_classes / class_weights.sum())
            
            logger.info(f"DATA CHECK: Using class weights: {class_weights}")
        else:
            class_weights = None
            logger.warning("DATA CHECK: Could not calculate class weights. Using uniform weighting.")
    else:
        class_weights = None
        logger.warning("DATA CHECK: Training dataloader not found. Using uniform weighting.")
    
    # Train model with enhanced logging
    logger.info("Starting training...")
    best_val_acc = 0
    prev_train_loss = float('inf')
    prev_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"EPOCH {epoch}: Beginning training")
        
        # Train for one epoch with class weights
        train_loss = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            class_weights=class_weights
        )
        
        # Check for learning progress
        loss_change = prev_train_loss - train_loss
        logger.info(f"TRAINING CHECK: Loss change from previous epoch: {loss_change:.6f}")
        if abs(loss_change) < 0.0001:
            logger.warning("TRAINING CHECK: Training loss barely changing. Model may not be learning effectively.")
        prev_train_loss = train_loss
        
        # Evaluate on validation set
        logger.info(f"EPOCH {epoch}: Evaluating on validation set")
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            device=device
        )
        
        # Check validation progress
        val_loss_change = prev_val_loss - val_metrics['loss']
        logger.info(f"VALIDATION CHECK: Loss change from previous epoch: {val_loss_change:.6f}")
        if val_metrics['accuracy'] <= 0.2:
            logger.warning("VALIDATION CHECK: Accuracy at or below random guessing level (0.2 for 5 classes)")
        elif val_metrics['accuracy'] <= 0.25:
            logger.warning("VALIDATION CHECK: Accuracy slightly above random guessing, model may be struggling")
        prev_val_loss = val_metrics['loss']
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(
            f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}"
        )
        
        # Save model if it's the best so far
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            # Save model
            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt"
            )
            logger.info(f"New best model saved with accuracy: {best_val_acc:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        device=device
    )
    
    logger.info(
        f"Test Loss: {test_metrics['loss']:.4f}, "
        f"Test Accuracy: {test_metrics['accuracy']:.4f}"
    )
    
    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_metrics": test_metrics,
            "args": vars(args)
        },
        output_dir / "final_model.pt"
    )
    
    logger.info(f"Training completed. Final model saved to {output_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()