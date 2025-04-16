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
    epoch: int
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # Get inputs
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    Evaluate model.
    
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
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels, strict=False)) / len(all_preds)
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy
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
    parser.add_argument("--num-epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
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
    
    # Check if custom path should be used and exists
    if use_custom_path:
        if os.path.exists(custom_path):
            logger.info(f"Using ModernBert from custom path: {custom_path}")
            # Register the model path with HuggingFace
            cache_dir = os.path.dirname(custom_path)
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            logger.info(f"Set TRANSFORMERS_CACHE to {cache_dir}")
        else:
            logger.warning(f"Custom path not found: {custom_path}. Will try to use from HuggingFace Hub.")
    
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
    
    # Train model
    logger.info("Starting training...")
    best_val_acc = 0
    for epoch in range(1, args.num_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            device=device
        )
        
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