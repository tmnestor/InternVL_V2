"""
Multimodal trainer implementation for the InternVL2 vision-language receipt counter.

Implements the Phase 3 training strategy with a two-stage approach:
1. Train with frozen vision encoder
2. End-to-end fine-tuning with careful learning rate scheduling
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vision_language.internvl2 import InternVL2MultimodalModel
from training.multimodal_loss import MultimodalLoss
from utils.device import get_device
from utils.logging import TensorboardLogger
from utils.metrics import compute_nlg_metrics


class MultimodalTrainer:
    """
    Trainer for the InternVL2 multimodal receipt counter model.
    
    Implements a two-stage training approach:
    1. Train with frozen vision encoder (focus on language and cross-attention)
    2. End-to-end fine-tuning with smaller learning rates for vision components
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: InternVL2MultimodalModel,
        dataloaders: Dict[str, DataLoader],
        output_dir: Path,
    ):
        """
        Initialize the multimodal trainer.
        
        Args:
            config: Training configuration
            model: InternVL2 multimodal model to train
            dataloaders: Dictionary with train and validation dataloaders
            output_dir: Directory to save outputs
        """
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get logger
        self.logger = logging.getLogger(__name__)
        
        # Configure checkpoint options
        self.use_safe_serialization = config["output"].get("safe_serialization", True)
        self.save_half_precision = config["output"].get("save_half_precision", False)
        
        # Environment setup for CUDA
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
            
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA capability: {torch.cuda.get_device_capability()}")
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
            # Log current memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(f"GPU Memory: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        # Setup device
        self.device = get_device()
        self.model.to(self.device)
        
        # Apply torch.compile if available and enabled
        if (torch.cuda.is_available() and hasattr(torch, 'compile') and 
                self.config["training"].get("torch_compile", False)):
            compile_mode = self.config["training"].get("compile_mode", "reduce-overhead")
            self.logger.info(f"Applying torch.compile with mode: {compile_mode}")
            self.model = torch.compile(self.model, mode=compile_mode)
            self.logger.info("Successfully applied torch.compile for GPU acceleration")
        
        # Setup loss function with configured weights
        loss_weights = config["training"]["loss_weights"]
        self.loss_fn = MultimodalLoss(
            classification_weight=loss_weights.get("classification", 1.0),
            language_weight=loss_weights.get("language", 1.0),
        )
        
        # Get total number of epochs from config
        self.epochs = config["training"].get("epochs", 15)
        
        # Initial optimizer setup - will be reconfigured during stage transitions
        self.optimizer = self._configure_optimizer(stage=1)
        self.scheduler = self._configure_scheduler()
        
        # Mixed precision setup
        self.use_mixed_precision = config["training"].get("fp16", False)
        if self.use_mixed_precision:
            self.logger.info("Using mixed precision training")
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Gradient clipping
        self.clip_grad_norm = config["training"].get("gradient_clip", 1.0)
        
        # Training parameters
        self.epochs = config["training"]["epochs"]
        self.current_epoch = 0
        
        # Setup TensorBoard logger if enabled in config
        self.tensorboard = None
        if config["output"].get("tensorboard", False):
            tensorboard_dir = Path(config["output"]["results_dir"]) / "tensorboard"
            self.tensorboard = TensorboardLogger(tensorboard_dir)
        else:
            self.logger.info("TensorBoard logging disabled in config")
        
        # Setup early stopping
        if "early_stopping" in config["training"]:
            self.patience = config["training"]["early_stopping"].get("patience", 5)
            self.min_delta = config["training"]["early_stopping"].get("min_delta", 0.01)
        else:
            self.patience = 5
            self.min_delta = 0.01
        
        # Setup multi-stage training
        self.three_stage = config["training"]["three_stage"]
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_bleu = 0.0
        self.no_improve_count = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_bleu': [],
            'val_loss': [],
            'val_acc': [],
            'val_bleu': [],
            'lr': []
        }
    
    def _configure_optimizer(self, stage: int = 1) -> optim.Optimizer:
        """
        Configure optimizer based on training stage.
        
        Args:
            stage: Current training stage (1 or 2)
            
        Returns:
            Configured optimizer
        """
        optimizer_config = self.config["training"]
        base_lr = float(optimizer_config["learning_rate"])
        weight_decay = float(optimizer_config.get("weight_decay", 1e-4))
        
        if stage == 1:
            # Stage 1: Train with frozen vision encoder
            # Create parameter groups: only train language model, cross-attention, and response generator
            param_groups = [
                # Classification head parameters
                {'params': self.model.classification_head.parameters(), 'lr': base_lr},
                
                # Cross-attention parameters
                {'params': self.model.cross_attention.parameters(), 'lr': base_lr},
                
                # Response generator parameters
                {'params': self.model.response_generator.parameters(), 'lr': base_lr},
                
                # Language model parameters (if not frozen)
                {'params': self.model.language_model.parameters(), 
                 'lr': base_lr * optimizer_config.get("language_lr_multiplier", 0.1)}
            ]
            
            # Freeze vision encoder
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
                
            self.logger.info(f"Stage 1: Vision encoder frozen, training other components with base LR: {base_lr}")
            
        elif stage == 2:
            # Stage 2: Unfreeze vision encoder with smaller learning rate
            vision_lr_multiplier = self.config["training"]["three_stage"]["stage2"].get("lr_multiplier", 0.01)
            
            # Unfreeze vision encoder (selectively for last few layers)
            vision_params = self._unfreeze_vision_encoder(lr_multiplier=vision_lr_multiplier)
            
            param_groups = [
                # Vision encoder parameters (small LR)
                {'params': vision_params, 'lr': base_lr * vision_lr_multiplier},
                
                # Classification head parameters
                {'params': self.model.classification_head.parameters(), 'lr': base_lr},
                
                # Cross-attention parameters
                {'params': self.model.cross_attention.parameters(), 'lr': base_lr},
                
                # Response generator parameters
                {'params': self.model.response_generator.parameters(), 'lr': base_lr},
                
                # Language model parameters
                {'params': self.model.language_model.parameters(), 
                 'lr': base_lr * optimizer_config.get("language_lr_multiplier", 0.1)}
            ]
            
            self.logger.info(f"Stage 2: Vision encoder unfrozen with LR multiplier: {vision_lr_multiplier}")
            
        elif stage == 3:
            # Stage 3: Full fine-tuning with balanced learning rates
            language_lr_multiplier = self.config["training"]["three_stage"]["stage3"].get("lr_multiplier", 0.1)
            vision_lr_multiplier = self.config["training"]["three_stage"]["stage2"].get("lr_multiplier", 0.01)
            
            # Unfreeze all components
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = True
                
            for param in self.model.language_model.parameters():
                param.requires_grad = True
            
            param_groups = [
                # Vision encoder parameters (small LR)
                {'params': self.model.vision_encoder.parameters(), 'lr': base_lr * vision_lr_multiplier},
                
                # Classification head parameters
                {'params': self.model.classification_head.parameters(), 'lr': base_lr},
                
                # Cross-attention parameters
                {'params': self.model.cross_attention.parameters(), 'lr': base_lr},
                
                # Response generator parameters
                {'params': self.model.response_generator.parameters(), 'lr': base_lr},
                
                # Language model parameters
                {'params': self.model.language_model.parameters(), 'lr': base_lr * language_lr_multiplier}
            ]
            
            self.logger.info(f"Stage 3: Full fine-tuning with vision LR multiplier: {vision_lr_multiplier}, " 
                           f"language LR multiplier: {language_lr_multiplier}")
        
        # Create optimizer
        optimizer_name = optimizer_config.get("name", "adamw")
        if optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adam":
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                param_groups, 
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _unfreeze_vision_encoder(self, lr_multiplier: float = 0.01) -> List[nn.Parameter]:
        """
        Selectively unfreeze vision encoder layers for fine-tuning.
        
        Args:
            lr_multiplier: Learning rate multiplier for vision encoder
            
        Returns:
            List of unfrozen parameters
        """
        # Start by freezing all vision encoder parameters
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False
        
        # Count total layers
        all_params = list(self.model.vision_encoder.named_parameters())
        total_layers = len(all_params)
        
        # Determine how many layers to unfreeze
        # For stage 2, we'll unfreeze only the last few transformer blocks
        last_n_layers = 2
        
        # Find layers to unfreeze (typically last few transformer blocks, 
        # attention blocks, and any pooling/output layers)
        unfrozen_params = []
        unfrozen_count = 0
        
        # Look for transformer blocks in reverse order
        for i, (name, param) in enumerate(reversed(all_params)):
            if i < total_layers // 3:  # Unfreeze about 1/3 of the model
                # Check if this is a key component worth unfreezing
                if any(key in name.lower() for key in ['block', 'layer', 'attention', 'mlp', 'norm', 'pool', 'output']):
                    param.requires_grad = True
                    unfrozen_params.append(param)
                    unfrozen_count += 1
                    self.logger.debug(f"Unfreezing layer: {name}")
        
        self.logger.info(f"Unfrozen {unfrozen_count} of {total_layers} vision encoder parameters")
        return unfrozen_params
    
    def _configure_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Configure learning rate scheduler.
        
        Returns:
            Learning rate scheduler
        """
        scheduler_config = self.config["training"].get("scheduler", {})
        scheduler_name = scheduler_config.get("name", "cosine")
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.optimizer.param_groups[0]["lr"] * scheduler_config.get("min_lr_factor", 0.1)
            )
        elif scheduler_name == "one_cycle":
            steps_per_epoch = len(self.dataloaders["train"])
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[pg["lr"] for pg in self.optimizer.param_groups],
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                pct_start=scheduler_config.get("pct_start", 0.3)
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 3),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_name == "warmup_cosine":
            warmup_steps = scheduler_config.get("warmup_steps", 500)
            return optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 
                        0.5 * (1 + torch.cos(torch.tensor((step - warmup_steps) / 
                        (self.epochs * len(self.dataloaders["train"]) - warmup_steps) * torch.pi)))
            )
        elif scheduler_name == "none" or not scheduler_name:
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_language_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize metric tracking for text generation
        all_predictions = []
        all_targets = []
        
        # Set up progress bar
        train_loader = self.dataloaders["train"]
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        # Enable gradient accumulation to reduce memory usage
        gradient_accumulation_steps = self.config["training"].get("gradient_accumulation_steps", 1)
        effective_batch_size = gradient_accumulation_steps * train_loader.batch_size
        self.logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        self.logger.info(f"Effective batch size: {effective_batch_size}")
        
        # Training loop
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients only at the beginning of accumulation steps
            if batch_idx % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                with autocast():
                    # Forward pass through model
                    outputs = self.model(
                        pixel_values=batch["pixel_values"],
                        text_input_ids=batch["text_input_ids"],
                        attention_mask=batch["text_attention_mask"]
                    )
                    
                    # Calculate loss
                    loss_dict = self.loss_fn(
                        model_outputs=outputs,
                        classification_labels=batch["classification_labels"],
                        language_labels=batch["labels"],
                        attention_mask=batch["labels_attention_mask"]
                    )
                    
                    # Scale loss by accumulation steps
                    loss = loss_dict["total_loss"] / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Only update weights at the end of accumulation steps or at the last batch
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Gradient clipping
                    if self.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Update weights with gradient scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Zero gradients after update
                    self.optimizer.zero_grad()
            else:
                # Standard precision training
                # Forward pass through model
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    text_input_ids=batch["text_input_ids"],
                    attention_mask=batch["text_attention_mask"]
                )
                
                # Calculate loss
                loss_dict = self.loss_fn(
                    model_outputs=outputs,
                    classification_labels=batch["classification_labels"],
                    language_labels=batch["labels"],
                    attention_mask=batch["labels_attention_mask"]
                )
                
                # Scale loss by accumulation steps
                loss = loss_dict["total_loss"] / gradient_accumulation_steps
                
                # Regular backward pass
                loss.backward()
                
                # Only update weights at the end of accumulation steps or at the last batch
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Gradient clipping
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    try:
                        # Update weights
                        self.optimizer.step()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            self.logger.error(f"CUDA OOM during optimizer step. Error: {e}")
                            self.logger.info("Trying to recover by clearing cache and reducing batch size...")
                            torch.cuda.empty_cache()
                            # Let it crash to prevent further issues
                            raise
                        else:
                            raise
                    
                    # Zero gradients after update
                    self.optimizer.zero_grad()
                    
                    # Periodically clear CUDA cache to prevent fragmentation (every 20 updates)
                    if (batch_idx + 1) % 20 == 0:
                        torch.cuda.empty_cache()
            
            # Update running losses
            running_loss += loss_dict["total_loss"].item()
            if "classification_loss" in loss_dict:
                running_class_loss += loss_dict["classification_loss"].item()
            if "language_loss" in loss_dict:
                running_language_loss += loss_dict["language_loss"].item()
            
            # Calculate classification accuracy
            if "logits" in outputs:
                _, predicted = outputs["logits"].max(1)
                total += batch["classification_labels"].size(0)
                correct += predicted.eq(batch["classification_labels"]).sum().item()
            
            # Collect text generation outputs for BLEU calculation (only every 5th batch)
            if "response_logits" in outputs and batch_idx % 5 == 0:
                # Get predictions (greedy decoding)
                with torch.no_grad():
                    pred_tokens = outputs["response_logits"].argmax(dim=-1)
                
                # Convert to text for metric calculation
                tokenizer = self.model.tokenizer
                for i in range(min(2, pred_tokens.size(0))):  # Only decode a couple samples
                    # Get token IDs within valid range
                    pred_tokens_valid = [t.item() for t in pred_tokens[i] if 0 <= t.item() < tokenizer.vocab_size]
                    label_tokens_valid = [t.item() for t in batch["labels"][i] if 0 <= t.item() < tokenizer.vocab_size]
                    
                    # Decode to text
                    pred_text = tokenizer.decode(pred_tokens_valid, skip_special_tokens=True)
                    target_text = tokenizer.decode(label_tokens_valid, skip_special_tokens=True)
                    
                    # Only add non-empty strings
                    if pred_text and target_text:
                        all_predictions.append(pred_text)
                        all_targets.append(target_text)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * gradient_accumulation_steps,
                'acc': 100. * correct / max(1, total)
            })
            
            # Clean up memory
            del outputs
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available() and batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_class_loss = running_class_loss / len(train_loader)
        epoch_language_loss = running_language_loss / len(train_loader)
        epoch_acc = 100. * correct / max(1, total)
        
        # Calculate BLEU score if we have predictions
        epoch_bleu = 0.0
        if all_predictions and all_targets:
            metrics = compute_nlg_metrics(all_predictions, all_targets)
            epoch_bleu = metrics.get("bleu4", 0.0)
        
        # Log to TensorBoard
        if self.tensorboard:
            self.tensorboard.log_scalar("train/loss", epoch_loss, epoch)
            self.tensorboard.log_scalar("train/class_loss", epoch_class_loss, epoch)
            self.tensorboard.log_scalar("train/language_loss", epoch_language_loss, epoch)
            self.tensorboard.log_scalar("train/accuracy", epoch_acc, epoch)
            self.tensorboard.log_scalar("train/bleu", epoch_bleu, epoch)
            
            # Log learning rate for each parameter group
            for i, pg in enumerate(self.optimizer.param_groups):
                self.tensorboard.log_scalar(f"train/lr_group_{i}", pg["lr"], epoch)
        
        # Return metrics
        return {
            "loss": epoch_loss,
            "class_loss": epoch_class_loss,
            "language_loss": epoch_language_loss,
            "accuracy": epoch_acc,
            "bleu": epoch_bleu
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        running_class_loss = 0.0
        running_language_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize metric tracking for text generation
        all_predictions = []
        all_targets = []
        
        # Validation loop
        val_loader = self.dataloaders["val"]
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    text_input_ids=batch["text_input_ids"],
                    attention_mask=batch["text_attention_mask"]
                )
                
                # Calculate loss
                loss_dict = self.loss_fn(
                    model_outputs=outputs,
                    classification_labels=batch["classification_labels"],
                    language_labels=batch["labels"],
                    attention_mask=batch["labels_attention_mask"]
                )
                
                loss = loss_dict["total_loss"]
                
                # Update running losses
                running_loss += loss.item()
                if "classification_loss" in loss_dict:
                    running_class_loss += loss_dict["classification_loss"].item()
                if "language_loss" in loss_dict:
                    running_language_loss += loss_dict["language_loss"].item()
                
                # Calculate classification accuracy
                if "logits" in outputs:
                    _, predicted = outputs["logits"].max(1)
                    total += batch["classification_labels"].size(0)
                    correct += predicted.eq(batch["classification_labels"]).sum().item()
                
                # Generate text responses for evaluation (only every 5th batch to save time)
                if batch_idx % 5 == 0:
                    # Generate responses with appropriate parameters
                    generate_params = {
                        "pixel_values": batch["pixel_values"],
                        "text_input_ids": batch["text_input_ids"],
                        "attention_mask": batch["text_attention_mask"],
                        "max_length": 50,
                        "temperature": 0.5,
                        "top_k": 5,
                        "top_p": 0.92,
                    }
                    
                    # Generate responses directly from the model
                    generated_ids, decoded_texts = self.model.generate_response(**generate_params)
                    
                    # Get target texts for comparison
                    tokenizer = self.model.tokenizer
                    target_texts = []
                    
                    for i in range(len(decoded_texts)):
                        # Only collect non-empty predictions
                        if decoded_texts[i].strip():
                            # Get corresponding target text
                            valid_tokens = [t for t in batch["labels"][i] if 0 <= t < tokenizer.vocab_size]
                            target_text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                            
                            # Only include if both prediction and target are non-empty
                            if target_text.strip():
                                all_predictions.append(decoded_texts[i])
                                all_targets.append(target_text)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / max(1, total)
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_class_loss = running_class_loss / len(val_loader)
        epoch_language_loss = running_language_loss / len(val_loader)
        epoch_acc = 100. * correct / max(1, total)
        
        # Calculate BLEU score if we have predictions
        epoch_bleu = 0.0
        if all_predictions and all_targets:
            metrics = compute_nlg_metrics(all_predictions, all_targets)
            epoch_bleu = metrics.get("bleu4", 0.0)
            
            # Log a few examples
            self.logger.info("Example validation predictions:")
            for i in range(min(3, len(all_predictions))):
                self.logger.info(f"Prediction: {all_predictions[i]}")
                self.logger.info(f"Target: {all_targets[i]}")
                self.logger.info("-" * 30)
        
        # Log to TensorBoard
        if self.tensorboard:
            self.tensorboard.log_scalar("val/loss", epoch_loss, epoch)
            self.tensorboard.log_scalar("val/class_loss", epoch_class_loss, epoch)
            self.tensorboard.log_scalar("val/language_loss", epoch_language_loss, epoch)
            self.tensorboard.log_scalar("val/accuracy", epoch_acc, epoch)
            self.tensorboard.log_scalar("val/bleu", epoch_bleu, epoch)
        
        # Return metrics
        return {
            "loss": epoch_loss,
            "class_loss": epoch_class_loss,
            "language_loss": epoch_language_loss,
            "accuracy": epoch_acc,
            "bleu": epoch_bleu
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save model checkpoint with direct file operations.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
        """
        import os
        
        # Check if we should save best model only
        save_best_only = self.config["output"].get("save_best_only", True)
        
        # If save_best_only is True and this is not the best model, don't save
        if save_best_only and not is_best:
            return
        
        # Prepare checkpoint contents
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history
        }
        
        # Add scheduler and scaler if they exist
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Convert to half precision if configured
        if self.save_half_precision:
            checkpoint["model_state_dict"] = {
                k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.logger.info("Saving checkpoint in half precision")
        
        if is_best:
            # Save best model directly to output directory
            best_path = self.output_dir / "best_model.pt"
            
            # Use direct save with configured serialization settings
            if self.use_safe_serialization:
                torch.save(checkpoint, str(best_path), _use_new_zipfile_serialization=False)
            else:
                torch.save(checkpoint, str(best_path))
                
            self.logger.info(f"Best model saved to {best_path}")
        else:
            # We only reach here if save_best_only is False
            # Create checkpoint directory for intermediate checkpoints
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine path for regular checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            
            # Save regular checkpoint
            if self.use_safe_serialization:
                torch.save(checkpoint, str(checkpoint_path), _use_new_zipfile_serialization=False)
            else:
                torch.save(checkpoint, str(checkpoint_path))
                
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self) -> Tuple[InternVL2MultimodalModel, Dict[str, List[float]]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Tuple of (trained model, training history)
        """
        start_time = time.time()
        self.logger.info("Starting multimodal training...")
        
        # Determine total epochs and stage transitions
        total_epochs = self.config["training"]["epochs"]
        stage2_start = self.config["training"]["three_stage"]["stage2"].get("start_epoch", total_epochs // 3)
        stage3_start = self.config["training"]["three_stage"]["stage3"].get("start_epoch", 2 * total_epochs // 3)
        
        # Initialize training stage
        current_stage = 1
        
        # Training loop
        for epoch in range(1, total_epochs + 1):
            self.current_epoch = epoch
            
            # Check for stage transitions
            if self.config["training"]["three_stage"]["enabled"]:
                # Stage 1: Initial training with frozen vision encoder
                if epoch == 1:
                    self.logger.info("Stage 1: Training with frozen vision encoder...")
                    # Vision encoder is frozen in model init
                
                # Stage 2: Unfreeze vision encoder with lower learning rate
                elif epoch == stage2_start:
                    self.logger.info("Transitioning to Stage 2: Unfreezing vision encoder...")
                    current_stage = 2
                    
                    # Reconfigure optimizer for stage 2
                    self.optimizer = self._configure_optimizer(stage=2)
                    self.scheduler = self._configure_scheduler()
                    
                    # Log learning rates
                    lr_info = "Learning rates for Stage 2: "
                    for i, group in enumerate(self.optimizer.param_groups):
                        lr_info += f"Group {i}: {group['lr']:.2e} "
                    self.logger.info(lr_info)
                
                # Stage 3: End-to-end fine-tuning (if needed)
                elif epoch == stage3_start:
                    self.logger.info("Transitioning to Stage 3: End-to-end fine-tuning...")
                    current_stage = 3
                    
                    # Reconfigure optimizer for stage 3
                    self.optimizer = self._configure_optimizer(stage=3)
                    self.scheduler = self._configure_scheduler()
                    
                    # Log learning rates
                    lr_info = "Learning rates for Stage 3: "
                    for i, group in enumerate(self.optimizer.param_groups):
                        lr_info += f"Group {i}: {group['lr']:.2e} "
                    self.logger.info(lr_info)
            
            # Train one epoch
            self.logger.info(f"Epoch {epoch}/{total_epochs} (Stage {current_stage})")
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["train_bleu"].append(train_metrics["bleu"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_bleu"].append(val_metrics["bleu"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            # Check for improvement
            is_best = False
            if val_metrics["loss"] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics["loss"]
                self.no_improve_count = 0
                is_best = True
                self.logger.info(f"New best validation loss: {val_metrics['loss']:.4f}")
            elif val_metrics["accuracy"] > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_metrics["accuracy"]
                self.no_improve_count = 0
                is_best = True
                self.logger.info(f"New best validation accuracy: {val_metrics['accuracy']:.2f}%")
            elif val_metrics["bleu"] > self.best_val_bleu + self.min_delta:
                self.best_val_bleu = val_metrics["bleu"]
                self.no_improve_count = 0
                is_best = True
                self.logger.info(f"New best validation BLEU: {val_metrics['bleu']:.4f}")
            else:
                self.no_improve_count += 1
                self.logger.info(f"No improvement for {self.no_improve_count} epochs")
            
            # Check config for save settings
            save_best_only = self.config["output"].get("save_best_only", True)
            
            # Only save checkpoint if it's the best model or if we're not in save_best_only mode
            if is_best or not save_best_only:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if self.no_improve_count >= self.patience:
                self.logger.info(f"Early stopping after {epoch} epochs")
                break
        
        # Calculate training time
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time / 60:.2f} minutes")
        
        # Close TensorBoard logger
        if self.tensorboard:
            self.tensorboard.close()
        
        return self.model, self.history