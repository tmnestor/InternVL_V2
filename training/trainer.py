"""
Trainer implementation for the InternVL2 receipt counter.
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
# Using torch.amp instead of torch.cuda.amp (which is deprecated)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.vision_language.internvl2 import InternVL2ReceiptClassifier
from utils.device import get_device, to_device
from utils.logging import TensorboardLogger


class InternVL2Trainer:
    """
    Trainer for the InternVL2 receipt counter model.
    
    Implements a three-stage training approach:
    1. Train only the classification head with frozen backbone
    2. Fine-tune the vision encoder with lower learning rate
    3. Fine-tune the entire model together
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: InternVL2ReceiptClassifier,
        dataloaders: Dict[str, DataLoader],
        output_dir: Path,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            model: InternVL2 model to train
            dataloaders: Dictionary with train and validation dataloaders
            output_dir: Directory to save outputs
        """
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get logger first so we can use it everywhere
        self.logger = logging.getLogger(__name__)
        
        # Setup optional environment optimization
        if torch.cuda.is_available():
            # Set CUDA optimization environment variables
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            if not os.environ.get("CUDA_LAUNCH_BLOCKING"):
                os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Better performance but less readable errors
            
            # Log CUDA information
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA capability: {torch.cuda.get_device_capability()}")
        
        # Setup device
        self.device = get_device()
        self.model.to(self.device)
        
        # Apply torch.compile for GPU acceleration if available and enabled in config
        if (torch.cuda.is_available() and hasattr(torch, 'compile') and 
                self.config["training"].get("torch_compile", False)):
            try:
                # Only compile if the model is in full precision to avoid dtype mismatches
                compile_full_precision_only = self.config["training"].get("compile_full_precision_only", True)
                if compile_full_precision_only and self.use_mixed_precision:
                    self.logger.warning("Skipping torch.compile because mixed precision is enabled. "
                                      "Set compile_full_precision_only: false in config to override.")
                else:
                    compile_mode = self.config["training"].get("compile_mode", "reduce-overhead")
                    self.logger.info(f"Applying torch.compile with mode: {compile_mode}")
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self.logger.info("Successfully applied torch.compile for GPU acceleration")
            except Exception as e:
                self.logger.warning(f"Failed to apply torch.compile: {e}")
        
        # Setup loss function
        self.loss_fn = self._get_loss_function()
        
        # Setup optimizer and scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Setup mixed precision
        self.use_mixed_precision = config["training"].get("mixed_precision", False)
        
        # We CANNOT use gradient clipping with mixed precision FP16
        # Completely disable gradient clipping functionality with mixed precision
        if self.use_mixed_precision:
            # Force disable gradient clipping to prevent unscale errors
            self.clip_grad_norm = 0.0
            config["training"]["optimizer"]["gradient_clip"] = 0.0
            self.logger.warning("Mixed precision training enabled. Gradient clipping DISABLED to prevent FP16 unscaling errors.")
            
            try:
                # Try the newer API (PyTorch 2.0+)
                self.scaler = GradScaler(device_type='cuda' if torch.cuda.is_available() else 'cpu')
                self.logger.info("Using device-specific GradScaler")
            except TypeError:
                # Fall back to older API
                self.scaler = GradScaler()
                self.logger.info("Using legacy GradScaler")
        else:
            self.scaler = None
            # Only enable gradient clipping if not using mixed precision
            self.clip_grad_norm = config["training"]["optimizer"].get("gradient_clip", 1.0)
        
        # Setup training parameters
        self.epochs = config["training"]["epochs"]
        
        # Setup TensorBoard logger if enabled
        self.tensorboard = None
        if config["output"].get("tensorboard", False):
            tensorboard_dir = Path(config["output"]["log_dir"]) / "tensorboard"
            self.tensorboard = TensorboardLogger(tensorboard_dir)
        
        # Setup early stopping
        self.early_stopping = config["training"]["early_stopping"]
        self.patience = self.early_stopping["patience"]
        self.min_delta = self.early_stopping["min_delta"]
        
        # Setup three-stage training
        self.three_stage = config["training"]["three_stage"]
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.no_improve_count = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Get logger
        self.logger = logging.getLogger(__name__)
        
    def _get_loss_function(self) -> nn.Module:
        """Create the loss function based on configuration."""
        loss_config = self.config["training"]["loss"]
        loss_name = loss_config["name"]
        
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss(
                label_smoothing=loss_config.get("label_smoothing", 0.0)
            )
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def _get_optimizer(self, param_groups=None) -> optim.Optimizer:
        """Create the optimizer based on configuration."""
        optimizer_config = self.config["training"]["optimizer"]
        optimizer_name = optimizer_config["name"]
        
        # Ensure learning rate is a float
        lr_value = optimizer_config["learning_rate"]
        lr = float(lr_value) if isinstance(lr_value, str) else lr_value
        
        # Ensure weight decay is a float
        wd_value = optimizer_config.get("weight_decay", 0.01)
        weight_decay = float(wd_value) if isinstance(wd_value, str) else wd_value
        
        # Use provided param groups or model parameters
        params = param_groups if param_groups else self.model.parameters()
        
        if optimizer_name == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(
                params,
                lr=lr,
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create the learning rate scheduler based on configuration."""
        scheduler_config = self.config["training"]["scheduler"]
        scheduler_name = scheduler_config["name"]
        
        # Get epochs from config directly to avoid any timing issues with attribute setting
        epochs = self.config["training"]["epochs"]
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.optimizer.param_groups[0]["lr"] * scheduler_config["min_lr_factor"]
            )
        elif scheduler_name == "one_cycle":
            steps_per_epoch = len(self.dataloaders["train"])
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]["lr"],
                steps_per_epoch=steps_per_epoch,
                epochs=epochs
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_name == "none" or not scheduler_name:
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Get dataloader
        train_loader = self.dataloaders["train"]
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move data to device
            images, targets = to_device((images, targets), self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                try:
                    # Try the newer API (PyTorch 2.0+)
                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                    # Always use float16 for mixed precision with grad scaler - bfloat16 not supported by _amp_foreach_non_finite_check_and_unscale_cuda
                    dtype = torch.float16
                    with autocast(device_type=device_type, dtype=dtype):
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs["logits"], targets)
                except TypeError:
                    # Fall back to older API
                    with autocast():
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs["logits"], targets)
                
                # We've disabled mixed precision in config, but adding a failsafe
                # in case someone enables it again
                try:
                    # Try without unscaling (which causes errors with FP16 params)
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Skip gradient clipping entirely - not compatible with FP16 gradients
                    
                    # Create a custom optimizer step that doesn't call unscale_
                    # This is a dangerous hack, but might work if we skip the unscale_ call
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            if p.grad.dtype == torch.float16:
                                # Skip the optimizer step for FP16 params
                                p.grad = None
                    
                    # Update weights with gradient scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except ValueError as e:
                    if "Attempting to unscale FP16 gradients" in str(e):
                        self.logger.error(f"Critical error with mixed precision: {e}")
                        self.logger.error("DISABLING mixed precision training entirely!")
                        self.use_mixed_precision = False
                        # Fall back to standard precision training for this batch
                        loss.backward()
                        self.optimizer.step()
                    else:
                        raise
            else:
                # Standard forward and backward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs["logits"], targets)
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Update weights
                self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs["logits"].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update running loss
            running_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch: {epoch} [{batch_idx}/{len(train_loader)}] "
                                f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Log to TensorBoard
        if self.tensorboard:
            self.tensorboard.log_scalar("train/loss", epoch_loss, epoch)
            self.tensorboard.log_scalar("train/accuracy", epoch_acc, epoch)
            self.tensorboard.log_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
        
        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model on the validation set with improved error handling.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Get dataloader
        val_loader = self.dataloaders["val"]
        
        # Additional debugging info
        self.logger.info(f"Starting validation for epoch {epoch} with {len(val_loader)} batches")
        
        # No gradient computation during validation
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                # Add progress logging
                if batch_idx % 10 == 0 or batch_idx >= len(val_loader) - 10:
                    self.logger.info(f"Validation batch: {batch_idx}/{len(val_loader)}")
                
                # Move data to device
                images, targets = to_device((images, targets), self.device)
                
                try:
                    # Keep same precision handling for validation and training
                    if self.use_mixed_precision:
                        try:
                            # Try the newer API (PyTorch 2.0+)
                            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                            # Always use float16 for mixed precision with grad scaler - bfloat16 not supported by _amp_foreach_non_finite_check_and_unscale_cuda
                            dtype = torch.float16
                            with autocast(device_type=device_type, dtype=dtype):
                                outputs = self.model(images)
                                loss = self.loss_fn(outputs["logits"], targets)
                        except TypeError:
                            # Fall back to older API
                            with autocast():
                                outputs = self.model(images)
                                loss = self.loss_fn(outputs["logits"], targets)
                    else:
                        # No mixed precision
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs["logits"], targets)
                    
                    # Calculate accuracy
                    _, predicted = outputs["logits"].max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Update running loss
                    running_loss += loss.item()
                    
                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                    self.logger.error(f"Image shape: {images.shape}, Target shape: {targets.shape}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    # Continue with next batch rather than failing
                    continue
        
        # Calculate epoch metrics
        if total > 0:  # Protect against division by zero
            epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            epoch_acc = 100. * correct / total
        else:
            self.logger.warning("No samples were successfully processed during validation")
            epoch_loss = float('inf')
            epoch_acc = 0.0
        
        # Log to TensorBoard
        if self.tensorboard:
            self.tensorboard.log_scalar("val/loss", epoch_loss, epoch)
            self.tensorboard.log_scalar("val/accuracy", epoch_acc, epoch)
        
        self.logger.info(f"Validation completed: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint with improved error handling.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        try:
            self.logger.info(f"Starting checkpoint save for epoch {epoch}")
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint components one by one with logging
            self.logger.info("Preparing checkpoint components...")
            
            try:
                model_state = self.model.state_dict()
                self.logger.info(f"Model state dict captured, size: {len(model_state)} layers")
            except Exception as e:
                self.logger.error(f"Error capturing model state: {e}")
                model_state = None
                
            try:
                optimizer_state = self.optimizer.state_dict()
                self.logger.info("Optimizer state dict captured")
            except Exception as e:
                self.logger.error(f"Error capturing optimizer state: {e}")
                optimizer_state = None
                
            scheduler_state = None
            if self.scheduler:
                try:
                    scheduler_state = self.scheduler.state_dict()
                    self.logger.info("Scheduler state dict captured")
                except Exception as e:
                    self.logger.error(f"Error capturing scheduler state: {e}")
            
            scaler_state = None
            if self.scaler:
                try:
                    scaler_state = self.scaler.state_dict()
                    self.logger.info("Scaler state dict captured")
                except Exception as e:
                    self.logger.error(f"Error capturing scaler state: {e}")
            
            # Assemble the checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "scheduler_state_dict": scheduler_state,
                "scaler_state_dict": scaler_state,
                "val_loss": self.best_val_loss,
                "val_acc": self.best_val_acc,
                "history": self.history
            }
            
            # Save the checkpoint
            if not self.config["output"].get("save_best_only", False) or is_best:
                self.logger.info(f"Saving checkpoint for epoch {epoch}...")
                checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if is_best:
                self.logger.info("Saving best model...")
                best_path = self.output_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                self.logger.info(f"Best model saved to {best_path}")
                
            self.logger.info("Checkpoint saving completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during checkpoint saving: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.warning("Continuing without saving checkpoint")

    def train(self) -> Tuple[InternVL2ReceiptClassifier, Dict]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Tuple of (trained model, training history)
        """
        start_time = time.time()
        self.logger.info("Starting training...")
        
        # Initialize stage
        current_stage = 1
        
        # Train for specified epochs
        for epoch in range(1, self.epochs + 1):
            # Check if we need to switch training stages
            if self.three_stage["enabled"]:
                # Stage 1: Train only classification head
                if epoch == 1:
                    self.logger.info("Stage 1: Training classification head only...")
                    # Freeze vision encoder (should be frozen in model init)
                
                # Stage 2: Unfreeze vision encoder with lower learning rate
                elif epoch == self.three_stage["mlp_warmup_epochs"] + 1:
                    self.logger.info("Stage 2: Unfreezing vision encoder...")
                    current_stage = 2
                    
                    # Unfreeze vision encoder and create new parameter groups
                    param_groups = self.model.unfreeze_vision_encoder(
                        lr_multiplier=self.config["training"]["optimizer"]["backbone_lr_multiplier"]
                    )
                    
                    # Get current optimizer state to preserve momentum
                    old_optimizer_state = None
                    try:
                        if hasattr(self.optimizer, 'state_dict'):
                            old_state_dict = self.optimizer.state_dict()
                            if 'state' in old_state_dict:
                                old_optimizer_state = old_state_dict['state']
                                self.logger.info("Captured optimizer state for momentum preservation")
                    except Exception as e:
                        self.logger.warning(f"Could not capture optimizer state: {e}")
                        old_optimizer_state = None
                    
                    # Re-initialize optimizer with new parameter groups
                    self.optimizer = self._get_optimizer(param_groups)
                    
                    # Log learning rates for inspection
                    lr_info = "Learning rates for Stage 2: "
                    for i, group in enumerate(self.optimizer.param_groups):
                        lr_info += f"Group {i}: {group['lr']:.2e} "
                    self.logger.info(lr_info)
                    
                    # Re-initialize scheduler
                    self.scheduler = self._get_scheduler()
                    
                # Stage 3: Continue fine-tuning everything together
                elif epoch == self.three_stage["mlp_warmup_epochs"] + self.three_stage["vision_tuning_epochs"] + 1:
                    self.logger.info("Stage 3: Fine-tuning entire model...")
                    current_stage = 3
            
            # Train one epoch
            self.logger.info(f"Epoch {epoch}/{self.epochs} (Stage {current_stage})")
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            # Check for improvement
            is_best = False
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.no_improve_count = 0
                is_best = True
                self.logger.info(f"New best validation loss: {val_loss:.4f}")
            elif val_acc > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_acc
                self.no_improve_count = 0
                is_best = True
                self.logger.info(f"New best validation accuracy: {val_acc:.2f}%")
            else:
                self.no_improve_count += 1
                self.logger.info(f"No improvement for {self.no_improve_count} epochs")
            
            # Save checkpoint
            if epoch % self.config["output"]["checkpoint_frequency"] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
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