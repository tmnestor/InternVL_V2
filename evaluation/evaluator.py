"""
Evaluation module for InternVL2 receipt counter.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models.vision_language.internvl2 import InternVL2ReceiptClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from utils.device import get_device, to_device


class InternVL2Evaluator:
    """
    Evaluator for the InternVL2 receipt counter model.
    
    Provides comprehensive evaluation metrics and visualizations.
    """
    
    def __init__(
        self,
        model: InternVL2ReceiptClassifier,
        dataloaders: Dict[str, DataLoader],
        config: Dict[str, Any],
        output_dir: Path,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained InternVL2 model
            dataloaders: Dictionary with dataloaders
            config: Evaluation configuration
            output_dir: Directory to save outputs
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()
        
        # Get metrics list
        self.metrics = config["evaluation"]["metrics"]
        
        # Setup loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Get logger
        self.logger = logging.getLogger(__name__)
        
        # For numerical labels to text mapping
        self.label_names = {
            0: "No Receipt",
            1: "One Receipt",
            2: "Multiple Receipts"
        }
        
    def evaluate(self, split: str = "val") -> Dict[str, Any]:
        """
        Evaluate the model on the specified data split.
        
        Args:
            split: Data split to evaluate on (train, val, or test)
            
        Returns:
            Dictionary with evaluation results
        """
        if split not in self.dataloaders:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(self.dataloaders.keys())}")
        
        # Get dataloader
        dataloader = self.dataloaders[split]
        
        # Initialize lists to store predictions and targets
        all_preds = []
        all_targets = []
        all_probs = []
        running_loss = 0.0
        
        # Collect predictions and targets
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                # Move data to device
                images, targets = to_device((images, targets), self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs["logits"], targets)
                
                # Convert to probabilities and predictions
                probs = torch.nn.functional.softmax(outputs["logits"], dim=1)
                _, preds = outputs["logits"].max(1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update running loss
                running_loss += loss.item()
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        results = {}
        results["loss"] = running_loss / len(dataloader)
        results["metrics"] = self._calculate_metrics(all_targets, all_preds, all_probs)
        
        # Generate and save visualizations if enabled
        if self.config["evaluation"].get("confusion_matrix", False):
            fig = self._plot_confusion_matrix(all_targets, all_preds)
            fig_path = self.output_dir / f"confusion_matrix_{split}.png"
            fig.savefig(fig_path)
            plt.close(fig)
        
        if self.config["evaluation"].get("class_report", False):
            report = self._generate_classification_report(all_targets, all_preds)
            report_path = self.output_dir / f"classification_report_{split}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
        
        # Save metrics to file
        metrics_path = self.output_dir / f"metrics_{split}.json"
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=4)
            
        # Log summary
        self.logger.info(f"Evaluation on {split} set complete.")
        self.logger.info(f"Loss: {results['loss']:.4f}")
        for metric, value in results["metrics"].items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return results
    
    def _calculate_metrics(self, targets: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            targets: Ground truth labels
            preds: Model predictions
            probs: Probability distributions
            
        Returns:
            Dictionary with metric values
        """
        metrics = {}
        
        if "accuracy" in self.metrics:
            metrics["accuracy"] = float(accuracy_score(targets, preds))
        
        if "balanced_accuracy" in self.metrics:
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
        
        if "precision" in self.metrics:
            metrics["precision"] = float(precision_score(targets, preds, average="macro", zero_division=0))
        
        if "recall" in self.metrics:
            metrics["recall"] = float(recall_score(targets, preds, average="macro", zero_division=0))
        
        if "f1_score" in self.metrics:
            metrics["f1_score"] = float(f1_score(targets, preds, average="macro", zero_division=0))
        
        # Calculate per-class metrics
        num_classes = self.config["model"]["num_classes"]
        for i in range(num_classes):
            class_name = self.label_names[i]
            
            # Convert to binary problem for each class
            binary_targets = (targets == i).astype(int)
            binary_preds = (preds == i).astype(int)
            
            # Class precision and recall
            if "precision" in self.metrics:
                metrics[f"precision_{class_name}"] = float(
                    precision_score(binary_targets, binary_preds, zero_division=0)
                )
            
            if "recall" in self.metrics:
                metrics[f"recall_{class_name}"] = float(
                    recall_score(binary_targets, binary_preds, zero_division=0)
                )
        
        return metrics
    
    def _plot_confusion_matrix(self, targets: np.ndarray, preds: np.ndarray) -> plt.Figure:
        """
        Create confusion matrix visualization.
        
        Args:
            targets: Ground truth labels
            preds: Model predictions
            
        Returns:
            Matplotlib figure
        """
        # Compute confusion matrix
        cm = confusion_matrix(targets, preds)
        
        # Normalize to get percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display the confusion matrix
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Setup tick marks and labels
        num_classes = self.config["model"]["num_classes"]
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([self.label_names[i] for i in range(num_classes)])
        ax.set_yticklabels([self.label_names[i] for i in range(num_classes)])
        
        # Label axes
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        
        # Add text annotations
        threshold = cm_norm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > threshold else "black")
        
        plt.tight_layout()
        return fig
    
    def _generate_classification_report(self, targets: np.ndarray, preds: np.ndarray) -> Dict[str, Any]:
        """
        Generate classification report.
        
        Args:
            targets: Ground truth labels
            preds: Model predictions
            
        Returns:
            Classification report as dictionary
        """
        # Get class names
        num_classes = self.config["model"]["num_classes"]
        target_names = [self.label_names[i] for i in range(num_classes)]
        
        # Generate classification report
        report = classification_report(
            targets, preds, target_names=target_names, 
            output_dict=True, zero_division=0
        )
        
        return report
    
    def visualize_attention(self, split: str = "val", num_samples: int = 5) -> None:
        """
        Visualize attention maps for a few samples.
        
        Args:
            split: Data split to use for visualization
            num_samples: Number of samples to visualize
        """
        if split not in self.dataloaders:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(self.dataloaders.keys())}")
        
        # Get dataloader and sample a batch
        dataloader = self.dataloaders[split]
        images, targets = next(iter(dataloader))
        
        # Limit to specified number of samples
        images = images[:num_samples]
        targets = targets[:num_samples]
        
        # Move to device
        images, targets = to_device((images, targets), self.device)
        
        # Get attention maps from the model
        with torch.no_grad():
            attention_maps = self.model.get_attention_maps(images)
            outputs = self.model(images)
            _, preds = outputs["logits"].max(1)
        
        # Setup directory for attention visualizations
        viz_dir = self.output_dir / "attention_maps"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to numpy for plotting
        images_np = images.cpu().numpy()
        targets_np = targets.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        # Normalize images for display
        images_np = np.transpose(images_np, (0, 2, 3, 1))  # BCHW -> BHWC
        images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
        
        # Choose one attention layer to visualize (typically the last layer)
        last_layer_attn = attention_maps[-1]  # Shape: [B, num_heads, seq_len, seq_len]
        
        # For each sample
        for i in range(min(num_samples, len(images))):
            # Create figure with subplots
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot original image
            axs[0].imshow(images_np[i])
            axs[0].set_title(f"True: {self.label_names[targets_np[i]]}, Pred: {self.label_names[preds_np[i]]}")
            axs[0].axis('off')
            
            # Average attention across heads
            avg_attn = last_layer_attn[i].mean(0).cpu().numpy()  # Shape: [seq_len, seq_len]
            
            # Plot attention map
            im = axs[1].imshow(avg_attn, cmap='viridis')
            axs[1].set_title("Attention Map (Avg across heads)")
            plt.colorbar(im, ax=axs[1])
            
            # Save figure
            plt.tight_layout()
            plt.savefig(viz_dir / f"attention_sample_{i}.png")
            plt.close(fig)
        
        self.logger.info(f"Attention maps saved to {viz_dir}")
        
    def evaluate_all_splits(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate the model on all available data splits.
        
        Returns:
            Dictionary with evaluation results for each split
        """
        results = {}
        
        for split in self.dataloaders.keys():
            self.logger.info(f"Evaluating on {split} split...")
            split_results = self.evaluate(split)
            results[split] = split_results
            
            self.logger.info(f"{split.capitalize()} Results:")
            for metric, value in split_results["metrics"].items():
                self.logger.info(f"  {metric}: {value:.4f}")
        
        # Save combined results
        results_path = self.output_dir / "all_results.json"
        with open(results_path, "w") as f:
            # Convert float values to be JSON serializable
            json_results = {
                split: {
                    "loss": float(res["loss"]),
                    "metrics": res["metrics"]
                }
                for split, res in results.items()
            }
            json.dump(json_results, f, indent=4)
        
        return results