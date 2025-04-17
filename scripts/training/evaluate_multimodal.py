"""
Evaluation script for multimodal vision-language receipt counter.

Evaluates the model on:
1. Receipt classification accuracy
2. Language generation quality (BLEU, ROUGE)
3. Visualization of attention maps and model outputs
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from data.dataset import create_dataloaders
from models.vision_language.internvl2 import InternVL2MultimodalModel
from utils.device import get_device
from utils.metrics import compute_classification_metrics, compute_nlg_metrics


def load_model(model_path: str, config: Dict) -> InternVL2MultimodalModel:
    """
    Load a trained multimodal model.
    
    Args:
        model_path: Path to model checkpoint
        config: Model configuration
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {model_path}")
    
    # Initialize model
    model = InternVL2MultimodalModel(config=config, pretrained=True)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device
    device = get_device()
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model: InternVL2MultimodalModel, dataloaders: Dict, output_dir: Path) -> Dict:
    """
    Evaluate the model on all metrics.
    
    Args:
        model: Trained model
        dataloaders: Dictionary of dataloaders
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    device = get_device()
    
    # Initialize metrics
    metrics = {
        "classification": {},
        "generation": {}
    }
    
    # Initialize prediction collection
    all_predictions = []
    all_targets = []
    all_generated_texts = []
    all_reference_texts = []
    all_class_predictions = []
    all_class_targets = []
    
    # Evaluate on validation and test sets
    for split, dataloader in dataloaders.items():
        if split == "train":
            continue  # Skip training set evaluation for speed
        
        logger.info(f"Evaluating on {split} set")
        
        # Iterate through batches
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}")):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    text_input_ids=batch["text_input_ids"],
                    attention_mask=batch["text_attention_mask"]
                )
                
                # Classification predictions
                if "logits" in outputs:
                    _, class_preds = outputs["logits"].max(1)
                    all_class_predictions.extend(class_preds.cpu().tolist())
                    all_class_targets.extend(batch["classification_labels"].cpu().tolist())
                
                # Text generation (for a subset to save time)
                if batch_idx % 2 == 0:  # Increase frequency of evaluation
                    # Generate responses
                    try:
                        logger.info(f"Generating responses for batch {batch_idx}")
                        generated_ids, decoded_texts = model.generate_response(
                            pixel_values=batch["pixel_values"],
                            text_input_ids=batch["text_input_ids"],
                            attention_mask=batch["text_attention_mask"],
                            max_length=50
                        )
                        
                        # Decode reference texts
                        tokenizer = model.tokenizer
                        reference_texts = [
                            tokenizer.decode(batch["labels"][i], skip_special_tokens=True)
                            for i in range(len(decoded_texts))
                        ]
                        
                        # Log raw generation results
                        logger.info(f"Generated {len(decoded_texts)} responses")
                        for i in range(min(2, len(decoded_texts))):
                            logger.info(f"Sample {i}: Q: {tokenizer.decode(batch['text_input_ids'][i], skip_special_tokens=True)}")
                            logger.info(f"         P: {decoded_texts[i]}")
                            logger.info(f"         R: {reference_texts[i]}")
                        
                        # Add to collection
                        all_generated_texts.extend(decoded_texts)
                        all_reference_texts.extend(reference_texts)
                        
                        # Save all examples for inspection
                        for i in range(len(decoded_texts)):
                            all_predictions.append({
                                "image_idx": batch_idx * dataloader.batch_size + i,
                                "question": tokenizer.decode(
                                    batch["text_input_ids"][i], skip_special_tokens=True
                                ),
                                "prediction": decoded_texts[i],
                                "reference": reference_texts[i],
                                "receipt_count": batch["receipt_count"][i].item() if "receipt_count" in batch else "N/A"
                            })
                    except Exception as e:
                        logger.warning(f"Error during text generation: {e}")
                        logger.warning(f"Exception details: {type(e).__name__}: {str(e)}")
    
    # Compute classification metrics
    if all_class_predictions and all_class_targets:
        metrics["classification"] = compute_classification_metrics(
            all_class_predictions, all_class_targets
        )
        logger.info(f"Classification metrics: {metrics['classification']}")
    
    # Compute text generation metrics
    if all_generated_texts and all_reference_texts:
        metrics["generation"] = compute_nlg_metrics(
            all_generated_texts, all_reference_texts
        )
        logger.info(f"Generation metrics: {metrics['generation']}")
    
    # Print some examples for debugging
    logger.info("Example predictions vs references:")
    for i, pred in enumerate(all_predictions[:5]):
        logger.info(f"Example {i+1}:")
        logger.info(f"  Question: {pred.get('question', 'N/A')}")
        logger.info(f"  Prediction: {pred.get('prediction', 'N/A')}")
        logger.info(f"  Reference: {pred.get('reference', 'N/A')}")
        logger.info(f"  Receipt count: {pred.get('receipt_count', 'N/A')}")
        logger.info("---")
    
    # Save metrics and examples
    metrics_path = output_dir / "metrics.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    examples_path = output_dir / "examples.json"
    with open(examples_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return metrics


def visualize_attention(
    model: InternVL2MultimodalModel,
    image_path: str,
    question: str,
    output_dir: Path
) -> None:
    """
    Visualize model attention maps.
    
    Args:
        model: Trained model
        image_path: Path to input image
        question: Question about the image
        output_dir: Directory to save output visualizations
    """
    logger = logging.getLogger(__name__)
    device = get_device()
    
    # Load and preprocess image
    from torchvision import transforms
    
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tokenize question
    inputs = model.prepare_inputs(image_tensor, [question])
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            text_input_ids=inputs["text_input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Generate response
        generated_ids, decoded_texts = model.generate_response(
            pixel_values=inputs["pixel_values"],
            text_input_ids=inputs["text_input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        response = decoded_texts[0]
        
        # Get attention maps if available
        attention_maps = model.get_attention_maps(inputs["pixel_values"])
    
    # Create figure for visualization
    fig = plt.figure(figsize=(12, 8))
    
    # Plot original image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis("off")
    
    # Plot attention map (if available)
    if attention_maps:
        # Use the last layer attention
        attn_map = attention_maps[-1]
        
        # If attention is multi-head, average over heads
        if len(attn_map.shape) > 3:
            attn_map = attn_map.mean(dim=1)
        
        # Average over batch dimension if needed
        if len(attn_map.shape) > 2:
            attn_map = attn_map[0]
        
        # Convert to numpy and reshape to image dimensions
        attn_np = attn_map.cpu().numpy()
        attn_reshaped = attn_np.mean(axis=0)  # Average over sequence dimension
        
        # Scale for better visualization
        attn_min, attn_max = attn_reshaped.min(), attn_reshaped.max()
        attn_normalized = (attn_reshaped - attn_min) / (attn_max - attn_min)
        
        # Plot attention map
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(attn_normalized.reshape(14, 14), cmap="hot")
        ax2.set_title("Attention Map")
        ax2.axis("off")
    
    # Add question and response as text
    plt.figtext(0.5, 0.02, f"Q: {question}", ha="center", fontsize=12)
    plt.figtext(0.5, 0.06, f"A: {response}", ha="center", fontsize=12)
    
    # Save figure
    output_path = output_dir / "attention_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    logger.info(f"Visualization saved to {output_path}")


def main():
    """Main function for model evaluation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate multimodal receipt counter")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="config/multimodal_config.yaml",
                        help="Path to model configuration")
    parser.add_argument("--output-dir", type=str, default="results/multimodal",
                        help="Directory to save evaluation results")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Optional path to a single image for visualization")
    parser.add_argument("--question", type=str, default="How many receipts are in this image?",
                        help="Question to ask about the image")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level)
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Load model
    model = load_model(args.model_path, config)
    
    # Evaluate model if not just visualizing
    if args.image_path is None:
        # Create dataloaders
        dataloaders = create_dataloaders(config)
        
        # Evaluate model
        metrics = evaluate_model(model, dataloaders, output_dir)
    
    # Visualize if image path provided
    if args.image_path:
        visualize_attention(model, args.image_path, args.question, output_dir)


if __name__ == "__main__":
    main()