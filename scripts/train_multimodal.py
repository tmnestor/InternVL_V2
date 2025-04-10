"""
Training script for multimodal vision-language receipt counter.

This script implements Phase 3 of the vision-language integration:
1. Training the InternVL-based multimodal model with a custom multi-stage strategy
2. Optimizing for both receipt counting and language generation tasks
3. Evaluating the model's performance with appropriate metrics
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from data.dataset import create_dataloaders
from models.internvl2 import InternVL2MultimodalModel
from training.multimodal_trainer import MultimodalTrainer
from utils.device import get_device
from utils.reproducibility import set_seed


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train multimodal InternVL2 receipt counter")
    parser.add_argument("--config", type=str, default="config/multimodal_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="models/multimodal",
                        help="Output directory for model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
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
    
    # Log system info
    logger.info(f"Using device: {get_device()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config_path = Path(args.config)
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update config with command-line arguments
    config["seed"] = args.seed
    config["log_level"] = args.log_level
    config["output"]["model_dir"] = args.output_dir
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save updated config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create dataloaders
    logger.info("Creating dataloaders")
    dataloaders = create_dataloaders(config)
    
    # Initialize model
    logger.info("Initializing multimodal model")
    model = InternVL2MultimodalModel(
        config=config,
        pretrained=True,
        freeze_vision_encoder=True,  # Will be unfrozen during stage 2
        freeze_language_model=False  # Train language model from the start
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
    
    # Initialize trainer
    logger.info("Initializing multimodal trainer")
    trainer = MultimodalTrainer(
        config=config,
        model=model,
        dataloaders=dataloaders,
        output_dir=output_dir
    )
    
    # Train model
    logger.info("Starting training")
    model, history = trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": config
    }, final_model_path)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()