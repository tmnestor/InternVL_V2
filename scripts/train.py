#!/usr/bin/env python3
"""
Train the InternVL2 receipt counter model.

This script provides a command-line interface for training the model
using the configuration specified in the config file.
"""
import argparse
import logging
from pathlib import Path

import torch
import yaml
from config import load_config
from data.dataset import create_dataloaders
from models.internvl2 import InternVL2ReceiptClassifier
from training.trainer import InternVL2Trainer
from utils.device import get_device_properties
from utils.logging import setup_logger
from utils.reproducibility import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train InternVL2 receipt counter model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", help="Override output directory from config")
    parser.add_argument("--model_path", help="Path to pre-downloaded model (overrides config)")
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.output_dir:
        config["output"]["model_dir"] = args.output_dir
    if args.model_path:
        config["model"]["pretrained_path"] = args.model_path
    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["optimizer"]["learning_rate"] = args.lr
    if args.debug:
        config["debug"] = True
        config["log_level"] = "debug"
    
    # Setup logging
    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    output_dir = Path(config["output"]["model_dir"])
    log_file = output_dir / "training.log" if not config.get("debug", False) else None
    logger = setup_logger(log_level, log_file)
    
    # Print device information
    from utils.device import print_gpu_info
    device_info = get_device_properties()
    logger.info(f"Device: {device_info}")
    
    # More readable GPU information
    logger.info("GPU Information:")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Capability: {torch.cuda.get_device_capability(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        logger.info("No CUDA-compatible GPU found")
    
    # Set random seed
    if "seed" in config:
        set_seed(config["seed"], config.get("deterministic", False))
        logger.info(f"Random seed set to {config['seed']}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    # Create data loaders
    logger.info("Creating data loaders")
    dataloaders = create_dataloaders(config)
    
    # Create model
    logger.info("Creating InternVL2 model")
    model = InternVL2ReceiptClassifier(
        config=config,
        pretrained=True,
        freeze_vision_encoder=config["training"]["three_stage"]["enabled"],
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create trainer
    logger.info("Setting up trainer")
    trainer = InternVL2Trainer(
        config=config,
        model=model,
        dataloaders=dataloaders,
        output_dir=output_dir,
    )
    
    # Train model
    logger.info(f"Starting training for {config['training']['epochs']} epochs")
    model, history = trainer.train()
    
    logger.info("Training complete")
    return model, history


if __name__ == "__main__":
    main()