#!/usr/bin/env python3
"""
Main entry point for InternVL2 receipt counter.
"""
import argparse
import logging
from pathlib import Path

import torch
from config import load_config
from data.dataset import create_dataloaders
from models.internvl2 import InternVL2ReceiptClassifier
from training.trainer import InternVL2Trainer
from utils.logging import setup_logger
from utils.reproducibility import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="InternVL2 Receipt Counter")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train",
                      help="Operation mode")
    parser.add_argument("--output_dir", help="Override output directory from config")
    parser.add_argument("--model_path", help="Path to pre-downloaded model (overrides config)")
    return parser.parse_args()


def main():
    """Main entry point for training and evaluation."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging first
    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    logger = setup_logger(log_level)
    
    # Override config with command-line arguments
    if args.output_dir:
        config["output"]["model_dir"] = args.output_dir
    if args.model_path:
        config["model"]["pretrained_path"] = args.model_path
        logger.info(f"Using model path from command-line: {args.model_path}")
    
    # Set random seed for reproducibility
    if "seed" in config:
        set_seed(config["seed"], config.get("deterministic", False))
        
    # Create output directories
    output_dir = Path(config["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    logger.info("Creating data loaders")
    dataloaders = create_dataloaders(config)
    
    if args.mode == "train":
        # Create model
        logger.info("Creating InternVL2 model")
        model = InternVL2ReceiptClassifier(
            config=config,
            pretrained=True,
            freeze_vision_encoder=config["training"]["three_stage"]["enabled"],
        )
        
        # Create trainer
        logger.info("Setting up trainer")
        trainer = InternVL2Trainer(
            config=config,
            model=model,
            dataloaders=dataloaders,
            output_dir=output_dir,
        )
        
        # Train model
        logger.info("Starting training")
        model, history = trainer.train()
        
    elif args.mode == "evaluate":
        # Load trained model
        model_path = config["evaluation"]["model_path"]
        logger.info(f"Loading model from {model_path}")
        model = InternVL2ReceiptClassifier(config=config)
        model.load_state_dict(torch.load(model_path))
        
        # Import evaluator module
        from evaluation.evaluator import InternVL2Evaluator
        
        # Create evaluator
        evaluator = InternVL2Evaluator(
            model=model,
            dataloaders=dataloaders,
            config=config,
            output_dir=Path(config["output"]["results_dir"]),
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Print summary of results
        logger.info("Evaluation Results:")
        for metric, value in results["metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()