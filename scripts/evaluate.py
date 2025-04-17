#!/usr/bin/env python3
"""
Evaluate the InternVL2 receipt counter model.

This script provides a command-line interface for evaluating the model
performance on a dataset and generating various metrics and visualizations.
"""
import argparse
import logging
from pathlib import Path

import torch
import yaml
from config import load_config
from data.dataset import create_dataloaders
from evaluation.evaluator import InternVL2Evaluator
from models.vision_language.internvl2 import InternVL2ReceiptClassifier
from utils.logging import setup_logger
from utils.reproducibility import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate InternVL2 receipt counter model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--pretrained_model_path", help="Path to pre-downloaded InternVL2 model (overrides config)")
    parser.add_argument("--output_dir", help="Override output directory from config")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                      help="Dataset split to evaluate on")
    parser.add_argument("--visualize", action="store_true", help="Generate attention visualizations")
    parser.add_argument("--num_samples", type=int, default=10, 
                      help="Number of samples for visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.output_dir:
        config["output"]["results_dir"] = args.output_dir
    if args.pretrained_model_path:
        config["model"]["pretrained_path"] = args.pretrained_model_path
    if args.debug:
        config["debug"] = True
        config["log_level"] = "debug"
    
    # Add model path to config
    config["evaluation"]["model_path"] = args.model_path
    
    # Setup logging
    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    output_dir = Path(config["output"]["results_dir"])
    log_file = output_dir / "evaluation.log" if not config.get("debug", False) else None
    logger = setup_logger(log_level, log_file)
    
    # Set random seed
    if "seed" in config:
        set_seed(config["seed"], config.get("deterministic", False))
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "eval_config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    # Create data loaders
    logger.info("Creating data loaders")
    dataloaders = create_dataloaders(config)
    
    # Check if specified split exists
    if args.split not in dataloaders:
        available_splits = list(dataloaders.keys())
        logger.error(f"Split '{args.split}' not found. Available splits: {available_splits}")
        raise ValueError(f"Split '{args.split}' not found")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = InternVL2ReceiptClassifier(config=config)
    
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Create evaluator
    evaluator = InternVL2Evaluator(
        model=model,
        dataloaders=dataloaders,
        config=config,
        output_dir=output_dir,
    )
    
    # Run evaluation
    logger.info(f"Evaluating on {args.split} set")
    results = evaluator.evaluate(args.split)
    
    # Generate attention visualizations if requested
    if args.visualize:
        logger.info(f"Generating attention visualizations for {args.num_samples} samples")
        evaluator.visualize_attention(args.split, args.num_samples)
    
    # Print summary of results
    logger.info("Evaluation Results:")
    logger.info(f"Loss: {results['loss']:.4f}")
    for metric, value in results["metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    main()