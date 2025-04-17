"""
Training orchestration script for Phase 4 of the InternVL Receipt Counter.

This script:
1. Manages the full training process with hyperparameter optimization
2. Runs multiple training configurations for ablation studies
3. Monitors metrics and manages experiments
4. Generates visualizations and reports
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tensorboard.backend.event_processing import event_accumulator

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.device import get_device
from utils.reproducibility import set_seed


class TrainingOrchestrator:
    """
    Orchestrates the training process for multiple experiments.
    
    Handles experiment tracking, hyperparameter optimization, 
    and ablation studies.
    """
    
    def __init__(
        self,
        base_config_path: str,
        experiments_dir: str,
        ablation_config_path: Optional[str] = None,
    ):
        """
        Initialize the training orchestrator.
        
        Args:
            base_config_path: Path to base configuration file
            experiments_dir: Directory to store experiments
            ablation_config_path: Optional path to ablation study configuration
        """
        self.base_config_path = Path(base_config_path)
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load base configuration
        with open(self.base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)
        
        # Load ablation configuration if provided
        self.ablation_config = None
        if ablation_config_path:
            with open(ablation_config_path, "r") as f:
                self.ablation_config = yaml.safe_load(f)
        
        # Experiments tracking
        self.experiments = []
        self.current_experiment = None
        
        # Results tracking
        self.results = {}
    
    def run_single_experiment(
        self,
        config: Dict,
        experiment_name: str,
        seed: int = 42,
    ) -> Dict:
        """
        Run a single training experiment.
        
        Args:
            config: Configuration dictionary
            experiment_name: Name of the experiment
            seed: Random seed
            
        Returns:
            Results dictionary
        """
        # Create experiment directory
        experiment_dir = self.experiments_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = experiment_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Set current experiment
        self.current_experiment = {
            "name": experiment_name,
            "config": config,
            "dir": experiment_dir,
            "seed": seed,
            "start_time": datetime.now().isoformat(),
        }
        
        # Run training process
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        try:
            # Execute training script as subprocess for isolation
            cmd = [
                sys.executable,
                str(project_root / "scripts" / "train_multimodal.py"),
                "--config", str(config_path),
                "--output-dir", str(experiment_dir / "model"),
                "--seed", str(seed),
                "--log-level", "INFO"
            ]
            
            # Run training
            training_process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log output
            with open(experiment_dir / "training_log.txt", "w") as f:
                f.write(training_process.stdout)
                if training_process.stderr:
                    f.write("\n\nERRORS:\n\n")
                    f.write(training_process.stderr)
            
            # Run evaluation
            self.logger.info(f"Training completed for {experiment_name}. Running evaluation...")
            eval_cmd = [
                sys.executable,
                str(project_root / "scripts" / "evaluate_multimodal.py"),
                "--model-path", str(experiment_dir / "model" / "best_model.pt"),
                "--config", str(config_path),
                "--output-dir", str(experiment_dir / "evaluation"),
                "--log-level", "INFO"
            ]
            
            eval_process = subprocess.run(
                eval_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log evaluation output
            with open(experiment_dir / "evaluation_log.txt", "w") as f:
                f.write(eval_process.stdout)
                if eval_process.stderr:
                    f.write("\n\nERRORS:\n\n")
                    f.write(eval_process.stderr)
            
            # Collect results
            results_path = experiment_dir / "evaluation" / "metrics.json"
            if results_path.exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
            else:
                results = {"error": "Evaluation results not found"}
            
            # Add training metrics from tensorboard
            tensorboard_dir = experiment_dir / "model" / "tensorboard"
            if tensorboard_dir.exists():
                try:
                    # Extract training curves
                    training_metrics = self._extract_tensorboard_metrics(tensorboard_dir)
                    results["training_metrics"] = training_metrics
                except Exception as e:
                    self.logger.error(f"Error extracting tensorboard metrics: {e}")
                    results["training_metrics"] = {"error": str(e)}
            
            # Update experiment record
            self.current_experiment["end_time"] = datetime.now().isoformat()
            self.current_experiment["results"] = results
            self.current_experiment["status"] = "completed"
            
            # Add to experiments list
            self.experiments.append(self.current_experiment)
            
            # Create experiment summary visualization
            self._create_experiment_summary(experiment_dir, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running experiment {experiment_name}: {e}")
            
            # Update experiment record
            if self.current_experiment:
                self.current_experiment["end_time"] = datetime.now().isoformat()
                self.current_experiment["error"] = str(e)
                self.current_experiment["status"] = "failed"
                self.experiments.append(self.current_experiment)
            
            return {"error": str(e)}
    
    def run_hyperparameter_optimization(
        self,
        param_grid: Dict[str, List],
        base_name: str = "hparam_opt",
        seeds: List[int] = [42],
    ) -> Dict:
        """
        Run hyperparameter optimization experiments.
        
        Args:
            param_grid: Dictionary of parameter names and values to search
            base_name: Base name for experiments
            seeds: Random seeds to use
            
        Returns:
            Results dictionary
        """
        # Generate experiment configurations
        experiment_configs = self._generate_grid_configs(param_grid)
        self.logger.info(f"Generated {len(experiment_configs)} hyperparameter configurations")
        
        # Run experiments
        all_results = []
        
        for i, config in enumerate(experiment_configs):
            for seed in seeds:
                # Create experiment name
                experiment_name = f"{base_name}_{i+1}_seed{seed}"
                
                # Create merged config
                merged_config = self._deep_merge(self.base_config, config)
                
                # Run experiment
                results = self.run_single_experiment(
                    config=merged_config,
                    experiment_name=experiment_name,
                    seed=seed
                )
                
                # Store results
                result_entry = {
                    "experiment": experiment_name,
                    "config": config,
                    "seed": seed,
                    "results": results
                }
                all_results.append(result_entry)
        
        # Analyze results
        analysis = self._analyze_hyperparameter_results(all_results)
        
        # Save analysis
        analysis_path = self.experiments_dir / f"{base_name}_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Create visualization
        self._visualize_hyperparameter_results(all_results, self.experiments_dir / f"{base_name}_results.png")
        
        return {
            "experiments": all_results,
            "analysis": analysis
        }
    
    def run_ablation_studies(self) -> Dict:
        """
        Run ablation studies based on configuration.
        
        Returns:
            Results of ablation studies
        """
        # Check if we have ablation configuration
        if not self.ablation_config:
            self.logger.error("No ablation configuration provided")
            return {"error": "No ablation configuration provided"}
        
        # Get ablation settings
        ablation_params = self.ablation_config.get("ablation_params", {})
        base_name = self.ablation_config.get("base_name", "ablation")
        seeds = self.ablation_config.get("seeds", [42])
        
        # Run baseline experiment first
        baseline_name = f"{base_name}_baseline"
        baseline_results = self.run_single_experiment(
            config=self.base_config,
            experiment_name=baseline_name,
            seed=seeds[0]  # Use first seed for baseline
        )
        
        # Run ablation experiments
        ablation_results = []
        
        for param_name, param_variations in ablation_params.items():
            for i, value in enumerate(param_variations):
                # Create experiment name
                experiment_name = f"{base_name}_{param_name}_{i+1}"
                
                # Create config with ablated parameter
                ablated_config = self._deep_copy(self.base_config)
                
                # Set the ablated parameter, supporting nested paths like "training.loss_weights.language"
                path_parts = param_name.split(".")
                current = ablated_config
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[path_parts[-1]] = value
                
                # Run experiment
                results = self.run_single_experiment(
                    config=ablated_config,
                    experiment_name=experiment_name,
                    seed=seeds[0]  # Use first seed for ablation studies
                )
                
                # Store results
                result_entry = {
                    "experiment": experiment_name,
                    "param_name": param_name,
                    "param_value": value,
                    "results": results
                }
                ablation_results.append(result_entry)
        
        # Analyze results
        analysis = self._analyze_ablation_results(baseline_results, ablation_results)
        
        # Save analysis
        analysis_path = self.experiments_dir / f"{base_name}_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Create visualization
        self._visualize_ablation_results(baseline_results, ablation_results, self.experiments_dir / f"{base_name}_results.png")
        
        return {
            "baseline": baseline_results,
            "ablation_experiments": ablation_results,
            "analysis": analysis
        }
    
    def save_experiments_summary(self) -> None:
        """
        Save summary of all experiments.
        """
        summary_path = self.experiments_dir / "experiments_summary.json"
        
        # Create summary dict
        summary = {
            "experiments_count": len(self.experiments),
            "experiments": self.experiments,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Experiments summary saved to {summary_path}")
    
    def _extract_tensorboard_metrics(self, tensorboard_dir: Path) -> Dict:
        """
        Extract metrics from tensorboard event files.
        
        Args:
            tensorboard_dir: Path to tensorboard directory
            
        Returns:
            Dictionary of extracted metrics
        """
        # Find event file
        event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return {"error": "No tensorboard event files found"}
        
        event_file = str(event_files[0])
        
        # Load event file
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Extract scalars
        scalar_tags = ea.Tags()["scalars"]
        
        metrics = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            metrics[tag] = {"steps": steps, "values": values}
        
        return metrics
    
    def _generate_grid_configs(self, param_grid: Dict[str, List]) -> List[Dict]:
        """
        Generate configurations for grid search.
        
        Args:
            param_grid: Dictionary of parameter names and values
            
        Returns:
            List of configuration dictionaries
        """
        # Generate all combinations
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Create config dictionaries
        configs = []
        for combination in all_combinations:
            config = {}
            
            # Create nested config structure
            for name, value in zip(param_names, combination):
                # Handle nested parameters like "training.optimizer.learning_rate"
                path_parts = name.split(".")
                current = config
                
                # Navigate to the correct level
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[path_parts[-1]] = value
            
            configs.append(config)
        
        return configs
    
    def _analyze_hyperparameter_results(self, results: List[Dict]) -> Dict:
        """
        Analyze results from hyperparameter optimization.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Analysis dictionary
        """
        # Extract key metrics
        metrics_data = []
        
        for result in results:
            if "results" in result and "error" not in result["results"]:
                # Extract config
                config = result["config"]
                
                # Flatten config for analysis
                flat_config = self._flatten_dict(config)
                
                # Extract metrics
                metrics = {}
                
                if "classification" in result["results"]:
                    for k, v in result["results"]["classification"].items():
                        metrics[f"classification_{k}"] = v
                
                if "generation" in result["results"]:
                    for k, v in result["results"]["generation"].items():
                        metrics[f"generation_{k}"] = v
                
                # Combine config and metrics
                entry = {**flat_config, **metrics, "experiment": result["experiment"]}
                metrics_data.append(entry)
        
        # Convert to DataFrame for analysis
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Find best configuration for each metric
            best_configs = {}
            
            # Classification metrics (higher is better)
            class_metrics = [col for col in df.columns if col.startswith("classification_")]
            for metric in class_metrics:
                if df[metric].notna().any():
                    best_idx = df[metric].idxmax()
                    best_configs[f"best_{metric}"] = {
                        "experiment": df.loc[best_idx, "experiment"],
                        "value": df.loc[best_idx, metric],
                        "config": {k: v for k, v in df.loc[best_idx].items() 
                                   if not k.startswith(("classification_", "generation_", "experiment"))}
                    }
            
            # Generation metrics (higher is better for BLEU/ROUGE)
            gen_metrics = [col for col in df.columns if col.startswith("generation_") and not col.endswith("_error")]
            for metric in gen_metrics:
                if df[metric].notna().any():
                    if "perplexity" in metric:
                        # Lower is better for perplexity
                        best_idx = df[metric].idxmin()
                    else:
                        # Higher is better for BLEU/ROUGE
                        best_idx = df[metric].idxmax()
                    
                    best_configs[f"best_{metric}"] = {
                        "experiment": df.loc[best_idx, "experiment"],
                        "value": df.loc[best_idx, metric],
                        "config": {k: v for k, v in df.loc[best_idx].items() 
                                   if not k.startswith(("classification_", "generation_", "experiment"))}
                    }
            
            # Compute correlation between hyperparameters and metrics
            param_columns = [col for col in df.columns 
                           if not col.startswith(("classification_", "generation_", "experiment"))]
            metric_columns = [col for col in df.columns 
                             if col.startswith(("classification_", "generation_"))]
            
            correlations = {}
            for param in param_columns:
                if df[param].nunique() > 1 and not df[param].dtype == 'object':  # Skip constant or non-numeric parameters
                    param_correlations = {}
                    for metric in metric_columns:
                        if df[metric].notna().all():  # Only compute if we have values
                            corr = df[[param, metric]].corr().iloc[0, 1]
                            param_correlations[metric] = corr
                    
                    if param_correlations:
                        correlations[param] = param_correlations
            
            return {
                "best_configurations": best_configs,
                "parameter_correlations": correlations,
                "experiment_count": len(metrics_data)
            }
        
        return {"error": "No valid results found for analysis"}
    
    def _analyze_ablation_results(self, baseline_results: Dict, ablation_results: List[Dict]) -> Dict:
        """
        Analyze results from ablation studies.
        
        Args:
            baseline_results: Results from baseline experiment
            ablation_results: Results from ablation experiments
            
        Returns:
            Analysis dictionary
        """
        # Extract baseline metrics
        baseline_metrics = {}
        
        if "classification" in baseline_results:
            for k, v in baseline_results["classification"].items():
                baseline_metrics[f"classification_{k}"] = v
        
        if "generation" in baseline_results:
            for k, v in baseline_results["generation"].items():
                baseline_metrics[f"generation_{k}"] = v
        
        # Analyze each ablation experiment
        ablation_analysis = []
        
        for ablation in ablation_results:
            if "results" in ablation and "error" not in ablation["results"]:
                # Extract metrics
                ablation_metrics = {}
                
                if "classification" in ablation["results"]:
                    for k, v in ablation["results"]["classification"].items():
                        ablation_metrics[f"classification_{k}"] = v
                
                if "generation" in ablation["results"]:
                    for k, v in ablation["results"]["generation"].items():
                        ablation_metrics[f"generation_{k}"] = v
                
                # Compare with baseline
                metric_diffs = {}
                for metric, value in ablation_metrics.items():
                    if metric in baseline_metrics:
                        if "perplexity" in metric:
                            # Lower is better for perplexity
                            diff = baseline_metrics[metric] - value
                        else:
                            # Higher is better for other metrics
                            diff = value - baseline_metrics[metric]
                        
                        metric_diffs[metric] = {
                            "baseline": baseline_metrics[metric],
                            "ablation": value,
                            "diff": diff,
                            "relative_diff": (diff / baseline_metrics[metric]) * 100 if baseline_metrics[metric] != 0 else 0
                        }
                
                # Add to analysis
                ablation_analysis.append({
                    "experiment": ablation["experiment"],
                    "param_name": ablation["param_name"],
                    "param_value": ablation["param_value"],
                    "metric_differences": metric_diffs
                })
        
        # Summarize importance of each parameter
        param_importance = {}
        
        for ablation in ablation_analysis:
            param_name = ablation["param_name"]
            
            if param_name not in param_importance:
                param_importance[param_name] = {
                    "max_impact": {},
                    "experiments": []
                }
            
            # Add experiment
            param_importance[param_name]["experiments"].append(ablation["experiment"])
            
            # Update max impact
            for metric, diff_data in ablation["metric_differences"].items():
                rel_diff = diff_data["relative_diff"]
                
                if metric not in param_importance[param_name]["max_impact"] or abs(rel_diff) > abs(param_importance[param_name]["max_impact"][metric]["relative_diff"]):
                    param_importance[param_name]["max_impact"][metric] = {
                        "relative_diff": rel_diff,
                        "experiment": ablation["experiment"],
                        "param_value": ablation["param_value"]
                    }
        
        return {
            "baseline_metrics": baseline_metrics,
            "ablation_analysis": ablation_analysis,
            "parameter_importance": param_importance
        }
    
    def _create_experiment_summary(self, experiment_dir: Path, results: Dict) -> None:
        """
        Create summary visualization for an experiment.
        
        Args:
            experiment_dir: Experiment directory
            results: Results dictionary
        """
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # Add title
        fig.suptitle(f"Experiment Summary: {self.current_experiment['name']}", fontsize=16)
        
        # Add training curves if available
        if "training_metrics" in results and "error" not in results["training_metrics"]:
            # Plot loss curves
            ax1 = fig.add_subplot(2, 2, 1)
            train_loss = results["training_metrics"].get("train/loss", {})
            val_loss = results["training_metrics"].get("val/loss", {})
            
            if train_loss and val_loss:
                ax1.plot(train_loss["steps"], train_loss["values"], label="Train Loss")
                ax1.plot(val_loss["steps"], val_loss["values"], label="Validation Loss")
                ax1.set_xlabel("Steps")
                ax1.set_ylabel("Loss")
                ax1.legend()
                ax1.set_title("Training & Validation Loss")
                ax1.grid(True)
            
            # Plot accuracy curves
            ax2 = fig.add_subplot(2, 2, 2)
            train_acc = results["training_metrics"].get("train/accuracy", {})
            val_acc = results["training_metrics"].get("val/accuracy", {})
            
            if train_acc and val_acc:
                ax2.plot(train_acc["steps"], train_acc["values"], label="Train Accuracy")
                ax2.plot(val_acc["steps"], val_acc["values"], label="Validation Accuracy")
                ax2.set_xlabel("Steps")
                ax2.set_ylabel("Accuracy (%)")
                ax2.legend()
                ax2.set_title("Training & Validation Accuracy")
                ax2.grid(True)
            
            # Plot BLEU scores
            ax3 = fig.add_subplot(2, 2, 3)
            train_bleu = results["training_metrics"].get("train/bleu", {})
            val_bleu = results["training_metrics"].get("val/bleu", {})
            
            if train_bleu and val_bleu:
                ax3.plot(train_bleu["steps"], train_bleu["values"], label="Train BLEU")
                ax3.plot(val_bleu["steps"], val_bleu["values"], label="Validation BLEU")
                ax3.set_xlabel("Steps")
                ax3.set_ylabel("BLEU Score")
                ax3.legend()
                ax3.set_title("Training & Validation BLEU")
                ax3.grid(True)
            
            # Plot learning rate
            ax4 = fig.add_subplot(2, 2, 4)
            lr = results["training_metrics"].get("train/lr", {})
            
            if lr:
                ax4.plot(lr["steps"], lr["values"])
                ax4.set_xlabel("Steps")
                ax4.set_ylabel("Learning Rate")
                ax4.set_title("Learning Rate Schedule")
                ax4.grid(True)
        
        # Add final metrics as text
        fig_text = "Final Metrics:\n"
        
        if "classification" in results:
            fig_text += "\nClassification:\n"
            for k, v in results["classification"].items():
                fig_text += f"  {k}: {v:.2f}\n"
        
        if "generation" in results:
            fig_text += "\nGeneration:\n"
            for k, v in results["generation"].items():
                if isinstance(v, (int, float)):
                    fig_text += f"  {k}: {v:.2f}\n"
        
        plt.figtext(0.5, 0.01, fig_text, ha="center", fontsize=10, 
                   bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        # Save figure
        plt.savefig(experiment_dir / "summary.png", dpi=150)
        plt.close()
    
    def _visualize_hyperparameter_results(self, results: List[Dict], output_path: Path) -> None:
        """
        Create visualization for hyperparameter optimization results.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save visualization
        """
        # Extract key metrics and parameters
        metrics_data = []
        
        for result in results:
            if "results" in result and "error" not in result["results"]:
                # Extract config
                config = result["config"]
                
                # Flatten config for analysis
                flat_config = self._flatten_dict(config)
                
                # Extract metrics
                metrics = {}
                
                if "classification" in result["results"]:
                    metrics["accuracy"] = result["results"]["classification"].get("accuracy", 0)
                
                if "generation" in result["results"]:
                    metrics["bleu"] = result["results"]["generation"].get("bleu", 0)
                    metrics["rouge1_f"] = result["results"]["generation"].get("rouge1_f", 0)
                
                # Combine config and metrics
                entry = {**flat_config, **metrics, "experiment": result["experiment"]}
                metrics_data.append(entry)
        
        # Convert to DataFrame for visualization
        if not metrics_data:
            self.logger.warning("No valid results found for visualization")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Identify numerical parameters with multiple values
        param_columns = [col for col in df.columns 
                       if not col.startswith(("accuracy", "bleu", "rouge", "experiment"))]
        varying_params = [col for col in param_columns 
                         if df[col].nunique() > 1 and pd.api.types.is_numeric_dtype(df[col])]
        
        # Create figure
        n_params = min(len(varying_params), 3)  # Limit to top 3 parameters
        
        if n_params == 0:
            self.logger.warning("No varying numerical parameters found for visualization")
            return
        
        fig, axes = plt.subplots(n_params, 2, figsize=(14, 4 * n_params))
        
        if n_params == 1:
            axes = [axes]
        
        fig.suptitle("Hyperparameter Optimization Results", fontsize=16)
        
        # Plot parameter vs. metrics
        for i, param in enumerate(varying_params[:n_params]):
            # Sort by parameter value
            df_sorted = df.sort_values(param)
            
            # Plot accuracy
            ax1 = axes[i][0]
            ax1.plot(df_sorted[param], df_sorted["accuracy"], "o-", label="Accuracy")
            ax1.set_xlabel(param)
            ax1.set_ylabel("Accuracy (%)")
            ax1.set_title(f"{param} vs. Accuracy")
            ax1.grid(True)
            
            # Plot BLEU/ROUGE
            ax2 = axes[i][1]
            ax2.plot(df_sorted[param], df_sorted["bleu"], "o-", label="BLEU")
            ax2.plot(df_sorted[param], df_sorted["rouge1_f"], "o-", label="ROUGE-1")
            ax2.set_xlabel(param)
            ax2.set_ylabel("Score")
            ax2.set_title(f"{param} vs. Text Generation Metrics")
            ax2.legend()
            ax2.grid(True)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def _visualize_ablation_results(
        self,
        baseline_results: Dict,
        ablation_results: List[Dict],
        output_path: Path
    ) -> None:
        """
        Create visualization for ablation study results.
        
        Args:
            baseline_results: Results from baseline experiment
            ablation_results: Results from ablation experiments
            output_path: Path to save visualization
        """
        # Extract baseline metrics
        baseline_metrics = {}
        
        if "classification" in baseline_results:
            baseline_metrics["accuracy"] = baseline_results["classification"].get("accuracy", 0)
        
        if "generation" in baseline_results:
            baseline_metrics["bleu"] = baseline_results["generation"].get("bleu", 0)
            baseline_metrics["rouge1_f"] = baseline_results["generation"].get("rouge1_f", 0)
        
        # Extract metrics for each ablation
        ablation_data = []
        
        for ablation in ablation_results:
            if "results" in ablation and "error" not in ablation["results"]:
                # Extract metrics
                metrics = {
                    "param_name": ablation["param_name"],
                    "param_value": str(ablation["param_value"]),
                    "experiment": ablation["experiment"]
                }
                
                if "classification" in ablation["results"]:
                    metrics["accuracy"] = ablation["results"]["classification"].get("accuracy", 0)
                
                if "generation" in ablation["results"]:
                    metrics["bleu"] = ablation["results"]["generation"].get("bleu", 0)
                    metrics["rouge1_f"] = ablation["results"]["generation"].get("rouge1_f", 0)
                
                ablation_data.append(metrics)
        
        # Convert to DataFrame
        if not ablation_data:
            self.logger.warning("No valid ablation results found for visualization")
            return
        
        df = pd.DataFrame(ablation_data)
        
        # Group by parameter name
        param_groups = df.groupby("param_name")
        n_params = len(param_groups)
        
        if n_params == 0:
            self.logger.warning("No parameter groups found for visualization")
            return
        
        # Create figure
        fig, axes = plt.subplots(n_params, 2, figsize=(14, 4 * n_params))
        
        if n_params == 1:
            axes = [axes]
        
        fig.suptitle("Ablation Study Results", fontsize=16)
        
        # Plot each parameter's impact
        for i, (param_name, group) in enumerate(param_groups):
            # Plot accuracy relative to baseline
            ax1 = axes[i][0]
            baseline_acc = baseline_metrics.get("accuracy", 0)
            
            # Add baseline
            ax1.axhline(y=baseline_acc, color="black", linestyle="--", label="Baseline")
            
            # Add ablation results
            param_values = group["param_value"].tolist()
            acc_values = group["accuracy"].tolist()
            
            # Set x-positions
            x_pos = np.arange(len(param_values))
            
            # Create bar chart
            bars = ax1.bar(x_pos, acc_values, label="Ablation")
            
            # Add baseline bar in different color
            ax1.bar(-1, baseline_acc, color="gray", label="Baseline")
            
            # Set labels
            ax1.set_xticks(np.arange(-1, len(param_values)))
            ax1.set_xticklabels(["Baseline"] + param_values, rotation=45)
            ax1.set_xlabel("Parameter Value")
            ax1.set_ylabel("Accuracy (%)")
            ax1.set_title(f"Impact of {param_name} on Accuracy")
            ax1.legend()
            ax1.grid(True)
            
            # Plot BLEU/ROUGE relative to baseline
            ax2 = axes[i][1]
            baseline_bleu = baseline_metrics.get("bleu", 0)
            baseline_rouge = baseline_metrics.get("rouge1_f", 0)
            
            # Add baselines
            ax2.axhline(y=baseline_bleu, color="blue", linestyle="--", label="Baseline BLEU")
            ax2.axhline(y=baseline_rouge, color="green", linestyle="--", label="Baseline ROUGE-1")
            
            # Add ablation results
            bleu_values = group["bleu"].tolist()
            rouge_values = group["rouge1_f"].tolist()
            
            width = 0.35
            ax2.bar(x_pos - width/2, bleu_values, width, label="BLEU")
            ax2.bar(x_pos + width/2, rouge_values, width, label="ROUGE-1")
            
            # Add baseline bars
            ax2.bar(-1 - width/2, baseline_bleu, width, color="lightblue", label="Baseline BLEU")
            ax2.bar(-1 + width/2, baseline_rouge, width, color="lightgreen", label="Baseline ROUGE-1")
            
            # Set labels
            ax2.set_xticks(np.arange(-1, len(param_values)))
            ax2.set_xticklabels(["Baseline"] + param_values, rotation=45)
            ax2.set_xlabel("Parameter Value")
            ax2.set_ylabel("Score")
            ax2.set_title(f"Impact of {param_name} on Text Generation")
            ax2.legend()
            ax2.grid(True)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursive calls
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _deep_copy(self, d: Dict) -> Dict:
        """
        Create a deep copy of a dictionary.
        
        Args:
            d: Dictionary to copy
            
        Returns:
            Deep copy of dictionary
        """
        return json.loads(json.dumps(d))
    
    def _deep_merge(self, d1: Dict, d2: Dict) -> Dict:
        """
        Deeply merge two dictionaries.
        
        Args:
            d1: First dictionary
            d2: Second dictionary
            
        Returns:
            Merged dictionary
        """
        result = self._deep_copy(d1)
        
        for k, v in d2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        
        return result


def main():
    """Main function for training orchestration."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training orchestration for Phase 4")
    parser.add_argument("--config", type=str, default="config/multimodal_config.yaml",
                        help="Path to base configuration")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Directory to store experiments")
    parser.add_argument("--ablation-config", type=str, default=None,
                        help="Path to ablation study configuration")
    parser.add_argument("--hyperparameter-config", type=str, default=None,
                        help="Path to hyperparameter optimization configuration")
    parser.add_argument("--mode", type=str, choices=["single", "ablation", "hyperparameter", "all"],
                        default="single", help="Experiment mode")
    parser.add_argument("--experiment-name", type=str, default="baseline",
                        help="Name for single experiment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
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
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(
        base_config_path=args.config,
        experiments_dir=args.experiments_dir,
        ablation_config_path=args.ablation_config
    )
    
    # Run experiments based on mode
    if args.mode == "single":
        logger.info(f"Running single experiment: {args.experiment_name}")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        orchestrator.run_single_experiment(
            config=config,
            experiment_name=args.experiment_name,
            seed=args.seed
        )
    
    elif args.mode == "ablation":
        if not args.ablation_config:
            logger.error("Ablation mode requires --ablation-config")
            return
        
        logger.info("Running ablation studies")
        orchestrator.run_ablation_studies()
    
    elif args.mode == "hyperparameter":
        if not args.hyperparameter_config:
            logger.error("Hyperparameter mode requires --hyperparameter-config")
            return
        
        logger.info("Running hyperparameter optimization")
        
        # Load hyperparameter grid
        with open(args.hyperparameter_config, "r") as f:
            hp_config = yaml.safe_load(f)
        
        param_grid = hp_config.get("param_grid", {})
        base_name = hp_config.get("base_name", "hparam_opt")
        seeds = hp_config.get("seeds", [args.seed])
        
        orchestrator.run_hyperparameter_optimization(
            param_grid=param_grid,
            base_name=base_name,
            seeds=seeds
        )
    
    elif args.mode == "all":
        # Run baseline experiment
        logger.info("Running baseline experiment")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        orchestrator.run_single_experiment(
            config=config,
            experiment_name="baseline",
            seed=args.seed
        )
        
        # Run ablation studies if config provided
        if args.ablation_config:
            logger.info("Running ablation studies")
            orchestrator.run_ablation_studies()
        
        # Run hyperparameter optimization if config provided
        if args.hyperparameter_config:
            logger.info("Running hyperparameter optimization")
            
            # Load hyperparameter grid
            with open(args.hyperparameter_config, "r") as f:
                hp_config = yaml.safe_load(f)
            
            param_grid = hp_config.get("param_grid", {})
            base_name = hp_config.get("base_name", "hparam_opt")
            seeds = hp_config.get("seeds", [args.seed])
            
            orchestrator.run_hyperparameter_optimization(
                param_grid=param_grid,
                base_name=base_name,
                seeds=seeds
            )
    
    # Save experiments summary
    orchestrator.save_experiments_summary()
    logger.info("Training orchestration completed")


if __name__ == "__main__":
    main()