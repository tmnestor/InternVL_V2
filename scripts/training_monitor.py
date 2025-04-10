"""
Training monitoring dashboard for Phase 4 of the InternVL Receipt Counter.

This script:
1. Provides a real-time dashboard for monitoring training progress
2. Compares multiple experiments
3. Visualizes training metrics
4. Generates reports
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from flask import Flask, render_template, jsonify, send_from_directory

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


class TrainingMonitor:
    """
    Monitor training progress across multiple experiments.
    
    Extracts metrics from tensorboard logs and creates visualizations.
    """
    
    def __init__(self, experiments_dir: str):
        """
        Initialize the training monitor.
        
        Args:
            experiments_dir: Directory containing experiments
        """
        self.experiments_dir = Path(experiments_dir)
        if not self.experiments_dir.exists():
            raise ValueError(f"Experiments directory does not exist: {experiments_dir}")
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Dictionary to store experiment data
        self.experiments_data = {}
        
        # Scan for experiments
        self._scan_experiments()
    
    def _scan_experiments(self) -> None:
        """
        Scan for experiments in the experiments directory.
        """
        self.logger.info(f"Scanning for experiments in {self.experiments_dir}")
        
        # Look for experiment directories
        experiment_dirs = [d for d in self.experiments_dir.iterdir() if d.is_dir()]
        
        # Skip special directories
        experiment_dirs = [d for d in experiment_dirs if not d.name.startswith(".")]
        
        self.logger.info(f"Found {len(experiment_dirs)} experiment directories")
        
        # Extract experiment data
        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            
            # Check for config file
            config_path = exp_dir / "config.yaml"
            if not config_path.exists():
                config_path = exp_dir / "model" / "config.yaml"
            
            # Check for tensorboard logs
            tensorboard_dir = exp_dir / "model" / "tensorboard"
            if not tensorboard_dir.exists():
                tensorboard_dir = exp_dir / "tensorboard"
            
            # Check for metrics
            metrics_path = exp_dir / "evaluation" / "metrics.json"
            
            # Store paths
            self.experiments_data[exp_name] = {
                "dir": exp_dir,
                "config_path": config_path if config_path.exists() else None,
                "tensorboard_dir": tensorboard_dir if tensorboard_dir.exists() else None,
                "metrics_path": metrics_path if metrics_path.exists() else None,
                "loaded": False
            }
    
    def load_experiment_data(self, exp_name: str) -> Dict:
        """
        Load data for a specific experiment.
        
        Args:
            exp_name: Name of the experiment
            
        Returns:
            Dictionary of experiment data
        """
        if exp_name not in self.experiments_data:
            self.logger.error(f"Experiment not found: {exp_name}")
            return {}
        
        # Skip if already loaded
        if self.experiments_data[exp_name]["loaded"]:
            return self.experiments_data[exp_name]
        
        exp_data = self.experiments_data[exp_name]
        
        # Load config if available
        if exp_data["config_path"]:
            try:
                import yaml
                with open(exp_data["config_path"], "r") as f:
                    exp_data["config"] = yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Error loading config for {exp_name}: {e}")
                exp_data["config"] = {}
        
        # Load tensorboard metrics if available
        if exp_data["tensorboard_dir"]:
            try:
                exp_data["metrics"] = self._extract_tensorboard_metrics(exp_data["tensorboard_dir"])
            except Exception as e:
                self.logger.error(f"Error loading tensorboard metrics for {exp_name}: {e}")
                exp_data["metrics"] = {}
        
        # Load evaluation metrics if available
        if exp_data["metrics_path"]:
            try:
                with open(exp_data["metrics_path"], "r") as f:
                    exp_data["evaluation"] = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading evaluation metrics for {exp_name}: {e}")
                exp_data["evaluation"] = {}
        
        # Mark as loaded
        exp_data["loaded"] = True
        
        return exp_data
    
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
    
    def get_all_experiments(self) -> List[str]:
        """
        Get list of all experiments.
        
        Returns:
            List of experiment names
        """
        return list(self.experiments_data.keys())
    
    def get_experiment_summary(self, exp_name: str) -> Dict:
        """
        Get summary of experiment metrics.
        
        Args:
            exp_name: Name of the experiment
            
        Returns:
            Dictionary of summarized metrics
        """
        # Load data if not loaded
        exp_data = self.load_experiment_data(exp_name)
        
        if not exp_data:
            return {"error": f"Experiment not found: {exp_name}"}
        
        # Create summary
        summary = {
            "name": exp_name,
        }
        
        # Extract final metrics from training
        if "metrics" in exp_data and exp_data["metrics"]:
            metrics = exp_data["metrics"]
            
            # Get final values for key metrics
            for metric_name, metric_data in metrics.items():
                if metric_data and "values" in metric_data and metric_data["values"]:
                    summary[f"final_{metric_name}"] = metric_data["values"][-1]
        
        # Add evaluation metrics if available
        if "evaluation" in exp_data and exp_data["evaluation"]:
            evaluation = exp_data["evaluation"]
            
            # Add classification metrics
            if "classification" in evaluation:
                for k, v in evaluation["classification"].items():
                    summary[f"eval_{k}"] = v
            
            # Add generation metrics
            if "generation" in evaluation:
                for k, v in evaluation["generation"].items():
                    if isinstance(v, (int, float)):
                        summary[f"eval_{k}"] = v
        
        return summary
    
    def compare_experiments(self, exp_names: List[str]) -> Dict:
        """
        Compare multiple experiments.
        
        Args:
            exp_names: List of experiment names to compare
            
        Returns:
            Comparison dictionary
        """
        # Validate experiments
        valid_exp_names = [name for name in exp_names if name in self.experiments_data]
        
        if not valid_exp_names:
            return {"error": "No valid experiments to compare"}
        
        # Load data for all experiments
        for exp_name in valid_exp_names:
            self.load_experiment_data(exp_name)
        
        # Get summaries
        summaries = {exp_name: self.get_experiment_summary(exp_name) for exp_name in valid_exp_names}
        
        # Collect all metrics
        all_metrics = set()
        for exp_name, summary in summaries.items():
            all_metrics.update(summary.keys())
        
        # Filter out non-numeric and metadata fields
        exclude_fields = {"name", "dir", "config_path", "tensorboard_dir", "metrics_path", "loaded", "error"}
        metric_names = [m for m in all_metrics if m not in exclude_fields and not m.startswith(("config_", "dir_"))]
        
        # Create comparison table
        comparison = {
            "experiments": valid_exp_names,
            "metrics": {},
            "best": {}
        }
        
        # Fill in metrics table
        for metric in sorted(metric_names):
            comparison["metrics"][metric] = {}
            
            valid_values = []
            
            for exp_name in valid_exp_names:
                value = summaries[exp_name].get(metric)
                comparison["metrics"][metric][exp_name] = value
                
                if value is not None and isinstance(value, (int, float)):
                    valid_values.append((exp_name, value))
            
            # Find best for this metric if we have values
            if valid_values:
                # Determine if higher is better
                higher_is_better = not any(m in metric.lower() for m in ["loss", "error", "perplexity"])
                
                if higher_is_better:
                    best_exp, best_value = max(valid_values, key=lambda x: x[1])
                else:
                    best_exp, best_value = min(valid_values, key=lambda x: x[1])
                
                comparison["best"][metric] = {
                    "experiment": best_exp,
                    "value": best_value
                }
        
        return comparison
    
    def create_comparison_plot(
        self,
        exp_names: List[str],
        metric_name: str,
        output_path: Optional[Path] = None
    ) -> Optional[plt.Figure]:
        """
        Create comparison plot for a specific metric.
        
        Args:
            exp_names: List of experiment names to compare
            metric_name: Name of the metric to plot
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure if successful, None otherwise
        """
        # Validate experiments
        valid_exp_names = [name for name in exp_names if name in self.experiments_data]
        
        if not valid_exp_names:
            self.logger.error("No valid experiments to compare")
            return None
        
        # Load data for all experiments
        for exp_name in valid_exp_names:
            self.load_experiment_data(exp_name)
        
        # Check if metric exists in any experiment
        metric_data = {}
        
        for exp_name in valid_exp_names:
            exp_data = self.experiments_data[exp_name]
            
            if "metrics" in exp_data and metric_name in exp_data["metrics"]:
                metric_data[exp_name] = exp_data["metrics"][metric_name]
        
        if not metric_data:
            self.logger.error(f"Metric {metric_name} not found in any experiment")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metric for each experiment
        for exp_name, data in metric_data.items():
            if "steps" in data and "values" in data:
                ax.plot(data["steps"], data["values"], label=exp_name)
        
        # Add labels and title
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Comparison of {metric_name}")
        ax.legend()
        ax.grid(True)
        
        # Save figure if requested
        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            return None
        
        return fig
    
    def create_summary_report(
        self,
        output_path: Path,
        exp_names: Optional[List[str]] = None
    ) -> None:
        """
        Create summary report of all experiments.
        
        Args:
            output_path: Path to save the report
            exp_names: Optional list of experiment names to include
        """
        # Use all experiments if none specified
        if exp_names is None:
            exp_names = self.get_all_experiments()
        
        # Validate experiments
        valid_exp_names = [name for name in exp_names if name in self.experiments_data]
        
        if not valid_exp_names:
            self.logger.error("No valid experiments to report")
            return
        
        # Load data for all experiments
        for exp_name in valid_exp_names:
            self.load_experiment_data(exp_name)
        
        # Get summaries
        summaries = {exp_name: self.get_experiment_summary(exp_name) for exp_name in valid_exp_names}
        
        # Create report
        report = {
            "generated_at": datetime.now().isoformat(),
            "experiments": valid_exp_names,
            "experiment_data": summaries
        }
        
        # Add comparison
        if len(valid_exp_names) > 1:
            report["comparison"] = self.compare_experiments(valid_exp_names)
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Summary report saved to {output_path}")


class MonitoringDashboard:
    """
    Flask-based dashboard for real-time monitoring of training.
    """
    
    def __init__(self, experiments_dir: str, host: str = "127.0.0.1", port: int = 8080):
        """
        Initialize the monitoring dashboard.
        
        Args:
            experiments_dir: Directory containing experiments
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.experiments_dir = Path(experiments_dir)
        self.host = host
        self.port = port
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Create monitor
        self.monitor = TrainingMonitor(experiments_dir)
        
        # Create Flask app
        self.app = Flask(
            __name__,
            template_folder=project_root / "scripts" / "templates",
            static_folder=project_root / "scripts" / "static"
        )
        
        # Create template and static directories if they don't exist
        (project_root / "scripts" / "templates").mkdir(exist_ok=True)
        (project_root / "scripts" / "static").mkdir(exist_ok=True)
        
        # Create basic HTML template
        self._create_templates()
        
        # Set up routes
        self._setup_routes()
    
    def _create_templates(self) -> None:
        """
        Create basic HTML templates for the dashboard.
        """
        templates_dir = project_root / "scripts" / "templates"
        static_dir = project_root / "scripts" / "static"
        
        # Create base template
        base_template = """<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Training Monitor{% endblock %}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.6.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body { padding-top: 70px; }
        .navbar { background-color: #343a40; }
        .navbar-brand { color: white; }
        .nav-link { color: rgba(255,255,255,.75); }
        .nav-link:hover { color: white; }
        .card { margin-bottom: 20px; }
    </style>
    {% block head %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">InternVL Training Monitor</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/experiments">Experiments</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/compare">Compare</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/refresh">Refresh</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container">
        {% block content %}{% endblock %}
    </div>

    <footer class="footer mt-5">
        <div class="container">
            <hr>
            <p class="text-muted text-center">InternVL Receipt Counter - Phase 4 Monitoring</p>
        </div>
    </footer>

    {% block scripts %}{% endblock %}
</body>
</html>
"""
        
        # Create index template
        index_template = """{% extends "base.html" %}

{% block title %}Training Monitor - Dashboard{% endblock %}

{% block content %}
    <h1 class="mb-4">Training Dashboard</h1>
    
    <div class="row">
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Experiments</h5>
                </div>
                <div class="card-body">
                    <p>Total Experiments: <span id="experiment-count">{{ experiments|length }}</span></p>
                    <div class="list-group">
                        {% for exp_name in experiments %}
                            <a href="/experiment/{{ exp_name }}" class="list-group-item list-group-item-action">{{ exp_name }}</a>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Recent Training Progress</h5>
                </div>
                <div class="card-body">
                    {% if experiments %}
                        <canvas id="progress-chart" width="600" height="400"></canvas>
                    {% else %}
                        <p class="text-center">No experiments found.</p>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <div class="row mt-4">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Best Performing Experiments</h5>
                </div>
                <div class="card-body">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Best Experiment</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody id="best-experiments">
                            {% if best_metrics %}
                                {% for metric, data in best_metrics.items() %}
                                    <tr>
                                        <td>{{ metric }}</td>
                                        <td><a href="/experiment/{{ data.experiment }}">{{ data.experiment }}</a></td>
                                        <td>{{ "%.4f"|format(data.value) }}</td>
                                    </tr>
                                {% endfor %}
                            {% else %}
                                <tr>
                                    <td colspan="3" class="text-center">No data available</td>
                                </tr>
                            {% endif %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        {% if experiments %}
        // Fetch data for chart
        $.getJSON('/api/metrics/train/loss', function(data) {
            var ctx = document.getElementById('progress-chart').getContext('2d');
            var datasets = [];
            
            for (var exp in data) {
                datasets.push({
                    label: exp,
                    data: data[exp].values.map((value, index) => ({
                        x: data[exp].steps[index],
                        y: value
                    })),
                    borderColor: getRandomColor(),
                    fill: false,
                    tension: 0.1
                });
            }
            
            var chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'Training Steps'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Training Loss'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Training Loss Across Experiments'
                        }
                    }
                }
            });
        });
        {% endif %}
        
        // Random color generator
        function getRandomColor() {
            var letters = '0123456789ABCDEF';
            var color = '#';
            for (var i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
            }
            return color;
        }
    });
</script>
{% endblock %}
"""
        
        # Create experiment template
        experiment_template = """{% extends "base.html" %}

{% block title %}Training Monitor - {{ experiment }}{% endblock %}

{% block content %}
    <h1 class="mb-4">{{ experiment }}</h1>
    
    <div class="row">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Training Loss</h5>
                </div>
                <div class="card-body">
                    <canvas id="loss-chart" width="400" height="300"></canvas>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Training Accuracy</h5>
                </div>
                <div class="card-body">
                    <canvas id="accuracy-chart" width="400" height="300"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mt-4">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">BLEU Score</h5>
                </div>
                <div class="card-body">
                    <canvas id="bleu-chart" width="400" height="300"></canvas>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Learning Rate</h5>
                </div>
                <div class="card-body">
                    <canvas id="lr-chart" width="400" height="300"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mt-4">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Evaluation Metrics</h5>
                </div>
                <div class="card-body">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% if evaluation %}
                                {% for category, metrics in evaluation.items() %}
                                    {% if category not in ['training_metrics', 'error'] %}
                                        {% for name, value in metrics.items() %}
                                            {% if value is number %}
                                                <tr>
                                                    <td>{{ category }}.{{ name }}</td>
                                                    <td>{{ "%.4f"|format(value) }}</td>
                                                </tr>
                                            {% endif %}
                                        {% endfor %}
                                    {% endif %}
                                {% endfor %}
                            {% else %}
                                <tr>
                                    <td colspan="2" class="text-center">No evaluation data available</td>
                                </tr>
                            {% endif %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        // Fetch training metrics
        $.getJSON('/api/experiment/{{ experiment }}/metrics', function(data) {
            // Loss chart
            if (data['train/loss'] && data['val/loss']) {
                createChart('loss-chart', 
                    [data['train/loss'], data['val/loss']], 
                    ['Training Loss', 'Validation Loss'],
                    'Steps', 'Loss', 'Training and Validation Loss');
            }
            
            // Accuracy chart
            if (data['train/accuracy'] && data['val/accuracy']) {
                createChart('accuracy-chart', 
                    [data['train/accuracy'], data['val/accuracy']], 
                    ['Training Accuracy', 'Validation Accuracy'],
                    'Steps', 'Accuracy (%)', 'Training and Validation Accuracy');
            }
            
            // BLEU chart
            if (data['train/bleu'] && data['val/bleu']) {
                createChart('bleu-chart', 
                    [data['train/bleu'], data['val/bleu']], 
                    ['Training BLEU', 'Validation BLEU'],
                    'Steps', 'BLEU Score', 'Training and Validation BLEU');
            }
            
            // Learning rate chart
            if (data['train/lr']) {
                createChart('lr-chart', 
                    [data['train/lr']], 
                    ['Learning Rate'],
                    'Steps', 'Learning Rate', 'Learning Rate Schedule');
            }
        });
        
        // Create chart with multiple datasets
        function createChart(chartId, datasets, labels, xLabel, yLabel, title) {
            var ctx = document.getElementById(chartId).getContext('2d');
            var colors = ['#007bff', '#dc3545', '#28a745', '#ffc107'];
            
            var chartDatasets = datasets.map((dataset, index) => ({
                label: labels[index],
                data: dataset.values.map((value, i) => ({
                    x: dataset.steps[i],
                    y: value
                })),
                borderColor: colors[index],
                fill: false,
                tension: 0.1
            }));
            
            var chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: chartDatasets
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: xLabel
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: yLabel
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: title
                        }
                    }
                }
            });
        }
    });
</script>
{% endblock %}
"""
        
        # Create compare template
        compare_template = """{% extends "base.html" %}

{% block title %}Training Monitor - Compare Experiments{% endblock %}

{% block content %}
    <h1 class="mb-4">Compare Experiments</h1>
    
    <div class="row">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Select Experiments to Compare</h5>
                </div>
                <div class="card-body">
                    <form id="compare-form">
                        <div class="mb-3">
                            <label class="form-label">Experiments:</label>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="select-all">
                                <label class="form-check-label" for="select-all">Select All</label>
                            </div>
                            <hr>
                            {% for exp_name in experiments %}
                                <div class="form-check">
                                    <input class="form-check-input experiment-checkbox" type="checkbox" name="experiments" value="{{ exp_name }}" id="exp-{{ exp_name }}">
                                    <label class="form-check-label" for="exp-{{ exp_name }}">{{ exp_name }}</label>
                                </div>
                            {% endfor %}
                        </div>
                        
                        <div class="mb-3">
                            <label for="metric" class="form-label">Metric:</label>
                            <select class="form-select" id="metric" name="metric">
                                <option value="train/loss">Training Loss</option>
                                <option value="val/loss">Validation Loss</option>
                                <option value="train/accuracy">Training Accuracy</option>
                                <option value="val/accuracy">Validation Accuracy</option>
                                <option value="train/bleu">Training BLEU</option>
                                <option value="val/bleu">Validation BLEU</option>
                                <option value="train/lr">Learning Rate</option>
                            </select>
                        </div>
                        
                        <button type="submit" class="btn btn-primary">Compare</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mt-4" id="comparison-chart-container" style="display: none;">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Comparison Chart</h5>
                </div>
                <div class="card-body">
                    <canvas id="comparison-chart" width="800" height="400"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mt-4" id="comparison-table-container" style="display: none;">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title mb-0">Comparison Table</h5>
                </div>
                <div class="card-body">
                    <table class="table table-striped" id="comparison-table">
                        <thead>
                            <tr id="comparison-table-header">
                                <th>Metric</th>
                            </tr>
                        </thead>
                        <tbody id="comparison-table-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
{% endblock %}

{% block scripts %}
<script>
    $(document).ready(function() {
        var comparisonChart;
        
        // Select/Deselect All
        $('#select-all').change(function() {
            $('.experiment-checkbox').prop('checked', $(this).prop('checked'));
        });
        
        // Form submission
        $('#compare-form').submit(function(e) {
            e.preventDefault();
            
            var selectedExps = [];
            $('input[name="experiments"]:checked').each(function() {
                selectedExps.push($(this).val());
            });
            
            var selectedMetric = $('#metric').val();
            
            if (selectedExps.length === 0) {
                alert('Please select at least one experiment to compare.');
                return;
            }
            
            // Fetch comparison data
            $.getJSON('/api/compare', {
                experiments: selectedExps.join(','),
                metric: selectedMetric
            }, function(data) {
                // Show chart container
                $('#comparison-chart-container').show();
                
                // Create chart
                createComparisonChart(selectedMetric, data);
                
                // Create comparison table
                createComparisonTable(selectedExps);
            });
        });
        
        // Create comparison chart
        function createComparisonChart(metricName, data) {
            var ctx = document.getElementById('comparison-chart').getContext('2d');
            
            // Destroy previous chart if exists
            if (comparisonChart) {
                comparisonChart.destroy();
            }
            
            var datasets = [];
            var colors = ['#007bff', '#dc3545', '#28a745', '#ffc107', '#6f42c1', '#fd7e14', '#20c997', '#e83e8c'];
            
            var i = 0;
            for (var exp in data) {
                datasets.push({
                    label: exp,
                    data: data[exp].values.map((value, index) => ({
                        x: data[exp].steps[index],
                        y: value
                    })),
                    borderColor: colors[i % colors.length],
                    fill: false,
                    tension: 0.1
                });
                i++;
            }
            
            // Get axis labels
            var metricParts = metricName.split('/');
            var yLabel = metricParts[1].charAt(0).toUpperCase() + metricParts[1].slice(1);
            
            comparisonChart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'Training Steps'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: yLabel
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Comparison of ' + metricName
                        }
                    }
                }
            });
        }
        
        // Create comparison table
        function createComparisonTable(experiments) {
            // Fetch comparison data for all metrics
            $.getJSON('/api/compare/metrics', {
                experiments: experiments.join(',')
            }, function(data) {
                // Show table container
                $('#comparison-table-container').show();
                
                // Create header
                var header = '<th>Metric</th>';
                experiments.forEach(function(exp) {
                    header += '<th>' + exp + '</th>';
                });
                $('#comparison-table-header').html(header);
                
                // Create rows
                var rows = '';
                var metrics = Object.keys(data);
                
                metrics.sort(); // Sort metrics alphabetically
                
                metrics.forEach(function(metric) {
                    rows += '<tr>';
                    rows += '<td>' + metric + '</td>';
                    
                    experiments.forEach(function(exp) {
                        var value = data[metric][exp];
                        if (value !== undefined && value !== null) {
                            if (typeof value === 'number') {
                                value = value.toFixed(4);
                                
                                // Highlight best value
                                var bestExp = data.best && data.best[metric] ? data.best[metric].experiment : null;
                                if (bestExp === exp) {
                                    rows += '<td class="table-success">' + value + '</td>';
                                } else {
                                    rows += '<td>' + value + '</td>';
                                }
                            } else {
                                rows += '<td>' + value + '</td>';
                            }
                        } else {
                            rows += '<td>-</td>';
                        }
                    });
                    
                    rows += '</tr>';
                });
                
                $('#comparison-table-body').html(rows);
            });
        }
    });
</script>
{% endblock %}
"""
        
        # Write templates
        with open(templates_dir / "base.html", "w") as f:
            f.write(base_template)
        
        with open(templates_dir / "index.html", "w") as f:
            f.write(index_template)
        
        with open(templates_dir / "experiment.html", "w") as f:
            f.write(experiment_template)
        
        with open(templates_dir / "compare.html", "w") as f:
            f.write(compare_template)
    
    def _setup_routes(self) -> None:
        """
        Set up routes for the dashboard.
        """
        # Main routes
        @self.app.route("/")
        def index():
            experiments = self.monitor.get_all_experiments()
            
            # Get best metrics if we have multiple experiments
            best_metrics = {}
            if len(experiments) > 1:
                comparison = self.monitor.compare_experiments(experiments)
                if "best" in comparison:
                    best_metrics = comparison["best"]
            
            return render_template("index.html", experiments=experiments, best_metrics=best_metrics)
        
        @self.app.route("/experiments")
        def experiments():
            experiments = self.monitor.get_all_experiments()
            return render_template("index.html", experiments=experiments)
        
        @self.app.route("/experiment/<exp_name>")
        def experiment(exp_name):
            exp_data = self.monitor.load_experiment_data(exp_name)
            evaluation = exp_data.get("evaluation", {})
            return render_template("experiment.html", experiment=exp_name, evaluation=evaluation)
        
        @self.app.route("/compare")
        def compare():
            experiments = self.monitor.get_all_experiments()
            return render_template("compare.html", experiments=experiments)
        
        @self.app.route("/refresh")
        def refresh():
            self.monitor = TrainingMonitor(self.experiments_dir)
            return self.app.redirect("/")
        
        # API routes
        @self.app.route("/api/experiments")
        def api_experiments():
            experiments = self.monitor.get_all_experiments()
            return jsonify(experiments)
        
        @self.app.route("/api/experiment/<exp_name>")
        def api_experiment(exp_name):
            exp_data = self.monitor.load_experiment_data(exp_name)
            return jsonify(exp_data)
        
        @self.app.route("/api/experiment/<exp_name>/metrics")
        def api_experiment_metrics(exp_name):
            exp_data = self.monitor.load_experiment_data(exp_name)
            return jsonify(exp_data.get("metrics", {}))
        
        @self.app.route("/api/experiment/<exp_name>/summary")
        def api_experiment_summary(exp_name):
            summary = self.monitor.get_experiment_summary(exp_name)
            return jsonify(summary)
        
        @self.app.route("/api/metrics/<category>/<name>")
        def api_metrics(category, name):
            metric_name = f"{category}/{name}"
            experiments = self.monitor.get_all_experiments()
            
            metrics_data = {}
            for exp_name in experiments:
                exp_data = self.monitor.load_experiment_data(exp_name)
                if "metrics" in exp_data and metric_name in exp_data["metrics"]:
                    metrics_data[exp_name] = exp_data["metrics"][metric_name]
            
            return jsonify(metrics_data)
        
        @self.app.route("/api/compare")
        def api_compare():
            experiments = request.args.get("experiments", "").split(",")
            metric = request.args.get("metric", "train/loss")
            
            metrics_data = {}
            for exp_name in experiments:
                exp_data = self.monitor.load_experiment_data(exp_name)
                if "metrics" in exp_data and metric in exp_data["metrics"]:
                    metrics_data[exp_name] = exp_data["metrics"][metric]
            
            return jsonify(metrics_data)
        
        @self.app.route("/api/compare/metrics")
        def api_compare_metrics():
            experiments = request.args.get("experiments", "").split(",")
            
            comparison = self.monitor.compare_experiments(experiments)
            return jsonify(comparison.get("metrics", {}))
    
    def run(self) -> None:
        """
        Run the dashboard.
        """
        self.logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        
        # Open browser
        webbrowser.open(f"http://{self.host}:{self.port}")
        
        # Run app
        self.app.run(host=self.host, port=self.port, debug=False)


def main():
    """Main function for the monitoring dashboard."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training monitoring dashboard")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Directory containing experiments")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to bind the server to")
    parser.add_argument("--report", action="store_true",
                        help="Generate summary report instead of starting dashboard")
    parser.add_argument("--output", type=str, default="experiments/summary_report.json",
                        help="Path to save summary report")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated list of experiments to include in report")
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
    
    # Create experiments directory if it doesn't exist
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        logger.warning(f"Creating experiments directory: {experiments_dir}")
        experiments_dir.mkdir(parents=True)
    
    # Generate report or start dashboard
    if args.report:
        logger.info(f"Generating summary report: {args.output}")
        
        monitor = TrainingMonitor(args.experiments_dir)
        
        if args.experiments:
            exp_names = args.experiments.split(",")
        else:
            exp_names = monitor.get_all_experiments()
        
        monitor.create_summary_report(Path(args.output), exp_names)
        
        logger.info(f"Report generated: {args.output}")
    else:
        logger.info(f"Starting monitoring dashboard at http://{args.host}:{args.port}")
        
        dashboard = MonitoringDashboard(args.experiments_dir, args.host, args.port)
        dashboard.run()


if __name__ == "__main__":
    main()