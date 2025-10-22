"""
Utility functions for MLOps features including logging and experiment tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

from src.config import Config


def setup_logging(config: Config) -> logging.Logger:
    """
    Set up logging configuration for MLOps compliance.

    Args:
        config: Configuration object

    Returns:
        Configured logger instance
    """
    log_dir = config.paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'clustering_pipeline_{timestamp}.log'

    # Configure logging
    log_level = getattr(logging, config.mlops.log_level)
    log_format = config.mlops.log_format

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if config.mlops.log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info("="*80)
    logger.info("Logging initialized")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    return logger


class ExperimentTracker:
    """Tracks experiments and metrics for MLOps compliance."""

    def __init__(self, config: Config):
        """
        Initialize experiment tracker.

        Args:
            config: Configuration object
        """
        self.config = config
        self.experiment_name = config.mlops.experiment_name
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.metrics = {}
        self.parameters = {}
        self.artifacts = {}

        # Create experiment directory
        self.experiment_dir = config.paths.outputs_dir / 'experiments' / self.run_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Experiment tracker initialized: {self.experiment_name} - Run {self.run_id}")

    def log_parameter(self, key: str, value: Any):
        """
        Log a parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if not self.config.mlops.track_parameters:
            return

        self.parameters[key] = value
        logging.debug(f"Parameter logged: {key} = {value}")

    def log_parameters(self, params: Dict[str, Any]):
        """
        Log multiple parameters.

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            self.log_parameter(key, value)

    def log_metric(self, key: str, value: float):
        """
        Log a metric.

        Args:
            key: Metric name
            value: Metric value
        """
        if not self.config.mlops.track_metrics:
            return

        self.metrics[key] = value
        logging.info(f"Metric logged: {key} = {value}")

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_artifact(self, name: str, path: Path):
        """
        Log an artifact (file).

        Args:
            name: Artifact name
            path: Path to artifact file
        """
        if not self.config.mlops.track_artifacts:
            return

        self.artifacts[name] = str(path)
        logging.debug(f"Artifact logged: {name} at {path}")

    def save_experiment(self):
        """Save experiment data to JSON file."""
        experiment_data = {
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'parameters': self.parameters,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'config': self.config.to_dict(),
        }

        output_file = self.experiment_dir / 'experiment.json'
        with open(output_file, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)

        logging.info(f"Experiment data saved to {output_file}")

    def print_summary(self):
        """Print experiment summary."""
        logging.info("="*80)
        logging.info("EXPERIMENT SUMMARY")
        logging.info("="*80)
        logging.info(f"Experiment: {self.experiment_name}")
        logging.info(f"Run ID: {self.run_id}")
        logging.info(f"\nParameters ({len(self.parameters)}):")
        for key, value in self.parameters.items():
            logging.info(f"  {key}: {value}")

        logging.info(f"\nMetrics ({len(self.metrics)}):")
        for key, value in self.metrics.items():
            logging.info(f"  {key}: {value}")

        logging.info(f"\nArtifacts ({len(self.artifacts)}):")
        for key, path in self.artifacts.items():
            logging.info(f"  {key}: {path}")
        logging.info("="*80)


def save_environment_info(config: Config):
    """
    Save environment information for reproducibility.

    Args:
        config: Configuration object
    """
    if not config.mlops.save_environment:
        return

    import platform
    import sys

    env_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'timestamp': datetime.now().isoformat(),
    }

    # Try to get package versions
    try:
        import pandas
        import numpy
        import sklearn

        env_info['packages'] = {
            'pandas': pandas.__version__,
            'numpy': numpy.__version__,
            'scikit-learn': sklearn.__version__,
        }
    except Exception as e:
        logging.warning(f"Could not capture package versions: {e}")

    output_file = config.paths.outputs_dir / 'environment.json'
    with open(output_file, 'w') as f:
        json.dump(env_info, f, indent=2)

    logging.info(f"Environment information saved to {output_file}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
