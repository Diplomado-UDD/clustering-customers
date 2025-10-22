"""
Configuration module for MLOps-compliant settings.

This module centralizes all configuration parameters for data generation,
preprocessing, clustering, and reporting to enable reproducibility and
easy experimentation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import yaml


@dataclass
class DataGeneratorConfig:
    """Configuration for customer data generation."""

    num_customers: int = 5000
    random_seed: int = 42
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"

    # Product catalog
    product_categories: list = None
    price_ranges: Dict[str, tuple] = None

    def __post_init__(self):
        if self.product_categories is None:
            self.product_categories = [
                "Electronics", "Clothing", "Home & Garden",
                "Books", "Sports", "Beauty", "Toys"
            ]
        if self.price_ranges is None:
            self.price_ranges = {
                "Electronics": (50, 2000),
                "Clothing": (20, 300),
                "Home & Garden": (30, 1000),
                "Books": (10, 100),
                "Sports": (25, 500),
                "Beauty": (15, 200),
                "Toys": (10, 150)
            }


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    scaling_method: str = "standard"  # standard, minmax, robust
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore
    outlier_threshold: float = 1.5

    # Feature engineering
    enable_rfm: bool = True
    enable_clv: bool = True
    enable_behavioral_features: bool = True


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""

    # K-means parameters
    kmeans_min_k: int = 2
    kmeans_max_k: int = 10
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300

    # DBSCAN parameters
    dbscan_eps_range: tuple = (0.3, 2.0)
    dbscan_eps_steps: int = 20
    dbscan_min_samples_range: tuple = (3, 10)
    dbscan_metric: str = "euclidean"

    # Evaluation
    use_silhouette: bool = True
    use_davies_bouldin: bool = True
    use_calinski_harabasz: bool = True


@dataclass
class ReportingConfig:
    """Configuration for visualization and reporting."""

    # Visualization settings
    figure_size: tuple = (12, 8)
    figure_dpi: int = 300
    plot_style: str = "seaborn-v0_8-darkgrid"
    color_palette: str = "husl"

    # Dashboard settings
    generate_interactive: bool = True
    dashboard_template: str = "plotly_white"

    # Report settings
    report_format: str = "markdown"
    include_business_insights: bool = True
    include_recommendations: bool = True


@dataclass
class MLOpsConfig:
    """Configuration for MLOps features."""

    # Experiment tracking
    experiment_name: str = "customer_clustering"
    track_metrics: bool = True
    track_parameters: bool = True
    track_artifacts: bool = True

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Versioning
    model_version: str = "v1.0.0"
    data_version: str = "v1.0.0"

    # Reproducibility
    enable_random_seed: bool = True
    save_environment: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths."""

    # Base directories
    base_dir: Path = Path(__file__).parent.parent
    outputs_dir: Path = None
    data_dir: Path = None
    viz_dir: Path = None
    reports_dir: Path = None
    logs_dir: Path = None

    def __post_init__(self):
        if self.outputs_dir is None:
            self.outputs_dir = self.base_dir / "outputs"
        if self.data_dir is None:
            self.data_dir = self.outputs_dir / "data"
        if self.viz_dir is None:
            self.viz_dir = self.outputs_dir / "visualizations"
        if self.reports_dir is None:
            self.reports_dir = self.outputs_dir / "reports"
        if self.logs_dir is None:
            self.logs_dir = self.outputs_dir / "logs"

        # Create directories if they don't exist
        for directory in [self.data_dir, self.viz_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class that aggregates all sub-configurations."""

    def __init__(self, config_file: str = None):
        """
        Initialize configuration from file or defaults.

        Args:
            config_file: Optional path to YAML configuration file
        """
        self.paths = PathConfig()
        self.data_generator = DataGeneratorConfig()
        self.preprocessing = PreprocessingConfig()
        self.clustering = ClusteringConfig()
        self.reporting = ReportingConfig()
        self.mlops = MLOpsConfig()

        if config_file:
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Update configurations from file
        for section, params in config_dict.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in params.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)

    def save_to_file(self, config_file: str):
        """Save current configuration to YAML file."""
        config_dict = {
            'data_generator': self.data_generator.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'clustering': self.clustering.__dict__,
            'reporting': self.reporting.__dict__,
            'mlops': self.mlops.__dict__,
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_generator': self.data_generator.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'clustering': self.clustering.__dict__,
            'reporting': self.reporting.__dict__,
            'mlops': self.mlops.__dict__,
        }


# Default configuration instance
default_config = Config()
