# Project Structure Documentation

This document provides a detailed explanation of the project's organization and architecture.

## Directory Structure

```
clustering-customers/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management (MLOps)
│   ├── data_generator.py        # Synthetic data generation
│   ├── preprocessing.py         # Data cleaning and feature engineering
│   ├── clustering.py            # Clustering algorithms (K-means, DBSCAN)
│   ├── reporting.py             # Visualization and reporting
│   └── utils.py                 # Utility functions (logging, tracking)
│
├── outputs/                      # Generated outputs
│   ├── data/                    # CSV data files
│   │   ├── customers.csv        # Customer profiles
│   │   ├── transactions.csv     # Transaction records
│   │   ├── raw_customer_data.csv            # Merged raw data
│   │   ├── processed_customer_data.csv      # Processed features
│   │   ├── customer_clusters_kmeans.csv     # K-means results
│   │   └── customer_clusters_dbscan.csv     # DBSCAN results
│   │
│   ├── visualizations/          # Charts and dashboards
│   │   ├── optimal_k_analysis.png           # K-means optimization
│   │   ├── kmeans_clustering_visualization.png
│   │   ├── kmeans_cluster_comparison.png
│   │   ├── dbscan_clustering_visualization.png
│   │   ├── dbscan_cluster_comparison.png
│   │   ├── kmeans_dashboard.html            # Interactive dashboard
│   │   └── dbscan_dashboard.html            # Interactive dashboard
│   │
│   ├── reports/                 # Business insights reports
│   │   ├── kmeans_clustering_report.md
│   │   └── dbscan_clustering_report.md
│   │
│   ├── logs/                    # Execution logs
│   │   └── clustering_pipeline_YYYYMMDD_HHMMSS.log
│   │
│   └── experiments/             # Experiment tracking data
│       └── YYYYMMDD_HHMMSS/    # Timestamped experiment run
│           └── experiment.json  # Parameters, metrics, artifacts
│
├── docs/                         # Documentation
│   ├── PROJECT_STRUCTURE.md     # This file
│   └── DBSCAN_NOISE_EXPLANATION.md
│
├── main.py                       # Main pipeline orchestration
├── pyproject.toml               # Project dependencies and metadata
├── uv.lock                      # Dependency lock file
├── README.md                    # Project overview
└── .gitignore                   # Git ignore rules
```

## Module Descriptions

### 1. `src/config.py` - Configuration Management

**Purpose**: Centralized configuration for MLOps compliance and reproducibility.

**Key Components**:
- `DataGeneratorConfig`: Data generation parameters
- `PreprocessingConfig`: Feature engineering settings
- `ClusteringConfig`: Algorithm hyperparameters
- `ReportingConfig`: Visualization preferences
- `MLOpsConfig`: Experiment tracking settings
- `PathConfig`: File path management

**MLOps Features**:
- Parameter versioning
- Configuration serialization (YAML)
- Environment reproducibility
- Experiment tracking integration

### 2. `src/data_generator.py` - Synthetic Data Generation

**Purpose**: Generate realistic customer data using Faker library.

**Key Functions**:
- `generate_customers()`: Create customer profiles
- `generate_transactions()`: Create purchase history
- `merge_and_aggregate()`: Combine and aggregate data
- `save_data()`: Persist data to CSV files

**Features**:
- Realistic demographic data
- Behavioral customer segments
- Transaction patterns
- Seeded random generation for reproducibility

### 3. `src/preprocessing.py` - Data Preprocessing

**Purpose**: Clean, transform, and engineer features for clustering.

**Key Functions**:
- `clean_data()`: Handle missing values and duplicates
- `engineer_features()`: Create derived features
- `calculate_rfm()`: Recency, Frequency, Monetary analysis
- `calculate_clv()`: Customer Lifetime Value estimation
- `create_behavioral_features()`: Engagement and churn metrics
- `handle_outliers()`: Outlier detection and treatment
- `scale_features()`: Feature normalization

**Feature Engineering**:
- Customer age and lifetime
- Purchase frequency and patterns
- RFM scores
- CLV (simple and advanced)
- Engagement and churn risk scores
- Behavioral segmentation

### 4. `src/clustering.py` - Clustering Algorithms

**Purpose**: Implement K-means and DBSCAN clustering with optimization.

**Key Functions**:
- `find_optimal_k()`: Determine best K for K-means
- `fit_kmeans()`: Train K-means model
- `optimize_dbscan()`: Find optimal DBSCAN parameters
- `fit_dbscan()`: Train DBSCAN model
- `analyze_clusters()`: Compute cluster statistics
- `assign_cluster_names()`: Generate meaningful cluster names

**Evaluation Metrics**:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Elbow method for K-means

**Features**:
- Automated hyperparameter optimization
- Multiple evaluation metrics
- PCA dimensionality reduction
- Cluster profiling and naming

### 5. `src/reporting.py` - Visualization and Reporting

**Purpose**: Generate comprehensive visualizations and business reports.

**Key Functions**:
- `plot_optimal_k_analysis()`: K-means optimization charts
- `plot_clusters_2d()`: PCA scatter plots
- `plot_cluster_comparison()`: Multi-metric comparisons
- `create_interactive_dashboard()`: Plotly dashboards
- `generate_business_report()`: Markdown reports with insights

**Visualizations**:
- Static plots (matplotlib, seaborn)
- Interactive dashboards (Plotly)
- Cluster comparisons
- Business metrics charts

**Reports**:
- Cluster profiles
- Business recommendations
- Key insights
- Executive summaries

### 6. `src/utils.py` - Utility Functions

**Purpose**: MLOps utilities for logging and experiment tracking.

**Key Functions**:
- `setup_logging()`: Configure logging system
- `ExperimentTracker`: Track parameters, metrics, and artifacts
- `save_environment_info()`: Capture environment details
- `format_duration()`: Human-readable time formatting

**MLOps Features**:
- Structured logging
- Experiment versioning
- Metric tracking
- Artifact management
- Environment reproducibility

### 7. `main.py` - Pipeline Orchestration

**Purpose**: Coordinate the complete clustering pipeline.

**Pipeline Stages**:
1. **Data Generation**: Create synthetic customer dataset
2. **Preprocessing**: Clean and engineer features
3. **Clustering**: Apply K-means and DBSCAN
4. **Reporting**: Generate visualizations and insights

**Command-Line Options**:
- `--demo`: Run with smaller dataset
- `--config`: Load custom configuration file
- `--num-customers`: Override customer count
- `--skip-kmeans`: Skip K-means clustering
- `--skip-dbscan`: Skip DBSCAN clustering

**MLOps Integration**:
- Comprehensive logging
- Experiment tracking
- Performance monitoring
- Error handling

## Data Flow

```
1. Data Generation
   CustomerDataGenerator
   ↓
   [customers.csv, transactions.csv, raw_customer_data.csv]

2. Preprocessing
   CustomerDataPreprocessor
   ↓
   [processed_customer_data.csv, scaled_features]

3. Clustering
   CustomerClusterer
   ├── K-means
   │   ↓
   │   [customer_clusters_kmeans.csv]
   └── DBSCAN
       ↓
       [customer_clusters_dbscan.csv]

4. Reporting
   ClusteringReporter
   ↓
   [visualizations/, reports/]

5. Experiment Tracking
   ExperimentTracker
   ↓
   [experiments/TIMESTAMP/experiment.json]
```

## MLOps Architecture

### Configuration Management
- Centralized configuration in `config.py`
- YAML-based configuration files
- Environment-specific settings
- Parameter versioning

### Experiment Tracking
- Automatic parameter logging
- Metric collection
- Artifact management
- Timestamp-based versioning

### Logging
- Structured logging format
- Console and file output
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Timestamped log files

### Reproducibility
- Seeded random number generation
- Configuration persistence
- Environment information capture
- Dependency management with `uv`

### Modularity
- Clean separation of concerns
- Reusable components
- Configurable pipeline stages
- Easy to extend and modify

## Best Practices Implemented

1. **Code Organization**
   - Modular architecture
   - Clear separation of concerns
   - Consistent naming conventions
   - Comprehensive documentation

2. **Configuration Management**
   - Centralized configuration
   - Easy parameterization
   - Environment-specific settings

3. **Data Management**
   - Organized output structure
   - Intermediate data saving
   - Clear data lineage

4. **Experiment Tracking**
   - Parameter logging
   - Metric tracking
   - Artifact management
   - Reproducible experiments

5. **Error Handling**
   - Try-catch blocks
   - Informative error messages
   - Graceful degradation
   - Logging of errors

6. **Performance**
   - Efficient data processing
   - Vectorized operations
   - Progress logging
   - Performance metrics

## Extending the Project

### Adding New Features
1. Add configuration parameters in `config.py`
2. Implement feature engineering in `preprocessing.py`
3. Update clustering logic if needed
4. Add visualizations in `reporting.py`

### Adding New Clustering Algorithms
1. Implement algorithm in `clustering.py`
2. Add configuration in `config.py`
3. Update pipeline in `main.py`
4. Add evaluation metrics
5. Create visualizations

### Customizing Reports
1. Modify templates in `reporting.py`
2. Add new metrics and insights
3. Customize visualization styles
4. Update business recommendations

## Dependencies

Core dependencies managed by `uv`:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **faker**: Synthetic data generation
- **plotly**: Interactive visualizations
- **pyyaml**: Configuration files

## Running the Pipeline

```bash
# Full analysis
uv run python main.py

# Quick demo
uv run python main.py --demo

# Custom configuration
uv run python main.py --config my_config.yaml

# Skip specific algorithms
uv run python main.py --skip-dbscan
```

## Output Artifacts

All outputs are timestamped and versioned for traceability:

- **Data Files**: CSV format for easy analysis
- **Visualizations**: PNG (static) and HTML (interactive)
- **Reports**: Markdown format for readability
- **Logs**: Detailed execution logs
- **Experiments**: JSON metadata for tracking

## Troubleshooting

Common issues and solutions:

1. **Import Errors**: Run `uv sync` to install dependencies
2. **Memory Issues**: Use `--demo` mode for smaller dataset
3. **Clustering Warnings**: Adjust parameters in `config.py`
4. **Visualization Errors**: Check matplotlib backend settings

## Version History

- **v1.0.0** (Current): Initial MLOps-compliant implementation
  - K-means and DBSCAN clustering
  - Comprehensive preprocessing
  - Interactive dashboards
  - Experiment tracking
  - Business insights reporting
