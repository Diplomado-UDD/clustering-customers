# Online Retailer Customer Clustering Project

This project demonstrates customer segmentation for an online retailer using machine learning clustering algorithms (K-means and DBSCAN). The project generates synthetic customer data and applies clustering techniques to identify distinct customer segments.

## ⚡ Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository-url>
cd clustering-customers

# Install dependencies
uv sync

# Run demo
uv run main.py --demo
```

## 📋 Prerequisites

- **Python 3.11+** - The project is tested with Python 3.11
- **Git** - For cloning the repository
- **uv** - Fast Python package manager (installation instructions below)

## 🚀 Features

- **Synthetic Data Generation**: Uses Faker library to create realistic customer data
- **Data Preprocessing**: Comprehensive data cleaning, feature engineering, and scaling
- **Clustering Algorithms**: 
  - K-means clustering with optimal k selection
  - DBSCAN for density-based clustering with parameter optimization
- **Advanced Analytics**: RFM analysis, customer lifetime value, behavioral segmentation
- **Visualization**: Interactive dashboards and comprehensive charts
- **Business Insights**: Actionable recommendations for marketing strategies
- **Modular Design**: Clean separation of concerns across 4 main modules

## 📦 Installation

### Step 1: Install uv

This project uses `uv` for fast Python package and project management.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/

### Step 2: Clone the Repository

```bash
git clone <repository-url>
cd clustering-customers
```

### Step 3: Install Dependencies

```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

## 🎯 Usage

### Full Analysis
Run the complete clustering pipeline:

```bash
uv run main.py
```

### Quick Demo
Run a quick demo with smaller dataset (recommended for first run):

```bash
uv run main.py --demo
```

### Advanced Options

**Skip specific clustering methods:**
```bash
uv run main.py --skip-kmeans  # Skip K-means clustering
uv run main.py --skip-dbscan  # Skip DBSCAN clustering
```

**Custom number of customers:**
```bash
uv run main.py --num-customers 5000
```

**Use custom configuration:**
```bash
uv run main.py --config path/to/config.yaml
```

## 🏗️ Project Architecture

The project follows a modular pipeline architecture:

```
Data Generator → Preprocessing → Clustering → Reporting
```

### Module Details

1. **`src/data_generator.py`** - Customer data generation using Faker
2. **`src/preprocessing.py`** - Data cleaning and feature engineering
3. **`src/clustering.py`** - Clustering algorithms and analysis
4. **`src/reporting.py`** - Business insights and visualization
5. **`main.py`** - Orchestrates the entire pipeline

## 🏗️ Project Structure

```
clustering-customers/
├── src/                    # Source code modules
├── outputs/                # Generated outputs
│   ├── data/              # CSV files
│   ├── visualizations/    # Charts and dashboards
│   └── reports/           # Analysis reports
├── docs/                   # Documentation
├── main.py                # Main execution script
└── README.md              # This file
```

See `docs/PROJECT_STRUCTURE.md` for detailed structure explanation.

## 📊 Customer Segments Identified

1. **🌟 High-Value Customers**: High spending, frequent purchases, loyal
2. **💰 Bargain Hunters**: Frequent but low-value purchases, price-sensitive
3. **🛍️ Occasional Buyers**: Medium spending, infrequent purchases
4. **🆕 New Customers**: Recent joiners, varied behavior patterns
5. **⚠️ At-Risk Customers**: Low engagement, potential churn candidates
6. **🔍 Outliers**: Unusual purchasing patterns requiring investigation

## 🛠️ Dependencies

- **pandas** (>=2.0.0): Data manipulation and analysis
- **numpy** (>=1.24.0): Numerical computing
- **scikit-learn** (>=1.3.0): Machine learning algorithms
- **matplotlib** (>=3.7.0): Static visualizations
- **seaborn** (>=0.12.0): Statistical visualizations
- **faker** (>=19.0.0): Synthetic data generation
- **plotly** (>=5.15.0): Interactive visualizations

## 📈 Generated Outputs

The project generates comprehensive outputs:

### Data Files (`outputs/data/`)
- `raw_customer_data.csv` - Original synthetic customer data
- `processed_customer_data.csv` - Cleaned and engineered features
- `customer_clusters_kmeans.csv` - K-means cluster assignments
- `customer_clusters_dbscan.csv` - DBSCAN cluster assignments

### Visualizations (`outputs/visualizations/`)
- `optimal_k_analysis.png` - K-means optimization charts
- `clustering_visualization.png` - PCA cluster visualizations
- `kmeans_cluster_comparison.png` - K-means cluster comparison
- `dbscan_cluster_comparison.png` - DBSCAN cluster comparison
- `kmeans_dashboard.html` - Interactive K-means analysis
- `dbscan_dashboard.html` - Interactive DBSCAN analysis

### Reports (`outputs/reports/`)
- `kmeans_clustering_report.md` - Comprehensive K-means analysis
- `dbscan_clustering_report.md` - Comprehensive DBSCAN analysis

### Documentation (`docs/`)
- `DBSCAN_NOISE_EXPLANATION.md` - Detailed explanation of DBSCAN noise points
- `PROJECT_STRUCTURE.md` - Detailed project organization guide

## 🔍 Understanding DBSCAN Noise Points

DBSCAN identifies "noise" points (labeled as -1) - customers that don't fit into any dense cluster. These are often the most interesting customers:

- **🌟 VIP Customers**: Exceptional spending patterns requiring personalized service
- **🚨 Anomalies**: Potential fraud or data quality issues needing investigation  
- **🎯 Niche Segments**: Unique purchasing behaviors worth exploring
- **📊 Outliers**: Extreme values that may represent opportunities or risks

**Key Insight**: Noise points aren't errors - they're valuable business intelligence requiring individual attention.

See `DBSCAN_NOISE_EXPLANATION.md` for detailed analysis and business recommendations.

## 📊 Cluster Numbering

- **K-means clusters**: Start from 1 (Cluster 1, Cluster 2, etc.)
- **DBSCAN clusters**: Start from 1, with noise points labeled as -1
- This makes business reporting more intuitive (no "Cluster 0")
