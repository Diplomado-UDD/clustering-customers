"""
Main pipeline orchestration script for customer clustering analysis.

This script orchestrates the complete MLOps-compliant pipeline including:
- Data generation
- Preprocessing and feature engineering
- Clustering (K-means and DBSCAN)
- Visualization and reporting
- Experiment tracking
"""

import argparse
import logging
import time
from pathlib import Path

from src.config import Config
from src.data_generator import CustomerDataGenerator
from src.preprocessing import CustomerDataPreprocessor
from src.clustering import CustomerClusterer
from src.reporting import ClusteringReporter
from src.utils import setup_logging, ExperimentTracker, save_environment_info, format_duration


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Customer Clustering Analysis with MLOps best practices"
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run quick demo with smaller dataset'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML)'
    )

    parser.add_argument(
        '--num-customers',
        type=int,
        default=None,
        help='Number of customers to generate'
    )

    parser.add_argument(
        '--skip-kmeans',
        action='store_true',
        help='Skip K-means clustering'
    )

    parser.add_argument(
        '--skip-dbscan',
        action='store_true',
        help='Skip DBSCAN clustering'
    )

    return parser.parse_args()


def run_pipeline(config: Config, skip_kmeans: bool = False, skip_dbscan: bool = False):
    """
    Run the complete clustering pipeline.

    Args:
        config: Configuration object
        skip_kmeans: Whether to skip K-means clustering
        skip_dbscan: Whether to skip DBSCAN clustering
    """
    logger = logging.getLogger(__name__)
    pipeline_start = time.time()

    # Initialize experiment tracker
    tracker = ExperimentTracker(config)

    # Log configuration parameters
    tracker.log_parameters(config.to_dict())

    logger.info("="*80)
    logger.info("STARTING CUSTOMER CLUSTERING PIPELINE")
    logger.info("="*80)

    # Save environment information
    save_environment_info(config)

    # -------------------------------------------------------------------------
    # STEP 1: Data Generation
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA GENERATION")
    logger.info("="*80)

    step_start = time.time()
    data_generator = CustomerDataGenerator(config)

    customers_df, transactions_df = data_generator.generate_dataset()
    merged_df = data_generator.merge_and_aggregate(customers_df, transactions_df)
    data_generator.save_data(customers_df, transactions_df, merged_df)

    step_duration = time.time() - step_start
    logger.info(f"Data generation completed in {format_duration(step_duration)}")

    # Log data metrics
    tracker.log_metric('num_customers', len(customers_df))
    tracker.log_metric('num_transactions', len(transactions_df))
    tracker.log_artifact('raw_data', config.paths.data_dir / 'raw_customer_data.csv')

    # -------------------------------------------------------------------------
    # STEP 2: Data Preprocessing
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("="*80)

    step_start = time.time()
    preprocessor = CustomerDataPreprocessor(config)

    processed_df, scaled_features, feature_names = preprocessor.preprocess(merged_df)
    preprocessor.save_processed_data(processed_df)

    step_duration = time.time() - step_start
    logger.info(f"Preprocessing completed in {format_duration(step_duration)}")

    # Log preprocessing metrics
    tracker.log_metric('num_features', len(feature_names))
    tracker.log_parameter('features', feature_names)
    tracker.log_artifact('processed_data', config.paths.data_dir / 'processed_customer_data.csv')

    # -------------------------------------------------------------------------
    # STEP 3: Clustering Analysis
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CLUSTERING ANALYSIS")
    logger.info("="*80)

    clusterer = CustomerClusterer(config)

    # Reduce dimensions for visualization
    X_reduced = clusterer.reduce_dimensions(scaled_features, n_components=2)

    # Initialize reporter
    reporter = ClusteringReporter(config)

    # K-means Clustering
    if not skip_kmeans:
        logger.info("\n" + "-"*80)
        logger.info("K-MEANS CLUSTERING")
        logger.info("-"*80)

        step_start = time.time()

        # Find optimal K and fit model
        kmeans_labels = clusterer.fit_kmeans(scaled_features)
        processed_df['kmeans_cluster'] = kmeans_labels

        # Analyze clusters
        kmeans_stats = clusterer.analyze_clusters(
            processed_df, kmeans_labels, feature_names, 'kmeans'
        )

        # Add temporary 'cluster' column for reporting methods
        processed_df['cluster'] = processed_df['kmeans_cluster']

        # Assign cluster names
        kmeans_names = clusterer.assign_cluster_names(
            processed_df, kmeans_stats, 'kmeans'
        )

        # Save results
        clusterer.save_cluster_results(processed_df, kmeans_labels, 'kmeans')

        step_duration = time.time() - step_start
        logger.info(f"K-means clustering completed in {format_duration(step_duration)}")

        # Log K-means metrics
        kmeans_metrics = clusterer.get_metrics().get('kmeans_evaluation', {})
        for key, value in kmeans_metrics.items():
            tracker.log_metric(f'kmeans_{key}', value)

        # Generate visualizations
        logger.info("Generating K-means visualizations...")

        if 'kmeans_optimization' in clusterer.get_metrics():
            reporter.plot_optimal_k_analysis(clusterer.get_metrics()['kmeans_optimization'])

        reporter.plot_clusters_2d(X_reduced, kmeans_labels, kmeans_names, 'kmeans')
        reporter.plot_cluster_comparison(
            processed_df[processed_df['kmeans_cluster'].notna()].copy(),
            kmeans_names,
            'kmeans'
        )
        reporter.create_interactive_dashboard(
            processed_df[processed_df['kmeans_cluster'].notna()].copy(),
            X_reduced,
            kmeans_stats,
            kmeans_names,
            'kmeans'
        )

        # Generate report
        logger.info("Generating K-means business report...")
        reporter.generate_business_report(
            processed_df[processed_df['kmeans_cluster'].notna()].copy(),
            kmeans_stats,
            kmeans_names,
            clusterer.get_metrics(),
            'kmeans'
        )

    # DBSCAN Clustering
    if not skip_dbscan:
        logger.info("\n" + "-"*80)
        logger.info("DBSCAN CLUSTERING")
        logger.info("-"*80)

        step_start = time.time()

        # Fit DBSCAN model
        dbscan_labels = clusterer.fit_dbscan(scaled_features)
        processed_df['dbscan_cluster'] = dbscan_labels

        # Analyze clusters
        dbscan_stats = clusterer.analyze_clusters(
            processed_df, dbscan_labels, feature_names, 'dbscan'
        )

        # Add temporary 'cluster' column for reporting methods
        processed_df['cluster'] = processed_df['dbscan_cluster']

        # Assign cluster names
        dbscan_names = clusterer.assign_cluster_names(
            processed_df, dbscan_stats, 'dbscan'
        )

        # Save results
        clusterer.save_cluster_results(processed_df, dbscan_labels, 'dbscan')

        step_duration = time.time() - step_start
        logger.info(f"DBSCAN clustering completed in {format_duration(step_duration)}")

        # Log DBSCAN metrics
        dbscan_metrics = clusterer.get_metrics().get('dbscan_evaluation', {})
        for key, value in dbscan_metrics.items():
            tracker.log_metric(f'dbscan_{key}', value)

        # Generate visualizations
        logger.info("Generating DBSCAN visualizations...")

        reporter.plot_clusters_2d(X_reduced, dbscan_labels, dbscan_names, 'dbscan')
        reporter.plot_cluster_comparison(
            processed_df[processed_df['dbscan_cluster'].notna()].copy(),
            dbscan_names,
            'dbscan'
        )
        reporter.create_interactive_dashboard(
            processed_df[processed_df['dbscan_cluster'].notna()].copy(),
            X_reduced,
            dbscan_stats,
            dbscan_names,
            'dbscan'
        )

        # Generate report
        logger.info("Generating DBSCAN business report...")
        reporter.generate_business_report(
            processed_df[processed_df['dbscan_cluster'].notna()].copy(),
            dbscan_stats,
            dbscan_names,
            clusterer.get_metrics(),
            'dbscan'
        )

    # -------------------------------------------------------------------------
    # PIPELINE COMPLETE
    # -------------------------------------------------------------------------
    pipeline_duration = time.time() - pipeline_start

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Total duration: {format_duration(pipeline_duration)}")

    # Log pipeline metrics
    tracker.log_metric('pipeline_duration_seconds', pipeline_duration)

    # Save experiment data
    tracker.save_experiment()

    # Print summary
    tracker.print_summary()

    logger.info("\n" + "="*80)
    logger.info("OUTPUT LOCATIONS")
    logger.info("="*80)
    logger.info(f"Data: {config.paths.data_dir}")
    logger.info(f"Visualizations: {config.paths.viz_dir}")
    logger.info(f"Reports: {config.paths.reports_dir}")
    logger.info(f"Logs: {config.paths.logs_dir}")
    logger.info("="*80)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config = Config(config_file=args.config)

    # Demo mode adjustments
    if args.demo:
        print("\n" + "="*80)
        print("RUNNING IN DEMO MODE - Using smaller dataset")
        print("="*80 + "\n")
        config.data_generator.num_customers = 1000
        config.clustering.kmeans_max_k = 6
        config.clustering.dbscan_eps_steps = 10

    # Command-line overrides
    if args.num_customers:
        config.data_generator.num_customers = args.num_customers

    # Setup logging
    setup_logging(config)

    # Run pipeline
    try:
        run_pipeline(
            config,
            skip_kmeans=args.skip_kmeans,
            skip_dbscan=args.skip_dbscan
        )

        print("\n" + "="*80)
        print("✅ Pipeline completed successfully!")
        print("="*80)

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        print("\n" + "="*80)
        print(f"❌ Pipeline failed: {e}")
        print("="*80)
        raise


if __name__ == "__main__":
    main()
