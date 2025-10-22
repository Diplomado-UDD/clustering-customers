"""
Clustering Module.

This module implements K-means and DBSCAN clustering algorithms with
hyperparameter optimization and evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import logging
from typing import Dict, Tuple, Any

from src.config import Config

logger = logging.getLogger(__name__)


class CustomerClusterer:
    """Performs clustering analysis on customer data."""

    def __init__(self, config: Config):
        """
        Initialize the clusterer.

        Args:
            config: Configuration object containing clustering parameters
        """
        self.config = config
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_model = None
        self.metrics = {}

        logger.info("CustomerClusterer initialized")

    def find_optimal_k(self, X: np.ndarray) -> Tuple[int, Dict[str, list]]:
        """
        Find optimal number of clusters for K-means using elbow method and silhouette score.

        Args:
            X: Scaled feature array

        Returns:
            Tuple of (optimal_k, metrics_dict)
        """
        logger.info("Finding optimal K for K-means")

        min_k = self.config.clustering.kmeans_min_k
        max_k = self.config.clustering.kmeans_max_k
        k_range = range(min_k, max_k + 1)

        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        for k in k_range:
            logger.info(f"Testing K={k}")

            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config.clustering.kmeans_random_state,
                n_init=self.config.clustering.kmeans_n_init,
                max_iter=self.config.clustering.kmeans_max_iter
            )
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)

            if self.config.clustering.use_silhouette and k > 1:
                silhouette_scores.append(silhouette_score(X, labels))

            if self.config.clustering.use_davies_bouldin and k > 1:
                davies_bouldin_scores.append(davies_bouldin_score(X, labels))

            if self.config.clustering.use_calinski_harabasz and k > 1:
                calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))

        # Find optimal K using silhouette score (higher is better)
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores) + 1]  # +1 because silhouette starts at k=2
        else:
            # Fallback to elbow method
            optimal_k = self._find_elbow_point(list(k_range), inertias)

        metrics = {
            'k_values': list(k_range),
            'inertia': inertias,
            'silhouette': silhouette_scores,
            'davies_bouldin': davies_bouldin_scores,
            'calinski_harabasz': calinski_harabasz_scores,
        }

        logger.info(f"Optimal K found: {optimal_k}")
        return optimal_k, metrics

    def _find_elbow_point(self, k_values: list, inertias: list) -> int:
        """Find elbow point in inertia curve using the elbow method."""
        # Simple elbow detection using rate of change
        diffs = np.diff(inertias)
        diffs_ratio = diffs[:-1] / diffs[1:]
        elbow_idx = np.argmax(diffs_ratio) + 2  # +2 because we lost 2 points in diff operations

        # Ensure within bounds
        elbow_idx = min(elbow_idx, len(k_values) - 1)

        return k_values[elbow_idx]

    def fit_kmeans(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """
        Fit K-means clustering model.

        Args:
            X: Scaled feature array
            n_clusters: Number of clusters (if None, will find optimal)

        Returns:
            Cluster labels
        """
        logger.info("Fitting K-means clustering")

        if n_clusters is None:
            n_clusters, optimization_metrics = self.find_optimal_k(X)
            self.metrics['kmeans_optimization'] = optimization_metrics
        else:
            logger.info(f"Using provided n_clusters: {n_clusters}")

        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.clustering.kmeans_random_state,
            n_init=self.config.clustering.kmeans_n_init,
            max_iter=self.config.clustering.kmeans_max_iter
        )

        labels = self.kmeans_model.fit_predict(X)

        # Adjust labels to start from 1 instead of 0
        labels = labels + 1

        # Calculate evaluation metrics
        self._evaluate_clustering(X, labels, 'kmeans')

        logger.info(f"K-means clustering complete with {n_clusters} clusters")
        return labels

    def optimize_dbscan(self, X: np.ndarray) -> Tuple[float, int]:
        """
        Find optimal DBSCAN parameters (eps and min_samples).

        Args:
            X: Scaled feature array

        Returns:
            Tuple of (optimal_eps, optimal_min_samples)
        """
        logger.info("Optimizing DBSCAN parameters")

        eps_min, eps_max = self.config.clustering.dbscan_eps_range
        eps_values = np.linspace(eps_min, eps_max, self.config.clustering.dbscan_eps_steps)

        min_samples_min, min_samples_max = self.config.clustering.dbscan_min_samples_range
        min_samples_values = range(min_samples_min, min_samples_max + 1)

        best_score = -1
        best_params = (eps_values[0], min_samples_values[0])
        results = []

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric=self.config.clustering.dbscan_metric
                )
                labels = dbscan.fit_predict(X)

                # Count number of clusters (excluding noise)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                # Only evaluate if we have at least 2 clusters and not too much noise
                if n_clusters >= 2 and n_noise < len(labels) * 0.5:
                    try:
                        # Use silhouette score for optimization (excluding noise points)
                        mask = labels != -1
                        if mask.sum() > 1:
                            score = silhouette_score(X[mask], labels[mask])

                            results.append({
                                'eps': eps,
                                'min_samples': min_samples,
                                'n_clusters': n_clusters,
                                'n_noise': n_noise,
                                'silhouette': score
                            })

                            if score > best_score:
                                best_score = score
                                best_params = (eps, min_samples)
                    except Exception as e:
                        logger.debug(f"Could not evaluate eps={eps}, min_samples={min_samples}: {e}")

        self.metrics['dbscan_optimization'] = results

        logger.info(f"Optimal DBSCAN parameters: eps={best_params[0]:.3f}, min_samples={best_params[1]}")
        return best_params

    def fit_dbscan(self, X: np.ndarray, eps: float = None, min_samples: int = None) -> np.ndarray:
        """
        Fit DBSCAN clustering model.

        Args:
            X: Scaled feature array
            eps: Maximum distance between samples (if None, will optimize)
            min_samples: Minimum samples in neighborhood (if None, will optimize)

        Returns:
            Cluster labels
        """
        logger.info("Fitting DBSCAN clustering")

        if eps is None or min_samples is None:
            eps, min_samples = self.optimize_dbscan(X)

        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=self.config.clustering.dbscan_metric
        )

        labels = self.dbscan_model.fit_predict(X)

        # Adjust labels: keep -1 for noise, but shift others to start from 1
        labels = np.where(labels == -1, -1, labels + 1)

        # Calculate evaluation metrics
        self._evaluate_clustering(X, labels, 'dbscan')

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"DBSCAN clustering complete with {n_clusters} clusters and {n_noise} noise points")
        return labels

    def _evaluate_clustering(self, X: np.ndarray, labels: np.ndarray, method: str):
        """
        Evaluate clustering quality using multiple metrics.

        Args:
            X: Scaled feature array
            labels: Cluster labels
            method: Clustering method name ('kmeans' or 'dbscan')
        """
        logger.info(f"Evaluating {method} clustering")

        metrics = {}

        # For DBSCAN, exclude noise points from evaluation
        if method == 'dbscan':
            mask = labels != -1
            if mask.sum() <= 1:
                logger.warning("Too few non-noise points for evaluation")
                return
            X_eval = X[mask]
            labels_eval = labels[mask]
        else:
            X_eval = X
            labels_eval = labels

        # Number of clusters
        n_clusters = len(set(labels_eval))
        metrics['n_clusters'] = n_clusters

        if n_clusters > 1:
            # Silhouette Score
            if self.config.clustering.use_silhouette:
                try:
                    metrics['silhouette_score'] = silhouette_score(X_eval, labels_eval)
                except Exception as e:
                    logger.warning(f"Could not calculate silhouette score: {e}")

            # Davies-Bouldin Index
            if self.config.clustering.use_davies_bouldin:
                try:
                    metrics['davies_bouldin_index'] = davies_bouldin_score(X_eval, labels_eval)
                except Exception as e:
                    logger.warning(f"Could not calculate Davies-Bouldin index: {e}")

            # Calinski-Harabasz Index
            if self.config.clustering.use_calinski_harabasz:
                try:
                    metrics['calinski_harabasz_index'] = calinski_harabasz_score(X_eval, labels_eval)
                except Exception as e:
                    logger.warning(f"Could not calculate Calinski-Harabasz index: {e}")

        # For DBSCAN, add noise statistics
        if method == 'dbscan':
            metrics['n_noise'] = int((labels == -1).sum())
            metrics['noise_ratio'] = float((labels == -1).sum() / len(labels))

        self.metrics[f'{method}_evaluation'] = metrics

        logger.info(f"{method} evaluation complete: {metrics}")

    def reduce_dimensions(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensionality using PCA for visualization.

        Args:
            X: Scaled feature array
            n_components: Number of components to keep

        Returns:
            Reduced feature array
        """
        logger.info(f"Reducing dimensions to {n_components} components using PCA")

        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_reduced = self.pca_model.fit_transform(X)

        explained_variance = self.pca_model.explained_variance_ratio_
        logger.info(f"Explained variance: {explained_variance}")
        logger.info(f"Total explained variance: {sum(explained_variance):.2%}")

        return X_reduced

    def analyze_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        features: list,
        method: str
    ) -> pd.DataFrame:
        """
        Analyze cluster characteristics.

        Args:
            df: Customer DataFrame
            labels: Cluster labels
            features: List of feature names used for clustering
            method: Clustering method name

        Returns:
            DataFrame with cluster statistics
        """
        logger.info(f"Analyzing {method} clusters")

        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        # Calculate cluster statistics
        cluster_stats = []

        for cluster_id in sorted(df_analysis['cluster'].unique()):
            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]

            stats = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_analysis) * 100,
            }

            # Calculate mean for key features
            for feature in features:
                if feature in df_analysis.columns:
                    stats[f'{feature}_mean'] = cluster_data[feature].mean()
                    stats[f'{feature}_std'] = cluster_data[feature].std()

            cluster_stats.append(stats)

        cluster_stats_df = pd.DataFrame(cluster_stats)

        logger.info(f"Cluster analysis complete for {len(cluster_stats)} clusters")
        return cluster_stats_df

    def assign_cluster_names(
        self,
        df: pd.DataFrame,
        cluster_stats: pd.DataFrame,
        method: str
    ) -> Dict[int, Dict[str, str]]:
        """
        Assign meaningful names and descriptions to clusters.

        Args:
            df: Customer DataFrame with cluster labels
            cluster_stats: Cluster statistics DataFrame
            method: Clustering method name

        Returns:
            Dictionary mapping cluster IDs to names and descriptions
        """
        logger.info(f"Assigning names to {method} clusters")

        cluster_names = {}

        for _, row in cluster_stats.iterrows():
            cluster_id = int(row['cluster'])

            if cluster_id == -1:
                cluster_names[cluster_id] = {
                    'name': 'Outliers',
                    'emoji': '🔍',
                    'description': 'Customers with unusual purchasing patterns requiring investigation'
                }
                continue

            # Analyze cluster characteristics
            cluster_data = df[df['cluster'] == cluster_id]

            avg_spent = cluster_data['total_spent'].mean()
            avg_frequency = cluster_data['total_transactions'].mean()
            avg_recency = cluster_data['days_since_last_purchase'].mean()

            # Determine cluster type based on characteristics
            if avg_spent > df['total_spent'].quantile(0.75) and avg_frequency > df['total_transactions'].quantile(0.75):
                name = "High-Value Customers"
                emoji = "🌟"
                description = "High spending, frequent purchases, loyal customers"
            elif avg_frequency > df['total_transactions'].quantile(0.75) and avg_spent < df['total_spent'].quantile(0.5):
                name = "Bargain Hunters"
                emoji = "💰"
                description = "Frequent but low-value purchases, price-sensitive"
            elif avg_recency < 90 and avg_frequency < df['total_transactions'].quantile(0.5):
                name = "New Customers"
                emoji = "🆕"
                description = "Recent joiners with emerging behavior patterns"
            elif avg_recency > 180:
                name = "At-Risk Customers"
                emoji = "⚠️"
                description = "Low engagement, potential churn candidates"
            elif avg_spent > df['total_spent'].quantile(0.5):
                name = "Occasional High Spenders"
                emoji = "🛍️"
                description = "Medium frequency but high-value purchases"
            else:
                name = f"Cluster {cluster_id}"
                emoji = "👥"
                description = "Standard customer segment"

            cluster_names[cluster_id] = {
                'name': name,
                'emoji': emoji,
                'description': description
            }

        logger.info(f"Cluster naming complete for {len(cluster_names)} clusters")
        return cluster_names

    def get_metrics(self) -> Dict[str, Any]:
        """Get all clustering metrics."""
        return self.metrics

    def save_cluster_results(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        method: str
    ):
        """
        Save clustering results to CSV.

        Args:
            df: Customer DataFrame
            labels: Cluster labels
            method: Clustering method name
        """
        df_result = df.copy()
        df_result['cluster'] = labels

        output_path = self.config.paths.data_dir / f'customer_clusters_{method}.csv'
        df_result.to_csv(output_path, index=False)

        logger.info(f"{method} clustering results saved to {output_path}")
