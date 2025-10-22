"""
Reporting and Visualization Module.

This module generates comprehensive visualizations, dashboards, and business
insights reports for clustering analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)


class ClusteringReporter:
    """Generates reports and visualizations for clustering results."""

    def __init__(self, config: Config):
        """
        Initialize the reporter.

        Args:
            config: Configuration object containing reporting parameters
        """
        self.config = config

        # Set matplotlib style
        try:
            plt.style.use(config.reporting.plot_style)
        except:
            plt.style.use('default')

        sns.set_palette(config.reporting.color_palette)

        logger.info("ClusteringReporter initialized")

    def plot_optimal_k_analysis(self, optimization_metrics: Dict[str, list]):
        """
        Plot K-means optimization metrics.

        Args:
            optimization_metrics: Dictionary containing optimization metrics
        """
        logger.info("Creating optimal K analysis plots")

        k_values = optimization_metrics['k_values']

        fig, axes = plt.subplots(2, 2, figsize=self.config.reporting.figure_size)
        fig.suptitle('K-means Optimization Analysis', fontsize=16, fontweight='bold')

        # Elbow curve
        axes[0, 0].plot(k_values, optimization_metrics['inertia'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
        axes[0, 0].set_ylabel('Inertia', fontsize=12)
        axes[0, 0].set_title('Elbow Method', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        # Silhouette score
        if optimization_metrics['silhouette']:
            # Silhouette scores are only calculated for k > 1
            silhouette_k_values = [k for k in k_values if k > 1]
            axes[0, 1].plot(
                silhouette_k_values,
                optimization_metrics['silhouette'],
                'go-',
                linewidth=2,
                markersize=8
            )
            axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
            axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
            axes[0, 1].set_title('Silhouette Analysis', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)

        # Davies-Bouldin Index
        if optimization_metrics['davies_bouldin']:
            # Davies-Bouldin scores are only calculated for k > 1
            davies_k_values = [k for k in k_values if k > 1]
            axes[1, 0].plot(
                davies_k_values,
                optimization_metrics['davies_bouldin'],
                'ro-',
                linewidth=2,
                markersize=8
            )
            axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
            axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=12)
            axes[1, 0].set_title('Davies-Bouldin Analysis (Lower is Better)', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)

        # Calinski-Harabasz Index
        if optimization_metrics['calinski_harabasz']:
            # Calinski-Harabasz scores are only calculated for k > 1
            calinski_k_values = [k for k in k_values if k > 1]
            axes[1, 1].plot(
                calinski_k_values,
                optimization_metrics['calinski_harabasz'],
                'mo-',
                linewidth=2,
                markersize=8
            )
            axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
            axes[1, 1].set_ylabel('Calinski-Harabasz Index', fontsize=12)
            axes[1, 1].set_title('Calinski-Harabasz Analysis (Higher is Better)', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.config.paths.viz_dir / 'optimal_k_analysis.png'
        plt.savefig(output_path, dpi=self.config.reporting.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Optimal K analysis plot saved to {output_path}")

    def plot_clusters_2d(
        self,
        X_reduced: np.ndarray,
        labels: np.ndarray,
        cluster_names: Dict[int, Dict[str, str]],
        method: str
    ):
        """
        Create 2D scatter plot of clusters.

        Args:
            X_reduced: PCA-reduced feature array
            labels: Cluster labels
            cluster_names: Dictionary mapping cluster IDs to names
            method: Clustering method name
        """
        logger.info(f"Creating 2D cluster visualization for {method}")

        fig, ax = plt.subplots(figsize=self.config.reporting.figure_size)

        # Plot each cluster
        unique_labels = sorted(set(labels))
        # Use seaborn color palette for consistent colors
        colors = sns.color_palette(self.config.reporting.color_palette, n_colors=len(unique_labels))

        for i, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id

            if cluster_id in cluster_names:
                label = f"{cluster_names[cluster_id]['emoji']} {cluster_names[cluster_id]['name']}"
            else:
                label = f"Cluster {cluster_id}"

            ax.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                c=[colors[i]],
                label=label,
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )

        ax.set_xlabel('First Principal Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
        ax.set_title(f'{method.upper()} Clustering Visualization (PCA)', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.config.paths.viz_dir / f'{method}_clustering_visualization.png'
        plt.savefig(output_path, dpi=self.config.reporting.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Cluster visualization saved to {output_path}")

    def plot_cluster_comparison(
        self,
        df: pd.DataFrame,
        cluster_names: Dict[int, Dict[str, str]],
        method: str
    ):
        """
        Create comprehensive cluster comparison plots.

        Args:
            df: Customer DataFrame with cluster labels
            cluster_names: Dictionary mapping cluster IDs to names
            method: Clustering method name
        """
        logger.info(f"Creating cluster comparison plots for {method}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{method.upper()} Cluster Comparison', fontsize=18, fontweight='bold')

        # Prepare cluster labels with names
        df_plot = df.copy()
        df_plot['cluster_name'] = df_plot['cluster'].map(
            lambda x: f"{cluster_names[x]['emoji']} {cluster_names[x]['name']}"
            if x in cluster_names else f"Cluster {x}"
        )

        # 1. Cluster size distribution
        cluster_sizes = df_plot['cluster_name'].value_counts()
        axes[0, 0].bar(range(len(cluster_sizes)), cluster_sizes.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_xticks(range(len(cluster_sizes)))
        axes[0, 0].set_xticklabels(cluster_sizes.index, rotation=45, ha='right', fontsize=8)
        axes[0, 0].set_ylabel('Number of Customers', fontsize=10)
        axes[0, 0].set_title('Cluster Size Distribution', fontsize=12)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Total spent by cluster
        sns.boxplot(data=df_plot, x='cluster_name', y='total_spent', ax=axes[0, 1])
        axes[0, 1].set_xlabel('')
        axes[0, 1].set_ylabel('Total Spent ($)', fontsize=10)
        axes[0, 1].set_title('Spending Distribution by Cluster', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)

        # 3. Transaction frequency by cluster
        sns.boxplot(data=df_plot, x='cluster_name', y='total_transactions', ax=axes[0, 2])
        axes[0, 2].set_xlabel('')
        axes[0, 2].set_ylabel('Total Transactions', fontsize=10)
        axes[0, 2].set_title('Transaction Frequency by Cluster', fontsize=12)
        axes[0, 2].tick_params(axis='x', rotation=45, labelsize=8)

        # 4. RFM Score by cluster
        if 'rfm_score' in df_plot.columns:
            sns.violinplot(data=df_plot, x='cluster_name', y='rfm_score', ax=axes[1, 0])
            axes[1, 0].set_xlabel('')
            axes[1, 0].set_ylabel('RFM Score', fontsize=10)
            axes[1, 0].set_title('RFM Score Distribution by Cluster', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)

        # 5. CLV by cluster
        if 'clv_advanced' in df_plot.columns:
            sns.boxplot(data=df_plot, x='cluster_name', y='clv_advanced', ax=axes[1, 1])
            axes[1, 1].set_xlabel('')
            axes[1, 1].set_ylabel('Customer Lifetime Value ($)', fontsize=10)
            axes[1, 1].set_title('CLV by Cluster', fontsize=12)
            axes[1, 1].tick_params(axis='x', rotation=45, labelsize=8)

        # 6. Days since last purchase by cluster
        sns.violinplot(data=df_plot, x='cluster_name', y='days_since_last_purchase', ax=axes[1, 2])
        axes[1, 2].set_xlabel('')
        axes[1, 2].set_ylabel('Days Since Last Purchase', fontsize=10)
        axes[1, 2].set_title('Recency by Cluster', fontsize=12)
        axes[1, 2].tick_params(axis='x', rotation=45, labelsize=8)

        plt.tight_layout()

        output_path = self.config.paths.viz_dir / f'{method}_cluster_comparison.png'
        plt.savefig(output_path, dpi=self.config.reporting.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Cluster comparison plot saved to {output_path}")

    def create_interactive_dashboard(
        self,
        df: pd.DataFrame,
        X_reduced: np.ndarray,
        cluster_stats: pd.DataFrame,
        cluster_names: Dict[int, Dict[str, str]],
        method: str
    ):
        """
        Create interactive Plotly dashboard.

        Args:
            df: Customer DataFrame with cluster labels
            X_reduced: PCA-reduced feature array
            cluster_stats: Cluster statistics DataFrame
            cluster_names: Dictionary mapping cluster IDs to names
            method: Clustering method name
        """
        if not self.config.reporting.generate_interactive:
            return

        logger.info(f"Creating interactive dashboard for {method}")

        # Prepare data
        df_plot = df.copy()
        df_plot['PC1'] = X_reduced[:, 0]
        df_plot['PC2'] = X_reduced[:, 1]
        df_plot['cluster_name'] = df_plot['cluster'].map(
            lambda x: f"{cluster_names[x]['name']}" if x in cluster_names else f"Cluster {x}"
        )

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cluster Distribution (PCA)',
                'Cluster Size',
                'Spending vs Frequency',
                'RFM Analysis'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )

        # 1. PCA scatter plot
        for cluster_id in sorted(df_plot['cluster'].unique()):
            cluster_data = df_plot[df_plot['cluster'] == cluster_id]
            cluster_name = cluster_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')

            fig.add_trace(
                go.Scatter(
                    x=cluster_data['PC1'],
                    y=cluster_data['PC2'],
                    mode='markers',
                    name=cluster_name,
                    text=cluster_data['customer_id'],
                    hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
                    marker=dict(size=8, opacity=0.6)
                ),
                row=1, col=1
            )

        # 2. Cluster size bar chart
        cluster_sizes = df_plot['cluster_name'].value_counts()
        fig.add_trace(
            go.Bar(
                x=cluster_sizes.index,
                y=cluster_sizes.values,
                name='Cluster Size',
                showlegend=False,
                marker=dict(color='lightblue')
            ),
            row=1, col=2
        )

        # 3. Spending vs Frequency scatter
        fig.add_trace(
            go.Scatter(
                x=df_plot['total_transactions'],
                y=df_plot['total_spent'],
                mode='markers',
                name='Customers',
                text=df_plot['cluster_name'],
                marker=dict(
                    size=8,
                    color=df_plot['cluster'],
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.6
                ),
                hovertemplate='<b>%{text}</b><br>Transactions: %{x}<br>Spent: $%{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. RFM scatter (if available)
        if 'rfm_frequency' in df_plot.columns and 'rfm_monetary' in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot['rfm_frequency'],
                    y=df_plot['rfm_monetary'],
                    mode='markers',
                    name='RFM',
                    text=df_plot['cluster_name'],
                    marker=dict(
                        size=8,
                        color=df_plot['cluster'],
                        colorscale='Plasma',
                        showscale=False,
                        opacity=0.6
                    ),
                    hovertemplate='<b>%{text}</b><br>Frequency: %{x:.0f}<br>Monetary: $%{y:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="Cluster", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Total Transactions", row=2, col=1)
        fig.update_yaxes(title_text="Total Spent ($)", row=2, col=1)
        fig.update_xaxes(title_text="RFM Frequency", row=2, col=2)
        fig.update_yaxes(title_text="RFM Monetary", row=2, col=2)

        fig.update_layout(
            title_text=f"{method.upper()} Clustering Dashboard",
            title_font_size=20,
            height=800,
            showlegend=True,
            template=self.config.reporting.dashboard_template
        )

        output_path = self.config.paths.viz_dir / f'{method}_dashboard.html'
        fig.write_html(str(output_path))

        logger.info(f"Interactive dashboard saved to {output_path}")

    def generate_business_report(
        self,
        df: pd.DataFrame,
        cluster_stats: pd.DataFrame,
        cluster_names: Dict[int, Dict[str, str]],
        metrics: Dict[str, Any],
        method: str
    ):
        """
        Generate comprehensive business insights report.

        Args:
            df: Customer DataFrame with cluster labels
            cluster_stats: Cluster statistics DataFrame
            cluster_names: Dictionary mapping cluster IDs to names
            metrics: Clustering metrics dictionary
            method: Clustering method name
        """
        logger.info(f"Generating business report for {method}")

        report_lines = []

        # Header
        report_lines.append(f"# {method.upper()} Clustering Analysis Report\n")
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("---\n")

        # Executive Summary
        report_lines.append("## Executive Summary\n")
        n_customers = len(df)
        n_clusters = len(cluster_stats)
        report_lines.append(f"- **Total Customers Analyzed**: {n_customers:,}\n")
        report_lines.append(f"- **Number of Clusters Identified**: {n_clusters}\n")

        if method == 'dbscan':
            n_noise = int((df['cluster'] == -1).sum())
            report_lines.append(f"- **Outlier/Noise Points**: {n_noise} ({n_noise/n_customers*100:.1f}%)\n")

        # Clustering Quality Metrics
        if f'{method}_evaluation' in metrics:
            eval_metrics = metrics[f'{method}_evaluation']
            report_lines.append("\n## Clustering Quality Metrics\n")

            if 'silhouette_score' in eval_metrics:
                report_lines.append(f"- **Silhouette Score**: {eval_metrics['silhouette_score']:.3f}\n")
                report_lines.append("  - Range: [-1, 1], Higher is better\n")
                report_lines.append("  - Interpretation: Measures how similar objects are to their own cluster\n")

            if 'davies_bouldin_index' in eval_metrics:
                report_lines.append(f"- **Davies-Bouldin Index**: {eval_metrics['davies_bouldin_index']:.3f}\n")
                report_lines.append("  - Range: [0, ∞), Lower is better\n")
                report_lines.append("  - Interpretation: Measures cluster separation\n")

            if 'calinski_harabasz_index' in eval_metrics:
                report_lines.append(f"- **Calinski-Harabasz Index**: {eval_metrics['calinski_harabasz_index']:.3f}\n")
                report_lines.append("  - Range: [0, ∞), Higher is better\n")
                report_lines.append("  - Interpretation: Measures cluster density and separation\n")

        # Cluster Profiles
        report_lines.append("\n## Cluster Profiles\n")

        for _, row in cluster_stats.iterrows():
            cluster_id = int(row['cluster'])
            cluster_info = cluster_names.get(cluster_id, {'name': f'Cluster {cluster_id}', 'emoji': '👥', 'description': 'Customer segment'})

            report_lines.append(f"\n### {cluster_info['emoji']} {cluster_info['name']}\n")
            report_lines.append(f"**Description**: {cluster_info['description']}\n\n")
            report_lines.append(f"- **Size**: {int(row['size']):,} customers ({row['percentage']:.1f}% of total)\n")

            # Key metrics
            cluster_data = df[df['cluster'] == cluster_id]

            report_lines.append(f"- **Average Total Spent**: ${cluster_data['total_spent'].mean():,.2f}\n")
            report_lines.append(f"- **Average Transactions**: {cluster_data['total_transactions'].mean():.1f}\n")
            report_lines.append(f"- **Average Transaction Value**: ${cluster_data['avg_transaction_value'].mean():,.2f}\n")

            if 'days_since_last_purchase' in cluster_data.columns:
                report_lines.append(f"- **Avg Days Since Last Purchase**: {cluster_data['days_since_last_purchase'].mean():.0f}\n")

            if 'rfm_score' in cluster_data.columns:
                report_lines.append(f"- **Average RFM Score**: {cluster_data['rfm_score'].mean():.2f}/15\n")

            if 'clv_advanced' in cluster_data.columns:
                report_lines.append(f"- **Estimated CLV**: ${cluster_data['clv_advanced'].mean():,.2f}\n")

        # Business Recommendations
        if self.config.reporting.include_recommendations:
            report_lines.append("\n## Business Recommendations\n")
            report_lines.extend(self._generate_recommendations(df, cluster_names))

        # Key Insights
        if self.config.reporting.include_business_insights:
            report_lines.append("\n## Key Insights\n")
            report_lines.extend(self._generate_insights(df, cluster_stats, cluster_names))

        # Save report
        output_path = self.config.paths.reports_dir / f'{method}_clustering_report.md'
        with open(output_path, 'w') as f:
            f.writelines(report_lines)

        logger.info(f"Business report saved to {output_path}")

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        cluster_names: Dict[int, Dict[str, str]]
    ) -> list:
        """Generate business recommendations based on clusters."""
        recommendations = []

        for cluster_id, info in cluster_names.items():
            cluster_data = df[df['cluster'] == cluster_id]

            if cluster_id == -1:
                recommendations.append(f"\n### {info['emoji']} {info['name']}\n")
                recommendations.append("- **Action**: Investigate these customers individually\n")
                recommendations.append("- **Strategy**: Personalized outreach to understand unique needs\n")
                recommendations.append("- **Opportunity**: Potential VIP customers or fraud detection\n")
                continue

            recommendations.append(f"\n### {info['emoji']} {info['name']}\n")

            avg_spent = cluster_data['total_spent'].mean()
            avg_recency = cluster_data['days_since_last_purchase'].mean()
            avg_frequency = cluster_data['total_transactions'].mean()

            # Customize recommendations based on cluster characteristics
            if "High-Value" in info['name']:
                recommendations.append("- **Action**: VIP treatment and loyalty programs\n")
                recommendations.append("- **Strategy**: Exclusive offers, early access to new products\n")
                recommendations.append("- **Communication**: Personalized emails, dedicated support\n")

            elif "Bargain" in info['name']:
                recommendations.append("- **Action**: Targeted promotions and volume discounts\n")
                recommendations.append("- **Strategy**: Bundle offers, seasonal sales notifications\n")
                recommendations.append("- **Communication**: Discount alerts, flash sale announcements\n")

            elif "New" in info['name']:
                recommendations.append("- **Action**: Welcome campaigns and onboarding\n")
                recommendations.append("- **Strategy**: Tutorial content, first-purchase incentives\n")
                recommendations.append("- **Communication**: Educational emails, product guides\n")

            elif "At-Risk" in info['name']:
                recommendations.append("- **Action**: Re-engagement campaigns\n")
                recommendations.append("- **Strategy**: Win-back offers, feedback surveys\n")
                recommendations.append("- **Communication**: Personalized emails, special incentives\n")

            else:
                recommendations.append("- **Action**: Standard marketing campaigns\n")
                recommendations.append("- **Strategy**: Regular promotions, newsletter content\n")
                recommendations.append("- **Communication**: Email marketing, social media\n")

        return recommendations

    def _generate_insights(
        self,
        df: pd.DataFrame,
        cluster_stats: pd.DataFrame,
        cluster_names: Dict[int, Dict[str, str]]
    ) -> list:
        """Generate key business insights."""
        insights = []

        # Revenue concentration
        total_revenue = df['total_spent'].sum()
        insights.append("\n### Revenue Distribution\n")

        for _, row in cluster_stats.iterrows():
            cluster_id = int(row['cluster'])
            if cluster_id == -1:
                continue

            cluster_data = df[df['cluster'] == cluster_id]
            cluster_revenue = cluster_data['total_spent'].sum()
            revenue_pct = (cluster_revenue / total_revenue) * 100

            cluster_name = cluster_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')
            insights.append(f"- **{cluster_name}**: {revenue_pct:.1f}% of total revenue from {row['percentage']:.1f}% of customers\n")

        # Customer engagement
        insights.append("\n### Customer Engagement\n")
        active_customers = df[df['days_since_last_purchase'] < 90]
        insights.append(f"- **Active Customers** (purchased in last 90 days): {len(active_customers):,} ({len(active_customers)/len(df)*100:.1f}%)\n")

        dormant_customers = df[df['days_since_last_purchase'] > 180]
        insights.append(f"- **Dormant Customers** (no purchase in 180+ days): {len(dormant_customers):,} ({len(dormant_customers)/len(df)*100:.1f}%)\n")

        # Value segmentation
        if 'clv_advanced' in df.columns:
            insights.append("\n### Customer Value\n")
            high_clv = df[df['clv_advanced'] > df['clv_advanced'].quantile(0.75)]
            insights.append(f"- **High CLV Customers**: {len(high_clv):,} customers with avg CLV of ${high_clv['clv_advanced'].mean():,.2f}\n")

            total_clv = df['clv_advanced'].sum()
            high_clv_revenue = high_clv['clv_advanced'].sum()
            insights.append(f"- **CLV Concentration**: Top 25% of customers represent {high_clv_revenue/total_clv*100:.1f}% of total CLV\n")

        return insights
