"""
Evaluation Framework for Lead Scoring System

Provides metrics calculation, distribution analysis, and performance evaluation.

Author: DeepAgent
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringEvaluator:
    """
    Comprehensive evaluation framework for lead scoring
    """
    
    def __init__(self, scored_df: pd.DataFrame):
        """
        Initialize evaluator with scored data
        
        Args:
            scored_df: DataFrame with scoring results
        """
        self.scored_df = scored_df.copy()
        self.metrics = {}
    
    def compute_score_distribution(self) -> Dict:
        """
        Analyze score distribution across all leads
        
        Returns:
            Dictionary of distribution statistics
        """
        logger.info("Computing score distribution...")
        
        scores = self.scored_df['score']
        
        distribution = {
            'mean': float(scores.mean()),
            'median': float(scores.median()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'q25': float(scores.quantile(0.25)),
            'q75': float(scores.quantile(0.75)),
            'iqr': float(scores.quantile(0.75) - scores.quantile(0.25)),
        }
        
        # Score bins
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
        self.scored_df['score_bin'] = pd.cut(
            self.scored_df['score'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        bin_counts = self.scored_df['score_bin'].value_counts().sort_index()
        distribution['bin_counts'] = bin_counts.to_dict()
        distribution['bin_percentages'] = (bin_counts / len(self.scored_df) * 100).round(1).to_dict()
        
        self.metrics['score_distribution'] = distribution
        return distribution
    
    def compute_territory_comparison(self) -> Optional[pd.DataFrame]:
        """
        Compare scoring across territories
        
        Returns:
            DataFrame with territory-level statistics or None if no territory column
        """
        if 'territory' not in self.scored_df.columns:
            logger.warning("No territory column found, skipping territory comparison")
            return None
        
        logger.info("Computing territory comparison...")
        
        territory_stats = self.scored_df.groupby('territory').agg({
            'score': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'confidence_flag': lambda x: {
                'high': (x == 'high').sum(),
                'medium': (x == 'medium').sum(),
                'low': (x == 'low').sum()
            }
        }).round(2)
        
        # Flatten multi-index columns
        territory_stats.columns = ['_'.join(col).strip('_') for col in territory_stats.columns.values]
        
        # Calculate high-confidence percentage
        territory_stats['high_confidence_pct'] = (
            self.scored_df[self.scored_df['confidence_flag'] == 'high']
            .groupby('territory')
            .size() / self.scored_df.groupby('territory').size() * 100
        ).round(1)
        
        self.metrics['territory_comparison'] = territory_stats
        return territory_stats
    
    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Aggregate feature importance across all leads
        
        Returns:
            DataFrame with feature importance statistics
        """
        logger.info("Computing feature importance...")
        
        importance_data = []
        
        # Collect from top contributor columns
        for idx in [1, 2, 3]:
            feature_col = f'top_{idx}_feature'
            pct_col = f'top_{idx}_contribution_pct'
            
            if feature_col in self.scored_df.columns and pct_col in self.scored_df.columns:
                for feature, pct in zip(self.scored_df[feature_col], self.scored_df[pct_col]):
                    if pd.notna(feature):
                        importance_data.append({
                            'feature': feature,
                            'contribution_pct': pct,
                            'rank': idx
                        })
        
        if not importance_data:
            logger.warning("No feature importance data found")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(importance_data)
        
        # Aggregate statistics
        feature_summary = importance_df.groupby('feature').agg({
            'contribution_pct': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'rank': 'mean'
        }).round(2)
        
        feature_summary.columns = ['_'.join(col).strip('_') for col in feature_summary.columns.values]
        feature_summary = feature_summary.sort_values('contribution_pct_mean', ascending=False)
        
        # Rename columns for clarity
        feature_summary.columns = [
            'appearance_count',
            'avg_contribution_pct',
            'median_contribution_pct',
            'std_contribution_pct',
            'min_contribution_pct',
            'max_contribution_pct',
            'avg_rank'
        ]
        
        self.metrics['feature_importance'] = feature_summary
        return feature_summary
    
    def simulate_precision_at_k(
        self,
        k_values: List[int] = [10, 20, 50],
        synthetic_label_col: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Simulate precision@k using synthetic high-value labels
        
        Args:
            k_values: List of k values to compute precision for
            synthetic_label_col: Column name with synthetic labels (if available)
            
        Returns:
            Dictionary mapping k to precision value
        """
        logger.info("Simulating precision@k...")
        
        # If no labels provided, create synthetic labels based on heuristics
        if synthetic_label_col is None or synthetic_label_col not in self.scored_df.columns:
            logger.info("Generating synthetic high-value labels...")
            self.scored_df['synthetic_label'] = self._generate_synthetic_labels()
            synthetic_label_col = 'synthetic_label'
        
        precision_scores = {}
        
        for k in k_values:
            if k > len(self.scored_df):
                logger.warning(f"k={k} exceeds dataset size, skipping")
                continue
            
            # Get top k by score
            top_k = self.scored_df.nsmallest(k, 'rank')
            
            # Calculate precision
            true_positives = top_k[synthetic_label_col].sum()
            precision = true_positives / k
            
            precision_scores[k] = float(precision)
            logger.info(f"Precision@{k}: {precision:.3f}")
        
        self.metrics['precision_at_k'] = precision_scores
        return precision_scores
    
    def _generate_synthetic_labels(self) -> pd.Series:
        """
        Generate synthetic high-value labels based on heuristics
        
        Creates realistic "high-value" labels using multiple signals:
        - Clinic size (staff, rooms)
        - Service offering quality
        - Growth signals
        - Engagement metrics
        
        Returns:
            Binary series (1 = high-value, 0 = regular)
        """
        scores = pd.Series(0, index=self.scored_df.index)
        
        # Signal 1: Large capacity (30% weight)
        if 'staff_count_est' in self.scored_df.columns:
            staff_score = (self.scored_df['staff_count_est'] > self.scored_df['staff_count_est'].quantile(0.75))
            scores += staff_score * 0.3
        
        # Signal 2: High engagement (25% weight)
        if 'instagram_followers' in self.scored_df.columns:
            follower_score = (self.scored_df['instagram_followers'] > self.scored_df['instagram_followers'].quantile(0.70))
            scores += follower_score * 0.25
        
        # Signal 3: Growth indicators (25% weight)
        if 'job_postings_last_90d' in self.scored_df.columns:
            growth_score = (self.scored_df['job_postings_last_90d'] > 0)
            scores += growth_score * 0.25
        
        # Signal 4: Service match (20% weight)
        service_match_score = 0
        for col in ['body_contouring_offered', 'laser_hair_removal_offered', 'injectables_offered']:
            if col in self.scored_df.columns:
                service_match_score += self.scored_df[col]
        service_score = (service_match_score >= 2)  # At least 2 key services
        scores += service_score * 0.20
        
        # Convert to binary label (top 30% are "high-value")
        threshold = scores.quantile(0.70)
        labels = (scores >= threshold).astype(int)
        
        logger.info(f"Generated {labels.sum()} high-value labels ({labels.mean():.1%} of total)")
        
        return labels
    
    def compute_confidence_breakdown(self) -> pd.DataFrame:
        """
        Analyze scoring by confidence level
        
        Returns:
            DataFrame with confidence-level breakdown
        """
        logger.info("Computing confidence breakdown...")
        
        if 'confidence_flag' not in self.scored_df.columns:
            logger.warning("No confidence_flag column found")
            return pd.DataFrame()
        
        confidence_stats = self.scored_df.groupby('confidence_flag').agg({
            'score': ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(2)
        
        confidence_stats.columns = ['_'.join(col).strip('_') for col in confidence_stats.columns.values]
        
        # Add percentage
        confidence_stats['percentage'] = (
            confidence_stats['score_count'] / len(self.scored_df) * 100
        ).round(1)
        
        self.metrics['confidence_breakdown'] = confidence_stats
        return confidence_stats
    
    def compute_data_quality_metrics(self) -> Dict:
        """
        Compute data quality metrics
        
        Returns:
            Dictionary of data quality statistics
        """
        logger.info("Computing data quality metrics...")
        
        quality_metrics = {}
        
        # Overall completeness
        if 'data_completeness' in self.scored_df.columns:
            quality_metrics['avg_completeness'] = float(self.scored_df['data_completeness'].mean())
            quality_metrics['median_completeness'] = float(self.scored_df['data_completeness'].median())
            quality_metrics['min_completeness'] = float(self.scored_df['data_completeness'].min())
        
        # Missing data by column
        critical_columns = [
            'instagram_followers', 'posts_90d', 'google_reviews_count',
            'staff_count_est', 'contact_email', 'contact_phone'
        ]
        
        missing_rates = {}
        for col in critical_columns:
            if col in self.scored_df.columns:
                missing_rate = self.scored_df[col].isna().mean()
                missing_rates[col] = float(missing_rate)
        
        quality_metrics['missing_rates_by_column'] = missing_rates
        
        self.metrics['data_quality'] = quality_metrics
        return quality_metrics
    
    def generate_evaluation_report(self) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("Generating Evaluation Report")
        logger.info("=" * 80)
        
        report = {
            'score_distribution': self.compute_score_distribution(),
            'territory_comparison': self.compute_territory_comparison(),
            'feature_importance': self.compute_feature_importance(),
            'precision_at_k': self.simulate_precision_at_k(),
            'confidence_breakdown': self.compute_confidence_breakdown(),
            'data_quality': self.compute_data_quality_metrics()
        }
        
        logger.info("Evaluation report complete")
        
        return report
    
    def print_summary(self) -> None:
        """Print formatted summary of evaluation metrics"""
        if not self.metrics:
            self.generate_evaluation_report()
        
        print("\n" + "=" * 80)
        print("LEAD SCORING EVALUATION SUMMARY")
        print("=" * 80)
        
        # Score distribution
        if 'score_distribution' in self.metrics:
            dist = self.metrics['score_distribution']
            print("\n--- Score Distribution ---")
            print(f"Mean Score: {dist['mean']:.2f}")
            print(f"Median Score: {dist['median']:.2f}")
            print(f"Std Dev: {dist['std']:.2f}")
            print(f"Range: {dist['min']:.2f} - {dist['max']:.2f}")
            print(f"\nScore Bins:")
            for bin_label, pct in dist['bin_percentages'].items():
                count = dist['bin_counts'][bin_label]
                print(f"  {bin_label}: {count} leads ({pct}%)")
        
        # Precision@k
        if 'precision_at_k' in self.metrics:
            print("\n--- Precision@k (Simulated) ---")
            for k, precision in self.metrics['precision_at_k'].items():
                print(f"  Precision@{k}: {precision:.3f}")
        
        # Feature importance (top 5)
        if 'feature_importance' in self.metrics:
            print("\n--- Top 5 Most Important Features ---")
            top_features = self.metrics['feature_importance'].head(5)
            for idx, (feature, row) in enumerate(top_features.iterrows(), 1):
                print(f"  {idx}. {feature}")
                print(f"     Avg Contribution: {row['avg_contribution_pct']:.1f}%")
                print(f"     Appears in: {int(row['appearance_count'])} leads")
        
        # Confidence breakdown
        if 'confidence_breakdown' in self.metrics:
            print("\n--- Confidence Level Breakdown ---")
            conf_df = self.metrics['confidence_breakdown']
            for confidence, row in conf_df.iterrows():
                print(f"  {confidence.upper()}: {int(row['score_count'])} leads ({row['percentage']:.1f}%)")
                print(f"    Avg Score: {row['score_mean']:.2f}")
        
        print("\n" + "=" * 80)


def evaluate_scoring_results(
    scored_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to evaluate scoring results
    
    Args:
        scored_df: DataFrame with scoring results
        output_path: Optional path to save evaluation report as JSON
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ScoringEvaluator(scored_df)
    report = evaluator.generate_evaluation_report()
    evaluator.print_summary()
    
    # Save to file if requested
    if output_path:
        import json
        
        # Convert DataFrames to dicts for JSON serialization
        serializable_report = {}
        for key, value in report.items():
            if isinstance(value, pd.DataFrame):
                serializable_report[key] = value.to_dict()
            elif value is None:
                serializable_report[key] = None
            else:
                serializable_report[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    return report
