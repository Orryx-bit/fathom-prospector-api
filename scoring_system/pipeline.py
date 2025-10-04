"""
End-to-End Lead Scoring Pipeline

Orchestrates data loading, feature engineering, scoring, and output generation.

Author: DeepAgent
Version: 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import yaml
import logging

from .feature_engineering import FeatureEngineer, FeatureComputer
from .scoring_engine import LeadScoringEngine
from .data_generator import generate_sample_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeadScoringPipeline:
    """
    Complete pipeline for lead scoring from raw data to ranked output
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config)
        self.feature_computer = FeatureComputer(self.config)
        self.scoring_engine = LeadScoringEngine(config_path)
        
        # Data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.engineered_data: Optional[pd.DataFrame] = None
        self.normalized_data: Optional[pd.DataFrame] = None
        self.scored_data: Optional[pd.DataFrame] = None
        
        logger.info("Pipeline initialized")
    
    def load_data(
        self,
        data_path: Optional[str] = None,
        generate_synthetic: bool = False
    ) -> pd.DataFrame:
        """
        Load data from file or generate synthetic data
        
        Args:
            data_path: Path to data file (CSV or Excel)
            generate_synthetic: Whether to generate synthetic data
            
        Returns:
            Loaded DataFrame
        """
        if generate_synthetic:
            logger.info("Generating synthetic data...")
            self.raw_data = generate_sample_data(self.config)
        elif data_path:
            logger.info(f"Loading data from {data_path}...")
            if data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(data_path)
            elif data_path.endswith(('.xlsx', '.xls')):
                self.raw_data = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        else:
            raise ValueError("Must provide either data_path or set generate_synthetic=True")
        
        logger.info(f"Loaded {len(self.raw_data)} records")
        return self.raw_data
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Apply feature engineering to raw data
        
        Returns:
            DataFrame with engineered features
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Engineering features...")
        
        df = self.raw_data.copy()
        
        # Compute derived features
        
        # 1. Encode service offerings
        df = self.feature_computer.encode_service_offerings(df)
        
        # 2. Extract device brands
        df['existing_device_brands'] = self.feature_computer.extract_device_brands(df)
        
        # 3. Compute engagement rate
        df['engagement_rate_computed'] = self.feature_computer.compute_engagement_rate(df)
        
        # 4. Compute rating trend
        df['rating_trend'] = self.feature_computer.compute_rating_trend(df)
        
        # 5. Compute review recency
        df['review_recency'] = self.feature_computer.compute_review_recency(df)
        
        # 6. Compute competitive score
        df['competitive_score'] = self.feature_computer.compute_competitive_score(df)
        
        # 7. Compute contactability score
        df['contactability_score'] = self.feature_computer.compute_contactability_score(df)
        
        # 8. Compute growth score
        df['growth_score'] = self.feature_computer.compute_growth_score(df)
        
        self.engineered_data = df
        logger.info("Feature engineering complete")
        
        return self.engineered_data
    
    def normalize_features(self) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range
        
        Returns:
            DataFrame with normalized features
        """
        if self.engineered_data is None:
            raise ValueError("No engineered data. Call engineer_features() first.")
        
        logger.info("Normalizing features...")
        
        df = self.engineered_data.copy()
        
        # Fit feature engineer on the data
        self.feature_engineer.fit(df)
        
        # Get log transform features from config
        log_features = self.config['normalization'].get('log_transform_features', [])
        
        # Normalize each numeric feature
        normalized_features = {}
        
        # List of features to normalize
        features_to_normalize = [
            # Social/demand signals
            'instagram_followers',
            'posts_90d',
            'engagement_rate_computed',
            'google_reviews_count',
            'treatment_hashtag_count',
            # Capacity
            'staff_count_est',
            'treatment_rooms',
            'credentialed_providers',
            # Financial proxy
            'google_rating',
            'rating_trend',
            'review_recency',
            # Growth
            'growth_score',
            # Other
            'contactability_score',
            'competitive_score'
        ]
        
        for feature in features_to_normalize:
            if feature in df.columns:
                use_log = feature in log_features
                normalized = self.feature_engineer.normalize_feature(
                    df,
                    feature,
                    use_log=use_log,
                    missing_strategy='zero'
                )
                normalized_features[feature] = normalized
        
        # Create normalized dataframe
        normalized_df = pd.DataFrame(normalized_features, index=df.index)
        
        # Add binary/categorical features (already 0-1)
        binary_features = [
            'body_contouring_offered',
            'laser_hair_removal_offered',
            'injectables_offered',
            'skin_rejuvenation_offered',
            'existing_device_brands',
            'website_updated_recently',
            'responds_within_24h'
        ]
        
        for feature in binary_features:
            if feature in df.columns:
                normalized_df[feature] = df[feature].fillna(0)
        
        self.normalized_data = normalized_df
        logger.info("Feature normalization complete")
        
        return self.normalized_data
    
    def score_leads(self) -> pd.DataFrame:
        """
        Score all leads using the scoring engine
        
        Returns:
            DataFrame with scores and rankings
        """
        if self.engineered_data is None or self.normalized_data is None:
            raise ValueError("Data not prepared. Run engineer_features() and normalize_features() first.")
        
        logger.info("Scoring leads...")
        
        # Score using the engine
        scored = self.scoring_engine.score_dataframe(
            self.engineered_data,
            self.normalized_data
        )
        
        # Rank leads by score
        scored = scored.sort_values('score', ascending=False)
        scored['rank'] = range(1, len(scored) + 1)
        
        self.scored_data = scored
        logger.info("Lead scoring complete")
        
        return self.scored_data
    
    def generate_output_csv(
        self,
        output_path: str,
        include_metadata: bool = False
    ) -> None:
        """
        Generate final output CSV with required columns
        
        Args:
            output_path: Path for output CSV file
            include_metadata: Whether to include metadata columns
        """
        if self.scored_data is None:
            raise ValueError("No scored data. Call score_leads() first.")
        
        logger.info(f"Generating output CSV: {output_path}")
        
        # Required output columns
        output_columns = [
            'business_id',
            'name',
            'address',
            'city',
            'latitude',
            'longitude',
            'main_services',
            'instagram_followers',
            'posts_90d',
            'google_reviews_count',
            'google_rating',
            'staff_count_est',
            'job_postings_last_90d',
            'contact_email',
            'contact_phone',
            'score',
            'rank',
            'top_3_contributors',
            'confidence_flag'
        ]
        
        # Optional metadata columns
        metadata_columns = [
            'territory',
            'data_completeness',
            'top_1_feature',
            'top_1_contribution_pct',
            'top_2_feature',
            'top_2_contribution_pct',
            'top_3_feature',
            'top_3_contribution_pct'
        ]
        
        # Select columns
        columns_to_export = [col for col in output_columns if col in self.scored_data.columns]
        
        if include_metadata:
            columns_to_export.extend([col for col in metadata_columns if col in self.scored_data.columns])
        
        output_df = self.scored_data[columns_to_export].copy()
        
        # Format numeric columns
        if 'score' in output_df.columns:
            output_df['score'] = output_df['score'].round(2)
        
        if 'latitude' in output_df.columns:
            output_df['latitude'] = output_df['latitude'].round(6)
        
        if 'longitude' in output_df.columns:
            output_df['longitude'] = output_df['longitude'].round(6)
        
        if 'google_rating' in output_df.columns:
            output_df['google_rating'] = output_df['google_rating'].round(1)
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Output CSV saved: {output_path}")
        logger.info(f"Total leads: {len(output_df)}")
        logger.info(f"Score range: {output_df['score'].min():.2f} - {output_df['score'].max():.2f}")
    
    def run_complete_pipeline(
        self,
        data_path: Optional[str] = None,
        generate_synthetic: bool = False,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run complete pipeline from data loading to output generation
        
        Args:
            data_path: Path to input data file
            generate_synthetic: Whether to generate synthetic data
            output_path: Path for output CSV file
            
        Returns:
            Scored DataFrame
        """
        logger.info("=" * 80)
        logger.info("Starting Lead Scoring Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Load data
        self.load_data(data_path=data_path, generate_synthetic=generate_synthetic)
        
        # Step 2: Engineer features
        self.engineer_features()
        
        # Step 3: Normalize features
        self.normalize_features()
        
        # Step 4: Score leads
        self.score_leads()
        
        # Step 5: Generate output
        if output_path:
            self.generate_output_csv(output_path, include_metadata=True)
        
        logger.info("=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        
        return self.scored_data
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics from scored data
        
        Returns:
            Dictionary of summary statistics
        """
        if self.scored_data is None:
            raise ValueError("No scored data available")
        
        stats = {
            'total_leads': len(self.scored_data),
            'score_mean': float(self.scored_data['score'].mean()),
            'score_median': float(self.scored_data['score'].median()),
            'score_std': float(self.scored_data['score'].std()),
            'score_min': float(self.scored_data['score'].min()),
            'score_max': float(self.scored_data['score'].max()),
            'high_confidence_count': int((self.scored_data['confidence_flag'] == 'high').sum()),
            'medium_confidence_count': int((self.scored_data['confidence_flag'] == 'medium').sum()),
            'low_confidence_count': int((self.scored_data['confidence_flag'] == 'low').sum()),
        }
        
        # Territory breakdown if available
        if 'territory' in self.scored_data.columns:
            territory_stats = self.scored_data.groupby('territory').agg({
                'score': ['count', 'mean', 'median'],
                'confidence_flag': lambda x: (x == 'high').sum()
            }).round(2)
            stats['territory_breakdown'] = territory_stats.to_dict()
        
        return stats


def create_pipeline(config_path: str) -> LeadScoringPipeline:
    """
    Factory function to create pipeline
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized pipeline
    """
    return LeadScoringPipeline(config_path)
