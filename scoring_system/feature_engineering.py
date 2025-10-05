"""
Feature Engineering Framework for Lead Scoring System

This module provides robust feature normalization, transformation, and computation
functions for aesthetic clinic lead scoring.

Author: DeepAgent
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for a feature used in normalization"""
    feature_name: str
    min_value: float
    max_value: float
    percentile_5: float
    percentile_95: float
    mean: float
    median: float
    std: float
    missing_count: int
    total_count: int
    
    @property
    def missing_rate(self) -> float:
        """Calculate missing data rate"""
        return self.missing_count / self.total_count if self.total_count > 0 else 0.0


class FeatureEngineer:
    """
    Feature engineering framework with robust scaling, missing data handling,
    and outlier treatment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer with configuration
        
        Args:
            config: Configuration dictionary containing normalization parameters
        """
        self.config = config
        self.feature_stats: Dict[str, FeatureStats] = {}
        self.fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data to compute statistics
        
        Args:
            df: DataFrame containing raw feature data
            
        Returns:
            self: Fitted feature engineer
        """
        logger.info("Fitting feature engineer on data...")
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col]
            valid_series = series.dropna()
            
            if len(valid_series) == 0:
                logger.warning(f"Column {col} has no valid values")
                continue
            
            self.feature_stats[col] = FeatureStats(
                feature_name=col,
                min_value=float(valid_series.min()),
                max_value=float(valid_series.max()),
                percentile_5=float(valid_series.quantile(0.05)),
                percentile_95=float(valid_series.quantile(0.95)),
                mean=float(valid_series.mean()),
                median=float(valid_series.median()),
                std=float(valid_series.std()),
                missing_count=int(series.isna().sum()),
                total_count=len(series)
            )
            
        self.fitted = True
        logger.info(f"Feature engineer fitted on {len(self.feature_stats)} numeric features")
        return self
    
    def robust_normalize(
        self, 
        values: Union[pd.Series, np.ndarray, float],
        feature_name: str,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95
    ) -> Union[pd.Series, np.ndarray, float]:
        """
        Robust normalization using percentile-based scaling to handle outliers
        
        Args:
            values: Values to normalize
            feature_name: Name of the feature (for stats lookup)
            lower_percentile: Lower percentile for clipping (default 5th)
            upper_percentile: Upper percentile for clipping (default 95th)
            
        Returns:
            Normalized values in [0, 1] range
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before normalization")
        
        if feature_name not in self.feature_stats:
            logger.warning(f"Feature {feature_name} not found in stats, using min-max")
            return self._minmax_normalize(values)
        
        stats = self.feature_stats[feature_name]
        
        # Get percentile bounds
        lower_bound = stats.percentile_5
        upper_bound = stats.percentile_95
        
        # Handle case where bounds are equal
        if upper_bound - lower_bound < 1e-6:
            logger.warning(f"Feature {feature_name} has no variance, returning 0.5")
            if isinstance(values, pd.Series):
                return pd.Series([0.5] * len(values), index=values.index)
            elif isinstance(values, np.ndarray):
                return np.full_like(values, 0.5, dtype=float)
            else:
                return 0.5
        
        # Clip and normalize
        if isinstance(values, pd.Series):
            clipped = values.clip(lower_bound, upper_bound)
            normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
        elif isinstance(values, np.ndarray):
            clipped = np.clip(values, lower_bound, upper_bound)
            normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
        else:
            clipped = np.clip(values, lower_bound, upper_bound)
            normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
        
        return normalized
    
    def _minmax_normalize(
        self, 
        values: Union[pd.Series, np.ndarray, float]
    ) -> Union[pd.Series, np.ndarray, float]:
        """Fallback min-max normalization"""
        if isinstance(values, (pd.Series, np.ndarray)):
            valid_vals = values[~np.isnan(values)] if isinstance(values, np.ndarray) else values.dropna()
            if len(valid_vals) == 0:
                return values * 0.0
            min_val = valid_vals.min()
            max_val = valid_vals.max()
            if max_val - min_val < 1e-6:
                return values * 0.0 + 0.5
            return (values - min_val) / (max_val - min_val)
        else:
            return 0.5
    
    def log_transform(
        self, 
        values: Union[pd.Series, np.ndarray, float],
        epsilon: float = 1.0
    ) -> Union[pd.Series, np.ndarray, float]:
        """
        Log transformation for skewed features
        
        Args:
            values: Values to transform
            epsilon: Small constant to avoid log(0)
            
        Returns:
            Log-transformed values
        """
        if isinstance(values, pd.Series):
            return np.log(values + epsilon)
        elif isinstance(values, np.ndarray):
            return np.log(values + epsilon)
        else:
            return np.log(values + epsilon)
    
    def handle_missing(
        self, 
        values: pd.Series,
        strategy: str = 'zero',
        fill_value: Optional[float] = None
    ) -> pd.Series:
        """
        Handle missing values with various strategies
        
        Args:
            values: Series with potential missing values
            strategy: 'zero', 'mean', 'median', 'forward_fill', or 'custom'
            fill_value: Custom fill value if strategy='custom'
            
        Returns:
            Series with missing values handled
        """
        if strategy == 'zero':
            return values.fillna(0.0)
        elif strategy == 'mean':
            return values.fillna(values.mean())
        elif strategy == 'median':
            return values.fillna(values.median())
        elif strategy == 'forward_fill':
            return values.fillna(method='ffill').fillna(0.0)
        elif strategy == 'custom' and fill_value is not None:
            return values.fillna(fill_value)
        else:
            return values.fillna(0.0)
    
    def normalize_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        use_log: bool = False,
        missing_strategy: str = 'zero'
    ) -> pd.Series:
        """
        Complete normalization pipeline for a single feature
        
        Args:
            df: DataFrame containing the feature
            feature_name: Name of the feature column
            use_log: Whether to apply log transformation before normalization
            missing_strategy: Strategy for handling missing values
            
        Returns:
            Normalized feature series in [0, 1] range
        """
        if feature_name not in df.columns:
            logger.warning(f"Feature {feature_name} not found, returning zeros")
            return pd.Series(0.0, index=df.index)
        
        values = df[feature_name].copy()
        
        # Handle missing values
        values = self.handle_missing(values, strategy=missing_strategy)
        
        # Apply log transformation if specified
        if use_log:
            values = self.log_transform(values)
        
        # Robust normalization
        normalized = self.robust_normalize(values, feature_name)
        
        return normalized


class FeatureComputer:
    """
    Compute derived features and categorical mappings
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature computer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def compute_engagement_rate(
        self,
        df: pd.DataFrame,
        posts_col: str = 'posts_90d',
        followers_col: str = 'instagram_followers',
        avg_engagement_col: str = 'avg_engagement_per_post'
    ) -> pd.Series:
        """
        Compute engagement rate normalized by follower count
        
        Args:
            df: DataFrame with social media data
            posts_col: Column name for post count
            followers_col: Column name for follower count
            avg_engagement_col: Column name for average engagement per post
            
        Returns:
            Engagement rate series
        """
        if avg_engagement_col in df.columns and followers_col in df.columns:
            # Engagement rate = avg_engagement / followers
            engagement_rate = df[avg_engagement_col] / (df[followers_col] + 1)
            # Cap at 1.0 (100%)
            engagement_rate = engagement_rate.clip(0, 1)
        else:
            engagement_rate = pd.Series(0.0, index=df.index)
        
        return engagement_rate
    
    def compute_rating_trend(
        self,
        df: pd.DataFrame,
        rating_col: str = 'google_rating',
        review_count_col: str = 'google_reviews_count'
    ) -> pd.Series:
        """
        Compute rating trend score (higher ratings with more reviews = higher score)
        
        Args:
            df: DataFrame with review data
            rating_col: Column name for rating
            review_count_col: Column name for review count
            
        Returns:
            Rating trend score
        """
        if rating_col in df.columns and review_count_col in df.columns:
            # Weight rating by review count (more reviews = higher confidence)
            # Normalize by max possible (5.0 rating)
            base_score = df[rating_col] / 5.0
            # Apply log-scaled review count boost (up to 2x multiplier)
            review_boost = 1 + (np.log1p(df[review_count_col]) / np.log1p(df[review_count_col].max() + 1))
            trend_score = (base_score * review_boost).clip(0, 1)
        else:
            trend_score = pd.Series(0.5, index=df.index)
        
        return trend_score
    
    def compute_review_recency(
        self,
        df: pd.DataFrame,
        last_review_days_col: str = 'days_since_last_review'
    ) -> pd.Series:
        """
        Compute review recency score (more recent = higher score)
        
        Args:
            df: DataFrame with review data
            last_review_days_col: Column name for days since last review
            
        Returns:
            Review recency score (0-1, where 1 = very recent)
        """
        if last_review_days_col in df.columns:
            # Exponential decay: score decreases as days increase
            # Half-life of 90 days
            decay_rate = np.log(2) / 90
            recency_score = np.exp(-decay_rate * df[last_review_days_col])
        else:
            recency_score = pd.Series(0.5, index=df.index)
        
        return recency_score
    
    def encode_service_offerings(
        self,
        df: pd.DataFrame,
        service_col: str = 'main_services'
    ) -> pd.DataFrame:
        """
        Encode service offerings as binary features
        
        Args:
            df: DataFrame with service data
            service_col: Column name containing service strings
            
        Returns:
            DataFrame with binary service columns
        """
        target_services = [
            'body_contouring',
            'laser_hair_removal',
            'injectables',
            'skin_rejuvenation'
        ]
        
        result = df.copy()
        
        if service_col not in df.columns:
            for service in target_services:
                result[f'{service}_offered'] = 0
            return result
        
        # Convert to lowercase for matching
        services_lower = df[service_col].fillna('').str.lower()
        
        # Body contouring keywords
        result['body_contouring_offered'] = services_lower.str.contains(
            'body|contouring|sculpting|coolsculpting|fat reduction|body shaping',
            case=False, regex=True
        ).astype(int)
        
        # Laser hair removal keywords
        result['laser_hair_removal_offered'] = services_lower.str.contains(
            'laser hair|hair removal|ipl|laser',
            case=False, regex=True
        ).astype(int)
        
        # Injectables keywords
        result['injectables_offered'] = services_lower.str.contains(
            'botox|filler|injectable|dermal|juvederm|restylane',
            case=False, regex=True
        ).astype(int)
        
        # Skin rejuvenation keywords
        result['skin_rejuvenation_offered'] = services_lower.str.contains(
            'skin|facial|rejuvenation|resurfacing|peel|microneedling',
            case=False, regex=True
        ).astype(int)
        
        return result
    
    def extract_device_brands(
        self,
        df: pd.DataFrame,
        text_cols: List[str] = ['main_services', 'about_text']
    ) -> pd.Series:
        """
        Extract and score device brands mentioned in text
        
        Args:
            df: DataFrame with text data
            text_cols: List of column names to search for brand mentions
            
        Returns:
            Device brand score series
        """
        brand_keywords = {
            'coolsculpting': 1.0,
            'sculptsure': 1.0,
            'cynosure': 0.9,
            'candela': 0.9,
            'syneron': 0.8,
            'lumenis': 0.8,
            'sciton': 0.8
        }
        
        scores = []
        
        for idx, row in df.iterrows():
            max_score = 0.0
            combined_text = ''
            
            # Combine available text columns
            for col in text_cols:
                if col in df.columns and pd.notna(row[col]):
                    combined_text += ' ' + str(row[col]).lower()
            
            # Check for brand mentions
            for brand, score in brand_keywords.items():
                if brand in combined_text:
                    max_score = max(max_score, score)
            
            scores.append(max_score)
        
        return pd.Series(scores, index=df.index)
    
    def compute_competitive_score(
        self,
        df: pd.DataFrame,
        competitor_col: str = 'competitors_5mi_radius',
        ideal_count: int = 3,
        penalty_above: int = 5
    ) -> pd.Series:
        """
        Compute competitive context score
        Lower is better for crowded markets, but some competition is good
        
        Args:
            df: DataFrame with competitor data
            competitor_col: Column name for competitor count
            ideal_count: Ideal number of competitors
            penalty_above: Apply penalty above this threshold
            
        Returns:
            Competitive score (higher = better market position)
        """
        if competitor_col not in df.columns:
            return pd.Series(0.5, index=df.index)
        
        comp_count = df[competitor_col].fillna(0)
        
        # Score calculation:
        # - Too few competitors (0-1): moderate score (untested market)
        # - Ideal range (2-5): high score (validated market)
        # - Too many (>5): decreasing score (crowded market)
        
        scores = []
        for count in comp_count:
            if count <= 1:
                score = 0.6  # Untested market
            elif count <= ideal_count:
                score = 0.9  # Good market
            elif count <= penalty_above:
                score = 0.7  # Moderate competition
            else:
                # Linear decay after penalty threshold
                score = max(0.3, 0.7 - (count - penalty_above) * 0.05)
            scores.append(score)
        
        return pd.Series(scores, index=df.index)
    
    def compute_contactability_score(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Compute overall contactability score based on available contact methods
        
        Args:
            df: DataFrame with contact information
            
        Returns:
            Contactability score
        """
        score = pd.Series(0.0, index=df.index)
        
        # Weight each contact method
        if 'contact_phone' in df.columns:
            score += df['contact_phone'].notna().astype(float) * 0.35
        
        if 'contact_email' in df.columns:
            score += df['contact_email'].notna().astype(float) * 0.35
        
        if 'booking_link' in df.columns:
            score += df['booking_link'].notna().astype(float) * 0.20
        
        if 'responds_within_24h' in df.columns:
            score += df['responds_within_24h'].fillna(0).astype(float) * 0.10
        
        return score.clip(0, 1)
    
    def compute_growth_score(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Compute overall growth signal score
        
        Args:
            df: DataFrame with growth indicators
            
        Returns:
            Growth score
        """
        score = pd.Series(0.0, index=df.index)
        
        # Job postings (40% weight)
        if 'job_postings_last_90d' in df.columns:
            job_norm = (df['job_postings_last_90d'].fillna(0) / 5).clip(0, 1)
            score += job_norm * 0.40
        
        # New providers (35% weight)
        if 'new_providers_90d' in df.columns:
            provider_norm = (df['new_providers_90d'].fillna(0) / 3).clip(0, 1)
            score += provider_norm * 0.35
        
        # Website updates (25% weight)
        if 'website_updated_recently' in df.columns:
            score += df['website_updated_recently'].fillna(0).astype(float) * 0.25
        
        return score.clip(0, 1)


def create_feature_pipeline(config: Dict) -> Tuple[FeatureEngineer, FeatureComputer]:
    """
    Factory function to create feature engineering pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (FeatureEngineer, FeatureComputer)
    """
    engineer = FeatureEngineer(config)
    computer = FeatureComputer(config)
    
    return engineer, computer
