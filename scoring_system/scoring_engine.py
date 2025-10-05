"""
Rule-Based Scoring Engine for Lead Scoring System

Implements weighted sum model with feature contribution tracking,
confidence scoring, and explainability.

Author: DeepAgent
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Results from scoring a single lead"""
    business_id: str
    score: float
    confidence: str
    contributions: Dict[str, float]
    top_contributors: List[Tuple[str, float]]
    data_completeness: float
    

class LeadScoringEngine:
    """
    Core scoring engine with weighted sum model and explainability
    """
    
    def __init__(self, config_path: str):
        """
        Initialize scoring engine with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._validate_config()
        
        self.category_weights = self.config['category_weights']
        self.feature_weights = self.config['feature_weights']
        self.confidence_thresholds = self.config['confidence_thresholds']
        
        logger.info("Scoring engine initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration weights sum to 1.0"""
        category_sum = sum(self.config['category_weights'].values())
        tolerance = self.config['validation']['tolerance']
        
        if abs(category_sum - 1.0) > tolerance:
            raise ValueError(
                f"Category weights sum to {category_sum}, must sum to 1.0 (±{tolerance})"
            )
        
        # Validate feature weights within each category
        for category, features in self.config['feature_weights'].items():
            feature_sum = sum(features.values())
            if abs(feature_sum - 1.0) > tolerance:
                raise ValueError(
                    f"Feature weights in {category} sum to {feature_sum}, must sum to 1.0"
                )
        
        logger.info("Configuration validation passed")
    
    def compute_category_score(
        self,
        normalized_features: pd.Series,
        category: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute score for a single category
        
        Args:
            normalized_features: Series of normalized feature values
            category: Category name
            
        Returns:
            Tuple of (category_score, feature_contributions)
        """
        if category not in self.feature_weights:
            return 0.0, {}
        
        feature_weights = self.feature_weights[category]
        contributions = {}
        category_score = 0.0
        
        for feature_name, feature_weight in feature_weights.items():
            # Get feature value (default to 0 if missing)
            feature_value = normalized_features.get(feature_name, 0.0)
            
            # Handle NaN values
            if pd.isna(feature_value):
                feature_value = 0.0
            
            # Compute contribution
            contribution = feature_value * feature_weight
            contributions[feature_name] = contribution
            category_score += contribution
        
        return category_score, contributions
    
    def compute_overall_score(
        self,
        normalized_features: pd.Series
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Compute overall lead score using weighted sum model
        
        Args:
            normalized_features: Series of normalized feature values [0, 1]
            
        Returns:
            Tuple of (overall_score, category_scores, all_contributions)
        """
        category_scores = {}
        all_contributions = {}
        overall_score = 0.0
        
        # Score each category
        for category, category_weight in self.category_weights.items():
            cat_score, contributions = self.compute_category_score(
                normalized_features, category
            )
            
            # Weight by category importance
            weighted_cat_score = cat_score * category_weight
            category_scores[category] = weighted_cat_score
            overall_score += weighted_cat_score
            
            # Track individual feature contributions (weighted by category)
            for feature_name, contribution in contributions.items():
                weighted_contribution = contribution * category_weight
                all_contributions[feature_name] = weighted_contribution
        
        # Scale to 0-100 range
        overall_score = overall_score * 100.0
        
        return overall_score, category_scores, all_contributions
    
    def get_top_contributors(
        self,
        contributions: Dict[str, float],
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top N contributing features with contribution percentages
        
        Args:
            contributions: Dictionary of feature contributions
            top_n: Number of top contributors to return
            
        Returns:
            List of (feature_name, contribution_percentage) tuples
        """
        # Sort by contribution value
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top N
        top_contributors = sorted_contributions[:top_n]
        
        # Calculate total contribution for percentage
        total_contribution = sum(contributions.values())
        
        # Convert to percentages
        if total_contribution > 0:
            top_with_pct = [
                (name, (contrib / total_contribution) * 100.0)
                for name, contrib in top_contributors
            ]
        else:
            top_with_pct = [(name, 0.0) for name, _ in top_contributors]
        
        return top_with_pct
    
    def compute_data_completeness(
        self,
        row: pd.Series,
        all_feature_names: List[str]
    ) -> float:
        """
        Compute data completeness score
        
        Args:
            row: Row of data
            all_feature_names: List of all expected feature names
            
        Returns:
            Completeness score [0, 1]
        """
        available_features = []
        
        for feature in all_feature_names:
            if feature in row.index and pd.notna(row[feature]):
                available_features.append(feature)
        
        completeness = len(available_features) / len(all_feature_names)
        return completeness
    
    def determine_confidence(
        self,
        completeness: float
    ) -> str:
        """
        Determine confidence level based on data completeness
        
        Args:
            completeness: Data completeness score [0, 1]
            
        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        high_threshold = self.confidence_thresholds['high_confidence_threshold']
        medium_threshold = self.confidence_thresholds['medium_confidence_threshold']
        
        if completeness >= high_threshold:
            return 'high'
        elif completeness >= medium_threshold:
            return 'medium'
        else:
            return 'low'
    
    def score_lead(
        self,
        row: pd.Series,
        normalized_features: pd.Series,
        all_feature_names: List[str]
    ) -> ScoringResult:
        """
        Score a single lead with full explainability
        
        Args:
            row: Raw data row
            normalized_features: Normalized feature values
            all_feature_names: List of all expected features
            
        Returns:
            ScoringResult object
        """
        # Compute overall score and contributions
        score, category_scores, contributions = self.compute_overall_score(
            normalized_features
        )
        
        # Get top contributors
        top_contributors = self.get_top_contributors(contributions)
        
        # Compute data completeness
        completeness = self.compute_data_completeness(row, all_feature_names)
        
        # Determine confidence
        confidence = self.determine_confidence(completeness)
        
        # Create result
        result = ScoringResult(
            business_id=row.get('business_id', 'UNKNOWN'),
            score=float(score),
            confidence=confidence,
            contributions=contributions,
            top_contributors=top_contributors,
            data_completeness=float(completeness)
        )
        
        return result
    
    def score_dataframe(
        self,
        df: pd.DataFrame,
        normalized_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Score entire dataframe of leads
        
        Args:
            df: Raw data DataFrame
            normalized_df: DataFrame with normalized features
            
        Returns:
            DataFrame with scores and metadata
        """
        logger.info(f"Scoring {len(df)} leads...")
        
        # Get all feature names from feature_weights
        all_feature_names = []
        for category, features in self.feature_weights.items():
            all_feature_names.extend(features.keys())
        
        results = []
        
        for idx, row in df.iterrows():
            # Get normalized features for this row
            if idx in normalized_df.index:
                normalized_row = normalized_df.loc[idx]
            else:
                normalized_row = pd.Series()
            
            # Score the lead
            result = self.score_lead(row, normalized_row, all_feature_names)
            
            results.append({
                'business_id': result.business_id,
                'score': result.score,
                'confidence_flag': result.confidence,
                'data_completeness': result.data_completeness,
                'top_3_contributors': self._format_top_contributors(result.top_contributors),
                'top_1_feature': result.top_contributors[0][0] if result.top_contributors else None,
                'top_1_contribution_pct': result.top_contributors[0][1] if result.top_contributors else 0.0,
                'top_2_feature': result.top_contributors[1][0] if len(result.top_contributors) > 1 else None,
                'top_2_contribution_pct': result.top_contributors[1][1] if len(result.top_contributors) > 1 else 0.0,
                'top_3_feature': result.top_contributors[2][0] if len(result.top_contributors) > 2 else None,
                'top_3_contribution_pct': result.top_contributors[2][1] if len(result.top_contributors) > 2 else 0.0,
            })
        
        results_df = pd.DataFrame(results)
        
        # Merge with original data
        output_df = df.copy()
        output_df = output_df.merge(results_df, on='business_id', how='left')
        
        logger.info("Scoring complete")
        
        return output_df
    
    def _format_top_contributors(
        self,
        top_contributors: List[Tuple[str, float]]
    ) -> str:
        """
        Format top contributors for CSV output
        
        Args:
            top_contributors: List of (feature_name, percentage) tuples
            
        Returns:
            Formatted string
        """
        formatted = []
        for feature_name, percentage in top_contributors:
            # Clean feature name for readability
            clean_name = feature_name.replace('_', ' ').title()
            formatted.append(f"{clean_name} ({percentage:.1f}%)")
        
        return " | ".join(formatted)
    
    def get_feature_importance(
        self,
        scored_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate feature importance across all leads
        
        Args:
            scored_df: DataFrame with scoring results
            
        Returns:
            DataFrame with feature importance statistics
        """
        importance_data = []
        
        # Collect feature contributions from all leads
        feature_contributions = {}
        
        for col in ['top_1_feature', 'top_2_feature', 'top_3_feature']:
            if col in scored_df.columns:
                feature_col = col
                pct_col = col.replace('feature', 'contribution_pct')
                
                for feature, pct in zip(scored_df[feature_col], scored_df[pct_col]):
                    if pd.notna(feature):
                        if feature not in feature_contributions:
                            feature_contributions[feature] = []
                        feature_contributions[feature].append(pct)
        
        # Calculate statistics
        for feature, contributions in feature_contributions.items():
            importance_data.append({
                'feature': feature,
                'appearance_count': len(contributions),
                'avg_contribution_pct': np.mean(contributions),
                'max_contribution_pct': np.max(contributions),
                'min_contribution_pct': np.min(contributions),
                'std_contribution_pct': np.std(contributions)
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('avg_contribution_pct', ascending=False)
        
        return importance_df


def create_scoring_engine(config_path: str) -> LeadScoringEngine:
    """
    Factory function to create scoring engine
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized LeadScoringEngine
    """
    return LeadScoringEngine(config_path)
