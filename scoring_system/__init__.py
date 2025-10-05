"""
Lead Scoring System Package

A production-ready lead scoring system for aesthetic clinics.
"""

__version__ = '1.0.0'
__author__ = 'DeepAgent'

from .feature_engineering import FeatureEngineer, FeatureComputer
from .scoring_engine import LeadScoringEngine
from .pipeline import LeadScoringPipeline
from .evaluation import ScoringEvaluator
from .data_generator import ClinicDataGenerator, generate_sample_data

__all__ = [
    'FeatureEngineer',
    'FeatureComputer',
    'LeadScoringEngine',
    'LeadScoringPipeline',
    'ScoringEvaluator',
    'ClinicDataGenerator',
    'generate_sample_data'
]
