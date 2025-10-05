"""
Synthetic Data Generator for Aesthetic Clinic Lead Scoring

Generates realistic sample data with proper distributions, correlations,
and missing data patterns.

Author: DeepAgent
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)
random.seed(42)


class ClinicDataGenerator:
    """Generate realistic synthetic data for aesthetic clinics"""
    
    def __init__(self, config: Dict):
        """
        Initialize data generator
        
        Args:
            config: Configuration dictionary with territory definitions
        """
        self.config = config
        self.territories = config.get('territories', {})
        
        # Service templates with realistic combinations
        self.service_templates = [
            "Body Contouring, Laser Hair Removal, Injectables, Skin Rejuvenation",
            "Botox, Dermal Fillers, Laser Hair Removal, Facials",
            "CoolSculpting, SculpSure, Laser Treatments",
            "Injectables, Skin Care, Chemical Peels",
            "Body Sculpting, Fat Reduction, Skin Tightening",
            "Laser Hair Removal, IPL Treatments, Facials",
            "Comprehensive Medical Spa Services",
            "Anti-Aging Treatments, Injectables, Laser",
            "Body Contouring, Cellulite Reduction, Facials",
            "Cosmetic Injectables, Laser Treatments, Microneedling"
        ]
        
        # Device brands for mentions
        self.device_brands = [
            'CoolSculpting', 'SculpSure', 'Cynosure', 'Candela',
            'Syneron', 'Lumenis', 'Sciton', None
        ]
        
        # Clinic name patterns
        self.name_patterns = [
            "Elite", "Premium", "Luxury", "Advanced", "Modern",
            "Radiant", "Revive", "Rejuvenate", "Transform", "Sculpt",
            "Beauty", "Aesthetic", "Med Spa", "Wellness", "Glow"
        ]
        
        self.name_suffixes = [
            "Med Spa", "Aesthetics", "Clinic", "Wellness Center",
            "Beauty Lab", "Medical Spa", "Rejuvenation Center",
            "Body Sculpting", "Laser Center", "Cosmetic Center"
        ]
    
    def generate_clinic_name(self) -> str:
        """Generate realistic clinic name"""
        pattern = random.choice(self.name_patterns)
        suffix = random.choice(self.name_suffixes)
        return f"{pattern} {suffix}"
    
    def generate_address(self, city: str, state: str = "TX") -> str:
        """Generate realistic address"""
        street_num = random.randint(100, 9999)
        street_names = [
            "Main St", "Oak Ave", "Elm St", "Park Blvd", "Medical Dr",
            "Wellness Way", "Beauty Ln", "Healthcare Plaza", "Center Dr"
        ]
        street = random.choice(street_names)
        
        if random.random() < 0.3:  # 30% have suite numbers
            suite = f", Suite {random.randint(100, 500)}"
        else:
            suite = ""
        
        return f"{street_num} {street}{suite}, {city}, {state}"
    
    def generate_coordinates(
        self, 
        territory: str
    ) -> Tuple[float, float]:
        """
        Generate coordinates within territory bounds
        
        Args:
            territory: Territory name (austin, san_antonio, etc.)
            
        Returns:
            Tuple of (latitude, longitude)
        """
        if territory not in self.territories:
            # Default to Austin if territory not found
            center_lat, center_lon = 30.2672, -97.7431
            radius_miles = 30
        else:
            t = self.territories[territory]
            center_lat = t['center_lat']
            center_lon = t['center_lon']
            radius_miles = t['radius_miles']
        
        # Convert miles to degrees (rough approximation)
        # 1 degree latitude ≈ 69 miles
        # 1 degree longitude ≈ 54 miles (at Texas latitude)
        radius_lat = radius_miles / 69.0
        radius_lon = radius_miles / 54.0
        
        # Generate random point within circle
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(0, 1) ** 0.5  # Square root for uniform distribution
        
        lat = center_lat + radius * radius_lat * np.cos(angle)
        lon = center_lon + radius * radius_lon * np.sin(angle)
        
        return round(lat, 6), round(lon, 6)
    
    def generate_services(self) -> str:
        """Generate service offerings with realistic patterns"""
        # 60% use templates, 40% custom combinations
        if random.random() < 0.6:
            services = random.choice(self.service_templates)
        else:
            service_options = [
                "Body Contouring", "Laser Hair Removal", "Botox", "Dermal Fillers",
                "Injectables", "Skin Rejuvenation", "CoolSculpting", "Chemical Peels",
                "Microneedling", "Facials", "Anti-Aging Treatments"
            ]
            num_services = random.randint(2, 5)
            services = ", ".join(random.sample(service_options, num_services))
        
        # 30% mention device brands
        if random.random() < 0.3:
            brand = random.choice([b for b in self.device_brands if b is not None])
            services = f"{services} ({brand})"
        
        return services
    
    def generate_social_metrics(
        self,
        clinic_size: str
    ) -> Dict:
        """
        Generate correlated social media metrics based on clinic size
        
        Args:
            clinic_size: 'small', 'medium', or 'large'
            
        Returns:
            Dictionary of social metrics
        """
        # Base distributions by size
        if clinic_size == 'small':
            followers_mean, followers_std = 500, 400
            posts_mean, posts_std = 15, 10
            engagement_base = 0.05
        elif clinic_size == 'medium':
            followers_mean, followers_std = 2500, 1500
            posts_mean, posts_std = 35, 15
            engagement_base = 0.04
        else:  # large
            followers_mean, followers_std = 8000, 4000
            posts_mean, posts_std = 60, 20
            engagement_base = 0.03
        
        # Generate with log-normal distribution for followers
        followers = int(max(0, np.random.lognormal(
            np.log(followers_mean), 0.7
        )))
        
        # Posts correlated with followers
        posts = int(max(0, np.random.normal(posts_mean, posts_std)))
        
        # Engagement rate decreases slightly with follower count
        engagement_rate = max(0.01, np.random.normal(
            engagement_base, 0.015
        ))
        
        # Average engagement per post
        avg_engagement = int(followers * engagement_rate)
        
        # Treatment hashtags
        hashtag_count = random.randint(5, 30)
        
        return {
            'instagram_followers': followers,
            'posts_90d': posts,
            'avg_engagement_per_post': avg_engagement,
            'engagement_rate': round(engagement_rate, 4),
            'treatment_hashtag_count': hashtag_count
        }
    
    def generate_review_metrics(
        self,
        clinic_size: str,
        quality_tier: str
    ) -> Dict:
        """
        Generate Google review metrics
        
        Args:
            clinic_size: 'small', 'medium', or 'large'
            quality_tier: 'low', 'medium', or 'high'
            
        Returns:
            Dictionary of review metrics
        """
        # Review count by size
        if clinic_size == 'small':
            review_mean, review_std = 25, 20
        elif clinic_size == 'medium':
            review_mean, review_std = 100, 50
        else:
            review_mean, review_std = 250, 100
        
        review_count = int(max(0, np.random.normal(review_mean, review_std)))
        
        # Rating by quality tier
        if quality_tier == 'high':
            rating_mean = 4.6
        elif quality_tier == 'medium':
            rating_mean = 4.1
        else:
            rating_mean = 3.5
        
        rating = round(np.clip(
            np.random.normal(rating_mean, 0.3),
            1.0, 5.0
        ), 1)
        
        # Days since last review (recency)
        days_since = int(np.random.exponential(30))  # Exponential distribution
        
        return {
            'google_reviews_count': review_count,
            'google_rating': rating,
            'days_since_last_review': days_since
        }
    
    def generate_capacity_metrics(
        self,
        clinic_size: str
    ) -> Dict:
        """
        Generate capacity and scale metrics
        
        Args:
            clinic_size: 'small', 'medium', or 'large'
            
        Returns:
            Dictionary of capacity metrics
        """
        if clinic_size == 'small':
            staff = random.randint(2, 6)
            rooms = random.randint(1, 3)
            providers = random.randint(1, 2)
        elif clinic_size == 'medium':
            staff = random.randint(7, 15)
            rooms = random.randint(3, 6)
            providers = random.randint(2, 4)
        else:
            staff = random.randint(15, 40)
            rooms = random.randint(6, 12)
            providers = random.randint(4, 8)
        
        return {
            'staff_count_est': staff,
            'treatment_rooms': rooms,
            'credentialed_providers': providers
        }
    
    def generate_growth_signals(
        self,
        growth_stage: str
    ) -> Dict:
        """
        Generate growth indicators
        
        Args:
            growth_stage: 'stable', 'growing', or 'expanding'
            
        Returns:
            Dictionary of growth metrics
        """
        if growth_stage == 'expanding':
            job_postings = random.randint(2, 8)
            new_providers = random.randint(1, 3)
            website_updated = 1
        elif growth_stage == 'growing':
            job_postings = random.randint(1, 3)
            new_providers = random.randint(0, 2)
            website_updated = random.choice([0, 1])
        else:  # stable
            job_postings = random.randint(0, 1)
            new_providers = 0
            website_updated = random.choice([0, 0, 1])
        
        return {
            'job_postings_last_90d': job_postings,
            'new_providers_90d': new_providers,
            'website_updated_recently': website_updated
        }
    
    def generate_contact_info(
        self,
        sophistication_level: str
    ) -> Dict:
        """
        Generate contact information
        
        Args:
            sophistication_level: 'low', 'medium', or 'high'
            
        Returns:
            Dictionary of contact info
        """
        # Phone number (most have this)
        has_phone = random.random() < 0.95
        phone = f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}" if has_phone else None
        
        # Email (varies by sophistication)
        email_prob = 0.6 if sophistication_level == 'low' else (0.8 if sophistication_level == 'medium' else 0.95)
        has_email = random.random() < email_prob
        
        if has_email:
            domains = ['gmail.com', 'yahoo.com', 'medspa.com', 'clinic.com']
            email = f"contact@example{random.randint(1, 100)}.{random.choice(domains)}"
        else:
            email = None
        
        # Booking link (higher sophistication = more likely)
        booking_prob = 0.3 if sophistication_level == 'low' else (0.6 if sophistication_level == 'medium' else 0.85)
        booking_link = f"https://booking{random.randint(1, 100)}.com" if random.random() < booking_prob else None
        
        # Response indicators
        responds_24h = 1 if sophistication_level == 'high' and random.random() < 0.7 else 0
        
        return {
            'contact_phone': phone,
            'contact_email': email,
            'booking_link': booking_link,
            'responds_within_24h': responds_24h
        }
    
    def generate_competitive_context(
        self,
        territory: str
    ) -> int:
        """
        Generate competitor count (varies by territory and location)
        
        Args:
            territory: Territory name
            
        Returns:
            Number of competitors within 5-mile radius
        """
        # Urban areas have more competition
        if territory in ['austin', 'san_antonio']:
            comp_mean = 8
        elif territory == 'el_paso':
            comp_mean = 5
        else:  # waco
            comp_mean = 3
        
        competitors = int(max(0, np.random.normal(comp_mean, 3)))
        return competitors
    
    def introduce_missing_data(
        self,
        df: pd.DataFrame,
        missing_rate: float = 0.15
    ) -> pd.DataFrame:
        """
        Introduce realistic missing data patterns
        
        Args:
            df: Complete dataframe
            missing_rate: Overall missing rate (default 15%)
            
        Returns:
            DataFrame with missing values
        """
        df_copy = df.copy()
        
        # Columns that can have missing data (not critical fields)
        optional_cols = [
            'instagram_followers', 'posts_90d', 'avg_engagement_per_post',
            'treatment_hashtag_count', 'job_postings_last_90d',
            'new_providers_90d', 'contact_email', 'booking_link',
            'days_since_last_review'
        ]
        
        for col in optional_cols:
            if col in df_copy.columns:
                # Randomly set values to NaN
                mask = np.random.random(len(df_copy)) < missing_rate
                df_copy.loc[mask, col] = np.nan
        
        # Some clinics have no social media presence (complete missingness)
        no_social_mask = np.random.random(len(df_copy)) < 0.1  # 10% have no social
        social_cols = ['instagram_followers', 'posts_90d', 'avg_engagement_per_post', 'treatment_hashtag_count']
        for col in social_cols:
            if col in df_copy.columns:
                df_copy.loc[no_social_mask, col] = np.nan
        
        return df_copy
    
    def generate_dataset(
        self,
        territory_distribution: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Generate complete synthetic dataset
        
        Args:
            territory_distribution: Dict mapping territory name to clinic count
            
        Returns:
            DataFrame with synthetic clinic data
        """
        all_clinics = []
        clinic_id = 1
        
        for territory, count in territory_distribution.items():
            print(f"Generating {count} clinics for {territory}...")
            
            for _ in range(count):
                # Randomly assign characteristics
                clinic_size = random.choices(
                    ['small', 'medium', 'large'],
                    weights=[0.4, 0.45, 0.15]
                )[0]
                
                quality_tier = random.choices(
                    ['low', 'medium', 'high'],
                    weights=[0.15, 0.55, 0.30]
                )[0]
                
                growth_stage = random.choices(
                    ['stable', 'growing', 'expanding'],
                    weights=[0.50, 0.35, 0.15]
                )[0]
                
                sophistication = random.choices(
                    ['low', 'medium', 'high'],
                    weights=[0.25, 0.50, 0.25]
                )[0]
                
                # Generate all components
                lat, lon = self.generate_coordinates(territory)
                city = territory.replace('_', ' ').title()
                
                clinic = {
                    'business_id': f"CLINIC_{clinic_id:04d}",
                    'name': self.generate_clinic_name(),
                    'address': self.generate_address(city),
                    'city': city,
                    'latitude': lat,
                    'longitude': lon,
                    'territory': territory,
                    'main_services': self.generate_services(),
                }
                
                # Add metrics
                clinic.update(self.generate_social_metrics(clinic_size))
                clinic.update(self.generate_review_metrics(clinic_size, quality_tier))
                clinic.update(self.generate_capacity_metrics(clinic_size))
                clinic.update(self.generate_growth_signals(growth_stage))
                clinic.update(self.generate_contact_info(sophistication))
                
                # Add competitive context
                clinic['competitors_5mi_radius'] = self.generate_competitive_context(territory)
                
                # Metadata for analysis
                clinic['_clinic_size'] = clinic_size
                clinic['_quality_tier'] = quality_tier
                clinic['_growth_stage'] = growth_stage
                
                all_clinics.append(clinic)
                clinic_id += 1
        
        # Create DataFrame
        df = pd.DataFrame(all_clinics)
        
        # Introduce realistic missing data
        df = self.introduce_missing_data(df)
        
        return df


def generate_sample_data(config: Dict) -> pd.DataFrame:
    """
    Convenience function to generate sample data
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with synthetic clinic data
    """
    generator = ClinicDataGenerator(config)
    
    # Territory distribution as specified
    distribution = {
        'austin': 40,
        'san_antonio': 30,
        'el_paso': 20,
        'waco': 10
    }
    
    df = generator.generate_dataset(distribution)
    
    print(f"\nGenerated {len(df)} clinic records")
    print(f"Missing data rate: {df.isna().sum().sum() / (df.shape[0] * df.shape[1]):.1%}")
    
    return df
