
#!/usr/bin/env python3
"""
Venus Medical Device Prospecting System
Comprehensive tool for finding and scoring medical practices for Venus device sales
Version 2.0 - Now with advanced ML-based scoring engine
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import googlemaps
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
import tldextract
import yaml

# Import the new scoring system
from scoring_system.scoring_engine import LeadScoringEngine
from scoring_system.venus_adapter import VenusScoringAdapter

# Load environment variables
load_dotenv()

# Configure logging
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prospector.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VenusProspector:
    """Main prospecting system for Venus medical devices"""
    
    def __init__(self, demo_mode=False, existing_customers_csv=None):
        # Initialize Google Maps client
        self.gmaps_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not self.gmaps_key:
            raise ValueError('GOOGLE_PLACES_API_KEY environment variable is required. Please set it in your .env file.')
        self.demo_mode = demo_mode
        self.existing_customers = set()
        
        # Initialize the new scoring engine
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'scoring_config.yaml')
        try:
            self.scoring_engine = LeadScoringEngine(config_path)
            self.scoring_adapter = VenusScoringAdapter()
            logger.info("✓ New scoring engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize scoring engine: {str(e)}")
            logger.warning("Falling back to legacy scoring system")
            self.scoring_engine = None
            self.scoring_adapter = None
        
        # Load existing customers for exclusion
        if existing_customers_csv and os.path.exists(existing_customers_csv):
            self.load_existing_customers(existing_customers_csv)
        
        # Initialize session for web scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if not demo_mode:
            try:
                self.gmaps = googlemaps.Client(key=self.gmaps_key)
                # Test the API key with a simple request
                test_result = self.gmaps.geocode("Austin, TX")
                if not test_result:
                    logger.warning("Google Places API test failed - switching to demo mode")
                    self.demo_mode = True
            except Exception as e:
                logger.warning(f"Google Places API initialization failed: {str(e)} - switching to demo mode")
                self.demo_mode = True
        
        if self.demo_mode:
            logger.info("Running in DEMO MODE with mock data")
        
        # Search templates for different practice types
        self.search_templates = {
            'med_spa': ['med spa', 'medical spa', 'aesthetic clinic', 'cosmetic clinic'],
            'plastic_surgeon': ['plastic surgeon', 'cosmetic surgeon', 'aesthetic surgeon'],
            'dermatologist': ['dermatologist', 'dermatology clinic', 'skin clinic'],
            'cosmetic_surgeon': ['cosmetic surgery', 'aesthetic surgery', 'beauty surgery'],
            'obgyn': ['obgyn', 'gynecologist', 'women\'s health clinic'],
            'wellness': ['wellness center', 'anti-aging clinic', 'rejuvenation clinic']
        }
        
        # Hospital/medical center exclusion patterns
        self.hospital_patterns = [
            r'\bhospital\b', r'\bmedical center\b', r'\bhealth system\b',
            r'\bhealthcare system\b', r'\bregional medical\b', r'\buniversity medical\b'
        ]
        
        # Venus device catalog
        self.venus_devices = {
            'Venus Versa': {
                'specialties': ['hair removal', 'photorejuvenation', 'skin resurfacing', 'acne treatment'],
                'keywords': ['laser hair removal', 'ipl', 'photo facial', 'skin tightening']
            },
            'Venus Legacy': {
                'specialties': ['body contouring', 'cellulite reduction', 'skin tightening'],
                'keywords': ['body sculpting', 'cellulite', 'radiofrequency', 'body tightening']
            },
            'Venus Bliss MAX': {
                'specialties': ['weight loss', 'muscle stimulation', 'body contouring'],
                'keywords': ['weight loss', 'ems', 'muscle building', 'fat reduction', 'glp-1']
            }
        }
        
        # Initialize session for web scraping
        
    def load_existing_customers(self, csv_file_path: str):
        """Load existing customers from CSV for exclusion from prospecting"""
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loading existing customers from {csv_file_path}")
            
            # Try common column names for practice names
            name_columns = ['name', 'practice_name', 'business_name', 'company_name', 'customer_name']
            name_column = None
            
            for col in name_columns:
                if col.lower() in [c.lower() for c in df.columns]:
                    name_column = col
                    break
            
            if name_column is None:
                # Use first column if no standard name found
                name_column = df.columns[0]
                logger.warning(f"No standard name column found, using '{name_column}'")
            
            # Clean and add names to exclusion set
            for name in df[name_column].dropna():
                # Normalize names for better matching
                cleaned_name = str(name).lower().strip()
                # Remove common business suffixes
                cleaned_name = re.sub(r'\b(llc|inc|corp|ltd|pllc|pa|md|pc)\b', '', cleaned_name).strip()
                self.existing_customers.add(cleaned_name)
            
            logger.info(f"Loaded {len(self.existing_customers)} existing customers for exclusion")
            
        except Exception as e:
            logger.error(f"Failed to load existing customers CSV: {str(e)}")
    
    def is_existing_customer(self, practice_name: str) -> bool:
        """Check if practice is an existing customer and should be excluded"""
        if not self.existing_customers:
            return False
        
        # Normalize practice name for comparison
        cleaned_name = practice_name.lower().strip()
        cleaned_name = re.sub(r'\b(llc|inc|corp|ltd|pllc|pa|md|pc)\b', '', cleaned_name).strip()
        
        # Check for exact match or partial match
        for existing_customer in self.existing_customers:
            if existing_customer in cleaned_name or cleaned_name in existing_customer:
                return True
        
        return False



    def get_mock_data(self, query: str, location: str) -> List[Dict]:
        """Generate mock data for demo purposes"""
        mock_practices = [
            {
                'name': 'Austin Aesthetic Center',
                'formatted_address': '123 Main St, Austin, TX 78701',
                'formatted_phone_number': '(512) 555-0101',
                'website': 'https://austinaesthetic.com',
                'rating': 4.8,
                'user_ratings_total': 127,
                'types': ['beauty_salon', 'health'],
                'place_id': 'mock_1'
            },
            {
                'name': 'Hill Country Med Spa',
                'formatted_address': '456 Oak Ave, Austin, TX 78704',
                'formatted_phone_number': '(512) 555-0102',
                'website': 'https://hillcountrymedspa.com',
                'rating': 4.6,
                'user_ratings_total': 89,
                'types': ['spa', 'health'],
                'place_id': 'mock_2'
            },
            {
                'name': 'Lone Star Dermatology',
                'formatted_address': '789 Cedar St, Austin, TX 78702',
                'formatted_phone_number': '(512) 555-0103',
                'website': 'https://lonestardermatology.com',
                'rating': 4.9,
                'user_ratings_total': 203,
                'types': ['doctor', 'health'],
                'place_id': 'mock_3'
            },
            {
                'name': 'Capital City Cosmetic Surgery',
                'formatted_address': '321 Elm St, Austin, TX 78703',
                'formatted_phone_number': '(512) 555-0104',
                'website': 'https://capitalcitycosmetic.com',
                'rating': 4.7,
                'user_ratings_total': 156,
                'types': ['doctor', 'health'],
                'place_id': 'mock_4'
            },
            {
                'name': 'Rejuvenation Wellness Center',
                'formatted_address': '654 Pine St, Austin, TX 78705',
                'formatted_phone_number': '(512) 555-0105',
                'website': 'https://rejuvenationwellness.com',
                'rating': 4.5,
                'user_ratings_total': 78,
                'types': ['spa', 'health'],
                'place_id': 'mock_5'
            }
        ]
        
        # Filter based on query
        filtered_practices = []
        query_lower = query.lower()
        for practice in mock_practices:
            if ('med spa' in query_lower and 'spa' in practice['name'].lower()) or \
               ('dermatolog' in query_lower and 'dermatology' in practice['name'].lower()) or \
               ('cosmetic' in query_lower and 'cosmetic' in practice['name'].lower()) or \
               ('aesthetic' in query_lower and 'aesthetic' in practice['name'].lower()):
                filtered_practices.append(practice)
        
        return filtered_practices if filtered_practices else mock_practices[:3]

    @sleep_and_retry
    @limits(calls=10, period=60)  # Rate limiting: 10 calls per minute
    def estimate_result_count(self, keywords: List[str], location: str, radius: int) -> int:
        """
        Quick estimate of total result count for dynamic timeout calculation
        Returns estimated number of businesses that will be found
        """
        if self.demo_mode:
            return 30  # Demo mode estimate
        
        try:
            logger.info(f"Estimating result count for {keywords} near {location}")
            
            # Geocode the location
            geocode_result = self.gmaps.geocode(location)
            if not geocode_result:
                logger.warning(f"Could not geocode location for estimate: {location}")
                return 50  # Default estimate
            
            lat_lng = geocode_result[0]['geometry']['location']
            
            # Do a quick count across all keywords
            total_estimate = 0
            for keyword in keywords:
                try:
                    # Quick search without fetching details
                    places_result = self.gmaps.places_nearby(
                        location=lat_lng,
                        radius=radius * 1000,  # Convert km to meters
                        keyword=keyword,
                        type='health'
                    )
                    
                    result_count = len(places_result.get('results', []))
                    total_estimate += result_count
                    logger.info(f"  • {keyword}: ~{result_count} results")
                    
                except Exception as e:
                    logger.warning(f"Error estimating results for '{keyword}': {str(e)}")
                    continue
            
            # Remove duplicate estimate (assume 30% overlap across keywords)
            estimated_unique = int(total_estimate * 0.7)
            logger.info(f"Estimated total unique results: {estimated_unique}")
            
            return estimated_unique
            
        except Exception as e:
            logger.warning(f"Error estimating result count: {str(e)}")
            return 50  # Default safe estimate
    
    def google_places_search(self, query: str, location: str, radius: int = 25000) -> List[Dict]:
        """Search Google Places API for medical practices"""
        
        if self.demo_mode:
            logger.info(f"DEMO MODE: Generating mock data for {query} near {location}")
            return self.get_mock_data(query, location)
        
        try:
            logger.info(f"Searching Google Places: {query} near {location}")
            
            # Geocode the location first
            geocode_result = self.gmaps.geocode(location)
            if not geocode_result:
                logger.error(f"Could not geocode location: {location}")
                return []
            
            lat_lng = geocode_result[0]['geometry']['location']
            
            # Search for places
            places_result = self.gmaps.places_nearby(
                location=lat_lng,
                radius=radius,
                keyword=query,
                type='health'
            )
            
            results = []
            for place in places_result.get('results', []):
                # Skip hospitals and medical centers
                if self.is_hospital_system(place.get('name', '')):
                    continue
                    
                place_details = self.get_place_details(place['place_id'])
                if place_details:
                    results.append(place_details)
                    
                time.sleep(0.1)  # Small delay between API calls
                
            return results
            
        except Exception as e:
            logger.error(f"Error in Google Places search: {str(e)}")
            logger.info("Falling back to demo mode")
            self.demo_mode = True
            return self.get_mock_data(query, location)

    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information for a specific place"""
        try:
            details = self.gmaps.place(
                place_id=place_id,
                fields=['name', 'formatted_address', 'formatted_phone_number', 
                       'website', 'rating', 'user_ratings_total', 'type',
                       'opening_hours', 'geometry', 'reviews']
            )
            
            return details.get('result', {})
            
        except Exception as e:
            logger.error(f"Error getting place details: {str(e)}")
            return None

    def is_hospital_system(self, name: str) -> bool:
        """Check if a practice name indicates a hospital system"""
        name_lower = name.lower()
        return any(re.search(pattern, name_lower, re.IGNORECASE) for pattern in self.hospital_patterns)

    def check_robots_txt(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch('*', url)
        except:
            return True  # If can't check, assume allowed

    def get_mock_website_data(self, url: str) -> Dict[str, str]:
        """Generate mock website data for demo purposes"""
        mock_data = {
            'https://austinaesthetic.com': {
                'title': 'Austin Aesthetic Center - Premier Med Spa',
                'description': 'Leading medical spa offering laser hair removal, botox, fillers, and advanced skin treatments in Austin, TX.',
                'services': ['laser hair removal', 'botox', 'fillers', 'skin tightening', 'photorejuvenation'],
                'social_links': ['https://facebook.com/austinaesthetic', 'https://instagram.com/austinaesthetic'],
                'staff_count': 8
            },
            'https://hillcountrymedspa.com': {
                'title': 'Hill Country Med Spa - Body Contouring & Wellness',
                'description': 'Full-service medical spa specializing in body contouring, cellulite treatment, and wellness services.',
                'services': ['body contouring', 'cellulite treatment', 'coolsculpting', 'weight loss'],
                'social_links': ['https://instagram.com/hillcountrymedspa'],
                'staff_count': 5
            },
            'https://lonestardermatology.com': {
                'title': 'Lone Star Dermatology - Advanced Skin Care',
                'description': 'Board-certified dermatologists providing medical and cosmetic dermatology services.',
                'services': ['laser hair removal', 'skin tightening', 'photorejuvenation', 'botox'],
                'social_links': ['https://facebook.com/lonestardermatology', 'https://linkedin.com/company/lonestardermatology'],
                'staff_count': 12
            },
            'https://capitalcitycosmetic.com': {
                'title': 'Capital City Cosmetic Surgery - Aesthetic Excellence',
                'description': 'Premier cosmetic surgery practice offering surgical and non-surgical aesthetic treatments.',
                'services': ['body contouring', 'fillers', 'botox', 'skin tightening'],
                'social_links': ['https://instagram.com/capitalcitycosmetic', 'https://facebook.com/capitalcitycosmetic'],
                'staff_count': 15
            },
            'https://rejuvenationwellness.com': {
                'title': 'Rejuvenation Wellness Center - Anti-Aging & Wellness',
                'description': 'Comprehensive wellness center offering anti-aging treatments, weight loss programs, and aesthetic services.',
                'services': ['weight loss', 'ems', 'muscle building', 'botox', 'fillers'],
                'social_links': ['https://facebook.com/rejuvenationwellness'],
                'staff_count': 6
            }
        }
        
        return mock_data.get(url, {
            'title': 'Medical Practice',
            'description': 'Professional medical and aesthetic services',
            'services': ['botox', 'fillers'],
            'social_links': [],
            'staff_count': 3
        })

    @sleep_and_retry
    @limits(calls=30, period=60)  # Rate limiting for web scraping
    def scrape_website(self, url: str) -> Dict[str, str]:
        """Scrape practice website for additional information"""
        
        if self.demo_mode:
            logger.info(f"DEMO MODE: Using mock website data for {url}")
            return self.get_mock_website_data(url)
        
        if not self.check_robots_txt(url):
            logger.warning(f"Robots.txt disallows scraping: {url}")
            return {
                'title': 'Not Available - Robots.txt Restricted',
                'description': 'Not Available - Robots.txt Restricted',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
            
        try:
            time.sleep(random.uniform(0.5, 1.0))  # Polite delay
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract relevant information with explicit "Not Available" indicators
            data = {
                'title': '',
                'description': '',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
            
            # Get page title
            if soup.title and soup.title.string:
                data['title'] = soup.title.string.strip()
            else:
                data['title'] = 'Not Available'
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                data['description'] = meta_desc.get('content', '').strip()
            else:
                # Try og:description as fallback
                og_desc = soup.find('meta', attrs={'property': 'og:description'})
                if og_desc and og_desc.get('content'):
                    data['description'] = og_desc.get('content', '').strip()
                else:
                    data['description'] = 'Not Available'
            
            # Extract services mentioned
            text_content = soup.get_text().lower()
            service_keywords = [
                'laser hair removal', 'botox', 'fillers', 'coolsculpting',
                'body contouring', 'skin tightening', 'photorejuvenation',
                'cellulite treatment', 'weight loss', 'ems', 'muscle building',
                'dermal fillers', 'juvederm', 'restylane', 'sculptra',
                'microneedling', 'chemical peels', 'hydrafacial', 'prp',
                'iv therapy', 'hormone therapy', 'medical weight loss'
            ]
            
            services_found = []
            for keyword in service_keywords:
                if keyword in text_content:
                    services_found.append(keyword)
            
            data['services'] = services_found if services_found else []
            
            # Find social media links from the business website
            # We only extract the links/names, NOT scraping the social media accounts themselves
            social_platforms = {
                'facebook.com': 'Facebook',
                'instagram.com': 'Instagram',
                'twitter.com': 'Twitter',
                'x.com': 'Twitter/X',
                'linkedin.com': 'LinkedIn',
                'youtube.com': 'YouTube',
                'tiktok.com': 'TikTok',
                'pinterest.com': 'Pinterest'
            }
            
            social_links = []
            seen_platforms = set()
            
            for link in soup.find_all('a', href=True):
                href = link['href'].lower()
                for domain, platform_name in social_platforms.items():
                    if domain in href and platform_name not in seen_platforms:
                        # Clean up the URL
                        full_url = href if href.startswith('http') else urljoin(url, href)
                        social_links.append(f"{platform_name}: {full_url}")
                        seen_platforms.add(platform_name)
                        logger.info(f"Found social media link: {platform_name} - {full_url}")
                        break
            
            data['social_links'] = social_links
            
            # Estimate staff count (rough heuristic)
            staff_indicators = soup.find_all(text=re.compile(r'\b(dr\.|doctor|physician|provider|practitioner|nurse practitioner|pa-c)\b', re.I))
            unique_staff = len(set(str(s).strip() for s in staff_indicators if len(str(s).strip()) > 5))
            data['staff_count'] = min(unique_staff, 20)  # Cap at 20
            
            logger.info(f"Website scrape complete for {url}: {len(data['services'])} services, {len(data['social_links'])} social links")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout scraping website {url}")
            return {
                'title': 'Not Available - Timeout',
                'description': 'Not Available - Timeout',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error scraping website {url}: {str(e)}")
            return {
                'title': 'Not Available - Connection Error',
                'description': 'Not Available - Connection Error',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
        except Exception as e:
            logger.error(f"Error scraping website {url}: {str(e)}")
            return {
                'title': 'Not Available - Scraping Error',
                'description': 'Not Available - Scraping Error',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }

    def scrape_yelp_data(self, business_name: str, location: str) -> Dict:
        """Scrape Yelp for additional business information"""
        try:
            # Search Yelp for the business
            search_query = f"{business_name} {location}".replace(' ', '+')
            yelp_search_url = f"https://www.yelp.com/search?find_desc={search_query}"
            
            time.sleep(random.uniform(0.5, 1.0))
            response = self.session.get(yelp_search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract basic info (this is a simplified version)
                data = {
                    'yelp_rating': 0,
                    'yelp_reviews': 0,
                    'yelp_url': ''
                }
                
                # Look for rating and review count
                rating_elem = soup.find('div', class_=re.compile(r'rating'))
                if rating_elem:
                    rating_text = rating_elem.get_text()
                    rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                    if rating_match:
                        data['yelp_rating'] = float(rating_match.group(1))
                
                return data
            
        except Exception as e:
            logger.error(f"Error scraping Yelp data: {str(e)}")
            
        return {}

    def calculate_ai_score(self, practice_data: Dict) -> Tuple[int, Dict[str, int]]:
        """
        Calculate AI-powered scoring using the new scoring engine
        Version 2.0 - Now with ML-based scoring system
        """
        
        # Use new scoring engine if available, otherwise fall back to legacy
        if self.scoring_engine and self.scoring_adapter:
            try:
                return self._calculate_score_with_engine(practice_data)
            except Exception as e:
                logger.error(f"Scoring engine failed: {str(e)}, falling back to legacy")
                return self._calculate_score_legacy(practice_data)
        else:
            return self._calculate_score_legacy(practice_data)
    
    def _calculate_score_with_engine(self, practice_data: Dict) -> Tuple[int, Dict[str, int]]:
        """Use the new scoring engine to calculate scores"""
        
        # Convert practice data to features
        features = self.scoring_adapter.convert_practice_to_features(practice_data)
        
        # Create a single-row DataFrame
        df = pd.DataFrame([features])
        
        # The features are already normalized (0 or 1), so we can use them directly
        # Score using the engine
        all_feature_names = list(features.keys())
        
        # Create a normalized version (features are already 0/1, so this is pass-through)
        normalized_df = df.copy()
        
        # Score the lead
        result = self.scoring_engine.score_lead(
            row=df.iloc[0],
            normalized_features=normalized_df.iloc[0],
            all_feature_names=all_feature_names
        )
        
        # Extract category scores from contributions
        # The contributions dict has feature-level contributions
        # We need to aggregate them by category
        category_scores = self._aggregate_category_scores(result.contributions)
        
        # Convert to the format expected by existing code (0-100 integer scores per category)
        score_breakdown = {}
        for category, score in category_scores.items():
            # Score is already 0-100, convert to int
            score_breakdown[category] = int(score)
        
        total_score = int(result.score)
        
        logger.info(f"New scoring engine: {total_score}/100 (confidence: {result.confidence})")
        
        return total_score, score_breakdown
    
    def _aggregate_category_scores(self, feature_contributions: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate feature-level contributions into category scores
        
        Args:
            feature_contributions: Dict of feature names to contribution values (0-1 range)
            
        Returns:
            Dict of category names to scores (0-100 range)
        """
        # Get the category weights and feature weights from the config
        config = self.scoring_engine.config
        category_weights = config['category_weights']
        feature_weights = config['feature_weights']
        
        # Initialize category scores
        category_scores = {category: 0.0 for category in category_weights.keys()}
        
        # Map features to categories and aggregate
        for category, features in feature_weights.items():
            category_score = 0.0
            for feature_name in features.keys():
                if feature_name in feature_contributions:
                    # Contribution is already weighted, just sum them
                    category_score += feature_contributions[feature_name]
            
            # The contribution is already in 0-1 range and weighted by category importance
            # Convert to 0-100 scale
            category_scores[category] = category_score * 100.0
        
        return category_scores
    
    def _calculate_score_legacy(self, practice_data: Dict) -> Tuple[int, Dict[str, int]]:
        """
        Legacy scoring method (fallback)
        Calculate AI-powered scoring based on weighted criteria (0-100 scale)
        """
        
        scores = {
            'specialty_match': 0,          # 20 points
            'aesthetic_services': 0,       # 15 points
            'competing_devices': 0,        # 12 points
            'social_activity': 0,          # 8 points
            'practice_size': 0,            # 15 points (prefer smaller)
            'reviews_rating': 0,           # 5 points
            'search_visibility': 0,        # 5 points
            'geography_fit': 0,            # 10 points (less populated areas)
            'weight_loss_glp1': 0,         # 8 points (enhanced keywords)
            'skin_laxity_triggers': 0,     # 2 points
        }
        
        # 1. Specialty exact match (20 points)
        practice_types = practice_data.get('types', [])
        practice_name = practice_data.get('name', '').lower()
        practice_desc = practice_data.get('description', '').lower()
        all_text = f"{practice_name} {practice_desc}".lower()
        
        # Hospital exclusion check - significantly reduce score
        hospital_exclusions = [
            'hospital', 'medical center', 'health system', 'healthcare system', 
            'regional medical', 'university medical', 'medical group'
        ]
        is_hospital = any(exclusion in all_text for exclusion in hospital_exclusions)
        
        specialty_keywords = [
            'dermatology', 'plastic surgery', 'cosmetic', 'aesthetic',
            'med spa', 'medical spa', 'skin care', 'beauty', 'rejuvenation'
        ]
        
        specialty_matches = sum(1 for keyword in specialty_keywords 
                              if keyword in practice_name or keyword in practice_desc)
        scores['specialty_match'] = min(specialty_matches * 7, 20)
        
        # Reduce score significantly if hospital system
        if is_hospital:
            scores['specialty_match'] = max(0, scores['specialty_match'] - 15)
        
        # 2. Explicit aesthetic services (15 points)
        services = practice_data.get('services', [])
        aesthetic_services = [
            'botox', 'fillers', 'laser', 'coolsculpting', 'body contouring',
            'skin tightening', 'photorejuvenation', 'hair removal', 'microneedling',
            'hydrafacial', 'chemical peels', 'ipl'
        ]
        
        service_matches = sum(1 for service in aesthetic_services 
                            if any(service in s.lower() for s in services))
        scores['aesthetic_services'] = min(service_matches * 3, 15)
        
        # 3. Competing device presence (12 points - context matters)
        # Multi-platform competitors (poor fit for Versa Pro, better for Bliss Max)
        multi_platform_competitors = [
            'lumenis m22', 'alma harmony xl pro', 'cutera xeo', 'syneron etwo',
            'cynosure elite+', 'vydence etherea-mx', 'btl exilis ultra', 'radiance pod'
        ]
        
        # Single-purpose devices (good signal for expansion)
        single_purpose_devices = [
            'coolsculpting', 'thermage', 'ultherapy', 'sculptra', 'morpheus8', 
            'potenza', 'vivace', 'secret rf'
        ]
        
        multi_platform_count = sum(1 for device in multi_platform_competitors 
                                 if device in practice_desc.lower())
        single_device_count = sum(1 for device in single_purpose_devices 
                                if device in practice_desc.lower())
        
        # Score based on device context
        if multi_platform_count > 0:
            scores['competing_devices'] = 8  # Good for Bliss Max, not Versa Pro
        elif single_device_count > 0:
            scores['competing_devices'] = 12  # Great expansion opportunity
        else:
            scores['competing_devices'] = 6   # Clean slate opportunity
        
        # 4. Social media activity (8 points)
        social_links = practice_data.get('social_links', [])
        scores['social_activity'] = min(len(social_links) * 2, 8)
        
        # 5. Practice size estimate (15 points - prefer smaller practices)
        staff_count = practice_data.get('staff_count', 0)
        
        # Check for startup/small practice indicators
        startup_indicators = [
            'new', 'recently opened', 'grand opening', 'now open', 
            'established 202', 'founded 202'  # Recent establishments
        ]
        
        is_startup = any(indicator in all_text for indicator in startup_indicators)
        
        # Prefer smaller practices (better fit for Venus devices)
        if is_startup or staff_count <= 2:
            scores['practice_size'] = 15  # Startups/very small - highest priority
        elif staff_count <= 4:
            scores['practice_size'] = 12  # Small med spa/family practice
        elif staff_count <= 8:
            scores['practice_size'] = 8   # Medium practice
        else:
            scores['practice_size'] = 3   # Large practice - historically harder
        
        # 6. Reviews & rating (5 points)
        rating = practice_data.get('rating', 0)
        review_count = practice_data.get('user_ratings_total', 0)
        
        if rating >= 4.5 and review_count >= 50:
            scores['reviews_rating'] = 5
        elif rating >= 4.0 and review_count >= 20:
            scores['reviews_rating'] = 3
        elif rating >= 3.5:
            scores['reviews_rating'] = 2
        
        # 7. Search visibility (5 points)
        website = practice_data.get('website', '')
        if website:
            scores['search_visibility'] = 5
        elif practice_data.get('formatted_phone_number'):
            scores['search_visibility'] = 2
        
        # 8. Geography fit (10 points - prefer less populated/affluent areas)
        address = practice_data.get('formatted_address', '').lower()
        
        # Less populated area indicators (higher value)
        small_city_indicators = [
            'rd', 'drive', 'country', 'rural', 'main street', 'main st',
            'town', 'village', 'small town', 'suburb'
        ]
        
        # Over-serviced areas (lower value) 
        high_competition_areas = [
            'beverly hills', 'manhattan', 'miami beach', 'scottsdale',
            'malibu', 'newport beach', 'la jolla', 'austin downtown'
        ]
        
        if any(indicator in address for indicator in small_city_indicators):
            scores['geography_fit'] = 10  # Less populated = higher value
        elif any(area in address for area in high_competition_areas):
            scores['geography_fit'] = 4   # Over-serviced areas
        else:
            scores['geography_fit'] = 7   # Standard suburban/urban
        
        # 9. Enhanced Weight-loss/GLP-1 offerings (8 points)
        high_value_weight_keywords = [
            'ideal protein', 'biote', 'hormone pellets', 'iv therapy',
            'medical weight management', 'semaglutide', 'tirzepatide',
            'compounded glp-1', 'ozempic', 'wegovy', 'mounjaro',
            'metabolic medicine', 'functional medicine', 'hormone optimization'
        ]
        
        weight_matches = sum(1 for keyword in high_value_weight_keywords 
                           if keyword in all_text)
        scores['weight_loss_glp1'] = min(weight_matches * 3, 8)
        
        # 10. Skin laxity triggers (7 points - NEW)
        skin_laxity_triggers = [
            'post-surgical', 'facelift', 'body contouring surgery', 'blepharoplasty',
            'significant weight loss', 'bariatric surgery', 'gastric bypass',
            'mommy makeover', 'post-pregnancy', 'postpartum',
            'chronic steroid', 'corticosteroid use', 'skin thinning',
            'post-cosmetic surgery', 'revision surgery'
        ]
        
        laxity_matches = sum(1 for trigger in skin_laxity_triggers 
                           if trigger in all_text)
        scores['skin_laxity_triggers'] = min(laxity_matches * 4, 7)
        
        # 11. Growth indicators (8 points - NEW)
        growth_indicators = [
            'expanding', 'multiple locations', 'new services', 'recently added',
            'now offering', 'state-of-the-art', 'investment', 'upgraded',
            'grand opening', 'newly renovated', 'professional branding',
            'active social media', 'social media presence', 'instagram',
            'before and after', 'patient testimonials'
        ]
        
        growth_matches = sum(1 for indicator in growth_indicators 
                           if indicator in all_text)
        scores['growth_indicators'] = min(growth_matches * 2, 8)
        
        # 12. High-value specialties (7 points - NEW)
        # Family practices and OB/GYNs especially valuable for Bliss Max
        high_value_specialties = [
            'family practice', 'family medicine', 'primary care',
            'obgyn', 'gynecologist', 'women\'s health', 'reproductive health'
        ]
        
        # Specialties facing insurance challenges (seeking cash-pay)
        cash_pay_specialties = [
            'concierge medicine', 'direct pay', 'cash only',
            'membership medicine', 'boutique practice'
        ]
        
        specialty_score = 0
        for specialty in high_value_specialties:
            if specialty in all_text:
                specialty_score += 4
                
        for specialty in cash_pay_specialties:
            if specialty in all_text:
                specialty_score += 3
                
        scores['high_value_specialties'] = min(specialty_score, 7)
        
        total_score = sum(scores.values())
        return total_score, scores

    def recommend_venus_device(self, practice_data: Dict, ai_scores: Dict) -> Dict:
        """Recommend top 3 Venus devices with enhanced business logic"""
        
        services = practice_data.get('services', [])
        description = practice_data.get('description', '').lower()
        practice_name = practice_data.get('name', '').lower()
        all_text = f"{practice_name} {description}".lower()
        staff_count = practice_data.get('staff_count', 0)
        
        # Detect multi-platform competitors
        multi_platform_competitors = [
            'lumenis m22', 'alma harmony xl pro', 'cutera xeo', 'syneron etwo',
            'cynosure elite+', 'vydence etherea-mx', 'btl exilis ultra', 'radiance pod'
        ]
        
        has_multi_platform = any(device in all_text for device in multi_platform_competitors)
        
        # Detect high-value specialties
        family_practice = any(term in all_text for term in 
                            ['family practice', 'family medicine', 'primary care'])
        obgyn_practice = any(term in all_text for term in 
                           ['obgyn', 'gynecologist', 'women\'s health'])
        
        # Practice size categories
        is_small_practice = staff_count <= 4 or ai_scores.get('practice_size', 0) >= 12
        is_startup = ai_scores.get('practice_size', 0) == 15
        
        device_scores = {}
        
        # Score each Venus device with enhanced business logic
        for device_name, device_info in self.venus_devices.items():
            score = 0
            reasons = []
            
            # Base scoring for specialty alignment
            for specialty in device_info['specialties']:
                if any(specialty.lower() in service.lower() for service in services):
                    score += 20
                    reasons.append(f"Offers {specialty}")
                elif specialty.lower() in description:
                    score += 15
                    reasons.append(f"Mentions {specialty}")
            
            # Keyword matches
            for keyword in device_info['keywords']:
                if keyword.lower() in description:
                    score += 10
                    reasons.append(f"Keyword match: {keyword}")
            
            # ENHANCED DEVICE-SPECIFIC LOGIC
            if device_name == 'Venus Versa':
                # POOR FIT: Multi-platform competitors already offering IPL/RF Microneedling
                if has_multi_platform:
                    score -= 25  # Significant penalty
                    reasons.append("Already has multi-platform device - poor Versa fit")
                
                # GREAT FIT: Small practices and startups
                if is_small_practice:
                    score += 20
                    reasons.append("Small practice - ideal for Versa Pro")
                
                if is_startup:
                    score += 10
                    reasons.append("Startup practice - creative funding options")
                
                # Aesthetic services bonus
                if ai_scores.get('aesthetic_services', 0) > 10:
                    score += 15
                    reasons.append("Strong aesthetic portfolio")
            
            elif device_name == 'Venus Legacy':
                # Body contouring focus
                if ai_scores.get('competing_devices', 0) > 8:
                    score += 15
                    reasons.append("Uses competing body contouring devices")
                
                # Small to medium practice preference
                if is_small_practice:
                    score += 12
                    reasons.append("Good practice size for Legacy")
            
            elif device_name == 'Venus Bliss MAX':
                # EXCELLENT FIT: Multi-platform competitors (better than Versa)
                if has_multi_platform:
                    score += 25  # Major bonus
                    reasons.append("Multi-platform practice - excellent Bliss Max fit")
                
                # HIGH-VALUE SPECIALTIES: Family practice and OB/GYN bonus
                if family_practice:
                    score += 30
                    reasons.append("Family practice - high-value specialty for Bliss Max")
                
                if obgyn_practice:
                    score += 35  # Even higher for OB/GYN
                    reasons.append("OB/GYN practice - premium specialty for Bliss Max")
                
                # Weight management/GLP-1 services
                if ai_scores.get('weight_loss_glp1', 0) > 5:
                    score += 25
                    reasons.append("Strong weight management services")
                elif ai_scores.get('weight_loss_glp1', 0) > 0:
                    score += 15
                    reasons.append("Offers weight loss services")
                
                # Small practice preference (only if complementary services)
                if is_small_practice and ai_scores.get('weight_loss_glp1', 0) > 0:
                    score += 15
                    reasons.append("Small practice with complementary services")
                elif not is_small_practice and ai_scores.get('weight_loss_glp1', 0) == 0:
                    score -= 10  # Penalty for large practices without complementary services
                    reasons.append("Large practice without weight management - challenging fit")
            
            # Growth indicators bonus (all devices)
            if ai_scores.get('growth_indicators', 0) > 5:
                score += 10
                reasons.append("Shows growth/investment mindset")
            
            # Skin laxity triggers bonus (Versa/Legacy)
            if device_name in ['Venus Versa', 'Venus Legacy'] and ai_scores.get('skin_laxity_triggers', 0) > 0:
                score += 12
                reasons.append("Skin laxity patient population")
            
            device_scores[device_name] = {
                'score': score,
                'reasons': reasons
            }
        
        # Sort devices by score and return top 3
        sorted_devices = sorted(device_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        recommendations = []
        for i, (device_name, data) in enumerate(sorted_devices[:3]):
            recommendations.append({
                'device': device_name,
                'fit_score': data['score'],
                'rationale': '; '.join(data['reasons']) if data['reasons'] else 'General aesthetic practice fit',
                'rank': i + 1
            })
        
        return {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'all_recommendations': recommendations
        }

    def craft_outreach_opener(self, practice_data: Dict, ai_score: int, device_rec: Dict) -> str:
        """Generate personalized outreach opener for HIGH-fit leads"""
        
        if ai_score < 70:  # Only for high-fit leads
            return ""
        
        practice_name = practice_data.get('name', 'Your Practice')
        services = practice_data.get('services', [])
        primary_device = device_rec.get('primary_recommendation', {}).get('device', 'Venus devices')
        
        # Personalization elements
        service_mentions = []
        if 'laser hair removal' in services:
            service_mentions.append("laser hair removal services")
        if 'body contouring' in services:
            service_mentions.append("body contouring offerings")
        if 'weight loss' in services:
            service_mentions.append("weight management programs")
        
        service_text = ", ".join(service_mentions[:2]) if service_mentions else "aesthetic services"
        
        # Generate opener based on device recommendation
        openers = {
            'Venus Versa': f"Hi! I noticed {practice_name}'s excellent reputation for {service_text}. Many practices like yours are seeing 40% revenue increases by adding our Venus Versa platform for comprehensive skin treatments. Would you be interested in a brief conversation about how this could complement your current offerings?",
            
            'Venus Legacy': f"Hello! {practice_name}'s focus on {service_text} caught my attention. Our Venus Legacy system is helping similar practices add $15K+ monthly revenue through non-invasive body contouring. Could we schedule a quick call to discuss how this might fit your practice goals?",
            
            'Venus Bliss MAX': f"Hi there! Given {practice_name}'s {service_text}, I thought you'd be interested in our Venus Bliss MAX - the only device combining fat reduction with muscle building. Practices are seeing incredible patient satisfaction and ROI. Would you be open to a brief discussion?"
        }
        
        return openers.get(primary_device, f"Hi! I'd love to discuss how Venus medical devices could enhance {practice_name}'s {service_text}. Many similar practices are seeing significant ROI improvements. Could we schedule a brief conversation?")

    def determine_confidence_level(self, ai_score: int, data_completeness: float) -> str:
        """Determine confidence level based on AI score and data completeness"""
        
        if ai_score >= 80 and data_completeness >= 0.8:
            return "High"
        elif ai_score >= 60 and data_completeness >= 0.6:
            return "Medium"
        else:
            return "Low"

    def process_practice(self, place_data: Dict) -> Dict:
        """Process a single practice through the full pipeline"""
        
        practice_name = place_data.get('name', 'Unknown')
        logger.info(f"Processing practice: {practice_name}")
        
        # Check if this is an existing customer (exclude if so)
        if self.is_existing_customer(practice_name):
            logger.info(f"Skipping existing customer: {practice_name}")
            return None
        
        # Initialize practice record with explicit "Not Available" defaults
        practice_record = {
            'name': place_data.get('name', ''),
            'address': place_data.get('formatted_address', ''),
            'phone': place_data.get('formatted_phone_number', ''),
            'website': place_data.get('website', ''),
            'rating': place_data.get('rating', 0),
            'review_count': place_data.get('user_ratings_total', 0),
            'types': place_data.get('types', []),
            'services': [],
            'social_links': [],
            'staff_count': 0,
            'description': 'Not Available - No Website',
            'title': 'Not Available - No Website',
            'yelp_data': {}
        }
        
        # Scrape website if available
        if practice_record['website']:
            logger.info(f"Scraping website: {practice_record['website']}")
            website_data = self.scrape_website(practice_record['website'])
            practice_record.update(website_data)
        else:
            logger.info(f"No website available for {practice_name} - marking data as 'Not Available'")
        
        # Scrape Yelp data
        if practice_record['name'] and practice_record['address']:
            yelp_data = self.scrape_yelp_data(practice_record['name'], practice_record['address'])
            practice_record['yelp_data'] = yelp_data
        
        # Calculate AI score
        ai_score, score_breakdown = self.calculate_ai_score(practice_record)
        
        # Get device recommendations
        device_recommendations = self.recommend_venus_device(practice_record, score_breakdown)
        
        # Generate outreach opener for high-fit leads
        outreach_opener = self.craft_outreach_opener(practice_record, ai_score, device_recommendations)
        
        # Calculate data completeness
        required_fields = ['name', 'address', 'phone', 'website']
        completeness = sum(1 for field in required_fields if practice_record.get(field)) / len(required_fields)
        
        # Determine confidence level
        confidence = self.determine_confidence_level(ai_score, completeness)
        
        # Compile final record
        final_record = {
            **practice_record,
            'ai_score': ai_score,
            'score_breakdown': score_breakdown,
            'primary_device_rec': device_recommendations.get('primary_recommendation', {}).get('device', ''),
            'device_rationale': device_recommendations.get('primary_recommendation', {}).get('rationale', ''),
            'all_device_recs': device_recommendations.get('all_recommendations', []),
            'outreach_opener': outreach_opener,
            'confidence_level': confidence,
            'data_completeness': completeness,
            'trade_in_opportunity': 'Yes' if score_breakdown.get('competing_devices', 0) > 5 else 'No',
            'complement_existing': 'Yes' if score_breakdown.get('aesthetic_services', 0) > 10 else 'No'
        }
        
        return final_record

    def export_to_csv(self, results: List[Dict], filename: str):
        """Export results to CSV with specified schema"""
        
        if not results:
            logger.warning("No results to export")
            return
        
        # Define CSV schema
        csv_columns = [
            'name', 'address', 'phone', 'website', 'rating', 'review_count',
            'ai_score', 'confidence_level', 'primary_device_rec', 'device_rationale',
            'outreach_opener', 'services', 'social_links', 'staff_count',
            'specialty_match_score', 'aesthetic_services_score', 'competing_devices_score',
            'social_activity_score', 'practice_size_score', 'reviews_rating_score',
            'search_visibility_score', 'geography_fit_score', 'weight_loss_glp1_score',
            'skin_laxity_triggers_score', 'growth_indicators_score', 'high_value_specialties_score',
            'trade_in_opportunity', 'complement_existing', 'data_completeness'
        ]
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            row = {}
            for col in csv_columns:
                if col.endswith('_score') and col != 'ai_score':
                    score_key = col.replace('_score', '')
                    row[col] = result.get('score_breakdown', {}).get(score_key, 0)
                elif col == 'services':
                    row[col] = '; '.join(result.get('services', []))
                elif col == 'social_links':
                    row[col] = '; '.join(result.get('social_links', []))
                else:
                    row[col] = result.get(col, '')
            csv_data.append(row)
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"Results exported to {filename}")

    def generate_summary_report(self, results: List[Dict], filename: str):
        """Generate summary report with top prospects and metrics"""
        
        if not results:
            return
        
        # Sort by AI score
        sorted_results = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)
        
        # Calculate metrics
        total_prospects = len(results)
        high_fit_prospects = len([r for r in results if r.get('ai_score', 0) >= 70])
        medium_fit_prospects = len([r for r in results if 50 <= r.get('ai_score', 0) < 70])
        low_fit_prospects = len([r for r in results if r.get('ai_score', 0) < 50])
        
        avg_score = sum(r.get('ai_score', 0) for r in results) / total_prospects if total_prospects > 0 else 0
        
        # Device recommendation distribution
        device_counts = {}
        for result in results:
            device = result.get('primary_device_rec', 'None')
            device_counts[device] = device_counts.get(device, 0) + 1
        
        # Generate report
        report = f"""
VENUS MEDICAL DEVICE PROSPECTING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Total Prospects Analyzed: {total_prospects}
Average AI Score: {avg_score:.1f}/100

PROSPECT DISTRIBUTION
====================
High-Fit Prospects (70-100): {high_fit_prospects} ({high_fit_prospects/total_prospects*100:.1f}%)
Medium-Fit Prospects (50-69): {medium_fit_prospects} ({medium_fit_prospects/total_prospects*100:.1f}%)
Low-Fit Prospects (0-49): {low_fit_prospects} ({low_fit_prospects/total_prospects*100:.1f}%)

DEVICE RECOMMENDATIONS
=====================
"""
        
        for device, count in sorted(device_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_prospects * 100 if total_prospects > 0 else 0
            report += f"{device}: {count} prospects ({percentage:.1f}%)\n"
        
        report += f"""

TOP 10 HIGH-PRIORITY PROSPECTS
==============================
"""
        
        for i, prospect in enumerate(sorted_results[:10], 1):
            report += f"""
{i}. {prospect.get('name', 'Unknown')}
   Score: {prospect.get('ai_score', 0)}/100 | Confidence: {prospect.get('confidence_level', 'Unknown')}
   Device Rec: {prospect.get('primary_device_rec', 'None')}
   Address: {prospect.get('address', 'Unknown')}
   Phone: {prospect.get('phone', 'Unknown')}
   Website: {prospect.get('website', 'Unknown')}
   Rationale: {prospect.get('device_rationale', 'N/A')}
"""
        
        # Write report to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Summary report generated: {filename}")

    def run_prospecting(self, keywords: List[str], location: str, radius: int, max_results: int):
        """Main prospecting workflow"""
        
        logger.info(f"Starting Venus prospecting for {keywords} near {location}")
        
        all_results = []
        
        # Search for each keyword
        for keyword in keywords:
            logger.info(f"Searching for: {keyword}")
            
            # Get places from Google Places API
            places = self.google_places_search(keyword, location, radius * 1000)  # Convert km to meters
            
            # Process each place
            for place in places[:max_results]:
                try:
                    processed_practice = self.process_practice(place)
                    all_results.append(processed_practice)
                    
                    logger.info(f"Processed: {processed_practice.get('name', 'Unknown')} - Score: {processed_practice.get('ai_score', 0)}")
                    
                except Exception as e:
                    logger.error(f"Error processing practice: {str(e)}")
                    continue
        
        # Remove duplicates based on name and address
        unique_results = []
        seen = set()
        for result in all_results:
            key = (result.get('name', ''), result.get('address', ''))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        logger.info(f"Found {len(unique_results)} unique prospects")
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export results
        csv_filename = f"venus_prospects_{timestamp}.csv"
        report_filename = f"venus_summary_{timestamp}.txt"
        
        self.export_to_csv(unique_results, csv_filename)
        self.generate_summary_report(unique_results, report_filename)
        
        return unique_results, csv_filename, report_filename


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(description='Venus Medical Device Prospecting System')
    parser.add_argument('--keywords', nargs='+', help='Search keywords (e.g., "med spa" "dermatologist")')
    parser.add_argument('--city', help='City or location to search (e.g., "Austin, TX")')
    parser.add_argument('--radius', type=int, default=25, help='Search radius in kilometers (default: 25)')
    parser.add_argument('--max-results', type=int, default=50, help='Maximum results per keyword (default: 50)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with mock data')
    parser.add_argument('--exclude-csv', help='CSV file containing existing customers to exclude')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize prospector with exclusion CSV if provided
    prospector = VenusProspector(demo_mode=args.demo, existing_customers_csv=args.exclude_csv)
    
    # Interactive mode or command line arguments
    if args.interactive or not all([args.keywords, args.city]):
        print("\n=== VENUS MEDICAL DEVICE PROSPECTING SYSTEM ===\n")
        
        # Get search parameters interactively
        if not args.keywords:
            print("Available search templates:")
            for key, templates in prospector.search_templates.items():
                print(f"  {key}: {', '.join(templates)}")
            
            keyword_input = input("\nEnter search keywords (comma-separated) or template name: ").strip()
            
            if keyword_input in prospector.search_templates:
                keywords = prospector.search_templates[keyword_input]
            else:
                keywords = [k.strip() for k in keyword_input.split(',')]
        else:
            keywords = args.keywords
        
        if not args.city:
            city = input("Enter city/location (e.g., 'Austin, TX'): ").strip()
        else:
            city = args.city
        
        radius = args.radius or int(input(f"Enter search radius in km (default: 25): ") or "25")
        max_results = args.max_results or int(input(f"Enter max results per keyword (default: 50): ") or "50")
        
    else:
        keywords = args.keywords
        city = args.city
        radius = args.radius
        max_results = args.max_results
    
    print(f"\nStarting prospecting with:")
    print(f"Keywords: {keywords}")
    print(f"Location: {city}")
    print(f"Radius: {radius} km")
    print(f"Max results per keyword: {max_results}")
    print(f"Google Places API Key: {'✓ Configured' if prospector.gmaps_key else '✗ Missing'}")
    
    try:
        # Run prospecting
        results, csv_file, report_file = prospector.run_prospecting(keywords, city, radius, max_results)
        
        print(f"\n=== PROSPECTING COMPLETE ===")
        print(f"Total prospects found: {len(results)}")
        print(f"Results exported to: {csv_file}")
        print(f"Summary report: {report_file}")
        
        # Show top 5 prospects
        if results:
            top_prospects = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)[:5]
            print(f"\nTOP 5 PROSPECTS:")
            for i, prospect in enumerate(top_prospects, 1):
                print(f"{i}. {prospect.get('name', 'Unknown')} - Score: {prospect.get('ai_score', 0)}/100")
        
    except Exception as e:
        logger.error(f"Error during prospecting: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
