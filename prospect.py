#!/usr/bin/env python3
"""
Fathom Medical Device Prospecting System
Comprehensive tool for finding and scoring medical practices
Production-Hardened Version 3.0
"""

import argparse
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

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

# Environment placeholder markers that should be treated as unset values
PLACEHOLDER_VALUES: Dict[str, set[str]] = {
    "GOOGLE_MAPS_API_KEY": {"replace-with-your-google-maps-api-key"},
    "GOOGLE_PLACES_API_KEY": {"replace-with-your-google-places-api-key"},
    "GEMINI_API_KEY": {"replace-with-your-gemini-api-key"},
    "FATHOM_API_KEY": {
        "generate-a-unique-shared-secret",
        "your-secret-api-key-change-this",
    },
}

# Load environment variables FIRST
load_dotenv()

# Configure logging BEFORE any other imports that might use it
handlers = [logging.StreamHandler(sys.stdout)]

# Try to add file logging if possible (development/local only)
try:
    log_dir = os.getenv('LOG_DIR', '/tmp/logs')
    os.makedirs(log_dir, exist_ok=True)
    handlers.append(logging.FileHandler(f'{log_dir}/prospector.log'))
except Exception:
    pass  # Continue with stdout only

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)


def normalize_secret(name: str, raw_value: Optional[str]) -> Optional[str]:
    """Return ``None`` for unset or placeholder secrets."""

    if raw_value is None:
        return None

    value = raw_value.strip()
    if not value:
        return None

    placeholders = PLACEHOLDER_VALUES.get(name)
    if placeholders and value in placeholders:
        logger.warning(
            "%s is using a placeholder value; treating it as unset",
            name,
        )
        return None

    return value

# Abacus RouteLLM Integration (OpenAI-compatible)
AI_AVAILABLE = False
openai_client = None

try:
    from openai import OpenAI
    abacus_key = normalize_secret('ABACUSAI_API_KEY', os.getenv('ABACUSAI_API_KEY'))

    if abacus_key:
        # Initialize OpenAI client with Abacus RouteLLM endpoint
        openai_client = OpenAI(
            api_key=abacus_key,
            base_url="https://apps.abacus.ai/v1"  # Abacus RouteLLM endpoint
        )
        AI_AVAILABLE = True
        logger.info("✓ Abacus RouteLLM initialized successfully - AI features enabled")
        logger.info("   Smart routing active: Access to GPT-4, Claude, Llama, and more")
    else:
        logger.warning("ABACUSAI_API_KEY not found - AI features disabled, using rule-based logic")
except ImportError:
    logger.warning("openai package not installed - AI features disabled, using rule-based logic")
    logger.warning("Run: pip install openai")
except Exception as e:
    logger.warning(f"Abacus RouteLLM initialization failed: {str(e)} - AI features disabled, using rule-based logic")



class GooglePlacesAPI:
    """Direct Google Places API wrapper with guaranteed timeouts"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
        # Connection timeout, Read timeout
        self.timeout = (10, 30)
        
    def _make_request(self, endpoint: str, params: dict, max_retries: int = 3) -> dict:
        """
        Make API request with timeout and retry logic
        
        Args:
            endpoint: API endpoint (e.g., '/place/nearbysearch/json')
            params: Query parameters
            max_retries: Number of retry attempts
            
        Returns:
            API response as dict
        """
        url = f"{self.base_url}{endpoint}"
        params['key'] = self.api_key
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"API request attempt {attempt + 1}/{max_retries}: {endpoint}")
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                # Check API status
                status = data.get('status')
                if status == 'OK' or status == 'ZERO_RESULTS':
                    return data
                elif status == 'OVER_QUERY_LIMIT':
                    logger.error("Google API quota exceeded")
                    raise Exception("API quota exceeded")
                elif status == 'REQUEST_DENIED':
                    logger.error("Google API request denied - check API key")
                    raise Exception("Invalid API key or permissions")
                elif status == 'INVALID_REQUEST':
                    logger.error(f"Invalid API request: {data.get('error_message', 'Unknown error')}")
                    raise Exception(f"Invalid request: {data.get('error_message')}")
                else:
                    logger.warning(f"API returned status: {status}")
                    return data
                    
            except requests.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise Exception("API request timed out after multiple attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"API request failed: {str(e)}")
                time.sleep(2 ** attempt)
                
        raise Exception("Max retries exceeded")
    
    def geocode(self, address: str) -> List[dict]:
        """
        Geocode an address to lat/lng
        
        Args:
            address: Address string to geocode
            
        Returns:
            List of geocoding results
        """
        endpoint = "/geocode/json"
        params = {'address': address}
        
        try:
            data = self._make_request(endpoint, params)
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Geocoding error for '{address}': {e}")
            return []
    
    def places_nearby(self, location: dict, radius: int, keyword: str) -> dict:
        """
        Search for places nearby with pagination support
        
        Args:
            location: Dict with 'lat' and 'lng' keys
            radius: Search radius in meters
            keyword: Search keyword
            
        Returns:
            Dict with 'results' key containing list of places found (handles pagination automatically)
        """
        endpoint = "/place/nearbysearch/json"
        all_results = []
        next_page_token = None
        
        while True:
            params = {
                'location': f"{location['lat']},{location['lng']}",
                'radius': radius,
                'keyword': keyword
            }
            
            if next_page_token:
                params['pagetoken'] = next_page_token
                # Google requires 2-second delay between paginated requests
                logger.debug("Waiting 2 seconds before next page request...")
                time.sleep(2)
            
            try:
                data = self._make_request(endpoint, params)
                results = data.get('results', [])
                all_results.extend(results)
                
                logger.info(f"Fetched {len(results)} results (total so far: {len(all_results)})")
                
                # Check for more pages
                next_page_token = data.get('next_page_token')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching places: {e}")
                break
        
        # Return in format compatible with calling code
        return {'results': all_results}
    
    def place_details(self, place_id: str, fields: List[str]) -> Optional[dict]:
        """
        Get detailed information for a specific place
        
        Args:
            place_id: Google Place ID
            fields: List of fields to retrieve
            
        Returns:
            Place details dict or None
        """
        endpoint = "/place/details/json"
        params = {
            'place_id': place_id,
            'fields': ','.join(fields)
        }
        
        try:
            data = self._make_request(endpoint, params)
            return data.get('result', {})
        except Exception as e:
            logger.error(f"Error fetching place details for {place_id}: {e}")
            return None



class FathomProspector:
    """Main prospecting system for medical devices"""
    
    def __init__(self, demo_mode=False, existing_customers_csv=None, progress_callback=None):
        self.gmaps_key = normalize_secret('GOOGLE_PLACES_API_KEY', os.getenv('GOOGLE_PLACES_API_KEY'))
        if not self.gmaps_key:
            logger.warning('GOOGLE_PLACES_API_KEY not found - switching to demo mode')
            demo_mode = True
        
        # Progress callback for real-time updates
        self.progress_callback = progress_callback
        
        # Initialize Abacus RouteLLM if available
        self.ai_enabled = AI_AVAILABLE
        self.openai_client = openai_client
        self.model_name = "gpt-4o-mini"  # Default model: fast and cheap ($0.15/1M input tokens)
        
        if AI_AVAILABLE:
            logger.info(f'✓ Abacus RouteLLM ready - AI features enabled')
            logger.info(f'   Using model: {self.model_name} for cost efficiency')
        else:
            logger.info('Using rule-based scoring and outreach (Abacus RouteLLM not available)')
        
        self.demo_mode = demo_mode
        self.existing_customers = set()
        
        # Load existing customers for exclusion
        if existing_customers_csv and os.path.exists(existing_customers_csv):
            self.load_existing_customers(existing_customers_csv)
        
        # Initialize session for web scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize Google Places API
        if not self.demo_mode:
            try:
                self.gmaps_api = GooglePlacesAPI(self.gmaps_key)
                test_result = self.gmaps_api.geocode("Austin, TX")
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
        
        # Device catalog - Venus product line
        self.device_catalog = {
            'Venus Versa': {
                'specialties': ['hair removal', 'photorejuvenation', 'skin resurfacing', 'acne treatment', 'vascular lesions'],
                'keywords': ['laser hair removal', 'ipl', 'photo facial', 'photofacial', 'skin rejuvenation', 'pigmentation']
            },
            'Venus Legacy': {
                'specialties': ['body contouring', 'cellulite reduction', 'skin tightening', 'wrinkle reduction'],
                'keywords': ['body sculpting', 'cellulite', 'radiofrequency', 'rf', 'skin tightening', 'body shaping']
            },
            'Venus Bliss MAX': {
                'specialties': ['weight loss', 'muscle stimulation', 'body contouring', 'fat reduction'],
                'keywords': ['weight loss', 'ems', 'muscle building', 'fat reduction', 'body sculpting', 'lipolysis']
            },
            'Venus Viva': {
                'specialties': ['skin resurfacing', 'scar treatment', 'texture improvement', 'wrinkle reduction'],
                'keywords': ['skin resurfacing', 'nano fractional', 'scar reduction', 'texture', 'fine lines']
            }
        }
        
        # Specialty-specific scoring configurations
        self.specialty_scoring = {
            'dermatology': {
                'weights': {
                    'specialty_match': 20,
                    'decision_autonomy': 20,
                    'aesthetic_services': 15,
                    'competing_devices': 10,
                    'social_activity': 10,
                    'reviews_rating': 10,
                    'search_visibility': 10,
                    'financial_indicators': 10,
                    'weight_loss_services': 5
                },
                'high_value_keywords': [
                    'laser', 'ipl', 'photorejuvenation', 'hair removal',
                    'botox', 'fillers', 'skin tightening', 'acne treatment'
                ]
            },
            'plastic_surgery': {
                'weights': {
                    'specialty_match': 20,
                    'decision_autonomy': 20,
                    'aesthetic_services': 18,
                    'competing_devices': 12,
                    'social_activity': 10,
                    'reviews_rating': 8,
                    'search_visibility': 7,
                    'financial_indicators': 10,
                    'weight_loss_services': 5
                },
                'high_value_keywords': [
                    'body contouring', 'liposuction', 'coolsculpting',
                    'breast augmentation', 'tummy tuck', 'fat reduction'
                ]
            },
            'obgyn': {
                'weights': {
                    'specialty_match': 20,
                    'decision_autonomy': 25,
                    'aesthetic_services': 12,
                    'competing_devices': 8,
                    'social_activity': 8,
                    'reviews_rating': 12,
                    'search_visibility': 10,
                    'financial_indicators': 10,
                    'weight_loss_services': 8
                },
                'high_value_keywords': [
                    'women\'s wellness', 'aesthetic gynecology', 'vaginal rejuvenation',
                    'hormone therapy', 'postpartum', 'wellness', 'cosmetic gynecology'
                ]
            },
            'medspa': {
                'weights': {
                    'specialty_match': 18,
                    'decision_autonomy': 22,
                    'aesthetic_services': 20,
                    'competing_devices': 15,
                    'social_activity': 12,
                    'reviews_rating': 8,
                    'search_visibility': 8,
                    'financial_indicators': 12,
                    'weight_loss_services': 10
                },
                'high_value_keywords': [
                    'body sculpting', 'coolsculpting', 'emsculpt', 'laser',
                    'botox', 'fillers', 'skin tightening', 'cellulite',
                    'inch loss', 'fat reduction'
                ]
            },
            'familypractice': {
                'weights': {
                    'specialty_match': 18,
                    'decision_autonomy': 25,
                    'aesthetic_services': 12,
                    'competing_devices': 8,
                    'social_activity': 5,
                    'reviews_rating': 8,
                    'search_visibility': 5,
                    'financial_indicators': 12,
                    'weight_loss_services': 20  # HIGHEST - key entry point
                },
                'high_value_keywords': [
                    'weight loss', 'glp-1', 'semaglutide', 'tirzepatide',
                    'wellness', 'longevity', 'preventive care', 'functional medicine',
                    'integrative medicine', 'anti-aging', 'hormone therapy'
                ]
            }
        }
    
    def call_ai(self, prompt: str, system_message: str = None, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Unified method to call Abacus RouteLLM API
        
        Args:
            prompt: User prompt
            system_message: Optional system message for context
            max_tokens: Max response length
            temperature: Creativity level (0.0-1.0)
            
        Returns:
            Response text or empty string if failed
        """
        if not self.ai_enabled or not self.openai_client:
            logger.debug("AI not available, skipping AI call")
            return ""
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30.0  # Prevent hanging
            )
            
            result = response.choices[0].message.content.strip()
            logger.debug(f"🤖 AI response received ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"AI API error: {str(e)}")
            return ""
    
    def load_existing_customers(self, csv_file_path: str):
        """Load existing customers from CSV for exclusion"""
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loading existing customers from {csv_file_path}")
            
            name_columns = ['name', 'practice_name', 'business_name', 'company_name']
            name_column = None
            
            for col in name_columns:
                if col.lower() in [c.lower() for c in df.columns]:
                    name_column = col
                    break
            
            if name_column is None:
                name_column = df.columns[0]
                logger.warning(f"No standard name column found, using '{name_column}'")
            
            for name in df[name_column].dropna():
                cleaned_name = str(name).lower().strip()
                cleaned_name = re.sub(r'\b(llc|inc|corp|ltd|pllc|pa|md|pc)\b', '', cleaned_name).strip()
                self.existing_customers.add(cleaned_name)
            
            logger.info(f"Loaded {len(self.existing_customers)} existing customers for exclusion")
            
        except Exception as e:
            logger.error(f"Failed to load existing customers CSV: {str(e)}")
    
    def is_existing_customer(self, practice_name: str) -> bool:
        """Check if practice is an existing customer"""
        if not self.existing_customers:
            return False
        
        cleaned_name = practice_name.lower().strip()
        cleaned_name = re.sub(r'\b(llc|inc|corp|ltd|pllc|pa|md|pc)\b', '', cleaned_name).strip()
        
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
            }
        ]
        
        return mock_practices
    
    @sleep_and_retry
    @limits(calls=10, period=60)
    def google_places_search(self, query: str, location: str, radius: int = 25000) -> List[Dict]:
        """Search Google Places API for medical practices"""
        
        if self.demo_mode:
            logger.info(f"DEMO MODE: Generating mock data for {query} near {location}")
            return self.get_mock_data(query, location)
        
        try:
            logger.info(f"Searching Google Places: {query} near {location}")
            
            geocode_result = self.gmaps_api.geocode(location)
            if not geocode_result:
                logger.error(f"Could not geocode location: {location}")
                return []
            
            lat_lng = geocode_result[0]['geometry']['location']
            
            places_result = self.gmaps_api.places_nearby(
                location=lat_lng,
                radius=radius,
                keyword=query
            )
            
            results = []
            for place in places_result.get('results', []):
                # Filter out hospital systems
                if self.is_hospital_system(place.get('name', '')):
                    logger.info(f"Skipping hospital system: {place.get('name', '')}")
                    continue
                
                # Get full place details
                place_details = self.get_place_details(place['place_id'])
                
                if not place_details or not place_details.get('name'):
                    logger.debug(f"Skipping place with no details: {place.get('place_id', 'unknown')}")
                    continue
                
                # CRITICAL: Apply strict medical business filtering
                if not self.is_medical_business(place_details):
                    continue
                
                results.append(place_details)
                time.sleep(0.1)
            
            logger.info(f"✅ {len(results)} medical businesses passed all filters")
            return results
            
        except Exception as e:
            logger.error(f"Error in Google Places search: {str(e)}")
            logger.info("Falling back to demo mode")
            self.demo_mode = True
            return self.get_mock_data(query, location)
    
    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information for a specific place"""
        try:
            details = self.gmaps_api.place_details(
                place_id=place_id,
                fields=['name', 'formatted_address', 'formatted_phone_number', 
                       'website', 'rating', 'user_ratings_total', 'types']
            )
            
            # place_details() already returns the result dict, not the full response
            return details if details else None
            
        except Exception as e:
            logger.error(f"Error getting place details: {str(e)}")
            return None
    
    def is_medical_business(self, place_data: Dict) -> bool:
        """
        Strict filtering: Check if a place is a legitimate medical/aesthetic business
        using Google Places types
        """
        place_types = place_data.get('types', [])
        name = place_data.get('name', '').lower()
        
        # REQUIRED: Must have at least ONE of these medical-related types
        medical_types = {
            'doctor', 'health', 'spa', 'beauty_salon', 'hair_care',
            'physiotherapist', 'dentist', 'hospital'
        }
        
        has_medical_type = any(ptype in place_types for ptype in medical_types)
        
        if not has_medical_type:
            logger.info(f"❌ FILTERED OUT (no medical type): {name} - Types: {place_types}")
            return False
        
        # EXCLUDE: General stores, restaurants, etc.
        exclude_types = {
            'store', 'food', 'restaurant', 'cafe', 'bar', 'grocery_or_supermarket',
            'shopping_mall', 'clothing_store', 'jewelry_store', 'shoe_store',
            'electronics_store', 'furniture_store', 'home_goods_store',
            'hardware_store', 'car_dealer', 'car_repair', 'gas_station',
            'gym', 'night_club', 'movie_theater', 'bowling_alley',
            'amusement_park', 'aquarium', 'art_gallery', 'museum',
            'library', 'school', 'university', 'real_estate_agency',
            'travel_agency', 'insurance_agency', 'accounting', 'lawyer',
            'general_contractor', 'electrician', 'plumber', 'roofing_contractor',
            'locksmith', 'moving_company', 'storage', 'laundry', 'car_wash'
        }
        
        has_exclude_type = any(ptype in place_types for ptype in exclude_types)
        
        if has_exclude_type:
            logger.info(f"❌ FILTERED OUT (non-medical business): {name} - Types: {place_types}")
            return False
        
        # EXCLUDE: Keywords indicating non-medical businesses
        exclude_keywords = [
            'pharmacy', 'drugstore', 'cvs', 'walgreens', 'walmart', 'target',
            'urgent care', 'emergency room', 'laboratory', 'imaging center',
            'physical therapy', 'chiropractor', 'massage', 'acupuncture',
            'veterinary', 'pet', 'animal', 'dentist', 'orthodont'
        ]
        
        if any(keyword in name for keyword in exclude_keywords):
            logger.info(f"❌ FILTERED OUT (excluded keyword): {name}")
            return False
        
        logger.info(f"✅ PASSED FILTER: {name} - Types: {place_types}")
        return True
    
    def is_hospital_system(self, name: str) -> bool:
        """Check if a practice name indicates a hospital system"""
        name_lower = name.lower()
        return any(re.search(pattern, name_lower, re.IGNORECASE) for pattern in self.hospital_patterns)
    
    def check_robots_txt(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt"""
        try:
            if not url:
                return False
                
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
                
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch('*', url)
        except Exception:
            return True
    
    def get_mock_website_data(self, url: str) -> Dict[str, any]:
        """Generate mock website data for demo purposes"""
        mock_data = {
            'https://austinaesthetic.com': {
                'title': 'Austin Aesthetic Center - Premier Med Spa',
                'description': 'Leading medical spa offering laser hair removal, botox, fillers, and advanced skin treatments.',
                'services': ['laser hair removal', 'botox', 'fillers', 'skin tightening'],
                'social_links': ['Facebook', 'Instagram'],
                'staff_count': 8
            },
            'https://hillcountrymedspa.com': {
                'title': 'Hill Country Med Spa - Body Contouring & Wellness',
                'description': 'Full-service medical spa specializing in body contouring and wellness services.',
                'services': ['body contouring', 'cellulite treatment', 'weight loss'],
                'social_links': ['Instagram'],
                'staff_count': 5
            },
            'https://lonestardermatology.com': {
                'title': 'Lone Star Dermatology - Advanced Skin Care',
                'description': 'Board-certified dermatologists providing medical and cosmetic dermatology services.',
                'services': ['laser hair removal', 'skin tightening', 'botox'],
                'social_links': ['Facebook', 'LinkedIn'],
                'staff_count': 12
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
    @limits(calls=30, period=60)
    def scrape_website(self, url: str) -> Dict[str, any]:
        """Scrape practice website for additional information"""
        
        if self.demo_mode:
            logger.info(f"DEMO MODE: Using mock website data for {url}")
            return self.get_mock_website_data(url)
        
        if not url:
            return {
                'title': 'Not Available - No Website',
                'description': 'Not Available - No Website',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
        
        if not self.check_robots_txt(url):
            logger.warning(f"Robots.txt disallows scraping: {url}")
            return {
                'title': 'Not Available - Restricted',
                'description': 'Not Available - Restricted',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
            
        try:
            time.sleep(random.uniform(0.5, 1.0))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
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
                'cellulite treatment', 'weight loss', 'ems', 'muscle building'
            ]
            
            services_found = []
            for keyword in service_keywords:
                if keyword in text_content:
                    services_found.append(keyword)
            
            data['services'] = services_found
            
            # Find social media links - capture full URLs
            social_platforms = {
                'facebook.com': 'Facebook',
                'fb.com': 'Facebook',
                'instagram.com': 'Instagram',
                'twitter.com': 'Twitter',
                'x.com': 'Twitter',
                'linkedin.com': 'LinkedIn',
                'youtube.com': 'YouTube',
                'youtu.be': 'YouTube',
                'tiktok.com': 'TikTok',
                'pinterest.com': 'Pinterest'
            }
            
            social_links = []
            seen_platforms = set()
            
            # Look for social media links in all anchor tags
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                href_lower = href.lower()
                
                # Skip empty or invalid hrefs
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # Check each social platform
                for domain, platform_name in social_platforms.items():
                    if domain in href_lower and platform_name not in seen_platforms:
                        # Clean up the URL
                        full_url = href
                        if href.startswith('//'):
                            full_url = 'https:' + href
                        elif href.startswith('/') or not href.startswith('http'):
                            # Relative URL - skip it as it's not a social link
                            continue
                        
                        # Add the full URL to the list
                        social_links.append(full_url)
                        seen_platforms.add(platform_name)
                        logger.info(f"Found {platform_name} link: {full_url}")
                        break
            
            data['social_links'] = social_links
            
            # Estimate staff count
            staff_indicators = soup.find_all(text=re.compile(
                r'\b(dr\.|doctor|physician|provider|practitioner)\b', re.I))
            unique_staff = len(set(str(s).strip() for s in staff_indicators if len(str(s).strip()) > 5))
            data['staff_count'] = min(unique_staff, 20)
            
            logger.info(f"Website scrape complete for {url}: {len(data['services'])} services found")
            
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
        except Exception as e:
            logger.error(f"Error scraping website {url}: {str(e)}")
            return {
                'title': 'Not Available - Error',
                'description': 'Not Available - Error',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
    
    def detect_specialty(self, practice_data: Dict) -> str:
        """
        Detect the primary specialty of a practice (keyword-based)
        Returns: 'dermatology', 'plastic_surgery', 'obgyn', 'medspa', 'familypractice', or 'general'
        """
        # Keyword-based detection
        name = practice_data.get('name', '').lower()
        desc = practice_data.get('description', '').lower()
        services = ' '.join(practice_data.get('services', [])).lower()
        all_text = f"{name} {desc} {services}"

    def detect_specialty_ai(self, practice_data: Dict) -> str:
        """
        AI-powered specialty detection using Abacus RouteLLM
        Analyzes practice data with context awareness
        Falls back to rule-based if AI unavailable
        """
        if not self.ai_enabled:
            return self.detect_specialty(practice_data)
        
        try:
            name = practice_data.get('name', 'Unknown Practice')
            desc = practice_data.get('description', '')
            services = ', '.join(practice_data.get('services', []))
            website_text = practice_data.get('website_text', '')[:500]  # First 500 chars
            
            prompt = f"""Analyze this medical practice and determine their PRIMARY specialty.

Practice Name: {name}
Description: {desc}
Services Offered: {services}
Website Content: {website_text}

Based on ALL the context above, determine the single best specialty classification:
- dermatology: Skin doctors, medical dermatology, cosmetic dermatology
- plastic_surgery: Plastic surgeons, cosmetic surgeons, reconstructive surgery
- obgyn: OB/GYN, gynecology, women's health, obstetrics
- medspa: Medical spas, aesthetic centers, cosmetic clinics (non-physician owned)
- familypractice: Family medicine, primary care, general practice, internal medicine
- general: Everything else

Return ONLY ONE WORD from the list above. No explanation."""

            system_msg = "You are an expert medical practice analyst specializing in practice classification."
            
            result = self.call_ai(prompt, system_msg, max_tokens=20, temperature=0.3).lower().strip()
            
            # Validate result
            valid_specialties = ['dermatology', 'plastic_surgery', 'obgyn', 'medspa', 'familypractice', 'general']
            if result in valid_specialties:
                logger.info(f"🤖 AI-enhanced specialty detection: {result}")
                return result
            else:
                logger.warning(f"AI returned invalid specialty '{result}', using rule-based fallback")
                return self.detect_specialty(practice_data)
                
        except Exception as e:
            logger.error(f"AI specialty detection failed: {str(e)}, using rule-based fallback")
            return self.detect_specialty(practice_data)
    
    def analyze_pain_points_ai(self, practice_data: Dict, specialty: str) -> Dict:
        """
        AI-powered pain point analysis using Abacus RouteLLM
        Generates personalized insights based on practice context
        Falls back to rule-based if AI unavailable
        """
        if not self.ai_enabled:
            return self.analyze_pain_points_rule_based(practice_data, specialty)
        
        try:
            name = practice_data.get('name', 'Unknown Practice')
            services = ', '.join(practice_data.get('services', []))
            website_text = practice_data.get('website_text', '')[:800]
            has_website = bool(practice_data.get('website'))
            devices_found = practice_data.get('devices_found', [])
            
            prompt = f"""Analyze this medical practice and identify their top 3-4 business pain points related to aesthetic services and growth.

Practice: {name}
Specialty: {specialty}
Services: {services}
Has Website: {has_website}
Competing Devices: {devices_found}
Website Content: {website_text}

Generate:
1. 3-4 specific pain points this practice likely faces
2. A readiness score (0-100) for adopting Venus aesthetic technology

Format your response as:
PAIN_POINTS:
- [pain point 1]
- [pain point 2]
- [pain point 3]
- [pain point 4]

READINESS_SCORE: [number 0-100]

Consider factors like competition, market position, current services, and growth potential."""

            system_msg = "You are a medical device sales strategist specializing in identifying practice needs."
            
            response = self.call_ai(prompt, system_msg, max_tokens=400, temperature=0.7)
            
            if not response:
                return self.analyze_pain_points_rule_based(practice_data, specialty)
            
            # Parse the AI response
            pain_points = []
            readiness_score = 50
            
            # Extract pain points
            if 'PAIN_POINTS:' in response:
                pain_section = response.split('PAIN_POINTS:')[1].split('READINESS_SCORE:')[0]
                pain_points = [p.strip('- ').strip() for p in pain_section.split('\n') if p.strip().startswith('-')]
            
            # Extract readiness score
            if 'READINESS_SCORE:' in response:
                try:
                    score_text = response.split('READINESS_SCORE:')[1].strip().split()[0]
                    readiness_score = int(''.join(filter(str.isdigit, score_text)))
                    readiness_score = max(0, min(100, readiness_score))  # Clamp to 0-100
                except:
                    readiness_score = 50
            
            if not pain_points:
                logger.warning("AI returned no pain points, using rule-based fallback")
                return self.analyze_pain_points_rule_based(practice_data, specialty)
            
            logger.info(f"🤖 AI-generated pain points: {len(pain_points)} insights, readiness: {readiness_score}")
            
            return {
                'pain_points': pain_points,
                'readiness_score': readiness_score,
                'ai_generated': True
            }
            
        except Exception as e:
            logger.error(f"AI pain point analysis failed: {str(e)}, using rule-based fallback")
            return self.analyze_pain_points_rule_based(practice_data, specialty)
    
    def generate_outreach_ai(self, practice_data: Dict, specialty: str, pain_analysis: Dict) -> Dict:
        """
        AI-powered outreach generation using Abacus RouteLLM
        Creates personalized cold call script, Instagram DM, and email
        Falls back to template-based if AI unavailable
        """
        if not self.ai_enabled:
            return self.generate_outreach_template_based(practice_data, specialty, pain_analysis)
        
        try:
            name = practice_data.get('name', 'Unknown Practice')
            contact = practice_data.get('contact_name', 'Doctor')
            location = practice_data.get('location', '')
            pain_points = pain_analysis.get('pain_points', [])
            readiness_score = pain_analysis.get('readiness_score', 50)
            services = ', '.join(practice_data.get('services', []))[:200]
            rating = practice_data.get('rating', 0)
            device_rec = self.recommend_device(practice_data, {}).get('primary_recommendation', {}).get('device', 'Venus Bliss MAX')
            
            pain_points_text = '\n'.join([f"- {p}" for p in pain_points[:3]])
            
            # Single prompt to generate all 3 outreach types
            prompt = f"""Generate personalized sales outreach for a medical device sales team reaching out to this practice. Create 3 different versions optimized for each channel.

Practice: {name}
Contact: {contact}
Location: {location}
Specialty: {specialty}
Services: {services}
Rating: {rating}/5
Readiness Score: {readiness_score}/100

Key Pain Points:
{pain_points_text}

Product: Venus aesthetic technologies (body contouring, skin tightening, cellulite reduction, wrinkle reduction)
Recommended Device: {device_rec}

Generate exactly 3 outreach versions:

1. COLD CALL SCRIPT (2-3 minute phone conversation opener)
- Start with a warm introduction
- Reference their reputation or specific details
- Mention 1-2 pain points naturally
- Position {device_rec} as a solution
- End with soft ask for 15-min meeting
- Keep it conversational, not scripted

2. INSTAGRAM DM (casual, social media appropriate, 2-3 sentences max)
- Brief and friendly
- Reference their Instagram/social presence
- Mention Venus and results other practices see
- Ask if they'd be open to learn more
- Keep it light and non-salesy

3. EMAIL (professional, 120-150 words)
- Compelling subject line
- Professional but warm tone
- Reference specific practice details
- Mention how Venus addresses their challenges
- Soft call-to-action
- Real person tone, not corporate

Format your response EXACTLY like this:
COLD_CALL:
[cold call script here]

INSTAGRAM:
[Instagram DM here]

EMAIL_SUBJECT:
[subject line]

EMAIL_BODY:
[email body]"""

            system_msg = "You are a top-performing medical device sales professional known for personalized, effective multi-channel outreach."
            
            response = self.call_ai(prompt, system_msg, max_tokens=1200, temperature=0.8)
            
            if not response:
                return self.generate_outreach_template_based(practice_data, specialty, pain_analysis)
            
            # Parse all 3 outreach types
            cold_call = ""
            instagram = ""
            email_subject = ""
            email_body = ""
            
            # Extract cold call
            if 'COLD_CALL:' in response:
                parts = response.split('INSTAGRAM:', 1)
                cold_call = parts[0].replace('COLD_CALL:', '').strip()
            
            # Extract Instagram DM
            if 'INSTAGRAM:' in response:
                parts = response.split('INSTAGRAM:', 1)[1].split('EMAIL_SUBJECT:', 1)
                instagram = parts[0].strip()
            
            # Extract email subject and body
            if 'EMAIL_SUBJECT:' in response:
                parts = response.split('EMAIL_SUBJECT:', 1)[1].split('EMAIL_BODY:', 1)
                email_subject = parts[0].strip()
                if len(parts) > 1:
                    email_body = parts[1].strip()
            
            # Validation - ensure we got at least email
            if not email_body:
                return self.generate_outreach_template_based(practice_data, specialty, pain_analysis)
            
            logger.info(f"🤖 AI outreach generated successfully for {name}")
            
            return {
                'outreachColdCall': cold_call if cold_call else None,
                'outreachInstagram': instagram if instagram else None,
                'outreachEmail': email_body if email_body else None,
                'outreachEmailSubject': email_subject if email_subject else None,
                'talking_points': pain_points[:3],
                'ai_generated': True
            }
            
        except Exception as e:
            logger.error(f"AI outreach generation failed: {str(e)}, using template-based fallback")
            return self.generate_outreach_template_based(practice_data, specialty, pain_analysis)

        
        # Priority order (most specific first)
        if any(kw in all_text for kw in ['dermatology', 'dermatologist', 'skin doctor']):
            return 'dermatology'
        elif any(kw in all_text for kw in ['plastic surgery', 'plastic surgeon', 'cosmetic surgeon']):
            return 'plastic_surgery'
        elif any(kw in all_text for kw in ['obgyn', 'ob/gyn', 'ob-gyn', 'gynecologist', 'women\'s health', 'womens health']):
            return 'obgyn'
        elif any(kw in all_text for kw in ['med spa', 'medspa', 'medical spa']):
            return 'medspa'
        elif any(kw in all_text for kw in ['family medicine', 'family practice', 'family physician', 'primary care', 'general practice', 'functional medicine', 'integrative medicine']):
            return 'familypractice'
        else:
            return 'general'
    
    def analyze_pain_points_rule_based(self, practice_data: Dict, specialty: str) -> Dict:
        """
        Rule-based pain point analysis (replaces AI version)
        Returns pain points and readiness score based on specialty and practice data
        """
        pain_points = []
        readiness_score = 50  # Base score
        
        # Get practice details
        services = [s.lower() for s in practice_data.get('services', [])]
        website_text = practice_data.get('website_text', '').lower()
        devices_found = practice_data.get('devices_found', [])
        has_website = bool(practice_data.get('website'))
        
        # Specialty-specific pain points and scoring
        if specialty == 'dermatology':
            pain_points = [
                'Competitive pressure from med spas offering aesthetic services',
                'Patient demand for non-invasive body contouring',
                'Need to expand beyond medical dermatology',
                'Revenue growth opportunities in aesthetics'
            ]
            readiness_score += 20
            
        elif specialty == 'plastic_surgery':
            pain_points = [
                'Pre and post-surgical care revenue opportunities',
                'Non-invasive alternatives for surgical-averse patients',
                'Patient retention between surgical procedures',
                'Complementary services for body contouring'
            ]
            readiness_score += 25
            
        elif specialty == 'obgyn':
            pain_points = [
                'Postpartum body contouring demand',
                'Womens wellness and aesthetics integration',
                'Patient satisfaction and retention',
                'Additional revenue streams beyond traditional services'
            ]
            readiness_score += 15
            
        elif specialty == 'medspa':
            pain_points = [
                'Need for advanced technology to compete',
                'Equipment upgrade or expansion',
                'Attracting higher-value clients',
                'Expanding treatment menu'
            ]
            readiness_score += 30
            
        elif specialty == 'familypractice':
            pain_points = [
                'Differentiation in crowded primary care market',
                'Additional revenue streams beyond insurance',
                'Patient retention and satisfaction',
                'Aesthetic services as practice differentiator'
            ]
            readiness_score += 10
            
        else:  # general
            pain_points = [
                'Practice growth and differentiation',
                'New revenue opportunities',
                'Patient demand for aesthetic services',
                'Competitive market positioning'
            ]
            readiness_score += 5
        
        # Boost score for positive indicators
        if has_website:
            readiness_score += 10
        
        if any(aesthetic_kw in website_text for aesthetic_kw in ['botox', 'filler', 'laser', 'aesthetic', 'cosmetic']):
            readiness_score += 15
            pain_points.append('Already offering aesthetics - ready for advanced equipment')
        
        if len(devices_found) > 0:
            readiness_score += 10
            pain_points.append('Existing aesthetic equipment - potential upgrade opportunity')
        
        if len(services) > 5:
            readiness_score += 5
        
        # Cap score at 100
        readiness_score = min(100, readiness_score)
        
        return {
            'pain_points': pain_points[:5],  # Top 5
            'revenue_opportunity': f'High-value aesthetic services for {specialty} practices',
            'readiness_score': readiness_score,
            'competing_services': [s for s in services if any(kw in s.lower() for kw in ['botox', 'laser', 'filler', 'aesthetic', 'cosmetic'])],
            'gap_analysis': 'Venus devices can complement existing services or create new revenue stream',
            'decision_maker_profile': 'Practice owner, medical director, or office manager',
            'best_approach': 'Focus on ROI, patient satisfaction, and competitive differentiation'
        }
    
    def generate_outreach_template_based(self, practice_data: Dict, specialty: str, pain_analysis: Dict) -> Dict:
        """
        Template-based outreach generation (replaces AI version)
        Returns personalized email and talking points based on specialty
        """
        name = practice_data.get('name', 'Practice')
        contact = practice_data.get('contact_name', 'Doctor')
        pain_points = pain_analysis.get('pain_points', [])
        readiness_score = pain_analysis.get('readiness_score', 50)
        
        # Specialty-specific email templates
        email_templates = {
            'dermatology': {
                'subject': f'Expand Your Aesthetic Services - {name}',
                'body': f"""Dear {contact},

I hope this message finds you well. I am reaching out because {name} is exactly the type of leading dermatology practice that benefits most from Venus technologies.

Many dermatologists we work with face similar challenges: med spas encroaching on aesthetic services, patients requesting non-invasive body contouring, and the need to stay competitive while maintaining medical excellence.

Our Venus systems complement your existing practice by adding high-demand services like body contouring, cellulite reduction, and skin tightening—all with clinically proven, FDA-cleared technology.

Would you be open to a brief conversation about how we have helped practices like yours increase aesthetic revenue by 30-40 percent?

Best regards,
Venus Sales Team"""
            },
            'plastic_surgery': {
                'subject': f'Enhance Pre/Post-Surgical Care Revenue - {name}',
                'body': f"""Dear {contact},

I specialize in working with plastic surgery practices like {name} that want to maximize patient value and retention.

Venus technologies are perfect for pre and post-surgical care, non-invasive alternatives for patients not ready for surgery, and maintenance between procedures.

Our devices complement your surgical practice by capturing patients throughout their aesthetic journey, not just during surgical windows.

I would love to show you how practices similar to yours have increased annual revenue by over $200K with Venus systems.

Can we schedule 15 minutes to discuss?

Best regards,
Venus Sales Team"""
            },
            'obgyn': {
                'subject': f'Womens Wellness + Aesthetics - {name}',
                'body': f"""Dear {contact},

I am reaching out to OB/GYN practices like {name} that are expanding into womens wellness and aesthetics.

Postpartum body contouring is one of the fastest-growing service requests in womens health, and Venus technologies allow you to serve this need without invasive procedures.

Many of our OB/GYN partners have successfully integrated Venus treatments for body contouring, skin tightening, and cellulite reduction—perfect for your patient demographic.

Would you be interested in learning how we have helped practices like yours add $100-150K in annual aesthetic revenue?

Best regards,
Venus Sales Team"""
            },
            'medspa': {
                'subject': f'Upgrade Your Technology - {name}',
                'body': f"""Dear {contact},

{name} caught my attention as a forward-thinking med spa, and I wanted to reach out about Venus technologies.

The med spa market is competitive, and having state-of-the-art equipment is critical for attracting and retaining high-value clients.

Venus systems deliver clinical results your clients will love: body contouring, cellulite reduction, skin tightening, and wrinkle reduction—all FDA-cleared and backed by extensive clinical studies.

Many med spas we work with see 30-50 percent increase in treatment bookings after adding Venus technologies.

Can we schedule a brief demo or discussion?

Best regards,
Venus Sales Team"""
            },
            'familypractice': {
                'subject': f'Differentiate Your Practice - {name}',
                'body': f"""Dear {contact},

I work with forward-thinking family practices like {name} that want to differentiate in a competitive primary care market.

Adding aesthetic services like body contouring and skin tightening creates a powerful practice differentiator while generating cash-pay revenue streams beyond insurance reimbursements.

Venus technologies are perfect for family practices because they are easy to integrate, require minimal training, and patients love the results.

Would you be open to a conversation about how we have helped practices like yours add $75-100K in annual aesthetic revenue?

Best regards,
Venus Sales Team"""
            }
        }
        
        # Get specialty-specific template or use general
        template = email_templates.get(specialty, {
            'subject': f'Aesthetic Technology for {name}',
            'body': f"""Dear {contact},

I am reaching out because {name} could benefit from Venus aesthetic technologies.

Our FDA-cleared systems offer body contouring, skin tightening, cellulite reduction, and wrinkle reduction—high-demand services that generate excellent revenue.

Many practices similar to yours have successfully integrated Venus technologies to enhance patient satisfaction and increase revenue.

Would you be interested in a brief conversation?

Best regards,
Venus Sales Team"""
        })
        
        # Generate talking points based on pain points and specialty
        talking_points = [
            f'Addresses key pain point: {pain_points[0] if pain_points else "practice growth"}',
            'FDA-cleared technology with proven clinical results',
            f'Typical ROI: $100-200K+ annual revenue for {specialty} practices',
            'Comprehensive training and ongoing support included',
            'Flexible financing options available'
        ]
        
        # Determine follow-up timeline based on readiness score
        if readiness_score >= 70:
            follow_up_days = 3
            call_to_action = 'Schedule in-office demo this week'
        elif readiness_score >= 50:
            follow_up_days = 5
            call_to_action = 'Schedule virtual demo or call'
        else:
            follow_up_days = 7
            call_to_action = 'Send additional information and case studies'
        
        # Generate simple cold call and Instagram versions from email
        cold_call = f"Hi, this is [Your Name] from Venus Medical. I'm reaching out because {name} would be a great fit for our aesthetic technologies. {pain_points[0] if pain_points else 'Many practices like yours'} - and Venus addresses this with proven body contouring and skin tightening solutions. Could we schedule a brief 15-minute call to discuss?"
        
        instagram = f"Hi! Noticed {name}'s excellent reputation. Many practices like yours are seeing great results with Venus aesthetic technologies. Would you be open to learning more?"
        
        return {
            'outreachColdCall': cold_call,
            'outreachInstagram': instagram,
            'outreachEmail': template['body'],
            'outreachEmailSubject': template['subject'],
            'talking_points': talking_points,
            'follow_up_days': follow_up_days,
            'call_to_action': call_to_action
        }
    
    def discover_site_pages(self, base_url: str) -> List[str]:
        """
        Discover high-value pages on a practice website
        Returns list of URLs to scrape
        """
        common_paths = [
            '',  # Homepage
            '/about', '/about-us', '/our-practice', '/who-we-are',
            '/services', '/treatments', '/procedures', '/what-we-offer',
            '/team', '/our-team', '/providers', '/doctors', '/staff', '/meet-the-team',
            '/contact', '/locations', '/visit-us'
        ]
        
        urls_to_scrape = []
        
        for path in common_paths:
            url = urljoin(base_url, path)
            # Add to list (we'll check if it exists when we try to scrape it)
            if url not in urls_to_scrape:
                urls_to_scrape.append(url)
        
        return urls_to_scrape[:7]  # Limit to 7 URLs max
    
    def scrape_single_page(self, url: str) -> Dict:
        """
        Scrape a single page and return extracted data
        Returns dict with services, staff_mentions, text, social_links
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text().lower()
            
            # Find services
            service_keywords = [
                'laser hair removal', 'botox', 'fillers', 'coolsculpting',
                'body contouring', 'skin tightening', 'photorejuvenation',
                'cellulite treatment', 'weight loss', 'ems', 'muscle building',
                'emsculpt', 'vaginal rejuvenation', 'hormone therapy',
                'chemical peel', 'microneedling', 'hydrafacial'
            ]
            
            services_found = [kw for kw in service_keywords if kw in text_content]
            
            # Find social links
            social_platforms = {
                'facebook.com': 'Facebook',
                'instagram.com': 'Instagram',
                'twitter.com': 'Twitter',
                'linkedin.com': 'LinkedIn',
                'youtube.com': 'YouTube'
            }
            
            social_links = []
            for link in soup.find_all('a', href=True):
                href = link['href'].lower()
                for domain, platform_name in social_platforms.items():
                    if domain in href and platform_name not in social_links:
                        social_links.append(platform_name)
            
            # Find staff mentions
            staff_indicators = soup.find_all(text=re.compile(
                r'\b(dr\.|doctor|physician|provider|practitioner)\b', re.I))
            
            return {
                'services': services_found,
                'staff_mentions': list(set(str(s).strip() for s in staff_indicators if len(str(s).strip()) > 5)),
                'text': text_content,
                'social_links': social_links
            }
            
        except Exception as e:
            logger.debug(f"Error scraping {url}: {str(e)}")
            return {
                'services': [],
                'staff_mentions': [],
                'text': '',
                'social_links': []
            }
    
    def scrape_website_deep(self, base_url: str, max_pages: int = 5) -> Dict[str, any]:
        """
        Intelligently scrape multiple pages from a medical practice website
        Returns aggregated data from all pages
        """
        if self.demo_mode:
            logger.info(f"DEMO MODE: Using mock website data for {base_url}")
            return self.get_mock_website_data(base_url)
        
        if not base_url:
            return {
                'title': 'Not Available - No Website',
                'description': 'Not Available - No Website',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
        
        if not self.check_robots_txt(base_url):
            logger.warning(f"Robots.txt disallows scraping: {base_url}")
            # Fall back to single page
            return self.scrape_website(base_url)
        
        try:
            # First, get homepage for title and description
            time.sleep(random.uniform(0.5, 1.0))
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get title
            title = soup.title.string.strip() if soup.title and soup.title.string else 'Not Available'
            
            # Get description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = 'Not Available'
            if meta_desc and meta_desc.get('content'):
                description = meta_desc.get('content', '').strip()
            else:
                og_desc = soup.find('meta', attrs={'property': 'og:description'})
                if og_desc and og_desc.get('content'):
                    description = og_desc.get('content', '').strip()
            
            # Discover pages to scrape
            pages_to_scrape = self.discover_site_pages(base_url)[:max_pages]
            
            # Aggregate data from multiple pages
            all_services = set()
            all_social_links = set()
            all_staff_mentions = []
            all_text = []
            pages_scraped = 0
            
            for page_url in pages_to_scrape:
                if pages_scraped >= max_pages:
                    break
                
                try:
                    time.sleep(1.0)  # Respectful delay
                    page_data = self.scrape_single_page(page_url)
                    
                    all_services.update(page_data['services'])
                    all_social_links.update(page_data['social_links'])
                    all_staff_mentions.extend(page_data['staff_mentions'])
                    all_text.append(page_data['text'])
                    pages_scraped += 1
                    
                except Exception as e:
                    logger.debug(f"Skipping page {page_url}: {str(e)}")
                    continue
            
            # Calculate staff count from all mentions
            unique_staff = len(set(all_staff_mentions))
            staff_count = min(unique_staff, 20)
            
            logger.info(f"Deep scrape complete for {base_url}: {pages_scraped} pages, {len(all_services)} services found")
            
            return {
                'title': title,
                'description': description,
                'services': list(all_services),
                'social_links': list(all_social_links),
                'staff_count': staff_count
            }
            
        except Exception as e:
            logger.error(f"Error in deep scrape for {base_url}: {str(e)}")
            # Fall back to single-page scrape
            return self.scrape_website(base_url)
    
    def calculate_ai_score(self, practice_data: Dict) -> Tuple[int, Dict[str, int], str]:
        """
        Calculate AI-powered scoring with specialty-specific weights
        
        CORRECTED LOGIC FOR VENUS DEVICE SALES:
        - SOLO/SMALL practices score HIGHER (single decision maker)
        - NO hospital affiliation scores HIGHER (purchasing freedom)
        - OFFICE-BASED scores HIGHER (no permission needed)
        - PRIVATE ownership scores HIGHER (financial autonomy)
        
        Returns:
            (total_score, score_breakdown, detected_specialty)
        """
        
        # STEP 1: Detect specialty
        # Use AI-enhanced detection (falls back to rule-based automatically)
        specialty = self.detect_specialty_ai(practice_data)
        
        # STEP 2: Get specialty-specific config (or use default)
        if specialty in self.specialty_scoring:
            config = self.specialty_scoring[specialty]
            logger.info(f"🎯 Using {specialty.upper()} scoring profile")
        else:
            # Default to dermatology weights for general practices
            config = self.specialty_scoring['dermatology']
            logger.info(f"Using DEFAULT scoring profile for {specialty}")
        
        weights = config['weights']
        high_value_keywords = config['high_value_keywords']
        
        scores = {
            'specialty_match': 0,
            'decision_autonomy': 0,
            'aesthetic_services': 0,
            'competing_devices': 0,
            'social_activity': 0,
            'reviews_rating': 0,
            'search_visibility': 0,
            'financial_indicators': 0,
            'weight_loss_services': 0
        }
        
        practice_name = practice_data.get('name', '').lower()
        practice_desc = practice_data.get('description', '').lower()
        address = practice_data.get('formatted_address', '').lower()
        all_text = f"{practice_name} {practice_desc} {address}"
        
        # ═══════════════════════════════════════════════════════════
        # 1. Specialty Match (20 points)
        # ═══════════════════════════════════════════════════════════
        specialty_keywords = [
            'dermatology', 'dermatologist', 'plastic surgery', 'plastic surgeon',
            'cosmetic', 'aesthetic', 'med spa', 'medical spa', 'medspa',
            'skin care', 'skincare', 'beauty', 'obgyn', 'ob/gyn', 'gynecologist',
            'women\'s health', 'family practice', 'family medicine'
        ]
        
        specialty_matches = sum(1 for keyword in specialty_keywords if keyword in all_text)
        scores['specialty_match'] = min(specialty_matches * 4, 20)
        
        # ═══════════════════════════════════════════════════════════
        # 2. Decision-Making Autonomy (20 points) 🔥 CRITICAL
        # CORRECTED: Solo/small = HIGH score (single decision maker)
        # ═══════════════════════════════════════════════════════════
        staff_count = practice_data.get('staff_count', 0)
        
        # Check for hospital affiliation indicators
        hospital_indicators = [
            'hospital', 'medical center', 'health system', 'healthcare system',
            'affiliated', 'network', 'regional medical', 'university medical'
        ]
        has_hospital_affiliation = any(indicator in all_text for indicator in hospital_indicators)
        
        # Check for corporate/chain indicators
        corporate_indicators = [
            'corporate', 'chain', 'franchise', 'national', 'locations',
            'branches', 'group practice', 'associates'
        ]
        is_corporate = any(indicator in all_text for indicator in corporate_indicators)
        
        # Base score by staff size (REVERSED - smaller is better)
        if staff_count == 0 or staff_count == 1:
            autonomy_score = 20  # ✅ Solo practice - PERFECT
        elif staff_count == 2:
            autonomy_score = 18  # ✅ Very small - EXCELLENT
        elif staff_count <= 4:
            autonomy_score = 15  # ✅ Small - VERY GOOD
        elif staff_count <= 6:
            autonomy_score = 10  # 🟡 Medium - OK
        elif staff_count <= 10:
            autonomy_score = 5   # 🟠 Large - Harder
        else:
            autonomy_score = 2   # ❌ Very large - Avoid
        
        # Penalize hospital affiliation (no purchasing freedom)
        if has_hospital_affiliation:
            autonomy_score = max(0, autonomy_score - 10)
        
        # Penalize corporate/chain (centralized purchasing)
        if is_corporate:
            autonomy_score = max(0, autonomy_score - 8)
        
        scores['decision_autonomy'] = autonomy_score
        
        # ═══════════════════════════════════════════════════════════
        # 3. Aesthetic Services (15 points)
        # ═══════════════════════════════════════════════════════════
        services = practice_data.get('services', [])
        aesthetic_services = [
            'botox', 'fillers', 'laser', 'coolsculpting', 'body contouring',
            'skin tightening', 'hair removal', 'ipl', 'radiofrequency',
            'body sculpting', 'cellulite', 'fat reduction'
        ]
        
        service_matches = sum(1 for service in aesthetic_services 
                            if any(service in s.lower() for s in services))
        scores['aesthetic_services'] = min(service_matches * 3, 15)
        
        # ═══════════════════════════════════════════════════════════
        # 4. Competing Devices (10 points)
        # ═══════════════════════════════════════════════════════════
        competing_devices = [
            'coolsculpting', 'thermage', 'ultherapy', 'sculptra', 
            'emsculpt', 'vanquish', 'exilis'
        ]
        
        device_count = sum(1 for device in competing_devices if device in practice_desc)
        scores['competing_devices'] = min(device_count * 5, 10)
        
        # ═══════════════════════════════════════════════════════════
        # 5. Social Media Activity (10 points)
        # ═══════════════════════════════════════════════════════════
        social_links = practice_data.get('social_links', [])
        scores['social_activity'] = min(len(social_links) * 3, 10)
        
        # ═══════════════════════════════════════════════════════════
        # 6. Reviews & Rating (10 points)
        # ═══════════════════════════════════════════════════════════
        rating = practice_data.get('rating', 0)
        review_count = practice_data.get('user_ratings_total', 0)
        
        if rating >= 4.5 and review_count >= 50:
            scores['reviews_rating'] = 10
        elif rating >= 4.0 and review_count >= 25:
            scores['reviews_rating'] = 7
        elif rating >= 3.5 and review_count >= 10:
            scores['reviews_rating'] = 4
        else:
            scores['reviews_rating'] = 1
        
        # ═══════════════════════════════════════════════════════════
        # 7. Search Visibility (10 points)
        # ═══════════════════════════════════════════════════════════
        website = practice_data.get('website', '')
        if website and 'http' in website:
            scores['search_visibility'] = 10
        elif website:
            scores['search_visibility'] = 7
        elif practice_data.get('formatted_phone_number'):
            scores['search_visibility'] = 4
        else:
            scores['search_visibility'] = 1
        
        # ═══════════════════════════════════════════════════════════
        # 8. Financial Indicators (10 points)
        # Affluent area + cash-pay readiness
        # ═══════════════════════════════════════════════════════════
        
        # Check for affluent area indicators
        affluent_indicators = [
            'hills', 'park', 'lake', 'estates', 'plaza', 'center',
            'avenue', 'boulevard', 'suite'
        ]
        is_affluent_area = any(indicator in address for indicator in affluent_indicators)
        
        # Check for cash-pay service keywords
        cashpay_keywords = [
            'aesthetic', 'cosmetic', 'elective', 'spa', 'beauty',
            'anti-aging', 'wellness', 'rejuvenation'
        ]
        offers_cashpay = any(keyword in all_text for keyword in cashpay_keywords)
        
        financial_score = 0
        if is_affluent_area:
            financial_score += 5
        if offers_cashpay:
            financial_score += 5
        
        scores['financial_indicators'] = financial_score
        
        # ═══════════════════════════════════════════════════════════
        # 9. Weight Loss Services (5 points)
        # ═══════════════════════════════════════════════════════════
        weight_keywords = [
            'weight loss', 'medical weight', 'hormone therapy', 'iv therapy',
            'body contouring', 'fat reduction', 'inch loss'
        ]
        
        weight_matches = sum(1 for keyword in weight_keywords if keyword in all_text)
        scores['weight_loss_services'] = min(weight_matches * 2, 5)
        
        # ═══════════════════════════════════════════════════════════
        # 10. Specialty-Specific Keyword Bonus (up to +10 points)
        # ═══════════════════════════════════════════════════════════
        services = practice_data.get('services', [])
        services_text = ' '.join(services).lower()
        all_text_with_services = f"{practice_name} {practice_desc} {services_text}"
        
        keyword_bonus = 0
        for keyword in high_value_keywords:
            if keyword in all_text_with_services:
                keyword_bonus += 2
        keyword_bonus = min(keyword_bonus, 10)
        
        # ═══════════════════════════════════════════════════════════
        # TOTAL SCORE (out of 110, normalized to 100)
        # ═══════════════════════════════════════════════════════════
        base_score = sum(scores.values())
        total_score = min(base_score + keyword_bonus, 100)
        
        return total_score, scores, specialty
    
    def recommend_device(self, practice_data: Dict, ai_scores: Dict) -> Dict:
        """Recommend top device based on practice profile"""
        
        services = practice_data.get('services', [])
        description = practice_data.get('description', '').lower()
        
        device_scores = {}
        
        for device_name, device_info in self.device_catalog.items():
            score = 0
            reasons = []
            
            # Base scoring for specialty alignment
            for specialty in device_info['specialties']:
                if any(specialty.lower() in service.lower() for service in services):
                    score += 20
                    reasons.append(f"Offers {specialty}")
            
            # Keyword matches
            for keyword in device_info['keywords']:
                if keyword.lower() in description:
                    score += 10
                    reasons.append(f"Keyword: {keyword}")
            
            device_scores[device_name] = {
                'score': score,
                'reasons': reasons
            }
        
        sorted_devices = sorted(device_scores.items(), 
                              key=lambda x: x[1]['score'], reverse=True)
        
        recommendations = []
        for i, (device_name, data) in enumerate(sorted_devices[:3]):
            recommendations.append({
                'device': device_name,
                'fit_score': data['score'],
                'rationale': '; '.join(data['reasons']) if data['reasons'] else 'General practice fit',
                'rank': i + 1
            })
        
        return {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'all_recommendations': recommendations
        }
    
    def craft_outreach_opener(self, practice_data: Dict, ai_score: int, 
                            device_rec: Dict) -> str:
        """Generate personalized outreach opener"""
        
        if ai_score < 70:
            return ""
        
        practice_name = practice_data.get('name', 'Your Practice')
        primary_device = device_rec.get('primary_recommendation', {}).get('device', 'our devices')
        
        opener = f"Hi! I noticed {practice_name}'s excellent reputation. Many practices like yours are seeing significant results with {primary_device}. Would you be interested in a brief conversation?"
        
        return opener
    
    def process_practice(self, place_data: Dict) -> Optional[Dict]:
        """Process a single practice through the full pipeline"""
        
        practice_name = place_data.get('name', 'Unknown')
        logger.info(f"Processing practice: {practice_name}")
        
        # Check if existing customer
        if self.is_existing_customer(practice_name):
            logger.info(f"Skipping existing customer: {practice_name}")
            return None
        
        # Initialize practice record
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
            'description': 'Not Available',
            'title': 'Not Available'
        }
        
        # Scrape website if available (using deep multi-page scraping)
        if practice_record['website']:
            logger.info(f"Deep scraping website: {practice_record['website']}")
            website_data = self.scrape_website_deep(practice_record['website'], max_pages=5)
            practice_record.update(website_data)
        
        # Calculate AI score with specialty detection
        ai_score, score_breakdown, specialty = self.calculate_ai_score(practice_record)
        
        # ON-DEMAND OUTREACH: AI outreach generation moved to separate API call
        # This makes search faster and more reliable
        logger.info(f"📊 Scoring complete for {practice_name} - AI outreach available on-demand")
        
        # Get device recommendations
        device_recommendations = self.recommend_device(practice_record, score_breakdown)
        
        # Generate outreach opener
        outreach_opener = self.craft_outreach_opener(
            practice_record, ai_score, device_recommendations)
        
        # Calculate data completeness
        required_fields = ['name', 'address', 'phone', 'website']
        completeness = sum(1 for field in required_fields 
                         if practice_record.get(field)) / len(required_fields)
        
        # Determine confidence level
        if ai_score >= 70 and completeness >= 0.75:
            confidence = "High"
        elif ai_score >= 50 and completeness >= 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Compile final record (without AI outreach - generated on-demand)
        final_record = {
            **practice_record,
            'specialty': specialty,
            'ai_score': ai_score,
            'score_breakdown': score_breakdown,
            'primary_device_rec': device_recommendations.get('primary_recommendation', {}).get('device', ''),
            'device_rationale': device_recommendations.get('primary_recommendation', {}).get('rationale', ''),
            'all_device_recs': device_recommendations.get('all_recommendations', []),
            'outreach_opener': outreach_opener,
            'confidence_level': confidence,
            'data_completeness': completeness,
            # AI outreach fields set to None - will be populated on-demand via API
            'outreachColdCall': None,
            'outreachInstagram': None,
            'outreachEmail': None,
            'outreachEmailSubject': None
        }
        
        return final_record
    
    def generate_outreach_for_prospect_standalone(self, prospect_data: Dict) -> Dict:
        """
        Generate AI-powered outreach for a single prospect ON-DEMAND
        
        This is called AFTER search completes, when a sales rep clicks "Generate AI Outreach"
        Uses all the business intelligence gathered during prospecting to create personalized messages
        
        Args:
            prospect_data: Complete prospect record from database/CSV with all fields
        
        Returns:
            Dict with outreachColdCall, outreachInstagram, outreachEmail, outreachEmailSubject
        """
        try:
            practice_name = prospect_data.get('name', 'Unknown')
            specialty = prospect_data.get('specialty', 'general')
            
            logger.info(f"🤖 Generating on-demand AI outreach for {practice_name} ({specialty})")
            
            # Step 1: Analyze pain points using AI (or rule-based fallback)
            pain_analysis = self.analyze_pain_points_ai(prospect_data, specialty)
            
            # Step 2: Generate AI outreach using all available data
            outreach_content = self.generate_outreach_ai(prospect_data, specialty, pain_analysis)
            
            logger.info(f"✅ On-demand outreach generated for {practice_name}")
            
            return {
                'success': True,
                'outreach': {
                    'outreachColdCall': outreach_content.get('outreachColdCall', ''),
                    'outreachInstagram': outreach_content.get('outreachInstagram', ''),
                    'outreachEmail': outreach_content.get('outreachEmail', ''),
                    'outreachEmailSubject': outreach_content.get('outreachEmailSubject', ''),
                    'talkingPoints': outreach_content.get('talking_points', []),
                    'painPoints': pain_analysis.get('pain_points', []),
                    'readinessScore': pain_analysis.get('readiness_score', 0)
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating on-demand outreach: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'outreach': {
                    'outreachColdCall': 'Error generating outreach',
                    'outreachInstagram': 'Error generating outreach',
                    'outreachEmail': 'Error generating outreach',
                    'outreachEmailSubject': 'Error generating outreach'
                }
            }
    
    def export_to_csv(self, results: List[Dict], filename: str):
        """Export results to CSV"""
        
        if not results:
            logger.warning("No results to export")
            return
        
        csv_columns = [
            'name', 'specialty', 'address', 'phone', 'website', 'rating', 'review_count',
            'ai_score', 'confidence_level', 'primary_device_rec', 'device_rationale',
            'outreach_opener', 'services', 'social_links', 'staff_count',
            'specialty_match_score', 'decision_autonomy_score', 'aesthetic_services_score', 
            'competing_devices_score', 'social_activity_score', 'reviews_rating_score',
            'search_visibility_score', 'financial_indicators_score', 'weight_loss_services_score',
            'data_completeness',
            # AI-enhanced fields (Option B: Full AI integration)
            'ai_readiness_score', 'revenue_opportunity', 'gap_analysis', 'best_approach',
            'decision_maker_profile', 'email_subject', 'email_body', 'pain_points',
            'competing_services', 'talking_points', 'follow_up_days', 'call_to_action'
        ]
        
        csv_data = []
        for result in results:
            row = {}
            for col in csv_columns:
                if col.endswith('_score') and col != 'ai_score' and col != 'ai_readiness_score':
                    score_key = col.replace('_score', '')
                    row[col] = result.get('score_breakdown', {}).get(score_key, 0)
                elif col in ['services', 'social_links', 'pain_points', 'competing_services', 'talking_points']:
                    # Join list fields with semicolons
                    row[col] = '; '.join(str(x) for x in result.get(col, []))
                else:
                    row[col] = result.get(col, '')
            csv_data.append(row)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"Results exported to {filename}")
    
    def generate_summary_report(self, results: List[Dict], filename: str):
        """Generate summary report"""
        
        if not results:
            return
        
        sorted_results = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)
        
        total_prospects = len(results)
        high_fit = len([r for r in results if r.get('ai_score', 0) >= 70])
        medium_fit = len([r for r in results if 50 <= r.get('ai_score', 0) < 70])
        low_fit = len([r for r in results if r.get('ai_score', 0) < 50])
        
        avg_score = sum(r.get('ai_score', 0) for r in results) / total_prospects if total_prospects > 0 else 0
        
        report = f"""
MEDICAL DEVICE PROSPECTING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Total Prospects Analyzed: {total_prospects}
Average AI Score: {avg_score:.1f}/100

PROSPECT DISTRIBUTION
====================
High-Fit (70-100): {high_fit} ({high_fit/total_prospects*100:.1f}%)
Medium-Fit (50-69): {medium_fit} ({medium_fit/total_prospects*100:.1f}%)
Low-Fit (0-49): {low_fit} ({low_fit/total_prospects*100:.1f}%)

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
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Summary report generated: {filename}")
    
    def run_prospecting(self, keywords: List[str], location: str, 
                       radius: int, max_results: int):
        """Main prospecting workflow"""
        
        logger.info(f"Starting prospecting for {keywords} near {location}")
        
        all_results = []
        
        for keyword in keywords:
            logger.info(f"Searching for: {keyword}")
            
            places = self.google_places_search(keyword, location, radius * 1000)
            
            for place in places[:max_results]:
                try:
                    processed_practice = self.process_practice(place)
                    if processed_practice:
                        all_results.append(processed_practice)
                        logger.info(f"Processed: {processed_practice.get('name', 'Unknown')} - Score: {processed_practice.get('ai_score', 0)}")
                    
                except Exception as e:
                    logger.error(f"Error processing practice: {str(e)}")
                    continue
        
        # Remove duplicates
        unique_results = []
        seen = set()
        for result in all_results:
            key = (result.get('name', ''), result.get('address', ''))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        logger.info(f"Found {len(unique_results)} unique prospects")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"prospects_{timestamp}.csv"
        report_filename = f"summary_{timestamp}.txt"
        
        self.export_to_csv(unique_results, csv_filename)
        self.generate_summary_report(unique_results, report_filename)
        
        return unique_results, csv_filename, report_filename


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(description='Medical Device Prospecting System')
    parser.add_argument('--keywords', nargs='+', help='Search keywords')
    parser.add_argument('--city', help='City or location to search')
    parser.add_argument('--radius', type=int, default=25, help='Search radius in km')
    parser.add_argument('--max-results', type=int, default=150, help='Max results per keyword')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('--exclude-csv', help='CSV file with existing customers')
    
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    
    prospector = FathomProspector(
        demo_mode=args.demo, 
        existing_customers_csv=args.exclude_csv
    )
    
    if args.interactive or not all([args.keywords, args.city]):
        print("\n=== MEDICAL DEVICE PROSPECTING SYSTEM ===\n")
        
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
        max_results = args.max_results or int(input(f"Enter max results per keyword (default: 150): ") or "150")
        
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
    print(f"Google Places API: {'✓ Configured' if prospector.gmaps_key else '✗ Missing'}")
    sys.stdout.flush()
    
    try:
        results, csv_file, report_file = prospector.run_prospecting(
            keywords, city, radius, max_results)
        
        print(f"\n=== PROSPECTING COMPLETE ===")
        print(f"Total prospects found: {len(results)}")
        print(f"Results exported to: {csv_file}")
        print(f"Summary report: {report_file}")
        
        if results:
            top_prospects = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)[:5]
            print(f"\nTOP 5 PROSPECTS:")
            for i, prospect in enumerate(top_prospects, 1):
                print(f"{i}. {prospect.get('name', 'Unknown')} - Score: {prospect.get('ai_score', 0)}/100")
        
    except Exception as e:
        logger.error(f"Error during prospecting: {str(e)}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
