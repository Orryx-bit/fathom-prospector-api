#!/usr/bin/env python3
"""
Fathom Medical Device Prospecting System
Comprehensive tool for finding and scoring medical practices
Production-Hardened Version 3.0
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

import aiohttp
import pandas as pd
import requests
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

# Import blacklist manager for filtering problematic sites
from blacklist_manager import get_blacklist_manager

# Async HTTP for concurrent scraping


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

# ScrapingBee for JavaScript-heavy sites (replaces Playwright)
try:
    from scrapingbee_integration import (
        scrape_with_scrapingbee,
        is_scrapingbee_available
    )
    SCRAPINGBEE_AVAILABLE = is_scrapingbee_available()
    if SCRAPINGBEE_AVAILABLE:
        logger.info("✅ ScrapingBee enabled for JavaScript rendering")
except ImportError:
    SCRAPINGBEE_AVAILABLE = False
    logger.warning("⚠️  ScrapingBee not available - JavaScript rendering disabled")

# Domain whitelist for known JavaScript-heavy website builders
# These sites will automatically use ScrapingBee for better scraping
JS_HEAVY_DOMAINS = [
    'squarespace.com',
    'wix.com',
    'webflow.com',
    'weebly.com',
    'shopify.com',
    'wordpress.com',  # WordPress.com (hosted, often JS-heavy)
    'godaddy.com',    # GoDaddy website builder
    'site123.com',
    'jimdo.com',
    'strikingly.com'
]

# Abacus RouteLLM Integration (OpenAI-compatible)
AI_AVAILABLE = False
openai_client = None

try:
    from openai import OpenAI
    abacus_key = os.getenv('ABACUSAI_API_KEY')
    
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
        self.gmaps_api = None  # Initialize early to prevent AttributeError
        
        self.gmaps_key = os.getenv('GOOGLE_PLACES_API_KEY')
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
        
        # Initialize blacklist manager for filtering problematic sites
        try:
            self.blacklist_manager = get_blacklist_manager()
            logger.info(f"✅ Blacklist system loaded: {self.blacklist_manager.get_stats()['total_domains']} domains, {self.blacklist_manager.get_stats()['total_patterns']} patterns")
        except Exception as e:
            logger.warning(f"⚠️  Blacklist system not available: {str(e)} - continuing without filtering")
            self.blacklist_manager = None
        
        # Initialize Google Maps API if not in demo mode
        if not demo_mode:
            try:
                self.gmaps_api = GooglePlacesAPI(self.gmaps_key)
                test_result = self.gmaps_api.geocode('Austin, TX')
                if not test_result:
                    logger.warning('Google Places API test failed - switching to demo mode')
                    self.demo_mode = True
                    self.gmaps_api = None
            except Exception as e:
                logger.warning(f'Google Places API initialization failed: {str(e)} - switching to demo mode')
                self.demo_mode = True
                self.gmaps_api = None
        
        if self.demo_mode:
            logger.info('Running in DEMO MODE with mock data')
        
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
        
        # Defensive check: ensure gmaps_api is initialized
        if not self.gmaps_api:
            logger.error("Google Maps API not initialized - switching to demo mode")
            self.demo_mode = True
            return self.get_mock_data(query, location)
        
        try:
            logger.info(f"Searching Google Places for '{query}' near '{location}'")
            
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
        # Defensive check: ensure gmaps_api is initialized
        if not self.gmaps_api:
            logger.debug("Google Maps API not initialized, cannot get place details")
            return None
        
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
                'staff_count': 8,
                'text': 'Mock website text for Austin Aesthetic Center.'
            },
            'https://hillcountrymedspa.com': {
                'title': 'Hill Country Med Spa - Body Contouring & Wellness',
                'description': 'Full-service medical spa specializing in body contouring and wellness services.',
                'services': ['body contouring', 'cellulite treatment', 'weight loss'],
                'social_links': ['Instagram'],
                'staff_count': 5,
                'text': 'Mock website text for Hill Country Med Spa.'
            },
            'https://lonestardermatology.com': {
                'title': 'Lone Star Dermatology - Advanced Skin Care',
                'description': 'Board-certified dermatologists providing medical and cosmetic dermatology services.',
                'services': ['laser hair removal', 'skin tightening', 'botox'],
                'social_links': ['Facebook', 'LinkedIn'],
                'staff_count': 12,
                'text': 'Mock website text for Lone Star Dermatology.'
            }
        }
        
        return mock_data.get(url, {
            'title': 'Medical Practice',
            'description': 'Professional medical and aesthetic services',
            'services': ['botox', 'fillers'],
            'social_links': [],
            'staff_count': 3,
            'text': 'Default mock website content.'
        })
    
    @sleep_and_retry
    @limits(calls=30, period=60)

    def extract_emails(self, soup: BeautifulSoup, url: str) -> List[str]:
        """
        Extract email addresses from website
        
        Args:
            soup: BeautifulSoup object
            url: Website URL for context
            
        Returns:
            List of unique email addresses (up to 5)
        """
        emails = set()
        
        try:
            # Method 1: mailto: links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('mailto:'):
                    # Extract email from mailto: link
                    email = href.replace('mailto:', '').split('?')[0].strip()
                    if '@' in email and '.' in email:
                        emails.add(email.lower())
            
            # Method 2: Email pattern in visible text
            text_content = soup.get_text()
            # Email regex pattern
            email_pattern = r' [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,} '
            found_emails = re.findall(email_pattern, text_content)
            
            # Filter out common false positives
            spam_keywords = ['example', 'test', 'noreply', 'no-reply', 'donotreply']
            for email in found_emails:
                email_lower = email.lower()
                if not any(spam in email_lower for spam in spam_keywords):
                    if '@' in email_lower and '.' in email_lower:
                        emails.add(email_lower)
            
            # Method 3: Schema.org markup
            for meta in soup.find_all('meta', attrs={'itemprop': 'email'}):
                if meta.get('content'):
                    email = meta['content'].strip().lower()
                    if '@' in email and '.' in email:
                        emails.add(email)
            
            # Convert to list and limit to top 5
            email_list = list(emails)[:5]
            
            if email_list:
                logger.info(f"📧 Found {len(email_list)} email(s): {', '.join(email_list)}")
            
            return email_list
            
        except Exception as e:
            logger.warning(f"Error extracting emails from {url}: {str(e)}")
            return []
    
    def extract_contact_names(self, soup: BeautifulSoup, business_name: str, url: str) -> List[str]:
        """
        Extract doctor/owner names from website
        
        Args:
            soup: BeautifulSoup object
            business_name: Business name for reference
            url: Website URL for context
            
        Returns:
            List of contact names (up to 5)
        """
        names = []
        seen_names = set()
        
        try:
            # Pattern 1: Extract from business name itself
            if business_name:
                # Match "Dr. FirstName LastName" or "FirstName LastName MD/DDS/etc"
                name_patterns = [
                    r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Dr. John Smith
                    r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:MD|DDS|DMD|DO|DPM)',  # John Smith MD
                ]
                
                for pattern in name_patterns:
                    matches = re.findall(pattern, business_name)
                    for match in matches:
                        clean_name = match.strip()
                        if clean_name and clean_name not in seen_names and len(clean_name) > 5:
                            names.append(f"Dr. {clean_name}" if not clean_name.startswith('Dr') else clean_name)
                            seen_names.add(clean_name)
            
            # Pattern 2: Check headings for staff names
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                text = heading.get_text().strip()
                
                # Look for doctor/physician indicators
                if any(keyword in text.lower() for keyword in ['dr.', 'doctor', 'meet', 'physician', 'about']):
                    # Try to extract name
                    name_match = re.search(r'(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
                    if name_match:
                        clean_name = name_match.group(1).strip()
                        if clean_name not in seen_names and len(clean_name) > 5:
                            names.append(clean_name)
                            seen_names.add(clean_name)
            
            # Pattern 3: Look for "Meet Dr." or "About Dr." sections
            text_content = soup.get_text()
            meet_patterns = [
                r'Meet\s+(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'About\s+(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            ]
            
            for pattern in meet_patterns:
                matches = re.findall(pattern, text_content)
                for match in matches:
                    clean_name = match.strip()
                    if clean_name not in seen_names and len(clean_name) > 5:
                        names.append(clean_name)
                        seen_names.add(clean_name)
            
            # Pattern 4: Schema.org Person markup
            for person_div in soup.find_all(attrs={'itemtype': re.compile(r'schema.org/Person')}):
                name_span = person_div.find(attrs={'itemprop': 'name'})
                if name_span:
                    clean_name = name_span.get_text().strip()
                    if clean_name not in seen_names and len(clean_name) > 5:
                        if 'dr' in clean_name.lower() or 'doctor' in clean_name.lower():
                            names.append(clean_name)
                            seen_names.add(clean_name)
            
            # Limit to top 5 names
            names = names[:5]
            
            if names:
                logger.info(f"👤 Found {len(names)} contact name(s): {', '.join(names)}")
            
            return names
            
        except Exception as e:
            logger.warning(f"Error extracting contact names from {url}: {str(e)}")
            return []
    
    def extract_additional_phones(self, soup: BeautifulSoup, primary_phone: str, url: str) -> List[str]:
        """
        Extract additional phone numbers from website
        
        Args:
            soup: BeautifulSoup object
            primary_phone: Primary phone from Google Places (to avoid duplicates)
            url: Website URL for context
            
        Returns:
            List of additional phone numbers (up to 3)
        """
        phones = set()
        
        try:
            # Normalize primary phone for comparison
            primary_normalized = re.sub(r'[^0-9]', '', primary_phone) if primary_phone else ''
            
            text_content = soup.get_text()
            
            # Phone number patterns
            phone_patterns = [
                r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (123) 456-7890 or 123-456-7890
                r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',           # 123.456.7890
                r'1[-.\s]\d{3}[-.\s]\d{3}[-.\s]\d{4}',    # 1-123-456-7890
            ]
            
            for pattern in phone_patterns:
                matches = re.findall(pattern, text_content)
                for match in matches:
                    # Normalize phone for comparison
                    normalized = re.sub(r'[^0-9]', '', match)
                    
                    # Must be exactly 10 or 11 digits
                    if len(normalized) in [10, 11]:
                        # Skip if it's the primary phone
                        if normalized != primary_normalized and normalized[-10:] != primary_normalized[-10:]:
                            phones.add(match.strip())
            
            # Also check tel: links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('tel:'):
                    phone = href.replace('tel:', '').strip()
                    normalized = re.sub(r'[^0-9]', '', phone)
                    
                    if len(normalized) in [10, 11]:
                        if normalized != primary_normalized and normalized[-10:] != primary_normalized[-10:]:
                            phones.add(phone)
            
            # Limit to top 3 additional phones
            phone_list = list(phones)[:3]
            
            if phone_list:
                logger.info(f"📞 Found {len(phone_list)} additional phone(s): {', '.join(phone_list)}")
            
            return phone_list
            
        except Exception as e:
            logger.warning(f"Error extracting additional phones from {url}: {str(e)}")
            return []
    
    def extract_contact_form_url(self, soup: BeautifulSoup, base_url: str) -> str:
        """
        Extract contact form URL from website
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL of the website
            
        Returns:
            Contact form URL or empty string
        """
        try:
            # Look for contact/schedule/appointment links
            contact_keywords = ['contact', 'schedule', 'appointment', 'book', 'consult']
            
            for link in soup.find_all('a', href=True):
                href = link['href'].lower()
                link_text = link.get_text().lower()
                
                # Check if link or text contains contact keywords
                if any(keyword in href or keyword in link_text for keyword in contact_keywords):
                    # Build full URL
                    if href.startswith('http'):
                        return link['href']
                    elif href.startswith('/'):
                        return urljoin(base_url, href)
            
            return ''
            
        except Exception as e:
            logger.warning(f"Error extracting contact form URL: {str(e)}")
            return ''
    
    def extract_team_members(self, soup: BeautifulSoup, url: str) -> List[str]:
        """
        Extract team member names from website
        
        Args:
            soup: BeautifulSoup object
            url: Website URL for context
            
        Returns:
            List of team member names with titles (up to 10)
        """
        team_members = []
        seen_members = set()
        
        try:
            # Look for team/staff sections
            team_sections = soup.find_all(['div', 'section'], class_=re.compile(r'team|staff|doctor|provider', re.I))
            
            for section in team_sections:
                # Look for names with titles
                for heading in section.find_all(['h2', 'h3', 'h4', 'h5']):
                    text = heading.get_text().strip()
                    
                    # Match patterns like "Dr. John Smith - Board Certified Surgeon"
                    if any(keyword in text.lower() for keyword in ['dr.', 'doctor', 'md', 'dds', 'dmd']):
                        if text not in seen_members and len(text) > 5 and len(text) < 100:
                            team_members.append(text)
                            seen_members.add(text)
            
            # Limit to top 10 team members
            team_members = team_members[:10]
            
            if team_members:
                logger.info(f"👥 Found {len(team_members)} team member(s)")
            
            return team_members
            
        except Exception as e:
            logger.warning(f"Error extracting team members from {url}: {str(e)}")
            return []


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
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': [],
                'text': 'Not Available - No Website'
            }
        
        if not self.check_robots_txt(url):
            logger.warning(f"Robots.txt disallows scraping: {url}")
            return {
                'title': 'Not Available - Restricted',
                'description': 'Not Available - Restricted',
                'services': [],
                'social_links': [],
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': [],
                'text': 'Not Available - Restricted'
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
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': [],
                'text': ''
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
            data['text'] = text_content
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
            staff_indicators = soup.find_all(string=re.compile(
                r'\b(dr\.|doctor|physician|provider|practitioner)\b', re.I))
            unique_staff = len(set(str(s).strip() for s in staff_indicators if len(str(s).strip()) > 5))
            data['staff_count'] = min(unique_staff, 20)
            
            # Extract contact intelligence
            data['emails'] = self.extract_emails(soup, url)
            data['contact_names'] = self.extract_contact_names(soup, '', url)
            data['additional_phones'] = self.extract_additional_phones(soup, '', url)
            data['contact_form_url'] = self.extract_contact_form_url(soup, url)
            data['team_members'] = self.extract_team_members(soup, url)
            
            logger.info(f"Website scrape complete for {url}: {len(data['services'])} services, {len(data['emails'])} emails, {len(data['contact_names'])} contacts found")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout scraping website {url}")
            return {
                'title': 'Not Available - Timeout',
                'description': 'Not Available - Timeout',
                'services': [],
                'social_links': [],
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': [],
                'text': 'Not Available - Timeout'
            }
        except Exception as e:
            logger.error(f"Error scraping website {url}: {str(e)}")
            return {
                'title': 'Not Available - Error',
                'description': 'Not Available - Error',
                'services': [],
                'social_links': [],
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': [],
                'text': 'Not Available - Error'
            }
    
    def is_js_heavy_site(self, url: str) -> bool:
        """
        Check if URL is from a known JavaScript-heavy platform
        
        Args:
            url: Website URL to check
            
        Returns:
            True if site is from JS-heavy platform, False otherwise
        """
        try:
            url_lower = url.lower()
            for domain in JS_HEAVY_DOMAINS:
                if domain in url_lower:
                    logger.info(f"Detected JS-heavy platform: {domain} in {url}")
                    return True
            return False
        except Exception:
            return False
    
    def is_scrape_successful(self, result: Dict) -> bool:
        """
        Check if scraping returned meaningful content
        
        Args:
            result: Scraping result dictionary
            
        Returns:
            True if scrape was successful, False otherwise
        """
        # Check 1: Did we get a real title?
        title = result.get('title', '')
        if not title or title in ['Not Available', 'Not Available - No Website', 
                                  'Not Available - Restricted', 'Not Available - Timeout',
                                  'Not Available - Error']:
            return False
        
        # Check 2: Did we get meaningful description?
        description = result.get('description', '')
        if not description or description in ['Not Available', 'Not Available - No Website',
                                              'Not Available - Restricted', 'Not Available - Timeout',
                                              'Not Available - Error']:
            return False
        
        # Check 3: Did we find any content at all?
        services = result.get('services', [])
        social_links = result.get('social_links', [])
        
        if len(services) == 0 and len(social_links) == 0:
            logger.debug(f"Scrape unsuccessful: no services or social links found")
            return False
        
        # Looks good!
        return True
    
    def scrape_with_scrapingbee_wrapper(self, url: str) -> Dict[str, any]:
        """
        Scrape website using ScrapingBee for JavaScript-rendered content, with exponential backoff.
        
        Args:
            url: Website URL to scrape
            
        Returns:
            Dictionary with scraped data
        """
        if not SCRAPINGBEE_AVAILABLE:
            logger.warning(f"ScrapingBee not available, falling back to BeautifulSoup for {url}")
            return self.scrape_website(url)
        
        if self.demo_mode:
            logger.info(f"DEMO MODE: Using mock website data for {url}")
            return self.get_mock_website_data(url)
        
        if not url:
            return {
                'title': 'Not Available - No Website',
                'description': 'Not Available - No Website',
                'services': [],
                'social_links': [],
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': []
            }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = scrape_with_scrapingbee(
                    url,
                    self.extract_emails,
                    self.extract_contact_names,
                    self.extract_additional_phones,
                    self.extract_contact_form_url,
                    self.extract_team_members
                )
                
                if result:
                    return result
                else:
                    # ScrapingBee failed, fallback
                    logger.warning(f"ScrapingBee returned no data for {url} on attempt {attempt + 1}, will fallback if all retries fail.")
                    if attempt == max_retries - 1:
                        break  # exit loop to fallback
                    time.sleep(1)  # wait before retrying on empty result
                    
            except Exception as e:
                # Check for 429 error
                if '429' in str(e):
                    if attempt < max_retries - 1:
                        backoff_time = (2 ** attempt) + random.uniform(0.5, 1.0)
                        logger.warning(f"ScrapingBee rate limit hit (429) for {url}. Retrying in {backoff_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(backoff_time)
                        continue  # continue to next attempt
                    else:
                        logger.error(f"ScrapingBee rate limit hit on final attempt for {url}. Falling back.")
                        break  # exit loop to fallback
                else:
                    logger.error(f"Error in ScrapingBee scraping {url} on attempt {attempt+1}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # brief pause for other errors
                        continue
                    else:
                        logger.error(f"ScrapingBee failed on final attempt for {url}. Falling back.")
                        break  # exit loop to fallback
        
        # Fallback to BeautifulSoup if all retries fail
        logger.warning(f"ScrapingBee failed for {url} after {max_retries} attempts, falling back to BeautifulSoup.")
        return self.scrape_website(url)

    
    def scrape_website_smart(self, url: str) -> Dict[str, any]:
        """
        Smart scraping with automatic method selection
        
        Strategy:
        1. Check if URL is from JS-heavy platform (whitelist) → use ScrapingBee
        2. Try BeautifulSoup first (fast)
        3. If BeautifulSoup gets poor data → retry with ScrapingBee
        
        Args:
            url: Website URL to scrape
            
        Returns:
            Dictionary with scraped data
        """
        if self.demo_mode:
            logger.info(f"DEMO MODE: Using mock website data for {url}")
            return self.get_mock_website_data(url)
        
        if not url:
            return {
                'title': 'Not Available - No Website',
                'description': 'Not Available - No Website',
                'services': [],
                'social_links': [],
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': []
            }
        
        # Check 1: Is this a known JS-heavy platform?
        if SCRAPINGBEE_AVAILABLE and self.is_js_heavy_site(url):
            logger.info(f"JS-heavy platform detected, using ScrapingBee immediately for {url}")
            return self.scrape_with_scrapingbee_wrapper(url)
        
        # Check 2: Try BeautifulSoup first (fast path)
        logger.info(f"Trying BeautifulSoup first for {url}")
        bs_result = self.scrape_website(url)
        
        # Check 3: Did BeautifulSoup get good data?
        if self.is_scrape_successful(bs_result):
            logger.info(f"✅ BeautifulSoup successful for {url}")
            return bs_result
        
        # Check 4: BeautifulSoup failed, try ScrapingBee if available
        if SCRAPINGBEE_AVAILABLE:
            logger.info(f"BeautifulSoup data incomplete for {url}, retrying with ScrapingBee...")
            return self.scrape_with_scrapingbee_wrapper(url)
        else:
            logger.warning(f"BeautifulSoup data incomplete but ScrapingBee not available for {url}")
            return bs_result
    
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
        CREATIVE AI outreach generation using Abacus RouteLLM
        Creates personalized cold call script, Instagram DM, and email with creative approaches
        All outreach focuses on SETTING MEETINGS with creative engagement
        """
        if not self.ai_enabled:
            return self.generate_outreach_template_based(practice_data, specialty, pain_analysis)
        
        try:
            # Extract practice data
            practice_name = practice_data.get('name', 'Unknown Practice')
            contact_name = practice_data.get('contact_name', '')
            contact_email = practice_data.get('contact_email', '')
            location = practice_data.get('location', '')
            rating = practice_data.get('rating', 0)
            phone_number = practice_data.get('phone', '')
            services = practice_data.get('services', [])
            review_count = practice_data.get('review_count', 0)
            
            # Handle missing contact info gracefully
            contact_display = contact_name if contact_name else 'the practice'
            
            # Creative frameworks
            creative_frameworks = [
                "curiosity_gap", "puzzle_solver", "trend_spotter", "results_tease"
            ]
            selected_framework = random.choice(creative_frameworks)
            
            # Specialty analogies
            specialty_analogies = {
                'dermatology': [
                    "Like finding the perfect skincare regimen, but for your practice's revenue",
                    "Similar to how laser treatments target specific skin concerns, this targets specific revenue gaps"
                ],
                'plastic_surgery': [
                    "Like precision surgery for your practice's revenue streams", 
                    "Similar to how reconstructive surgery restores function, this restores practice growth"
                ],
                'obgyn': [
                    "Like prenatal care for your practice's new service lines",
                    "Similar to how hormone therapy balances systems, this balances your service portfolio"
                ],
                'medspa': [
                    "Like the perfect chemical peel for your revenue - reveals what's underneath",
                    "Similar to how Botox smooths wrinkles, this smooths out revenue fluctuations"
                ],
                'familypractice': [
                    "Like preventive care for your practice's financial health",
                    "Similar to how family medicine treats the whole person, this treats the whole practice"
                ]
            }
            
            analogy = random.choice(specialty_analogies.get(specialty, [
                "Like finding the right diagnostic tool for your practice's growth potential",
                "Similar to how the right treatment plan transforms patient outcomes, this transforms practice revenue"
            ]))
            
            # ========================================
            # CREATIVE EMAIL PROMPT 
            # ========================================
            email_prompt = f"""
You are "Alex Rivera," a top-performing medical device consultant known for:
- Creating irresistible curiosity gaps
- Using unexpected analogies that make doctors stop and think  
- Building immediate rapport through personalized insights
- Crafting emails that get 3x higher response rates than industry average

YOUR CREATIVE MISSION:
Transform boring sales email into something a busy doctor would actually WANT to read.

PRACTICE CONTEXT:
- Practice: {practice_name} in {location}
- Specialty: {specialty} 
- Rating: {rating}/5 ({review_count} reviews)
- Recent Services: {services[:3] if services else 'General services'}

USE CREATIVE FRAMEWORK: {selected_framework}

PSYCHOLOGICAL TRIGGERS: Use scarcity, social proof, curiosity, and reciprocity naturally.

SPECIALTY ANALOGY: {analogy}

CREATIVE CONSTRAINTS:
- MUST create an "Aha!" moment in first 15 words
- MUST use the specialty analogy above
- MUST include one curiosity-building question
- CAN use humor, storytelling, or industry insights
- CAN break conventional email structure if it creates engagement
- CANNOT sound like a typical sales email
- CANNOT use corporate jargon like "leverage," "synergy," etc.

FORMAT:
- Subject line (intriguing, 5-8 words)
- Body (conversational, 80-120 words)
- CTA (specific but low-pressure)

Generate the email in the following format:

SUBJECT: [subject line]

BODY:
[email body]

Remember: Sound like a helpful colleague, not a salesperson.
"""

            # ========================================
            # CREATIVE COLD CALL PROMPT
            # ========================================
            cold_call_prompt = f"""
You are "Alex Rivera," a top-performing medical device consultant making a cold call to {practice_name}.

CREATIVE MISSION: Create a 30-second opening that makes the gatekeeper WANT to connect you.

PRACTICE CONTEXT:
- Practice: {practice_name} in {location}
- Specialty: {specialty}
- Rating: {rating}/5

USE PSYCHOLOGY:
- Scarcity: "Only 2 consultation slots left in {location} this month"
- Social Proof: "Other {specialty} practices in the area are seeing amazing results"
- Curiosity: "There's a specific pattern I'm seeing with successful {specialty} practices"
- Reciprocity: "I analyzed your practice and have one quick insight to share"

CREATIVE APPROACH:
- Start with a curiosity gap that makes them wonder
- Use the {specialty} analogy: {analogy}
- Sound like you're offering valuable market intelligence, not selling
- Keep it under 30 seconds when spoken

GENERATE A COLD CALL SCRIPT THAT:
1. Opens with a surprising industry insight about {specialty}
2. Creates immediate curiosity about what you know
3. Positions you as a valuable resource, not a salesperson
4. Asks for a specific meeting format (lunch, brief call, etc.)
5. Sounds confident and professional but not corporate

OUTPUT: A natural, conversational script ready to use.
"""

            # ========================================
            # CREATIVE INSTAGRAM DM PROMPT
            # ========================================
            instagram_dm_prompt = f"""
You are "Alex Rivera," sending an Instagram DM to {practice_name}.

CREATIVE MISSION: Craft a DM that doesn't get ignored or blocked.

PRACTICE CONTEXT:
- Practice: {practice_name} 
- Specialty: {specialty}
- Location: {location}

CREATIVE STRATEGY:
- Compliment their Instagram presence authentically
- Create curiosity about market trends in {specialty}
- Use 1-2 emojis maximum
- Sound like a real person, not a bot
- Keep it under 40 words total

PSYCHOLOGICAL HOOKS:
- "Noticed something interesting about {specialty} practices in {location}..."
- "Other top-rated practices are shifting their approach to..."
- "There's a trend in {specialty} that's creating huge opportunity..."

GENERATE AN INSTAGRAM DM THAT:
1. Opens with a genuine compliment or observation
2. Creates curiosity about industry insights
3. Suggests low-pressure connection (coffee, quick call)
4. Uses 1-2 relevant emojis
5. Feels like a human conversation

TONE: Friendly, curious, professional-but-not-stuffy

OUTPUT: A short, engaging DM ready to send.
"""

            # ========================================
            # Generate all three outreach types with creative approach
            # ========================================
            system_msg = "You are a creative medical device consultant who specializes in crafting outreach that actually gets responses. You avoid corporate jargon and focus on creating genuine curiosity and value."
            
            # Generate creative email
            email_response = self.call_ai(email_prompt, system_msg, max_tokens=400, temperature=0.8)
            email_subject = ""
            email_body = ""
            
            if email_response:
                if 'SUBJECT:' in email_response and 'BODY:' in email_response:
                    parts = email_response.split('BODY:', 1)
                    email_subject = parts[0].replace('SUBJECT:', '').strip()
                    email_body = parts[1].strip()
                else:
                    email_body = email_response.strip()
            
            # Generate creative cold call script
            cold_call_response = self.call_ai(cold_call_prompt, system_msg, max_tokens=300, temperature=0.7)
            cold_call = cold_call_response.strip() if cold_call_response else ""
            
            # Generate creative Instagram DM
            instagram_response = self.call_ai(instagram_dm_prompt, system_msg, max_tokens=200, temperature=0.8)
            instagram = instagram_response.strip() if instagram_response else ""
            
            # Validation
            if not email_body:
                logger.warning("Creative AI outreach generation incomplete, using template-based fallback")
                return self.generate_outreach_template_based(practice_data, specialty, pain_analysis)
            
            logger.info(f"🎨 CREATIVE AI outreach generated for {practice_name} using {selected_framework} framework")
            
            return {
                'outreachColdCall': cold_call if cold_call else None,
                'outreachInstagram': instagram if instagram else None,
                'outreachEmail': email_body if email_body else None,
                'outreachEmailSubject': email_subject if email_subject else None,
                'talking_points': pain_analysis.get('pain_points', [])[:3],
                'ai_generated': True,
                'creative_framework': selected_framework
            }
            
        except Exception as e:
            logger.error(f"Creative AI outreach generation failed: {str(e)}, using template-based fallback")
            return self.generate_outreach_template_based(practice_data, specialty, pain_analysis)

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
        Intelligently discover high-value pages by scraping the homepage for navigation links.
        Returns a prioritized list of URLs to scrape, reducing 404 errors.
        """
        if not base_url:
            return []

        logger.info(f"Intelligently discovering site pages from homepage: {base_url}")
        # Use a set to handle duplicates automatically; homepage is always included
        urls_to_scrape = {base_url}

        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Keywords to identify important pages
            page_keywords = [
                'about', 'team', 'staff', 'doctor', 'provider', 'physician',
                'service', 'treatment', 'procedure',
                'contact', 'location', 'visit', 'book', 'appointment'
            ]

            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text().strip().lower()

                # Filter out irrelevant links
                if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                    continue

                # Check if link text or URL path seems relevant
                href_path = urlparse(href).path.lower()
                if any(keyword in link_text or keyword in href_path for keyword in page_keywords):
                    full_url = urljoin(base_url, href)
                    
                    # Basic validation: ensure it's from the same domain and not a file download
                    parsed_url = urlparse(full_url)
                    if parsed_url.netloc == urlparse(base_url).netloc and not any(full_url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx']):
                        urls_to_scrape.add(full_url)

        except requests.RequestException as e:
            logger.warning(f"Could not scrape homepage {base_url} to discover pages: {e}. Falling back to guessing common paths.")
            # Fallback to a smaller, safer list of common paths
            common_paths = ['', '/about', '/services', '/team', '/contact']
            return [urljoin(base_url, path) for path in common_paths]

        # Convert set to list and ensure homepage is first
        final_urls = list(urls_to_scrape)
        if base_url in final_urls:
            final_urls.remove(base_url)
            final_urls.insert(0, base_url)
        
        logger.info(f"Discovered {len(final_urls)} relevant pages to check for {base_url}.")
        return final_urls[:7]  # Limit to 7 URLs max
    
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
            staff_indicators = soup.find_all(string=re.compile(
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
    
    # --------------------------------------------------------------------------
    # --- SURGICAL FIX: Replacing scrape_website_deep with concurrent version ---
    # --------------------------------------------------------------------------

    def scrape_website_deep(self, base_url: str, max_pages: int = 5) -> Dict[str, any]:
        """
        Intelligently scrape multiple pages from a medical practice website CONCURRENTLY
        Returns aggregated data from all pages
        """
        if self.demo_mode:
            logger.info(f"DEMO MODE: Using mock website data for {base_url}")
            return self.get_mock_website_data(base_url)
        
        if not base_url:
            return {
                'title': 'Not Available - No Website',
                'description': 'Not Available - No Website',
                'services': [], 'social_links': [], 'staff_count': 0, 'emails': [],
                'contact_names': [], 'additional_phones': [], 'contact_form_url': '', 'team_members': [],
                'website_text': 'Not Available - No Website'
            }
        
        if not self.check_robots_txt(base_url):
            logger.warning(f"Robots.txt disallows scraping: {base_url}")
            # Fall back to single page with smart detection
            return self.scrape_website_smart(base_url)
        
        try:
            # Step 1: Discover pages to scrape
            pages_to_scrape = self.discover_site_pages(base_url)[:max_pages]
            logger.info(f"Deep scrape (concurrent) found {len(pages_to_scrape)} pages to check for {base_url}")

            # Step 2: Scrape all pages concurrently
            # Use the synchronous wrapper that handles the event loop
            results = self.scrape_multiple_sync_wrapper(pages_to_scrape, max_concurrent=10) # Limit to 10 concurrent per site

            # Step 3: Aggregate data from all scraped pages
            all_services = set()
            all_social_links = set()
            all_emails = set()
            all_contact_names = set()
            all_additional_phones = set()
            all_team_members = set()
            all_staff_counts = []
            all_text_content = []
            
            homepage_result = results[0] if results else {} # Homepage is always first
            title = homepage_result.get('title', 'Not Available')
            description = homepage_result.get('description', 'Not Available')
            contact_form_url = homepage_result.get('contact_form_url', '') # Get from homepage

            for page_data in results:
                if not page_data or page_data.get('title') == 'Error':
                    continue # Skip failed pages
                    
                all_services.update(page_data.get('services', []))
                all_social_links.update(page_data.get('social_links', []))
                all_emails.update(page_data.get('emails', []))
                all_contact_names.update(page_data.get('contact_names', []))
                all_additional_phones.update(page_data.get('additional_phones', []))
                all_team_members.update(page_data.get('team_members', []))
                all_staff_counts.append(page_data.get('staff_count', 0))
                all_text_content.append(page_data.get('text', ''))

            # Aggregate staff count (use the max found)
            staff_count = max(all_staff_counts) if all_staff_counts else 0
            
            logger.info(f"Deep scrape complete for {base_url}: {len(results)} pages, {len(all_services)} services, {len(all_emails)} emails")
            
            return {
                'title': title,
                'description': description,
                'services': list(all_services),
                'social_links': list(all_social_links),
                'staff_count': staff_count,
                'emails': list(all_emails)[:5], # Limit emails
                'contact_names': list(all_contact_names)[:5], # Limit contacts
                'additional_phones': list(all_additional_phones)[:3],
                'contact_form_url': contact_form_url,
                'team_members': list(all_team_members)[:10],
                'website_text': " ".join(all_text_content)
            }
            
        except Exception as e:
            logger.error(f"Error in deep scrape for {base_url}: {str(e)}")
            # Fall back to single-page scrape with smart detection
            return self.scrape_website_smart(base_url)

    # --------------------------------------------------------------------------
    # --- END OF SURGICAL FIX ---
    # --------------------------------------------------------------------------
    
    def _analyze_services_ai(self, website_text: str, specialty: str) -> Dict[str, str]:
        """
        NEW (V4.0): AI-powered "engine" for scoring.
        Analyzes website text to classify the prospect's opportunity type.
        """
        if not self.ai_enabled or not self.openai_client:
            raise Exception("AI is not enabled.")
        
        if not website_text:
            logger.debug("No website text to analyze, skipping AI analysis.")
            return {"classification": "Low Priority"}
        
        # Truncate text to avoid excessive token usage
        max_text_len = 15000  # Approx 4k tokens
        if len(website_text) > max_text_len:
            logger.debug(f"Truncating website text from {len(website_text)} to {max_text_len} chars for AI analysis")
            website_text = website_text[:max_text_len]

        system_message = (
            "You are an expert medical device sales analyst. Your job is to read a practice's "
            "website text and classify its market opportunity based on its service menu. "
            "You must return only a single, valid JSON object with your analysis."
        )
        
        prompt = f"""
Analyze the medical practice data below.

**Detected Specialty:** `{specialty}`
**Website Text:** ```{website_text}```

---

First, using the **Detected Specialty**, determine which prospect bucket to use:

* **If Specialty is `familypractice`, `obgyn`, or `general`:** Use "Bucket 1: Crossover Market" logic.
* **If Specialty is `medspa`, `plastic_surgery`, or `dermatology`:** Use "Bucket 2: Aesthetic Core Market" logic.

---

### **Bucket 1: Crossover Market Logic**

Look for these Green Flags and Red Flags.

* **Green Flags:** `botox`, `fillers`, `hydrafacial`, `weight loss`, `weight management`, `glp-1's`, `semaglutide`, `HRT`, `Biot-e`, `Ideal protein`, or any general mention of "aesthetic services".
* **Red Flags:** `urgent care`, `pediatrics`, "corporate multi-location", or a large, existing menu of aesthetic *devices* (like CoolSculpting, Emsculpt, lasers).

**Analysis for Bucket 1:**
* If you find **Green Flags** and **No Red Flags**, classify as **"High-Potential Crossover"**.
* If you find **Red Flags**, classify as **"Low Priority"**.
* If you find no flags, classify as **"Low Priority"**.

---

### **Bucket 2: Aesthetic Core Market Logic**

Look for the business's age (e.g., "since 1999", "20 years") and its device menu. Classify it into ONE of these types.

* **Type 1: "New & Growing" (Green Flag):**
    * **Profile:** Has **No Devices** (only injectables/facials) AND **no signs of being old**.
    * **Conclusion:** This is a prime target for a first multi-platform device.

* **Type 2: "Face, No Body" (Green Flag):**
    * **Profile:** Has *face/skin* devices (IPL, Photofacial, Skin Resurfacing) but is **missing a body contouring solution**.
    * **Conclusion:** This is a "gap selling" opportunity for a body device.

* **Type 3: "Body, No Face" (Green Flag):**
    * **Profile:** Has *body* devices (CoolSculpting, Emsculpt) but is **missing a face/skin platform** (IPL, Skin Resurfacing).
    * **Conclusion:** This is a "gap selling" opportunity for a face/skin platform.

* **Type 4: "The Laggard" (Red Flag):**
    * **Profile:** Has **No Devices** (only injectables/facials) BUT the website shows they have been **open for many years**.
    * **Conclusion:** This is a "Laggard" who is resistant to new technology.

* **Type 5: "The Fully Saturated" (Red Flag):**
    * **Profile:** Has a full menu of *new, modern, competing devices* for **both** face AND body.
    * **Conclusion:** There is no gap to sell into.

---

### **REQUIRED JSON OUTPUT FORMAT**

Based on your analysis, return a single JSON object in this *exact* format:

```json
{{
  "classification": "..."
}}
```

(Possible values for "classification": "High-Potential Crossover", "New & Growing", "Face, No Body", "Body, No Face", "The Laggard", "The Fully Saturated", or "Low Priority")
"""
        
        response_text = self.call_ai(prompt, system_message, max_tokens=150, temperature=0.1)
        
        if not response_text:
            raise Exception("AI returned an empty response.")
        
        # Extract JSON from the response
        try:
            # Find the JSON block
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise Exception(f"No JSON object found in AI response: {response_text}")
            
            json_str = json_match.group(0)
            json_data = json.loads(json_str)
            
            if 'classification' not in json_data:
                raise Exception(f"JSON response missing 'classification' key: {json_str}")
                
            return json_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode AI JSON response: {e}")
            logger.error(f"Raw AI response: {response_text}")
            raise Exception(f"AI returned malformed JSON: {response_text}")
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            raise

    def _calculate_medium_smart_score(self, scores: Dict, practice_data: Dict, specialty: str, all_text: str) -> Dict:
        """
        NEW (V4.0): "Medium-Smart" rule-based fallback logic.
        This runs if the AI analysis fails. It's much smarter than the original
        V3 logic and uses our tiered, specialty-aware rules.
        """
        logger.info(f"Running 'Medium-Smart' fallback scoring for {practice_data.get('name')}")
        
        services = practice_data.get('services', [])
        services_text = ' '.join(services).lower()
        full_text = f"{all_text} {services_text}".lower()
        
        # ---
        # 1. DEFINE KEYWORD TIERS
        # ---
        
        # Crossover Green Flags
        crossover_flags = ['botox', 'fillers', 'hydrafacial', 'weight loss', 'glp-1', 'semaglutide', 'hrt', 'bio-t', 'ideal protein']
        
        # Core Market Red Flags
        red_flag_devices = ['coolsculpting elite', 'emsculpt neo', 'sciton mjoule', 'lumenis m22', 'inmode optimas']
        red_flag_practice = ['urgent care', 'pediatrics']
        
        # Core Market Tiers
        tier_1_devices = ['coolsculpting', 'emsculpt', 'morpheus8', 'sciton', 'lumenis', 'ultherapy', 'fraxel']
        tier_2_face = ['ipl', 'photofacial', 'skin resurfacing', 'laser hair removal']
        tier_3_body = ['body contouring', 'cellulite reduction', 'body sculpting']
        tier_4_injectables = ['botox', 'juvederm', 'restylane', 'fillers']
        
        # ---
        # 2. APPLY BUCKET LOGIC
        # ---
        
        if specialty in ['familypractice', 'obgyn', 'general']:
            # --- BUCKET 1: Crossover Market Fallback ---
            
            if any(kw in full_text for kw in red_flag_practice):
                scores['specialty_match'] = 2  # Disqualify
                scores['aesthetic_services'] = 0
            elif any(kw in full_text for kw in crossover_flags):
                scores['specialty_match'] = 18 # Great match!
                scores['aesthetic_services'] = 10 # Has basics
                scores['competing_devices'] = 0  # Opportunity
            else:
                scores['specialty_match'] = 5  # Low priority
                scores['aesthetic_services'] = 0
        
        else:
            # --- BUCKET 2: Aesthetic Core Market Fallback ---
            has_tier_1 = any(kw in full_text for kw in tier_1_devices)
            has_tier_2_face = any(kw in full_text for kw in tier_2_face)
            has_tier_3_body = any(kw in full_text for kw in tier_3_body)
            has_tier_4_injectables = any(kw in full_text for kw in tier_4_injectables)
            
            is_laggard = any(kw in full_text for kw in ['since 199', '20 years']) and not (has_tier_1 or has_tier_2_face or has_tier_3_body)
            is_saturated = any(kw in full_text for kw in red_flag_devices) or (has_tier_1 and has_tier_2_face and has_tier_3_body)

            if is_saturated:
                scores['specialty_match'] = 2
                scores['aesthetic_services'] = 15
                scores['competing_devices'] = 10 # Saturated
            elif is_laggard:
                scores['specialty_match'] = 2
                scores['aesthetic_services'] = 2
                scores['competing_devices'] = 0
            elif has_tier_2_face and not has_tier_3_body and not has_tier_1:
                # "Face, No Body"
                scores['specialty_match'] = 20
                scores['aesthetic_services'] = 15 # Has face
                scores['competing_devices'] = 8  # Proven buyer
            elif (has_tier_3_body or has_tier_1) and not has_tier_2_face:
                # "Body, No Face"
                scores['specialty_match'] = 20
                scores['aesthetic_services'] = 15 # Has body
                scores['competing_devices'] = 8  # Proven buyer
            elif has_tier_4_injectables and not (has_tier_1 or has_tier_2_face or has_tier_3_body):
                # "New & Growing"
                scores['specialty_match'] = 20
                scores['aesthetic_services'] = 5 # Basics only
                scores['competing_devices'] = 0  # Opportunity
            else:
                # Default for Core
                scores['specialty_match'] = 10
                scores['aesthetic_services'] = 5
                scores['competing_devices'] = 2
                
        return scores

    def calculate_ai_score(self, practice_data: Dict) -> Tuple[int, Dict[str, int], str, str]:
        """
        Calculate AI-powered scoring with specialty-specific weights (V4.0 LOGIC)
        
        Returns:
            (total_score, score_breakdown, detected_specialty, ai_classification)
        """
        
        # STEP 1: Detect specialty
        specialty = self.detect_specialty_ai(practice_data)
        
        # STEP 2: Get specialty-specific config (or use default)
        if specialty in self.specialty_scoring:
            config = self.specialty_scoring[specialty]
            logger.info(f"🎯 Using {specialty.upper()} scoring profile")
        else:
            config = self.specialty_scoring['dermatology']
            logger.info(f"Using DEFAULT scoring profile for {specialty}")
        
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
        website_text = practice_data.get('website_text', '')
        all_text = f"{practice_name} {practice_desc} {address}"
        
        # ═══════════════════════════════════════════════════════════
        # 1. V4.0 Service & Opportunity Scoring (AI or Medium-Smart Fallback)
        # This replaces specialty_match, aesthetic_services, and competing_devices
        # ═══════════════════════════════════════════════════════════
        ai_classification = "Fallback"
        try:
            if self.ai_enabled:
                analysis = self._analyze_services_ai(website_text, specialty)
                ai_classification = analysis.get("classification", "Fallback")
                logger.info(f"🤖 AI Service Analysis Classification: {ai_classification}")

                # Translate classification into scores
                if ai_classification == "High-Potential Crossover":
                    scores['specialty_match'] = 20
                    scores['aesthetic_services'] = 12
                    scores['competing_devices'] = 0
                elif ai_classification == "New & Growing":
                    scores['specialty_match'] = 20
                    scores['aesthetic_services'] = 5
                    scores['competing_devices'] = 0
                elif ai_classification == "Face, No Body":
                    scores['specialty_match'] = 20
                    scores['aesthetic_services'] = 15
                    scores['competing_devices'] = 8
                elif ai_classification == "Body, No Face":
                    scores['specialty_match'] = 20
                    scores['aesthetic_services'] = 15
                    scores['competing_devices'] = 8
                elif ai_classification in ["The Laggard", "The Fully Saturated", "Low Priority"]:
                    scores['specialty_match'] = 2
                    scores['aesthetic_services'] = 2
                    scores['competing_devices'] = 10 if ai_classification == "The Fully Saturated" else 0
                else:
                    raise Exception(f"AI returned unknown classification: {ai_classification}")
            else:
                raise Exception("AI not enabled, using fallback.")
                
        except Exception as e:
            logger.warning(f"AI service analysis failed ({e}), using 'Medium-Smart' fallback scoring.")
            scores = self._calculate_medium_smart_score(scores, practice_data, specialty, all_text)

        # ═══════════════════════════════════════════════════════════
        # 2. Decision-Making Autonomy (20 points) 🔥 CRITICAL
        # ═══════════════════════════════════════════════════════════
        staff_count = practice_data.get('staff_count', 0)
        
        hospital_indicators = [
            'hospital', 'medical center', 'health system', 'healthcare system',
            'affiliated', 'network', 'regional medical', 'university medical'
        ]
        has_hospital_affiliation = any(indicator in all_text for indicator in hospital_indicators)
        
        corporate_indicators = [
            'corporate', 'chain', 'franchise', 'national', 'locations',
            'branches', 'group practice', 'associates'
        ]
        is_corporate = any(indicator in all_text for indicator in corporate_indicators)
        
        if staff_count == 0 or staff_count == 1:
            autonomy_score = 20
        elif staff_count == 2:
            autonomy_score = 18
        elif staff_count <= 4:
            autonomy_score = 15
        elif staff_count <= 6:
            autonomy_score = 10
        elif staff_count <= 10:
            autonomy_score = 5
        else:
            autonomy_score = 2
        
        if has_hospital_affiliation:
            autonomy_score = max(0, autonomy_score - 10)
        
        if is_corporate:
            autonomy_score = max(0, autonomy_score - 8)
        
        scores['decision_autonomy'] = autonomy_score
        
        # ═══════════════════════════════════════════════════════════
        # 3. Social Media Activity (10 points)
        # ═══════════════════════════════════════════════════════════
        social_links = practice_data.get('social_links', [])
        scores['social_activity'] = min(len(social_links) * 3, 10)
        
        # ═══════════════════════════════════════════════════════════
        # 4. Reviews & Rating (10 points)
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
        # 5. Search Visibility (10 points)
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
        # 6. Financial Indicators (10 points)
        # ═══════════════════════════════════════════════════════════
        affluent_indicators = [
            'hills', 'park', 'lake', 'estates', 'plaza', 'center',
            'avenue', 'boulevard', 'suite'
        ]
        is_affluent_area = any(indicator in address for indicator in affluent_indicators)
        
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
        # 7. Weight Loss Services (5 points)
        # ═══════════════════════════════════════════════════════════
        weight_keywords = [
            'weight loss', 'medical weight', 'hormone therapy', 'iv therapy',
            'body contouring', 'fat reduction', 'inch loss'
        ]
        
        weight_matches = sum(1 for keyword in weight_keywords if keyword in all_text)
        scores['weight_loss_services'] = min(weight_matches * 2, 5)
        
        # ═══════════════════════════════════════════════════════════
        # 8. Specialty-Specific Keyword Bonus (up to +10 points)
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
        
        return total_score, scores, specialty, ai_classification
    
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
            'title': 'Not Available',
            'website_text': ''
        }
        
        # Scrape website if available (using deep multi-page scraping)
        if practice_record['website']:
            # Check if website is blacklisted before scraping
            is_blacklisted = False
            blacklist_reason = None
            
            if self.blacklist_manager:
                is_blacklisted, blacklist_reason = self.blacklist_manager.is_blacklisted(practice_record['website'])
            
            if is_blacklisted:
                logger.info(f"⚫ SKIPPING blacklisted website: {practice_record['website']} - Reason: {blacklist_reason}")
                # Set minimal data for blacklisted sites
                website_data = {
                    'title': 'Skipped - Blacklisted Site',
                    'description': f'Site filtered by blacklist: {blacklist_reason}',
                    'services': [],
                    'social_links': [],
                    'staff_count': 0,
                    'emails': [],
                    'contact_names': [],
                    'additional_phones': [],
                    'contact_form_url': '',
                    'team_members': []
                }
                practice_record.update(website_data)
            else:
                logger.info(f"Deep scraping website: {practice_record['website']}")
                # --- THIS NOW USES THE FIXED CONCURRENT VERSION ---
                website_data = self.scrape_website_deep(practice_record['website'], max_pages=5)
                practice_record.update(website_data)
        
        # Calculate AI score with specialty detection
        ai_score, score_breakdown, specialty, ai_prospect_class = self.calculate_ai_score(practice_record)
        
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
            'ai_prospect_class': ai_prospect_class,
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
            'emails', 'contact_names',  # Contact Intelligence fields
            'ai_score', 'ai_prospect_class', 'confidence_level', 'primary_device_rec', 'device_rationale',
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
                elif col in ['services', 'social_links', 'pain_points', 'competing_services', 'talking_points', 'emails', 'contact_names']:
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
    
    # ============================================================================
    # CONCURRENT SCRAPING METHODS (Added for Performance)
    # ============================================================================
    
    # --------------------------------------------------------------------------
    # --- SURGICAL FIX: Upgrading scrape_website_async with all extractors ---
    # --------------------------------------------------------------------------
    async def scrape_website_async(self, session: aiohttp.ClientSession, url: str) -> Dict[str, any]:
        """
        Async version of scrape_website for concurrent scraping
        Falls back to sync version on error
        """
        if not url or self.demo_mode:
            # Return full dict structure even for demo
            return self.get_mock_website_data(url) if self.demo_mode else {
                'title': 'Not Available - No Website', 'description': 'Not Available - No Website',
                'services': [], 'social_links': [], 'staff_count': 0, 'emails': [],
                'contact_names': [], 'additional_phones': [], 'contact_form_url': '', 'team_members': [],
                'text': 'Not Available - No Website'
            }
        
        if not self.check_robots_txt(url):
            logger.warning(f"Robots.txt disallows scraping: {url}")
            return {
                'title': 'Not Available - Restricted',
                'description': 'Not Available - Restricted',
                'services': [],
                'social_links': [],
                'staff_count': 0,
                'emails': [],
                'contact_names': [],
                'additional_phones': [],
                'contact_form_url': '',
                'team_members': [],
                'text': 'Not Available - Restricted'
            }
        
        try:
            await asyncio.sleep(random.uniform(0.5, 1.0))
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                html = await response.text()
                
                soup = BeautifulSoup(html, 'html.parser')
                
                data = {
                    'title': '',
                    'description': '',
                    'services': [],
                    'social_links': [],
                    'staff_count': 0,
                    'emails': [],
                    'contact_names': [],
                    'additional_phones': [],
                    'contact_form_url': '',
                    'team_members': [],
                    'text': ''
                }
                
                # Extract title
                if soup.title and soup.title.string:
                    data['title'] = soup.title.string.strip()
                else:
                    data['title'] = 'Not Available'
                
                # Extract description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    data['description'] = meta_desc.get('content', '').strip()
                else:
                    og_desc = soup.find('meta', attrs={'property': 'og:description'})
                    if og_desc and og_desc.get('content'):
                        data['description'] = og_desc.get('content', '').strip()
                    else:
                        data['description'] = 'Not Available'
                
                data['text'] = soup.get_text()
                
                # Extract services
                data['services'] = self._extract_services_from_soup(soup)
                
                # Extract social links
                data['social_links'] = self._extract_social_links_from_soup(soup, url)
                
                # Extract emails
                data['emails'] = self.extract_emails(soup, url)
                
                # Extract contact names
                data['contact_names'] = self.extract_contact_names(soup, '', url)
                
                # Count staff
                data['staff_count'] = self._count_staff_from_soup(soup)
                
                # --- ADDED FOR COMPLETENESS ---
                data['additional_phones'] = self.extract_additional_phones(soup, '', url)
                data['contact_form_url'] = self.extract_contact_form_url(soup, url)
                data['team_members'] = self.extract_team_members(soup, url)
                # --- END OF ADDITION ---
                
                logger.info(f"✓ Async scraped {url}: {len(data['services'])} services, {len(data['emails'])} emails")
                return data
                
        except asyncio.TimeoutError:
            logger.warning(f"Async timeout for {url}, falling back to sync")
            return self.scrape_website(url) # Fallback returns the full dict
        except Exception as e:
            logger.warning(f"Async error for {url}: {str(e)}, falling back to sync")
            return self.scrape_website(url) # Fallback returns the full dict

    # --------------------------------------------------------------------------
    # --- END OF SURGICAL FIX ---
    # --------------------------------------------------------------------------
    
    async def scrape_multiple_concurrent(self, urls: List[str], max_concurrent: int = 50) -> List[Dict[str, any]]:
        """
        Scrape multiple URLs concurrently - MASSIVE performance improvement
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests (default 5)
        
        Returns:
            List of scraped data in same order as URLs
        
        Example:
            # Instead of 10 seconds sequential, this takes ~2 seconds
            urls = [url1, url2, url3, url4, url5]
            results = await prospector.scrape_multiple_concurrent(urls)
        """
        if not urls:
            return []
        
        connector = TCPConnector(
            limit=max_concurrent,
            limit_per_host=2,
            ttl_dns_cache=300
        )
        
        timeout = ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.session.headers.get('User-Agent', 'FathomProspector/3.0')}
        ) as session:
            tasks = [self.scrape_website_async(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error scraping {urls[i]}: {str(result)}")
                    processed_results.append({
                        'title': 'Error',
                        'description': 'Error',
                        'services': [],
                        'social_links': [],
                        'staff_count': 0,
                        'emails': [],
                        'contact_names': [],
                        'additional_phones': [],
                        'contact_form_url': '',
                        'team_members': []
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def scrape_multiple_sync_wrapper(self, urls: List[str], max_concurrent: int = 50) -> List[Dict[str, any]]:
        """
        Synchronous wrapper for concurrent scraping
        Use this in existing code - it handles the async event loop for you
        
        Example:
            prospector = FathomProspector()
            urls = [result['website'] for result in results if result.get('website')]
            scraped_data = prospector.scrape_multiple_sync_wrapper(urls)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Event loop running, falling back to sequential")
                return [self.scrape_website(url) for url in urls]
            else:
                return loop.run_until_complete(self.scrape_multiple_concurrent(urls, max_concurrent))
        except RuntimeError:
            return asyncio.run(self.scrape_multiple_concurrent(urls, max_concurrent))
        except Exception as e:
            logger.error(f"Concurrent scraping failed: {str(e)}, falling back")
            return [self.scrape_website(url) for url in urls]
    
    # Helper methods for async scraping
    def _extract_services_from_soup(self, soup) -> List[str]:
        """Extract services from BeautifulSoup object"""
        services = []
        service_keywords = [
            'botox', 'filler', 'laser', 'facial', 'peel', 'dermabrasion',
            'microneedling', 'prp', 'coolsculpting', 'body contouring',
            'skin tightening', 'hair removal', 'vein treatment', 'skin care',
            'anti-aging', 'rejuvenation', 'aesthetic', 'cosmetic'
        ]
        
        text_content = soup.get_text().lower()
        for keyword in service_keywords:
            if keyword in text_content:
                services.append(keyword.title())
        
        return list(set(services))
    
    def _extract_social_links_from_soup(self, soup, base_url: str) -> List[str]:
        """Extract social media links"""
        social_links = []
        social_domains = ['facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com', 'youtube.com']
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(domain in href for domain in social_domains):
                social_links.append(href)
        
        return list(set(social_links))
    
    def _count_staff_from_soup(self, soup) -> int:
        """Count staff mentions"""
        staff_keywords = ['dr.', 'doctor', 'physician', 'surgeon', 'rn', 'pa', 'np', 'provider']
        text_content = soup.get_text().lower()
        
        staff_count = 0
        for keyword in staff_keywords:
            staff_count += text_content.count(keyword)
        
        return min(staff_count, 20)

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
            logger.info(f"Searching for: '{keyword}'")
            
            places = self.google_places_search(keyword, location, radius * 1000)
            
            # PARALLEL PROCESSING: Process up to 50 prospects concurrently
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            places_to_process = places[:max_results]
            logger.info(f"Processing {len(places_to_process)} places in parallel (max 50 concurrent)")
            
            def process_single_place(place):
                """Wrapper for thread pool processing"""
                try:
                    return self.process_practice(place)
                except Exception as e:
                    logger.error(f"Error processing practice: {str(e)}")
                    return None
            
            # Process prospects in parallel with up to 50 workers
            with ThreadPoolExecutor(max_workers=50) as executor:
                future_to_place = {executor.submit(process_single_place, place): place for place in places_to_process}
                
                for future in as_completed(future_to_place):
                    try:
                        processed_practice = future.result()
                        if processed_practice:
                            all_results.append(processed_practice)
                            logger.info(f"Processed: {processed_practice.get('name', 'Unknown')} - Score: {processed_practice.get('ai_score', 0)}")
                    except Exception as e:
                        logger.error(f"Error getting result: {str(e)}")
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
