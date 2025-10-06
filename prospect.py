
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
# === Non-breaking enrichment helpers (booking/financing/devices/social/gallery/careers, JSON-LD) ===
BOOKING_HINTS = [
    "vagaro.com","clients.mindbodyonline.com","boulevard.io","booksy.com",
    "schedulicity.com","square.site","janeapp.com","acuityscheduling.com","calendly.com"
]
FINANCE_HINTS = ["carecredit","patientfi","ally lending","cherry","united medical credit","sunbit"]
VENUS_HINTS = ["venus viva","venus legacy","venus versa","tribella","diamondpolar","mp2","venus concept","venus glow","venus bliss"]
COMP_HINTS = ["inmode","cynosure","cutera","sciton","lutronic","alma","btl","emsculpt","candela","deka"]
WEIGHT_LOSS_HINTS = ["semaglutide","tirzepatide","glp-1","medical weight","weight management","lipo-b","lipotropic","b12 injections","peptides"]
GALLERY_HINTS = ["before and after","results","gallery","case photos","patient results"]
CAREER_HINTS = ["careers","we’re hiring","we're hiring","join our team","apply now","open roles","job openings"]

def _extract_jsonld(soup):
    nodes = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            blob = json.loads(s.string or "{}")
            if isinstance(blob, dict): nodes.append(blob)
            elif isinstance(blob, list): nodes.extend([b for b in blob if isinstance(b, dict)])
        except Exception:
            continue
    return nodes

def _detect_booking_stack(soup):
    urls = set()
    for tag in soup.find_all(["a","script","iframe","link"]):
        u = (tag.get("href") or tag.get("src") or "") or ""
        u = u.lower()
        if any(k in u for k in BOOKING_HINTS):
            urls.add(u)
    return list(urls)

def _text_has_any(text, keys):
    s = (text or "").lower()
    return any(k in s for k in keys)

def _social_from_jsonld(nodes):
    links = []
    for n in nodes:
        same_as = n.get("sameAs")
        if isinstance(same_as, str): links.append(same_as)
        elif isinstance(same_as, list): links.extend([x for x in same_as if isinstance(x, str)])
    # dedupe preserve order
    seen=set(); out=[]
    for l in links:
        if l not in seen:
            seen.add(l); out.append(l)
    return out

from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

# Load environment variables
load_dotenv()

# Configure logging
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
    
    def __init__(self, demo_mode=False, existing_customers_csv=None):
        self.gmaps_key = os.getenv('GOOGLE_PLACES_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        
        if not self.gmaps_key:
            logger.warning('GOOGLE_PLACES_API_KEY not found - switching to demo mode')
            demo_mode = True
        
        if not self.gemini_key:
            logger.warning('GEMINI_API_KEY not found - AI features disabled')
        
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
        
        if not demo_mode:
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
            staff_indicators = soup.find_all(string=re.compile(
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
        Detect the primary specialty of a practice
        Returns: 'dermatology', 'plastic_surgery', 'obgyn', 'medspa', 'familypractice', or 'general'
        """
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
            
            # JSON-LD extraction
            try:
                jsonld_nodes = _extract_jsonld(soup)
            except Exception:
                jsonld_nodes = []
            social_links_jsonld = _social_from_jsonld(jsonld_nodes)

            
            # Extract text content
            text_content = soup.get_text(" ", strip=True).lower()
            
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
            
            # Merge JSON-LD social links
            try:
                for l in social_links_jsonld:
                    if l not in social_links:
                        social_links.append(l)
            except Exception:
                pass

            # Lightweight detectors
            booking_platforms = _detect_booking_stack(soup)
            financing_present = _text_has_any(text_content, FINANCE_HINTS)
            venus_present = _text_has_any(text_content, VENUS_HINTS)
            competitor_present = _text_has_any(text_content, COMP_HINTS)
            weight_loss_present = _text_has_any(text_content, WEIGHT_LOSS_HINTS)
            gallery_present = _text_has_any(text_content, GALLERY_HINTS)
            hiring_present = _text_has_any(text_content, CAREER_HINTS)
            # Merge JSON-LD social links
            try:
                for l in social_links_jsonld:
                    if l not in social_links:
                        social_links.append(l)
            except Exception:
                pass

            # Lightweight detectors
            booking_platforms = _detect_booking_stack(soup)
            financing_present = _text_has_any(text_content, FINANCE_HINTS)
            venus_present = _text_has_any(text_content, VENUS_HINTS)
            competitor_present = _text_has_any(text_content, COMP_HINTS)
            weight_loss_present = _text_has_any(text_content, WEIGHT_LOSS_HINTS)
            gallery_present = _text_has_any(text_content, GALLERY_HINTS)
            hiring_present = _text_has_any(text_content, CAREER_HINTS)

            return {
                'services': services_found,
                'staff_mentions': list(set(str(s).strip() for s in staff_indicators if len(str(s).strip()) > 5)),
                'text': text_content,
                'social_links': social_links,
                'booking_platforms': booking_platforms,
                'financing_present': financing_present,
                'venus_present': venus_present,
                'competitor_present': competitor_present,
                'weight_loss_present': weight_loss_present,
                'gallery_present': gallery_present,
                'hiring_present': hiring_present,
                'booking_platforms': booking_platforms,
                'financing_present': financing_present,
                'venus_present': venus_present,
                'competitor_present': competitor_present,
                'weight_loss_present': weight_loss_present,
                'gallery_present': gallery_present,
                'hiring_present': hiring_present
            }
            
        except Exception as e:
            logger.debug(f"Error scraping {url}: {str(e)}")
            # Merge JSON-LD social links
            try:
                for l in social_links_jsonld:
                    if l not in social_links:
                        social_links.append(l)
            except Exception:
                pass

            # Lightweight detectors
            booking_platforms = _detect_booking_stack(soup)
            financing_present = _text_has_any(text_content, FINANCE_HINTS)
            venus_present = _text_has_any(text_content, VENUS_HINTS)
            competitor_present = _text_has_any(text_content, COMP_HINTS)
            weight_loss_present = _text_has_any(text_content, WEIGHT_LOSS_HINTS)
            gallery_present = _text_has_any(text_content, GALLERY_HINTS)
            hiring_present = _text_has_any(text_content, CAREER_HINTS)
            # Merge JSON-LD social links
            try:
                for l in social_links_jsonld:
                    if l not in social_links:
                        social_links.append(l)
            except Exception:
                pass

            # Lightweight detectors
            booking_platforms = _detect_booking_stack(soup)
            financing_present = _text_has_any(text_content, FINANCE_HINTS)
            venus_present = _text_has_any(text_content, VENUS_HINTS)
            competitor_present = _text_has_any(text_content, COMP_HINTS)
            weight_loss_present = _text_has_any(text_content, WEIGHT_LOSS_HINTS)
            gallery_present = _text_has_any(text_content, GALLERY_HINTS)
            hiring_present = _text_has_any(text_content, CAREER_HINTS)

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
            return {
                'source_mode': 'api_only',
                'title': 'Not Available - Restricted',
                'description': 'Not Available - Restricted',
                'services': [],
                'social_links': [],
                'staff_count': 0
            }
        
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
            any_financing=False
            any_venus=False
            any_competitor=False
            any_weight_loss=False
            any_gallery=False
            any_hiring=False
            all_booking=set()
            
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
                'staff_count': staff_count,
                'booking_platforms': list(all_booking),
                'financing_present': any_financing,
                'venus_present': any_venus,
                'competitor_present': any_competitor,
                'weight_loss_present': any_weight_loss,
                'gallery_present': any_gallery,
                'hiring_present': any_hiring
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
        specialty = self.detect_specialty(practice_data)
        
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
        # Breadth/video-forward bonus (non-breaking)
        platforms = {
            'instagram': any('instagram.com' in u for u in social_links),
            'tiktok': any('tiktok.com' in u for u in social_links),
            'youtube': any(('youtube.com' in u) or ('youtu.be' in u) for u in social_links),
            'facebook': any('facebook.com' in u for u in social_links),
            'pinterest': any('pinterest.' in u for u in social_links),
            'linkedin': any('linkedin.com' in u for u in social_links),
            'x': any(('twitter.com' in u) or ('x.com' in u) for u in social_links),
        }
        bonus = 0
        if sum(platforms.values()) >= 2:
            bonus += 1
        if platforms.get('tiktok') or platforms.get('youtube'):
            bonus += 1
        scores['social_activity'] = min(10, scores['social_activity'] + bonus)

        
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
        
        
        # === Non-breaking detector boosts ===
        if practice_data.get('booking_platforms'):
            scores['decision_autonomy'] = min(10, scores['decision_autonomy'] + 1)
            scores['search_visibility'] = min(10, scores['search_visibility'] + 1)
        if practice_data.get('financing_present'):
            scores['financial_indicators'] = min(10, scores['financial_indicators'] + 1)
        if practice_data.get('gallery_present'):
            scores['aesthetic_services'] = min(10, scores['aesthetic_services'] + 1)
            scores['search_visibility'] = min(10, scores['search_visibility'] + 1)
        if practice_data.get('weight_loss_present'):
            scores['weight_loss_services'] = min(5, scores['weight_loss_services'] + 1)
        if practice_data.get('hiring_present'):
            scores['financial_indicators'] = min(10, scores['financial_indicators'] + 1)
        # Device brand signals
        combined_text = (practice_data.get('description', '') or '').lower()
        try:
            combined_text += ' ' + ' '.join(practice_data.get('services', []))
        except Exception:
            pass
        if any(k in combined_text for k in VENUS_HINTS) or practice_data.get('venus_present'):
            scores['aesthetic_services'] = min(10, scores['aesthetic_services'] + 1)
        if any(k in combined_text for k in COMP_HINTS) or practice_data.get('competitor_present'):
            scores['competing_devices'] = min(10, scores['competing_devices'] + 1)

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
        
        # Compile final record
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
            'data_completeness': completeness
        }
        
        return final_record
    
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
            'data_completeness'
        ]
        
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
    
    
    def estimate_result_count(self, keywords: str, location: str, radius_km: float) -> int:
        """Heuristic estimator for result count used by API server timeouts (non-breaking)."""
        base = 15
        radius_factor = max(1, min(5, (radius_km or 5) / 5.0))
        kw = (keywords or "").lower()
        richness = 1.0 + 0.3*sum(int(t in kw) for t in ["medspa","clinic","laser","ob/gyn","wellness","derm","spa","aesthetic"])
        try:
            return int(base * radius_factor * max(1.0, richness))
        except Exception:
            return 50
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


# Backward-compatibility alias for API server imports
MedicalProspector = FathomProspector
