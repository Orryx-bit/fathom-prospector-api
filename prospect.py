#!/usr/bin/env python3
"""
Fathom Medical Device Prospecting System
Comprehensive tool for finding and scoring medical practices
Production-Hardened Version 3.1 (Refactored for direct import)
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
        """
        url = f"{self.base_url}{endpoint}"
        params['key'] = self.api_key
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"API request attempt {attempt + 1}/{max_retries}: {endpoint}")
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                status = data.get('status')
                if status in ['OK', 'ZERO_RESULTS']:
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
                logger.debug("Waiting 2 seconds before next page request...")
                time.sleep(2)
            
            try:
                data = self._make_request(endpoint, params)
                results = data.get('results', [])
                all_results.extend(results)
                
                logger.info(f"Fetched {len(results)} results (total so far: {len(all_results)})")
                
                next_page_token = data.get('next_page_token')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching places: {e}")
                break
        
        return {'results': all_results}
    
    def place_details(self, place_id: str, fields: List[str]) -> Optional[dict]:
        """
        Get detailed information for a specific place
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
    
    def __init__(self, api_key=None, demo_mode=False, existing_customers_csv=None):
        self.gmaps_key = os.getenv("GOOGLE_PLACES_API_KEY") or api_key
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.demo_mode = demo_mode

        if not self.gmaps_key:
            logger.warning('GOOGLE_PLACES_API_KEY not found - switching to demo mode')
            self.demo_mode = True
        
        if not self.gemini_key:
            logger.warning('GEMINI_API_KEY not found - AI features disabled')

        if self.demo_mode:
            logger.info("Running in DEMO MODE with mock data")
        else:
            logger.info("Google Places API: ✓ Configured")

        self.existing_customers = set()
        if existing_customers_csv and os.path.exists(existing_customers_csv):
            self.load_existing_customers(existing_customers_csv)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if not self.demo_mode:
            try:
                self.gmaps_api = GooglePlacesAPI(self.gmaps_key)
                if not self.gmaps_api.geocode("Austin, TX"):
                    raise Exception("Geocode test failed")
            except Exception as e:
                logger.warning(f"Google Places API initialization failed: {str(e)} - switching to demo mode")
                self.demo_mode = True
        
        self.search_templates = {
            'med_spa': ['med spa', 'medical spa', 'aesthetic clinic', 'cosmetic clinic'],
            'plastic_surgeon': ['plastic surgeon', 'cosmetic surgeon', 'aesthetic surgeon'],
            'dermatologist': ['dermatologist', 'dermatology clinic', 'skin clinic'],
            'cosmetic_surgeon': ['cosmetic surgery', 'aesthetic surgery', 'beauty surgery'],
            'obgyn': ['obgyn', 'gynecologist', 'women\'s health clinic'],
            'wellness': ['wellness center', 'anti-aging clinic', 'rejuvenation clinic']
        }
        
        self.hospital_patterns = [
            r'\bhospital\b', r'\bmedical center\b', r'\bhealth system\b',
            r'\bhealthcare system\b', r'\bregional medical\b', r'\buniversity medical\b'
        ]
        
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
        
        self.specialty_scoring = {
            'dermatology': {
                'weights': {
                    'specialty_match': 20, 'decision_autonomy': 20, 'aesthetic_services': 15,
                    'competing_devices': 10, 'social_activity': 10, 'reviews_rating': 10,
                    'search_visibility': 10, 'financial_indicators': 10, 'weight_loss_services': 5
                },
                'high_value_keywords': [
                    'laser', 'ipl', 'photorejuvenation', 'hair removal',
                    'botox', 'fillers', 'skin tightening', 'acne treatment'
                ]
            },
            'plastic_surgery': {
                'weights': {
                    'specialty_match': 20, 'decision_autonomy': 20, 'aesthetic_services': 18,
                    'competing_devices': 12, 'social_activity': 10, 'reviews_rating': 8,
                    'search_visibility': 7, 'financial_indicators': 10, 'weight_loss_services': 5
                },
                'high_value_keywords': [
                    'body contouring', 'liposuction', 'coolsculpting',
                    'breast augmentation', 'tummy tuck', 'fat reduction'
                ]
            },
            'obgyn': {
                'weights': {
                    'specialty_match': 20, 'decision_autonomy': 25, 'aesthetic_services': 12,
                    'competing_devices': 8, 'social_activity': 8, 'reviews_rating': 12,
                    'search_visibility': 10, 'financial_indicators': 10, 'weight_loss_services': 8
                },
                'high_value_keywords': [
                    'women\'s wellness', 'aesthetic gynecology', 'vaginal rejuvenation',
                    'hormone therapy', 'postpartum', 'wellness', 'cosmetic gynecology'
                ]
            },
            'medspa': {
                'weights': {
                    'specialty_match': 18, 'decision_autonomy': 22, 'aesthetic_services': 20,
                    'competing_devices': 15, 'social_activity': 12, 'reviews_rating': 8,
                    'search_visibility': 8, 'financial_indicators': 12, 'weight_loss_services': 10
                },
                'high_value_keywords': [
                    'body sculpting', 'coolsculpting', 'emsculpt', 'laser',
                    'botox', 'fillers', 'skin tightening', 'cellulite',
                    'inch loss', 'fat reduction'
                ]
            },
            'familypractice': {
                'weights': {
                    'specialty_match': 18, 'decision_autonomy': 25, 'aesthetic_services': 12,
                    'competing_devices': 8, 'social_activity': 5, 'reviews_rating': 8,
                    'search_visibility': 5, 'financial_indicators': 12, 'weight_loss_services': 20
                },
                'high_value_keywords': [
                    'weight loss', 'glp-1', 'semaglutide', 'tirzepatide', 'wellness', 
                    'longevity', 'preventive care', 'functional medicine', 'integrative medicine', 
                    'anti-aging', 'hormone therapy'
                ]
            }
        }
    
    def load_existing_customers(self, csv_file_path: str):
        """Load existing customers from CSV for exclusion"""
        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loading existing customers from {csv_file_path}")
            
            name_columns = ['name', 'practice_name', 'business_name', 'company_name']
            name_column = next((col for col in name_columns if col.lower() in [c.lower() for c in df.columns]), None)
            
            if name_column is None:
                name_column = df.columns[0]
                logger.warning(f"No standard name column found, using '{name_column}'")
            
            for name in df[name_column].dropna():
                cleaned_name = re.sub(r'\b(llc|inc|corp|ltd|pllc|pa|md|pc)\b', '', str(name).lower().strip()).strip()
                self.existing_customers.add(cleaned_name)
            
            logger.info(f"Loaded {len(self.existing_customers)} existing customers for exclusion")
            
        except Exception as e:
            logger.error(f"Failed to load existing customers CSV: {str(e)}")
    
    def is_existing_customer(self, practice_name: str) -> bool:
        """Check if practice is an existing customer"""
        if not self.existing_customers:
            return False
        
        cleaned_name = re.sub(r'\b(llc|inc|corp|ltd|pllc|pa|md|pc)\b', '', practice_name.lower().strip()).strip()
        
        return any(existing in cleaned_name or cleaned_name in existing for existing in self.existing_customers)
    
    def get_mock_data(self, query: str, location: str) -> List[Dict]:
        """Generate mock data for demo purposes"""
        return [
            {'name': 'Austin Aesthetic Center', 'formatted_address': '123 Main St, Austin, TX 78701', 'formatted_phone_number': '(512) 555-0101', 'website': 'https://austinaesthetic.com', 'rating': 4.8, 'user_ratings_total': 127, 'types': ['beauty_salon', 'health'], 'place_id': 'mock_1'},
            {'name': 'Hill Country Med Spa', 'formatted_address': '456 Oak Ave, Austin, TX 78704', 'formatted_phone_number': '(512) 555-0102', 'website': 'https://hillcountrymedspa.com', 'rating': 4.6, 'user_ratings_total': 89, 'types': ['spa', 'health'], 'place_id': 'mock_2'},
            {'name': 'Lone Star Dermatology', 'formatted_address': '789 Cedar St, Austin, TX 78702', 'formatted_phone_number': '(512) 555-0103', 'website': 'https://lonestardermatology.com', 'rating': 4.9, 'user_ratings_total': 203, 'types': ['doctor', 'health'], 'place_id': 'mock_3'}
        ]
    
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
            
            places_result = self.gmaps_api.places_nearby(location=lat_lng, radius=radius, keyword=query)
            
            results = []
            for place in places_result.get('results', []):
                if self.is_hospital_system(place.get('name', '')):
                    logger.info(f"Skipping hospital system: {place.get('name', '')}")
                    continue
                
                place_details = self.get_place_details(place['place_id'])
                
                if not place_details or not self.is_medical_business(place_details):
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
            return self.gmaps_api.place_details(
                place_id=place_id,
                fields=['name', 'formatted_address', 'formatted_phone_number', 
                       'website', 'rating', 'user_ratings_total', 'types']
            )
        except Exception as e:
            logger.error(f"Error getting place details: {str(e)}")
            return None
    
    def is_medical_business(self, place_data: Dict) -> bool:
        """Strict filtering for legitimate medical/aesthetic businesses"""
        place_types = place_data.get('types', [])
        name = place_data.get('name', '').lower()
        
        medical_types = {'doctor', 'health', 'spa', 'beauty_salon', 'hair_care', 'physiotherapist', 'dentist', 'hospital'}
        if not any(ptype in place_types for ptype in medical_types):
            logger.info(f"❌ FILTERED (no medical type): {name} - Types: {place_types}")
            return False
        
        exclude_types = {'store', 'food', 'restaurant', 'gym', 'school', 'general_contractor'}
        if any(ptype in place_types for ptype in exclude_types):
            logger.info(f"❌ FILTERED (non-medical business): {name} - Types: {place_types}")
            return False
        
        exclude_keywords = ['pharmacy', 'urgent care', 'veterinary', 'dentist', 'orthodont']
        if any(keyword in name for keyword in exclude_keywords):
            logger.info(f"❌ FILTERED (excluded keyword): {name}")
            return False
        
        logger.info(f"✅ PASSED FILTER: {name} - Types: {place_types}")
        return True
    
    def is_hospital_system(self, name: str) -> bool:
        """Check if a practice name indicates a hospital system"""
        return any(re.search(pattern, name.lower(), re.IGNORECASE) for pattern in self.hospital_patterns)
    
    def check_robots_txt(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc: return False
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()
            return rp.can_fetch('*', url)
        except Exception:
            return True
    
    def get_mock_website_data(self, url: str) -> Dict[str, any]:
        """Generate mock website data for demo purposes"""
        mock_data = {
            'https://austinaesthetic.com': {'title': 'Austin Aesthetic Center', 'description': '...', 'services': ['laser hair removal', 'botox'], 'social_links': ['Facebook', 'Instagram'], 'staff_count': 8},
            'https://hillcountrymedspa.com': {'title': 'Hill Country Med Spa', 'description': '...', 'services': ['body contouring', 'weight loss'], 'social_links': ['Instagram'], 'staff_count': 5},
            'https://lonestardermatology.com': {'title': 'Lone Star Dermatology', 'description': '...', 'services': ['laser hair removal', 'botox'], 'social_links': ['Facebook', 'LinkedIn'], 'staff_count': 12}
        }
        return mock_data.get(url, {'title': 'Medical Practice', 'description': '...', 'services': ['botox'], 'social_links': [], 'staff_count': 3})
    
    def scrape_single_page(self, url: str) -> Dict:
        """Scrape a single page and return extracted data"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(" ", strip=True).lower()
            
            service_keywords = ['laser hair removal', 'botox', 'fillers', 'coolsculpting', 'body contouring', 'skin tightening', 'photorejuvenation', 'cellulite treatment', 'weight loss', 'emsculpt', 'vaginal rejuvenation', 'hormone therapy', 'microneedling']
            
            return {
                'services': [kw for kw in service_keywords if kw in text_content],
                'staff_mentions': [str(s).strip() for s in soup.find_all(string=re.compile(r'\b(dr\.|doctor|physician)\b', re.I))],
                'text': text_content,
                'social_links': list(set(_social_from_jsonld(_extract_jsonld(soup)) + [link.get('href') for link in soup.find_all('a', href=True) if any(domain in (link.get('href') or '').lower() for domain in ['facebook', 'instagram', 'twitter', 'linkedin', 'youtube', 'tiktok'])])),
                'booking_platforms': _detect_booking_stack(soup),
                'financing_present': _text_has_any(text_content, FINANCE_HINTS),
                'venus_present': _text_has_any(text_content, VENUS_HINTS),
                'competitor_present': _text_has_any(text_content, COMP_HINTS),
                'weight_loss_present': _text_has_any(text_content, WEIGHT_LOSS_HINTS),
                'gallery_present': _text_has_any(text_content, GALLERY_HINTS),
                'hiring_present': _text_has_any(text_content, CAREER_HINTS),
            }
        except Exception as e:
            logger.debug(f"Error scraping {url}: {str(e)}")
            return {'services': [], 'staff_mentions': [], 'text': '', 'social_links': [], 'booking_platforms': [], 'financing_present': False, 'venus_present': False, 'competitor_present': False, 'weight_loss_present': False, 'gallery_present': False, 'hiring_present': False}

    def scrape_website_deep(self, base_url: str, max_pages: int = 5) -> Dict[str, any]:
        """Intelligently scrape multiple pages from a medical practice website"""
        if self.demo_mode:
            return self.get_mock_website_data(base_url)
        
        if not base_url or not self.check_robots_txt(base_url):
            return {'title': 'Not Available', 'description': 'Not Available', 'services': [], 'social_links': [], 'staff_count': 0}
        
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.title.string.strip() if soup.title and soup.title.string else 'Not Available'
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc['content'].strip() if meta_desc and meta_desc.get('content') else 'Not Available'
            
            common_paths = ['', '/about', '/services', '/team', '/contact']
            pages_to_scrape = [urljoin(base_url, path) for path in common_paths][:max_pages]
            
            aggregated_data = {
                'all_services': set(), 'all_social_links': set(), 'all_staff_mentions': [],
                'any_financing': False, 'any_venus': False, 'any_competitor': False,
                'any_weight_loss': False, 'any_gallery': False, 'any_hiring': False,
                'all_booking': set()
            }

            for page_url in pages_to_scrape:
                page_data = self.scrape_single_page(page_url)
                aggregated_data['all_services'].update(page_data['services'])
                aggregated_data['all_social_links'].update(page_data['social_links'])
                aggregated_data['all_staff_mentions'].extend(page_data['staff_mentions'])
                aggregated_data['all_booking'].update(page_data['booking_platforms'])
                for key in ['financing', 'venus', 'competitor', 'weight_loss', 'gallery', 'hiring']:
                    if page_data[f'{key}_present']: aggregated_data[f'any_{key}'] = True

            return {
                'title': title, 'description': description,
                'services': list(aggregated_data['all_services']),
                'social_links': list(aggregated_data['all_social_links']),
                'staff_count': min(len(set(aggregated_data['all_staff_mentions'])), 20),
                'booking_platforms': list(aggregated_data['all_booking']),
                'financing_present': aggregated_data['any_financing'],
                'venus_present': aggregated_data['any_venus'],
                'competitor_present': aggregated_data['any_competitor'],
                'weight_loss_present': aggregated_data['any_weight_loss'],
                'gallery_present': aggregated_data['any_gallery'],
                'hiring_present': aggregated_data['any_hiring']
            }
        except Exception as e:
            logger.error(f"Error in deep scrape for {base_url}: {str(e)}")
            return self.scrape_single_page(base_url) # Fallback to single page

    def detect_specialty(self, practice_data: Dict) -> str:
        """Detect the primary specialty of a practice"""
        all_text = f"{practice_data.get('name', '').lower()} {practice_data.get('description', '').lower()} {' '.join(practice_data.get('services', [])).lower()}"
        if any(kw in all_text for kw in ['dermatology', 'dermatologist']): return 'dermatology'
        if any(kw in all_text for kw in ['plastic surgery', 'plastic surgeon']): return 'plastic_surgery'
        if any(kw in all_text for kw in ['obgyn', 'ob/gyn', 'gynecologist']): return 'obgyn'
        if any(kw in all_text for kw in ['med spa', 'medspa']): return 'medspa'
        if any(kw in all_text for kw in ['family medicine', 'family practice']): return 'familypractice'
        return 'general'

    def calculate_ai_score(self, practice_data: Dict) -> Tuple[int, Dict[str, int], str]:
        """Calculate AI-powered scoring with specialty-specific weights"""
        specialty = self.detect_specialty(practice_data)
        config = self.specialty_scoring.get(specialty, self.specialty_scoring['dermatology'])
        weights = config['weights']
        scores = {k: 0 for k in weights.keys()}
        
        # Simplified scoring logic for brevity
        scores['specialty_match'] = min(sum(1 for kw in ['med spa', 'aesthetic', 'cosmetic'] if kw in practice_data.get('name','').lower()) * 10, 20)
        
        staff_count = practice_data.get('staff_count', 0)
        autonomy_score = 20 if staff_count <= 1 else 15 if staff_count <= 4 else 10 if staff_count <= 6 else 5
        if any(ind in practice_data.get('name','').lower() for ind in ['hospital', 'medical center']): autonomy_score -= 10
        scores['decision_autonomy'] = max(0, autonomy_score)
        
        scores['aesthetic_services'] = min(len(practice_data.get('services', [])) * 2, 15)
        scores['social_activity'] = min(len(practice_data.get('social_links', [])) * 3, 10)
        
        rating, review_count = practice_data.get('rating', 0), practice_data.get('user_ratings_total', 0)
        scores['reviews_rating'] = 10 if rating >= 4.5 and review_count >= 50 else 7 if rating >= 4.0 and review_count >= 25 else 4
        
        scores['search_visibility'] = 10 if practice_data.get('website') else 4
        scores['financial_indicators'] = 5 if practice_data.get('financing_present') else 0
        scores['weight_loss_services'] = 5 if practice_data.get('weight_loss_present') else 0
        
        # Apply weights
        weighted_score = sum(scores[k] * (weights[k] / 100.0) for k in scores) * (100.0 / sum(weights.values()))
        total_score = min(int(weighted_score), 100)
        
        return total_score, scores, specialty

    def recommend_device(self, practice_data: Dict) -> Dict:
        """Recommend top device based on practice profile"""
        services = practice_data.get('services', [])
        device_scores = {}
        for device, info in self.device_catalog.items():
            score = sum(20 for specialty in info['specialties'] if any(specialty in s for s in services))
            device_scores[device] = {'score': score, 'reasons': [f"Offers services related to {', '.join(info['specialties'])}"]}
        
        sorted_devices = sorted(device_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        if not sorted_devices: return {'primary_recommendation': None, 'all_recommendations': []}
        
        recommendations = [{'device': d, 'fit_score': data['score'], 'rationale': '; '.join(data['reasons']), 'rank': i+1} for i, (d, data) in enumerate(sorted_devices[:3])]
        return {'primary_recommendation': recommendations[0], 'all_recommendations': recommendations}

    def process_practice(self, place_data: Dict) -> Optional[Dict]:
        """Process a single practice through the full pipeline"""
        practice_name = place_data.get('name', 'Unknown')
        if self.is_existing_customer(practice_name):
            logger.info(f"Skipping existing customer: {practice_name}")
            return None
        
        logger.info(f"Processing practice: {practice_name}")
        practice_record = {k: place_data.get(k, '') for k in ['name', 'formatted_address', 'formatted_phone_number', 'website', 'rating', 'user_ratings_total', 'types']}
        practice_record['phone'] = practice_record.pop('formatted_phone_number') # Rename key
        practice_record['address'] = practice_record.pop('formatted_address') # Rename key
        practice_record['review_count'] = practice_record.pop('user_ratings_total')

        if practice_record['website']:
            website_data = self.scrape_website_deep(practice_record['website'])
            practice_record.update(website_data)
        
        ai_score, score_breakdown, specialty = self.calculate_ai_score(practice_record)
        device_recs = self.recommend_device(practice_record)
        
        final_record = {
            **practice_record,
            'specialty': specialty,
            'ai_score': ai_score,
            'score_breakdown': score_breakdown,
            'primary_device_rec': (device_recs['primary_recommendation'] or {}).get('device', ''),
            'device_rationale': (device_recs['primary_recommendation'] or {}).get('rationale', ''),
        }
        return final_record

    def export_to_csv(self, results: List[Dict], filename: str):
        """Export results to CSV"""
        if not results:
            logger.warning("No results to export")
            return
        
        # Dynamically create headers from the first result object to ensure all keys are captured
        if results:
            headers = list(results[0].keys())
        else:
            return
            
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in results:
                # Sanitize row: convert lists to strings, handle missing keys
                sanitized_row = {}
                for key in headers:
                    val = row.get(key)
                    if isinstance(val, list) or isinstance(val, dict):
                        sanitized_row[key] = json.dumps(val)
                    else:
                        sanitized_row[key] = val
                writer.writerow(sanitized_row)
        logger.info(f"Results exported to {filename}")

    def generate_summary_report(self, results: List[Dict], filename: str):
        """Generate summary report"""
        if not results: return
        
        sorted_results = sorted(results, key=lambda x: x.get('ai_score', 0), reverse=True)
        total = len(results)
        high_fit = len([r for r in results if r.get('ai_score', 0) >= 70])
        avg_score = sum(r.get('ai_score', 0) for r in results) / total if total > 0 else 0
        
        report = f"MEDICAL DEVICE PROSPECTING REPORT\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"EXECUTIVE SUMMARY\n================\nTotal Prospects: {total}\nHigh-Fit (>=70): {high_fit}\nAverage Score: {avg_score:.1f}\n\n"
        report += "TOP PROSPECTS\n=============\n"
        for i, p in enumerate(sorted_results[:10], 1):
            report += f"{i}. {p.get('name')} (Score: {p.get('ai_score')})\n   {p.get('address')}\n"
            
        with open(filename, 'w', encoding='utf-8') as f: f.write(report)
        logger.info(f"Summary report generated: {filename}")
    
    def run_prospecting(self, keywords: List[str], location: str, radius: int, max_results: int):
        """Main prospecting workflow"""
        logger.info(f"Starting prospecting for {keywords} near {location} (radius: {radius}km)")
        
        all_results = []
        unique_places = set()

        for keyword in keywords:
            logger.info(f"Searching for: {keyword}")
            places = self.google_places_search(keyword, location, radius * 1000)
            
            for place in places:
                place_id = place.get('place_id')
                if not place_id or place_id in unique_places:
                    continue
                unique_places.add(place_id)

                try:
                    processed = self.process_practice(place)
                    if processed:
                        all_results.append(processed)
                        logger.info(f"Processed: {processed.get('name')} - Score: {processed.get('ai_score')}")
                    if len(all_results) >= max_results:
                        break
                except Exception as e:
                    logger.error(f"Error processing practice {place.get('name')}: {str(e)}")
            if len(all_results) >= max_results:
                logger.info(f"Reached max results limit of {max_results}.")
                break
        
        logger.info(f"Found {len(all_results)} unique prospects")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"prospects_{timestamp}.csv"
        report_filename = f"summary_{timestamp}.txt"
        
        self.export_to_csv(all_results, csv_filename)
        self.generate_summary_report(all_results, report_filename)
        
        return all_results, csv_filename, report_filename


def main():
    """Main function with argument parsing for command-line execution"""
    parser = argparse.ArgumentParser(description='Medical Device Prospecting System')
    parser.add_argument('--keywords', nargs='+', required=True, help='Search keywords')
    parser.add_argument('--city', required=True, help='City or location to search')
    parser.add_argument('--radius', type=int, default=25, help='Search radius in km')
    parser.add_argument('--max-results', type=int, default=150, help='Max results total')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('--exclude-csv', help='CSV file with existing customers')
    
    args = parser.parse_args()
    
    prospector = FathomProspector(
        demo_mode=args.demo, 
        existing_customers_csv=args.exclude_csv
    )
    
    try:
        results, csv_file, report_file = prospector.run_prospecting(
            args.keywords, args.city, args.radius, args.max_results
        )
        print(f"\n=== PROSPECTING COMPLETE ===")
        print(f"Total prospects found: {len(results)}")
        print(f"Results exported to: {csv_file}")
        print(f"Summary report: {report_file}")

    except Exception as e:
        logger.error(f"Error during prospecting: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
