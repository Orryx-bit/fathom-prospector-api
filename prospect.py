#!/usr/bin/env python3
"""
Fathom Medical Device Prospecting System
Comprehensive tool for finding and scoring medical practices
Production-Hardened Version 3.6 (Final Gemini Model Fix)
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- GEMINI SETUP ---
import google.generativeai as genai
# --- END GEMINI SETUP ---

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GooglePlacesAPI:
    """Wrapper for Google Places API with retry logic."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
        self.timeout = (10, 30)

    def _make_request(self, endpoint: str, params: dict, max_retries: int = 3) -> dict:
        url = f"{self.base_url}{endpoint}"
        params['key'] = self.api_key
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if data.get('status') in ['OK', 'ZERO_RESULTS']:
                    return data
                elif data.get('status') == 'OVER_QUERY_LIMIT':
                    raise Exception("Google API quota exceeded")
                time.sleep(2 ** attempt)
            except requests.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
        raise Exception("Max retries exceeded")

    def geocode(self, address: str) -> List[dict]:
        try:
            data = self._make_request("/geocode/json", {'address': address})
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Geocoding error for '{address}': {e}")
            return []

    def places_nearby(self, location: dict, radius: int, keyword: str) -> dict:
        all_results = []
        params = {'location': f"{location['lat']},{location['lng']}", 'radius': radius, 'keyword': keyword}
        while True:
            try:
                data = self._make_request("/place/nearbysearch/json", params)
                all_results.extend(data.get('results', []))
                next_page_token = data.get('next_page_token')
                if not next_page_token:
                    break
                params['pagetoken'] = next_page_token
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error fetching places page: {e}")
                break
        return {'results': all_results}

    def place_details(self, place_id: str, fields: List[str]) -> Optional[dict]:
        try:
            data = self._make_request("/place/details/json", {'place_id': place_id, 'fields': ','.join(fields)})
            return data.get('result')
        except Exception as e:
            logger.error(f"Error fetching details for {place_id}: {e}")
            return None


class FathomProspector:
    """Main prospecting system for medical devices"""
    def __init__(self, api_key=None, demo_mode=False, existing_customers_csv=None):
        self.gmaps_key = os.getenv("GOOGLE_PLACES_API_KEY") or api_key
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.demo_mode = demo_mode
        self.gemini_model = None

        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                # --- THIS IS THE ONE-LINE FIX ---
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini API: ✓ Configured with gemini-1.5-flash")
            except Exception as e:
                logger.error(f"Gemini API initialization failed: {e}")
        else:
            logger.warning('GEMINI_API_KEY not found - AI features will be disabled.')

        if not self.gmaps_key:
            logger.warning('GOOGLE_PLACES_API_KEY not found - falling back to demo mode.')
            self.demo_mode = True
        else:
            self.gmaps_api = GooglePlacesAPI(self.gmaps_key)
            logger.info("Google Places API: ✓ Configured")

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.existing_customers = set()
        exclusion_file = existing_customers_csv or 'exclusion_list.txt'
        if os.path.exists(exclusion_file):
            self.load_existing_customers(exclusion_file)

    def load_existing_customers(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                customers = [line.strip().lower() for line in f if line.strip() and not line.startswith('#')]
                self.existing_customers = set(customers)
            logger.info(f"Loaded {len(self.existing_customers)} customers from exclusion list: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load exclusion list {file_path}: {e}")

    def is_existing_customer(self, practice_name: str) -> bool:
        cleaned_name = practice_name.lower().strip()
        return any(customer_name in cleaned_name for customer_name in self.existing_customers)

    def google_places_search(self, query: str, location: str, radius: int) -> List[Dict]:
        if self.demo_mode:
            logger.info("DEMO MODE: Generating mock data.")
            return []

        geocode_result = self.gmaps_api.geocode(location)
        if not geocode_result:
            logger.error(f"Could not geocode location: {location}")
            return []
        
        lat_lng = geocode_result[0]['geometry']['location']
        places_result = self.gmaps_api.places_nearby(location=lat_lng, radius=radius, keyword=query)
        return places_result.get('results', [])

    def scrape_website_deep(self, base_url: str) -> Dict[str, any]:
        if not base_url: return {}
        try:
            response = self.session.get(base_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(" ", strip=True).lower()
            description_tag = soup.find('meta', attrs={'name': 'description'})
            return {
                "title": soup.title.string.strip() if soup.title else "",
                "description": description_tag.get('content', '') if description_tag else '',
                "services_text": text_content,
            }
        except requests.RequestException as e:
            logger.warning(f"Could not scrape {base_url}: {e}")
            return {}
            
    def calculate_ai_score(self, practice_data: Dict) -> Tuple[int, Dict, str]:
        specialty = self.detect_specialty(practice_data)
        if not self.gemini_model:
            return 50, {"summary": "Gemini not available. Basic score used."}, specialty

        logger.info(f"Performing Gemini AI analysis for: {practice_data.get('name')}")
        prompt_data = {key: practice_data.get(key) for key in ['name', 'title', 'description', 'rating', 'review_count']}
        prompt_data['website_content_summary'] = practice_data.get('services_text', '')[:2000]
        prompt_data['detected_specialty'] = specialty

        prompt = f"""You are a sales analyst for Venus Concepts, an aesthetic device company. Evaluate this lead based on the data provided. Respond ONLY with a valid JSON object.
        DATA: {json.dumps(prompt_data, indent=2)}
        CRITERIA:
        1. Decision Maker Autonomy (1-10): Is this an independent practice (high score) or a hospital/chain (low score)?
        2. Aesthetic Focus (1-10): Are they a dedicated medspa (high score) or general practice (low score)?
        3. Financial Readiness (1-10): Do they seem like a premium business that can afford a $100k device?
        4. Growth Potential (1-10): Are they hiring or marketing heavily?
        RESPONSE FORMAT: {{"scores": {{"decision_maker_autonomy": 0, "aesthetic_focus": 0, "financial_readiness": 0, "growth_potential": 0}}, "final_summary": "<Your 2-sentence analysis here.>"}}
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            cleaned_text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
            ai_analysis = json.loads(cleaned_text)
            scores = ai_analysis.get("scores", {})
            total_score = int(((scores.get("decision_maker_autonomy",0)*0.35) + (scores.get("aesthetic_focus",0)*0.3) + (scores.get("financial_readiness",0)*0.2) + (scores.get("growth_potential",0)*0.15)) * 10)
            return total_score, ai_analysis, specialty
        except Exception as e:
            logger.error(f"Error during Gemini analysis for {practice_data.get('name')}: {e}")
            return 0, {"summary": f"AI analysis failed: {e}"}, specialty

    def detect_specialty(self, practice_data: Dict) -> str:
        text = f"{practice_data.get('name', '')} {practice_data.get('description', '')}".lower()
        if 'dermatology' in text: return 'Dermatology'
        if 'plastic surgery' in text: return 'Plastic Surgery'
        if 'med spa' in text or 'medspa' in text: return 'MedSpa'
        return 'General'

    def process_practice(self, place_data: Dict) -> Optional[Dict]:
        practice_name = place_data.get('name')
        if not practice_name:
            return None
        
        if self.is_existing_customer(practice_name):
            logger.warning(f"⏭️ Skipping excluded: {practice_name}")
            return None

        logger.info(f"Processing: {practice_name}")
        practice_record = {
            'name': practice_name,
            'address': place_data.get('formatted_address'),
            'phone': place_data.get('formatted_phone_number'),
            'website': place_data.get('website'),
            'rating': place_data.get('rating'),
            'review_count': place_data.get('user_ratings_total'),
        }
        
        website_data = self.scrape_website_deep(practice_record['website'])
        practice_record.update(website_data)

        ai_score, score_breakdown, specialty = self.calculate_ai_score(practice_record)
        
        practice_record.update({'ai_score': ai_score, 'score_breakdown': score_breakdown, 'specialty': specialty})
        return practice_record

    def run_prospecting(self, keywords: List[str], location: str, radius: int, max_results: int):
        logger.info(f"Starting prospecting for '{', '.join(keywords)}' near '{location}'")
        
        all_results, seen_ids = [], set()

        for keyword in keywords:
            if len(all_results) >= max_results: break
            
            places = self.google_places_search(keyword, location, radius * 1000)
            
            for place in places:
                if len(all_results) >= max_results: break
                
                place_id = place.get('place_id')
                if not place_id or place_id in seen_ids:
                    continue
                seen_ids.add(place_id)
                
                details = self.gmaps_api.place_details(place_id, ['name', 'formatted_address', 'formatted_phone_number', 'website', 'rating', 'user_ratings_total', 'types'])
                
                if details:
                    processed = self.process_practice(details)
                    if processed:
                        all_results.append(processed)
                        logger.info(f"✅ Added to results: {processed.get('name')} (Score: {processed.get('ai_score')})")

        logger.info(f"Found {len(all_results)} unique prospects.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"prospects_{timestamp}.csv"
        
        if all_results:
            headers = list(all_results[0].keys())
            if 'ai_summary' not in headers: headers.extend(['ai_summary', 'ai_score_breakdown'])
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                for row in all_results:
                    analysis = row.get('score_breakdown', {})
                    row['ai_summary'] = analysis.get('final_summary', '')
                    row['ai_score_breakdown'] = json.dumps(analysis.get('scores', {}))
                    for k, v in row.items():
                        if isinstance(v, (dict, list)): row[k] = json.dumps(v)
                    writer.writerow(row)
            logger.info(f"Results exported to {csv_filename}")
        else:
            logger.warning("No results to export.")

        return all_results, csv_filename, "summary_report.txt"

def main():
    parser = argparse.ArgumentParser(description='Fathom Prospecting System with AI Scoring')
    parser.add_argument('--keywords', nargs='+', required=True)
    parser.add_argument('--city', required=True)
    parser.add_argument('--radius', type=int, default=25)
    parser.add_argument('--max-results', type=int, default=50)
    args = parser.parse_args()
    
    prospector = FathomProspector()
    prospector.run_prospecting(args.keywords, args.city, args.radius, args.max_results)

if __name__ == "__main__":
    main()
