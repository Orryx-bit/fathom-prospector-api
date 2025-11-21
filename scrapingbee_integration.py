"""
ScrapingBee Integration Module
Replaces Playwright for JavaScript-rendered content scraping
"""
import os
import logging
import time
import random
import re
from typing import Dict, List
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient

logger = logging.getLogger(__name__)

# Get ScrapingBee API key from environment
SCRAPINGBEE_API_KEY = os.getenv('SCRAPINGBEE_API_KEY', '')
SCRAPINGBEE_AVAILABLE = bool(SCRAPINGBEE_API_KEY)

if SCRAPINGBEE_AVAILABLE:
    scrapingbee_client = ScrapingBeeClient(api_key=SCRAPINGBEE_API_KEY)
    logger.info("✅ ScrapingBee client initialized")
else:
    scrapingbee_client = None
    logger.warning("⚠️  ScrapingBee API key not found - JavaScript rendering disabled")


def scrape_with_scrapingbee(url: str, extract_emails_fn, extract_names_fn, 
                             extract_phones_fn, extract_form_fn, extract_team_fn) -> Dict:
    """
    Scrape website using ScrapingBee for JavaScript-rendered content
    
    Args:
        url: Website URL to scrape
        extract_*_fn: Functions to extract contact intelligence
        
    Returns:
        Dictionary with scraped data
    """
    if not SCRAPINGBEE_AVAILABLE:
        logger.warning(f"ScrapingBee not available for {url}")
        return None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"🐝 Scraping with ScrapingBee: {url} (Attempt {attempt + 1}/{max_retries})")
            time.sleep(random.uniform(0.5, 1.0)) # Keep random delay
            
            # ScrapingBee request with JavaScript rendering
            response = scrapingbee_client.get(
                url,
                params={
                    'render_js': True,      # Enable JavaScript rendering
                    'premium_proxy': True,  # Use premium proxies
                    'country_code': 'us',   # US proxies
                    'wait': 3000,           # --- MODIFIED: Increased wait time to 3 seconds ---
                },
                timeout=30 # --- ADDED: Client-side timeout ---
            )
            
            if response.status_code == 200:
                # --- SUCCESS ---
                # Parse the rendered HTML
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
                    'found_keywords': []
                }
                staff_indicators = soup.find_all(text=re.compile(
                    r'\b(dr\.|doctor|physician|provider|practitioner)\b', re.I))
                unique_staff = len(set(str(s).strip() for s in staff_indicators if len(str(s).strip()) > 5))
                data['staff_count'] = min(unique_staff, 20)
                
                # Extract contact intelligence using provided functions
                try:
                    data['emails'] = extract_emails_fn(soup, url)
                except Exception as e:
                    logger.error(f"Error extracting emails: {e}")
                    data['emails'] = []
                
                try:
                    data['contact_names'] = extract_names_fn(soup, '', url)
                except Exception as e:
                    logger.error(f"Error extracting contact names: {e}")
                    data['contact_names'] = []
                
                try:
                    data['additional_phones'] = extract_phones_fn(soup, '', url)
                except Exception as e:
                    logger.error(f"Error extracting phones: {e}")
                    data['additional_phones'] = []
                
                try:
                    data['contact_form_url'] = extract_form_fn(soup, url)
                except Exception as e:
                    logger.error(f"Error extracting contact form: {e}")
                    data['contact_form_url'] = ''
                
                try:
                    data['team_members'] = extract_team_fn(soup, url)
                except Exception as e:
                    logger.error(f"Error extracting team members: {e}")
                    data['team_members'] = []
                
                logger.info(f"✅ ScrapingBee scrape complete for {url}: {len(data['services'])} services, "
                           f"{len(data['emails'])} emails, {len(data['contact_names'])} contacts found")
                
                return data # --- Return successful data ---
            
            elif response.status_code >= 500:
                # --- Server error, retry ---
                logger.warning(f"ScrapingBee error for {url}: Status {response.status_code}. Retrying...")
                time.sleep(2 ** attempt) # Exponential backoff
                continue # Go to next attempt
            
            else:
                # --- Client error (4xx), do not retry ---
                logger.error(f"ScrapingBee error for {url}: Status {response.status_code}. Not retrying.")
                return None
            
        except Exception as e:
            logger.error(f"Error in ScrapingBee scraping {url}: {str(e)}")
            if attempt == max_retries - 1:
                return None # Failed all retries
            time.sleep(2 ** attempt) # Exponential backoff before next retry
    
    return None # Return None if all retries fail


def is_scrapingbee_available() -> bool:
    """Check if ScrapingBee is available"""
    return SCRAPINGBEE_AVAILABLE
