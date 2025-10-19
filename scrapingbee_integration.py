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
    
    try:
        logger.info(f"🐝 Scraping with ScrapingBee: {url}")
        time.sleep(random.uniform(0.5, 1.0))
        
        # ScrapingBee request with JavaScript rendering
        response = scrapingbee_client.get(
            url,
            params={
                'render_js': True,  # Enable JavaScript rendering
                'premium_proxy': True,  # Use premium proxies
                'country_code': 'us',  # US proxies
                'wait': 2000,  # Wait 2 seconds for JS to load
            }
        )
        
        if response.status_code != 200:
            logger.error(f"ScrapingBee error for {url}: Status {response.status_code}")
            return None
        
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
            'team_members': []
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
        text_content = soup.get_text(' ', strip=True).lower()
        service_keywords = [
            'laser hair removal', 'botox', 'fillers', 'coolsculpting',
            'body contouring', 'skin tightening', 'photorejuvenation',
            'cellulite treatment', 'weight loss', 'ems', 'muscle building',
            'dermatology', 'skin care', 'facial', 'microneedling',
            'chemical peel', 'laser treatment', 'acne treatment'
        ]
        
        services_found = []
        for keyword in service_keywords:
            if keyword in text_content:
                services_found.append(keyword)
        
        data['services'] = services_found[:10]  # Limit to 10
        
        # Find social media links
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
        
        # Get all links
        all_links = soup.find_all('a', href=True)
        for link_element in all_links:
            try:
                href = link_element.get('href', '').strip()
                href_lower = href.lower()
                
                # Skip empty or invalid hrefs
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                for domain, platform_name in social_platforms.items():
                    if domain in href_lower and platform_name not in seen_platforms:
                        # Clean up the URL
                        full_url = href
                        if href.startswith('//'):
                            full_url = 'https:' + href
                        elif href.startswith('/') or not href.startswith('http'):
                            # Relative URL - skip it as it's not a social link
                            continue
                        
                        social_links.append(full_url)
                        seen_platforms.add(platform_name)
                        logger.info(f"Found {platform_name} link: {full_url}")
                        break
            except Exception:
                continue
        
        data['social_links'] = social_links
        
        # Estimate staff count
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
        
        return data
        
    except Exception as e:
        logger.error(f"Error in ScrapingBee scraping {url}: {str(e)}")
        return None


def is_scrapingbee_available() -> bool:
    """Check if ScrapingBee is available"""
    return SCRAPINGBEE_AVAILABLE
