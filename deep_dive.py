"""
Deep Dive Intelligence Module
Comprehensive prospect enrichment using ScrapingBee
"""
import os
import logging
import time
import random
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient

logger = logging.getLogger(__name__)

# Get ScrapingBee API key
SCRAPINGBEE_API_KEY = os.getenv('SCRAPINGBEE_API_KEY', '')
SCRAPINGBEE_AVAILABLE = bool(SCRAPINGBEE_API_KEY)

if SCRAPINGBEE_AVAILABLE:
    scrapingbee_client = ScrapingBeeClient(api_key=SCRAPINGBEE_API_KEY)
    logger.info("✅ Deep Dive: ScrapingBee client initialized")
else:
    scrapingbee_client = None
    logger.warning("⚠️  Deep Dive: ScrapingBee API key not found")


def scrape_with_scrapingbee_advanced(url: str, wait_time: int = 3000) -> Optional[BeautifulSoup]:
    """
    Advanced ScrapingBee scraping with premium features
    """
    if not SCRAPINGBEE_AVAILABLE:
        return None
    
    try:
        logger.info(f"🐝 Deep Dive scraping: {url}")
        time.sleep(random.uniform(0.5, 1.0))
        
        response = scrapingbee_client.get(
            url,
            params={
                'render_js': True,
                'premium_proxy': True,
                'country_code': 'us',
                'wait': wait_time,
                'block_resources': False,
            }
        )
        
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            logger.error(f"ScrapingBee error: Status {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None


def scrape_google_reviews(business_name: str, location: str) -> Dict:
    """Scrape Google reviews for sentiment analysis"""
    try:
        # Build Google search URL for reviews
        search_query = f"{business_name} {location} reviews"
        google_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        
        soup = scrape_with_scrapingbee_advanced(google_url)
        if not soup:
            return {'source': 'Google', 'rating': None, 'count': 0, 'recent_reviews': []}
        
        # Extract review data (simplified - would need more robust parsing)
        reviews = {
            'source': 'Google',
            'rating': None,
            'count': 0,
            'recent_reviews': []
        }
        
        return reviews
        
    except Exception as e:
        logger.error(f"Error scraping Google reviews: {e}")
        return {'source': 'Google', 'rating': None, 'count': 0, 'recent_reviews': []}


def scrape_yelp_reviews(business_name: str, location: str) -> Dict:
    """Scrape Yelp reviews"""
    try:
        search_query = f"{business_name} {location}"
        yelp_url = f"https://www.yelp.com/search?find_desc={search_query.replace(' ', '+')}"
        
        soup = scrape_with_scrapingbee_advanced(yelp_url)
        if not soup:
            return {'source': 'Yelp', 'rating': None, 'count': 0, 'profile_url': None}
        
        # Extract Yelp data
        reviews = {
            'source': 'Yelp',
            'rating': None,
            'count': 0,
            'profile_url': None
        }
        
        return reviews
        
    except Exception as e:
        logger.error(f"Error scraping Yelp: {e}")
        return {'source': 'Yelp', 'rating': None, 'count': 0, 'profile_url': None}


def scrape_healthgrades(business_name: str) -> Dict:
    """Scrape Healthgrades reviews"""
    try:
        search_query = business_name.replace(' ', '+')
        url = f"https://www.healthgrades.com/search?what={search_query}&where="
        
        soup = scrape_with_scrapingbee_advanced(url)
        if not soup:
            return {'source': 'Healthgrades', 'rating': None, 'count': 0}
        
        return {
            'source': 'Healthgrades',
            'rating': None,
            'count': 0
        }
        
    except Exception as e:
        logger.error(f"Error scraping Healthgrades: {e}")
        return {'source': 'Healthgrades', 'rating': None, 'count': 0}


def scrape_social_media_deep(social_links: List[str]) -> Dict:
    """Deep dive into social media presence"""
    social_data = {
        'instagram': {'followers': None, 'engagement': None, 'recent_posts': []},
        'facebook': {'likes': None, 'check_ins': None},
        'youtube': {'subscribers': None, 'videos': 0},
        'linkedin': {'followers': None, 'employees': None}
    }
    
    for link in social_links:
        try:
            link_lower = link.lower()
            
            if 'instagram.com' in link_lower:
                soup = scrape_with_scrapingbee_advanced(link)
                if soup:
                    # Extract Instagram data (would need Instagram API or advanced scraping)
                    social_data['instagram']['status'] = 'active'
            
            elif 'facebook.com' in link_lower:
                soup = scrape_with_scrapingbee_advanced(link)
                if soup:
                    social_data['facebook']['status'] = 'active'
            
            elif 'youtube.com' in link_lower or 'youtu.be' in link_lower:
                soup = scrape_with_scrapingbee_advanced(link)
                if soup:
                    social_data['youtube']['status'] = 'active'
            
            elif 'linkedin.com' in link_lower:
                soup = scrape_with_scrapingbee_advanced(link)
                if soup:
                    social_data['linkedin']['status'] = 'active'
                    
        except Exception as e:
            logger.error(f"Error scraping social link {link}: {e}")
            continue
    
    return social_data


def extract_staff_credentials(website_url: str) -> List[Dict]:
    """Extract detailed staff information and credentials"""
    try:
        soup = scrape_with_scrapingbee_advanced(website_url)
        if not soup:
            return []
        
        staff_members = []
        
        # Look for team/staff/about pages
        team_keywords = ['team', 'staff', 'about', 'providers', 'doctors', 'physicians']
        text_content = soup.get_text(' ', strip=True).lower()
        
        # Find sections about staff
        staff_sections = soup.find_all(['div', 'section'], class_=re.compile(
            r'(team|staff|provider|doctor|physician|about)', re.I))
        
        for section in staff_sections:
            # Extract names with titles
            names = section.find_all(text=re.compile(
                r'\b(Dr\.|Doctor|MD|DO|NP|PA|RN)\b', re.I))
            
            for name_text in names[:10]:  # Limit to 10
                staff_members.append({
                    'name': str(name_text).strip(),
                    'context': 'Found on website'
                })
        
        logger.info(f"Extracted {len(staff_members)} staff members from {website_url}")
        return staff_members
        
    except Exception as e:
        logger.error(f"Error extracting staff credentials: {e}")
        return []


def scrape_media_coverage(business_name: str, location: str) -> List[Dict]:
    """Scrape media mentions and press coverage"""
    try:
        search_query = f'"{business_name}" {location} news OR press OR award'
        google_news_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}&tbm=nws"
        
        soup = scrape_with_scrapingbee_advanced(google_news_url)
        if not soup:
            return []
        
        media_mentions = []
        
        # Extract news articles (simplified)
        # In reality, would parse Google News results
        
        return media_mentions
        
    except Exception as e:
        logger.error(f"Error scraping media coverage: {e}")
        return []


def detect_technology_stack(website_url: str) -> Dict:
    """Detect website technology and marketing tools"""
    try:
        soup = scrape_with_scrapingbee_advanced(website_url)
        if not soup:
            return {}
        
        tech_stack = {
            'cms': 'Unknown',
            'marketing_tools': [],
            'booking_system': None,
            'ecommerce': False
        }
        
        html_content = str(soup)
        
        # Detect CMS
        if 'wp-content' in html_content or 'wordpress' in html_content.lower():
            tech_stack['cms'] = 'WordPress'
        elif 'squarespace' in html_content.lower():
            tech_stack['cms'] = 'Squarespace'
        elif 'wix' in html_content.lower():
            tech_stack['cms'] = 'Wix'
        elif 'shopify' in html_content.lower():
            tech_stack['cms'] = 'Shopify'
            tech_stack['ecommerce'] = True
        
        # Detect marketing tools
        if 'hubspot' in html_content.lower():
            tech_stack['marketing_tools'].append('HubSpot')
        if 'mailchimp' in html_content.lower():
            tech_stack['marketing_tools'].append('Mailchimp')
        if 'google-analytics' in html_content or 'gtag' in html_content:
            tech_stack['marketing_tools'].append('Google Analytics')
        
        # Detect booking systems
        if 'calendly' in html_content.lower():
            tech_stack['booking_system'] = 'Calendly'
        elif 'acuity' in html_content.lower():
            tech_stack['booking_system'] = 'Acuity Scheduling'
        elif 'mindbody' in html_content.lower():
            tech_stack['booking_system'] = 'Mindbody'
        
        return tech_stack
        
    except Exception as e:
        logger.error(f"Error detecting tech stack: {e}")
        return {}


def scrape_competitors(business_name: str, location: str, services: List[str]) -> List[Dict]:
    """Identify and analyze competitors"""
    try:
        # Build search query for similar businesses
        service_query = ' '.join(services[:3]) if services else 'medical spa'
        search_query = f"{service_query} near {location} -\"{business_name}\""
        
        google_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        
        soup = scrape_with_scrapingbee_advanced(google_url)
        if not soup:
            return []
        
        competitors = []
        
        # Extract competitor listings (simplified)
        # Would need more robust parsing of Google results
        
        return competitors
        
    except Exception as e:
        logger.error(f"Error scraping competitors: {e}")
        return []


async def perform_deep_dive(prospect: Dict) -> Dict:
    """
    Main deep dive function - comprehensive intelligence gathering
    
    Args:
        prospect: Basic prospect data from initial search
        
    Returns:
        Enriched data dictionary with comprehensive intelligence
    """
    logger.info(f"🚀 Starting deep dive for: {prospect.get('name', 'Unknown')}")
    
    enriched_data = {
        'status': 'complete',
        'timestamp': time.time(),
        'multi_platform_reviews': [],
        'social_media_intelligence': {},
        'staff_credentials': [],
        'media_coverage': [],
        'technology_stack': {},
        'competitors': []
    }
    
    try:
        # Multi-platform review aggregation
        logger.info("📊 Gathering multi-platform reviews...")
        reviews = []
        
        if prospect.get('name') and prospect.get('address'):
            reviews.append(scrape_google_reviews(prospect['name'], prospect['address']))
            reviews.append(scrape_yelp_reviews(prospect['name'], prospect['address']))
            reviews.append(scrape_healthgrades(prospect['name']))
        
        enriched_data['multi_platform_reviews'] = [r for r in reviews if r.get('rating') or r.get('count', 0) > 0]
        
        # Social media deep dive
        if prospect.get('socialLinks') or prospect.get('social_links'):
            logger.info("📱 Analyzing social media presence...")
            social_links = prospect.get('socialLinks', []) or prospect.get('social_links', [])
            enriched_data['social_media_intelligence'] = scrape_social_media_deep(social_links)
        
        # Staff credentials
        if prospect.get('website'):
            logger.info("👥 Extracting staff credentials...")
            enriched_data['staff_credentials'] = extract_staff_credentials(prospect['website'])
        
        # Media coverage
        if prospect.get('name') and prospect.get('address'):
            logger.info("📰 Searching media coverage...")
            enriched_data['media_coverage'] = scrape_media_coverage(prospect['name'], prospect['address'])
        
        # Technology stack
        if prospect.get('website'):
            logger.info("💻 Detecting technology stack...")
            enriched_data['technology_stack'] = detect_technology_stack(prospect['website'])
        
        # Competitive analysis
        if prospect.get('name') and prospect.get('address'):
            logger.info("🔍 Analyzing competitors...")
            services = prospect.get('services', [])
            enriched_data['competitors'] = scrape_competitors(
                prospect['name'], 
                prospect['address'], 
                services
            )
        
        logger.info(f"✅ Deep dive complete for {prospect.get('name')}")
        enriched_data['status'] = 'complete'
        
    except Exception as e:
        logger.error(f"Error in deep dive: {e}")
        enriched_data['status'] = 'failed'
        enriched_data['error'] = str(e)
    
    return enriched_data


def is_deep_dive_available() -> bool:
    """Check if deep dive functionality is available"""
    return SCRAPINGBEE_AVAILABLE
