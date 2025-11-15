
"""
Deep Dive Intelligence Module
Comprehensive prospect enrichment with device intelligence and ICP classification
"""
import os
import logging
import time
import random
import re
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient
from urllib.parse import quote_plus

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


# ============================================================================
# DEVICE INTELLIGENCE DATABASE
# ============================================================================

AESTHETIC_DEVICES = {
    # InMode Devices
    'Morpheus8': {'manufacturer': 'InMode', 'category': 'RF Microneedling', 'modality': 'skin_tightening'},
    'EvolveX': {'manufacturer': 'InMode', 'category': 'Body Contouring', 'modality': 'multi'},
    'Evolve': {'manufacturer': 'InMode', 'category': 'Body Contouring', 'modality': 'multi'},
    'BodyTite': {'manufacturer': 'InMode', 'category': 'Body Contouring', 'modality': 'fat_reduction'},
    'FaceTite': {'manufacturer': 'InMode', 'category': 'Facial', 'modality': 'skin_tightening'},
    'EmpowerRF': {'manufacturer': 'InMode', 'category': 'Wellness', 'modality': 'skin_tightening'},
    'Lumecca': {'manufacturer': 'InMode', 'category': 'IPL', 'modality': 'skin_rejuvenation'},
    
    # BTL Devices
    'Emsculpt NEO': {'manufacturer': 'BTL', 'category': 'Body Contouring', 'modality': 'fat_muscle'},
    'Emsculpt': {'manufacturer': 'BTL', 'category': 'Muscle Building', 'modality': 'muscle_only'},
    'Emsella': {'manufacturer': 'BTL', 'category': 'Wellness', 'modality': 'pelvic_floor'},
    'Emface': {'manufacturer': 'BTL', 'category': 'Facial', 'modality': 'skin_muscle'},
    'Exilis Ultra': {'manufacturer': 'BTL', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    'Exilis Ultra 360': {'manufacturer': 'BTL', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    'Emtone': {'manufacturer': 'BTL', 'category': 'Cellulite', 'modality': 'cellulite'},
    
    # Allergan (AbbVie)
    'CoolSculpting Elite': {'manufacturer': 'Allergan', 'category': 'Fat Reduction', 'modality': 'fat_only'},
    'CoolSculpting': {'manufacturer': 'Allergan', 'category': 'Fat Reduction', 'modality': 'fat_only'},
    
    # Cynosure
    'PicoSure': {'manufacturer': 'Cynosure', 'category': 'Laser', 'modality': 'skin_rejuvenation'},
    'PicoSure Pro': {'manufacturer': 'Cynosure', 'category': 'Laser', 'modality': 'skin_rejuvenation'},
    'Potenza': {'manufacturer': 'Cynosure', 'category': 'RF Microneedling', 'modality': 'skin_tightening'},
    'Clarity II': {'manufacturer': 'Cynosure', 'category': 'Laser Hair Removal', 'modality': 'hair_removal'},
    'StimSure': {'manufacturer': 'Cynosure', 'category': 'Muscle Building', 'modality': 'muscle_only'},
    'SculpSure': {'manufacturer': 'Cynosure', 'category': 'Fat Reduction', 'modality': 'fat_only'},
    'TempSure': {'manufacturer': 'Cynosure', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    
    # Candela
    'Nordlys': {'manufacturer': 'Candela', 'category': 'Multi-Platform', 'modality': 'skin_rejuvenation'},
    'GentleMax Pro': {'manufacturer': 'Candela', 'category': 'Laser Hair Removal', 'modality': 'hair_removal'},
    'GentleMax Pro Plus': {'manufacturer': 'Candela', 'category': 'Laser Hair Removal', 'modality': 'hair_removal'},
    'Vbeam': {'manufacturer': 'Candela', 'category': 'Vascular Laser', 'modality': 'skin_rejuvenation'},
    'Vbeam Pro': {'manufacturer': 'Candela', 'category': 'Vascular Laser', 'modality': 'skin_rejuvenation'},
    'Vbeam Perfecta': {'manufacturer': 'Candela', 'category': 'Vascular Laser', 'modality': 'skin_rejuvenation'},
    'Matrix': {'manufacturer': 'Candela', 'category': 'RF Platform', 'modality': 'skin_tightening'},
    'PicoWay': {'manufacturer': 'Candela', 'category': 'Laser', 'modality': 'skin_rejuvenation'},
    
    # Solta Medical
    'Thermage FLX': {'manufacturer': 'Solta', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    'Thermage': {'manufacturer': 'Solta', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    'Fraxel': {'manufacturer': 'Solta', 'category': 'Fractional Laser', 'modality': 'skin_rejuvenation'},
    'Fraxel DUAL': {'manufacturer': 'Solta', 'category': 'Fractional Laser', 'modality': 'skin_rejuvenation'},
    'Clear + Brilliant': {'manufacturer': 'Solta', 'category': 'Fractional Laser', 'modality': 'skin_rejuvenation'},
    'VASERlipo': {'manufacturer': 'Solta', 'category': 'Liposuction', 'modality': 'fat_reduction'},
    
    # Lumenis
    'Stellar M22': {'manufacturer': 'Lumenis', 'category': 'Multi-Platform', 'modality': 'skin_rejuvenation'},
    'M22': {'manufacturer': 'Lumenis', 'category': 'Multi-Platform', 'modality': 'skin_rejuvenation'},
    'AcuPulse': {'manufacturer': 'Lumenis', 'category': 'CO2 Laser', 'modality': 'skin_rejuvenation'},
    'NuEra Tight': {'manufacturer': 'Lumenis', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    'Splendor X': {'manufacturer': 'Lumenis', 'category': 'Laser Hair Removal', 'modality': 'hair_removal'},
    
    # Sciton
    'JOULE': {'manufacturer': 'Sciton', 'category': 'Multi-Platform', 'modality': 'skin_rejuvenation'},
    'mJOULE': {'manufacturer': 'Sciton', 'category': 'Multi-Platform', 'modality': 'skin_rejuvenation'},
    'BBL': {'manufacturer': 'Sciton', 'category': 'IPL', 'modality': 'skin_rejuvenation'},
    'BBL HERO': {'manufacturer': 'Sciton', 'category': 'IPL', 'modality': 'skin_rejuvenation'},
    'BBL HEROic': {'manufacturer': 'Sciton', 'category': 'IPL', 'modality': 'skin_rejuvenation'},
    'HALO': {'manufacturer': 'Sciton', 'category': 'Hybrid Laser', 'modality': 'skin_rejuvenation'},
    'MOXI': {'manufacturer': 'Sciton', 'category': 'Fractional Laser', 'modality': 'skin_rejuvenation'},
    
    # Cutera
    'truSculpt iD': {'manufacturer': 'Cutera', 'category': 'Fat Reduction', 'modality': 'fat_only'},
    'truSculpt': {'manufacturer': 'Cutera', 'category': 'Fat Reduction', 'modality': 'fat_only'},
    'truFlex': {'manufacturer': 'Cutera', 'category': 'Muscle Building', 'modality': 'muscle_only'},
    'Secret PRO': {'manufacturer': 'Cutera', 'category': 'RF Microneedling', 'modality': 'skin_tightening'},
    'excel V': {'manufacturer': 'Cutera', 'category': 'Vascular Laser', 'modality': 'skin_rejuvenation'},
    'excel V+': {'manufacturer': 'Cutera', 'category': 'Vascular Laser', 'modality': 'skin_rejuvenation'},
    'enlighten': {'manufacturer': 'Cutera', 'category': 'Laser', 'modality': 'skin_rejuvenation'},
    'AviClear': {'manufacturer': 'Cutera', 'category': 'Acne Treatment', 'modality': 'acne'},
    
    # Venus Concepts
    'Venus NOVA': {'manufacturer': 'Venus Concepts', 'category': 'Body Contouring', 'modality': 'multi'},
    'Venus Nova': {'manufacturer': 'Venus Concepts', 'category': 'Body Contouring', 'modality': 'multi'},
    'Venus Bliss MAX': {'manufacturer': 'Venus Concepts', 'category': 'Body Contouring', 'modality': 'multi'},
    'Venus Bliss': {'manufacturer': 'Venus Concepts', 'category': 'Body Contouring', 'modality': 'fat_skin'},
    'Venus Legacy': {'manufacturer': 'Venus Concepts', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    'Venus Versa': {'manufacturer': 'Venus Concepts', 'category': 'Multi-Platform', 'modality': 'skin_rejuvenation'},
    'Venus Viva': {'manufacturer': 'Venus Concepts', 'category': 'RF Resurfacing', 'modality': 'skin_rejuvenation'},
    'Venus Freeze': {'manufacturer': 'Venus Concepts', 'category': 'Skin Tightening', 'modality': 'skin_tightening'},
    
    # Merz Aesthetics
    'Ultherapy': {'manufacturer': 'Merz', 'category': 'Ultrasound', 'modality': 'skin_tightening'},
    'Ultherapy PRIME': {'manufacturer': 'Merz', 'category': 'Ultrasound', 'modality': 'skin_tightening'},
    
    # Sofwave
    'Sofwave': {'manufacturer': 'Sofwave Medical', 'category': 'Ultrasound', 'modality': 'skin_tightening'},
    
    # Aerolase
    'Neo Elite': {'manufacturer': 'Aerolase', 'category': 'Laser', 'modality': 'skin_rejuvenation'},
    
    # Cartessa/DEKA
    'Tetra CO2': {'manufacturer': 'Cartessa', 'category': 'CO2 Laser', 'modality': 'skin_rejuvenation'},
    'SmartXide': {'manufacturer': 'Cartessa', 'category': 'CO2 Laser', 'modality': 'skin_rejuvenation'},
    
    # Apyx Medical
    'Renuvion': {'manufacturer': 'Apyx', 'category': 'RF Plasma', 'modality': 'skin_tightening'},
    'J-Plasma': {'manufacturer': 'Apyx', 'category': 'RF Plasma', 'modality': 'skin_tightening'},
}

# Treatment brand names that indicate device ownership
TREATMENT_BRANDS = {
    'Forever Young BBL': 'BBL HERO',
    'TriBella': 'Venus Versa',
    'Core to Floor': 'Emsculpt NEO',
    'CoolPeel': 'Tetra CO2',
}


# ============================================================================
# ICP CLASSIFICATION RULES
# ============================================================================

def classify_icp(detected_devices: List[str]) -> Dict:
    """
    Classify prospect based on their installed device base
    Returns ICP tag and sales intelligence
    """
    
    modalities = {
        'fat_reduction': False,
        'skin_tightening': False,
        'muscle_building': False,
        'skin_rejuvenation': False,
    }
    
    manufacturers = set()
    device_categories = []
    
    for device_name in detected_devices:
        device_info = AESTHETIC_DEVICES.get(device_name, {})
        modality = device_info.get('modality', '')
        manufacturer = device_info.get('manufacturer', '')
        category = device_info.get('category', '')
        
        if manufacturer:
            manufacturers.add(manufacturer)
        if category:
            device_categories.append(category)
        
        # Track modalities
        if modality in ['fat_only', 'fat_reduction', 'fat_muscle', 'fat_skin', 'multi']:
            modalities['fat_reduction'] = True
        if modality in ['skin_tightening', 'fat_skin', 'fat_muscle', 'multi']:
            modalities['skin_tightening'] = True
        if modality in ['muscle_only', 'fat_muscle', 'multi']:
            modalities['muscle_building'] = True
        if modality in ['skin_rejuvenation', 'multi']:
            modalities['skin_rejuvenation'] = True
    
    # Determine ICP classification
    icp_tag = 'GENERAL'
    icp_description = 'General aesthetic practice'
    sales_opportunity = ''
    target_priority = 5  # 1-10 scale
    
    # FAT_ONLY - Has fat reduction but no skin tightening or muscle
    if modalities['fat_reduction'] and not modalities['skin_tightening'] and not modalities['muscle_building']:
        icp_tag = 'FAT_ONLY'
        icp_description = 'High-Value Target: Missing Skin Tightening & Muscle Building'
        sales_opportunity = 'This practice has made capital equipment purchases for fat reduction but cannot address the resulting skin laxity or offer muscle toning. They understand ROI cycles and won\'t be shocked by pricing. Venus Nova/Legacy provides the missing muscle building + skin tightening modalities. Position as the perfect "stack" to complete their body contouring offering.'
        target_priority = 10
    
    # SKIN_ONLY - Has skin treatments but no body contouring
    elif modalities['skin_rejuvenation'] and not modalities['fat_reduction'] and not modalities['muscle_building']:
        icp_tag = 'SKIN_ONLY'
        icp_description = 'High-Value Target: Missing Body Contouring'
        sales_opportunity = 'This practice is 100% focused on facial/skin treatments. They\'re missing the fastest-growing market segment: body contouring. They already have sophisticated clientele willing to invest in aesthetic treatments. Venus Bliss MAX or Nova is a turnkey "body-in-a-box" solution that opens an entirely new revenue stream.'
        target_priority = 10
    
    # BTL_OWNER - Has BTL devices
    elif 'BTL' in manufacturers:
        icp_tag = 'BTL_OWNER'
        icp_description = 'Complement Strategy: BTL Ecosystem'
        sales_opportunity = 'This practice is invested in the BTL ecosystem. Emsculpt NEO\'s RF is for fat reduction, not true skin tightening. Venus Legacy offers dedicated (MP)² technology for superior skin laxity and cellulite treatment - a modality gap in their current offering. Position as complementary, not competitive.'
        target_priority = 9
    
    # MORPHEUS_OWNER - Has Morpheus8
    elif any('Morpheus8' in device for device in detected_devices):
        icp_tag = 'MORPHEUS_OWNER'
        icp_description = 'Complement Strategy: Add Comfort Options'
        sales_opportunity = 'This practice specializes in aggressive RF treatments. They lack "no-downtime" options for sensitive patients and cannot address body fat or muscle building. Venus Legacy offers the comfortable, relaxing alternative for patients who fear Morpheus8\'s pain. Venus Bliss MAX/Nova fills their body contouring gap.'
        target_priority = 8
    
    # COOLSCULPTING_OWNER - #1 Target
    elif any('CoolSculpting' in device for device in detected_devices):
        icp_tag = 'COOLSCULPTING_OWNER'
        icp_description = '#1 Priority Target: CoolSculpting Stack Opportunity'
        sales_opportunity = 'This is a TOP-TIER prospect. They\'ve invested $120k+ in CoolSculpting but can only address fat - not the skin laxity or muscle loss that patients want fixed. Venus Nova is the perfect "stacking" partner: adds muscle building + skin tightening in one platform. Every CoolSculpting patient is a Venus Nova candidate.'
        target_priority = 10
    
    # UPGRADE_TARGET - Has older devices
    elif any(device in ['Venus Freeze', 'GentleMax Pro', 'M22'] for device in detected_devices):
        icp_tag = 'UPGRADE_TARGET'
        icp_description = 'Upgrade Opportunity: Legacy Equipment'
        sales_opportunity = 'This practice has 5-10+ year old devices. They\'re proven buyers who understand aesthetic equipment but are falling behind the "platform stacking" trend. They\'re losing patients to newer clinics. A Venus Versa PRO + Venus Nova bundle offers a complete practice refresh and positions them as cutting-edge again.'
        target_priority = 10
    
    # SOPHISTICATED_BUYER - Has multiple high-end devices
    elif len(detected_devices) >= 3:
        icp_tag = 'SOPHISTICATED_BUYER'
        icp_description = 'Premium Prospect: Multi-Device Practice'
        sales_opportunity = 'This practice has invested in multiple capital devices. They understand platform stacking and are likely Early Adopters. They\'re not price-sensitive - they\'re ROI-focused. Position Venus devices as ecosystem expansion to fill any remaining modality gaps.'
        target_priority = 8
    
    # Detect modality gaps
    modality_gaps = []
    if not modalities['fat_reduction']:
        modality_gaps.append('No fat reduction device - Missing body contouring revenue')
    if not modalities['skin_tightening']:
        modality_gaps.append('No skin tightening device - Cannot treat post-fat-loss laxity')
    if not modalities['muscle_building']:
        modality_gaps.append('No muscle building device - Missing high-demand muscle toning')
    if not modalities['skin_rejuvenation']:
        modality_gaps.append('No skin rejuvenation platform - Missing facial treatment revenue')
    
    return {
        'icp_tag': icp_tag,
        'icp_description': icp_description,
        'sales_opportunity': sales_opportunity,
        'target_priority': target_priority,
        'modality_gaps': modality_gaps,
        'detected_modalities': [k for k, v in modalities.items() if v],
        'manufacturer_ecosystem': list(manufacturers),
        'buyer_sophistication': 'High' if len(detected_devices) >= 3 else 'Medium' if len(detected_devices) >= 1 else 'Unknown'
    }


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def detect_aesthetic_devices(website_url: str, soup: Optional[BeautifulSoup] = None) -> Tuple[List[str], Dict]:
    """
    Detect aesthetic devices from website content
    Returns (detected_devices, device_details)
    """
    try:
        if not soup:
            soup = scrape_with_scrapingbee_advanced(website_url)
        
        if not soup:
            return [], {}
        
        detected_devices = []
        device_details = {}
        
        # Get all text content
        text_content = soup.get_text(' ', strip=True)
        html_content = str(soup)
        
        # Search for device names
        for device_name, device_info in AESTHETIC_DEVICES.items():
            # Case-insensitive search
            pattern = re.compile(re.escape(device_name), re.IGNORECASE)
            if pattern.search(text_content) or pattern.search(html_content):
                detected_devices.append(device_name)
                device_details[device_name] = device_info
                logger.info(f"✅ Detected device: {device_name} ({device_info['manufacturer']})")
        
        # Search for treatment brand names
        for brand_name, device_name in TREATMENT_BRANDS.items():
            pattern = re.compile(re.escape(brand_name), re.IGNORECASE)
            if pattern.search(text_content):
                if device_name not in detected_devices:
                    detected_devices.append(device_name)
                    device_details[device_name] = AESTHETIC_DEVICES.get(device_name, {})
                    logger.info(f"✅ Detected device via brand: {device_name} (from '{brand_name}')")
        
        logger.info(f"Total devices detected: {len(detected_devices)}")
        return detected_devices, device_details
        
    except Exception as e:
        logger.error(f"Error detecting devices: {e}")
        return [], {}


# ============================================================================
# GLP-1 OPPORTUNITY DETECTION
# ============================================================================

def detect_glp1_opportunity(website_url: str, soup: Optional[BeautifulSoup] = None) -> Dict:
    """
    Detect if practice is targeting GLP-1 / weight loss market
    """
    try:
        if not soup:
            soup = scrape_with_scrapingbee_advanced(website_url)
        
        if not soup:
            return {'detected': False, 'signals': []}
        
        text_content = soup.get_text(' ', strip=True).lower()
        
        # GLP-1 keywords
        glp1_keywords = [
            'glp-1', 'glp1', 'ozempic', 'wegovy', 'mounjaro', 'zepbound',
            'semaglutide', 'tirzepatide',
            'post-weight loss', 'post weight loss', 'weight loss skin',
            'rapid weight loss', 'post-bariatric',
            'skin laxity', 'loose skin', 'skin tightening after weight loss'
        ]
        
        detected_signals = []
        for keyword in glp1_keywords:
            if keyword in text_content:
                detected_signals.append(keyword)
        
        is_glp1_opportunity = len(detected_signals) > 0
        
        return {
            'detected': is_glp1_opportunity,
            'signals': detected_signals,
            'opportunity_level': 'HIGH' if len(detected_signals) >= 3 else 'MEDIUM' if len(detected_signals) >= 1 else 'NONE',
            'sales_message': '🎯 GLP-1 OPPORTUNITY DETECTED - This practice is actively marketing post-weight-loss treatments. Venus Nova/Legacy\'s skin tightening technology is perfectly positioned for this massive patient funnel.' if is_glp1_opportunity else ''
        }
        
    except Exception as e:
        logger.error(f"Error detecting GLP-1 opportunity: {e}")
        return {'detected': False, 'signals': []}


# ============================================================================
# TREATMENT STACKING DETECTION
# ============================================================================

def detect_stacking_sophistication(website_url: str, soup: Optional[BeautifulSoup] = None) -> Dict:
    """
    Detect if practice understands treatment stacking / multi-modality approach
    """
    try:
        if not soup:
            soup = scrape_with_scrapingbee_advanced(website_url)
        
        if not soup:
            return {'detected': False, 'signals': []}
        
        text_content = soup.get_text(' ', strip=True).lower()
        
        stacking_keywords = [
            'stacked treatment', 'treatment stack', 'multi-modality',
            'combination therapy', 'combined treatment', '3-in-1', '2-in-1',
            'platform approach', 'comprehensive treatment', 'treatment protocol'
        ]
        
        detected_signals = []
        for keyword in stacking_keywords:
            if keyword in text_content:
                detected_signals.append(keyword)
        
        is_sophisticated = len(detected_signals) > 0
        
        return {
            'detected': is_sophisticated,
            'signals': detected_signals,
            'sophistication_level': 'HIGH' if len(detected_signals) >= 2 else 'MEDIUM' if len(detected_signals) == 1 else 'LOW',
            'sales_message': '✅ SOPHISTICATED BUYER - This practice understands platform stacking and multi-modality treatments. Position Venus as an ecosystem play, not a single-device sale.' if is_sophisticated else ''
        }
        
    except Exception as e:
        logger.error(f"Error detecting stacking sophistication: {e}")
        return {'detected': False, 'signals': []}


# ============================================================================
# PERSONAL SOCIAL PROFILE DETECTION
# ============================================================================

def search_personal_social_profiles(staff_name: str, location: str, practice_name: str) -> List[Dict]:
    """
    Search for personal social media profiles of staff members
    Returns list of potential profile matches
    """
    try:
        profiles = []
        
        # Clean up name for searching
        clean_name = staff_name.replace('Dr.', '').replace('MD', '').replace('DO', '').replace('NP', '').replace('PA', '').strip()
        
        # Search Instagram
        try:
            instagram_query = f"{clean_name} {location} instagram"
            google_url = f"https://www.google.com/search?q={quote_plus(instagram_query)}"
            soup = scrape_with_scrapingbee_advanced(google_url, wait_time=2000)
            
            if soup:
                # Look for Instagram links in search results
                instagram_links = soup.find_all('a', href=re.compile(r'instagram\.com/[^/]+/?$'))
                for link in instagram_links[:2]:  # Top 2 results
                    href = link.get('href', '')
                    if 'instagram.com/' in href and '/p/' not in href and '/reel/' not in href:
                        username = href.split('instagram.com/')[-1].strip('/')
                        profiles.append({
                            'platform': 'Instagram',
                            'username': username,
                            'url': f"https://instagram.com/{username}",
                            'confidence': 'medium',
                            'staff_name': staff_name
                        })
        except Exception as e:
            logger.error(f"Error searching Instagram for {staff_name}: {e}")
        
        # Search Facebook
        try:
            facebook_query = f"{clean_name} {practice_name} facebook"
            google_url = f"https://www.google.com/search?q={quote_plus(facebook_query)}"
            soup = scrape_with_scrapingbee_advanced(google_url, wait_time=2000)
            
            if soup:
                # Look for Facebook profile links
                facebook_links = soup.find_all('a', href=re.compile(r'facebook\.com/[^/]+/?$'))
                for link in facebook_links[:2]:
                    href = link.get('href', '')
                    if 'facebook.com/' in href and '/posts/' not in href and '/videos/' not in href:
                        username = href.split('facebook.com/')[-1].strip('/')
                        profiles.append({
                            'platform': 'Facebook',
                            'username': username,
                            'url': href,
                            'confidence': 'medium',
                            'staff_name': staff_name
                        })
        except Exception as e:
            logger.error(f"Error searching Facebook for {staff_name}: {e}")
        
        # Search LinkedIn
        try:
            linkedin_query = f"{clean_name} {practice_name} linkedin"
            google_url = f"https://www.google.com/search?q={quote_plus(linkedin_query)}"
            soup = scrape_with_scrapingbee_advanced(google_url, wait_time=2000)
            
            if soup:
                linkedin_links = soup.find_all('a', href=re.compile(r'linkedin\.com/in/[^/]+/?$'))
                for link in linkedin_links[:1]:  # Top result only
                    href = link.get('href', '')
                    if 'linkedin.com/in/' in href:
                        username = href.split('linkedin.com/in/')[-1].strip('/')
                        profiles.append({
                            'platform': 'LinkedIn',
                            'username': username,
                            'url': href,
                            'confidence': 'high',  # LinkedIn is usually accurate
                            'staff_name': staff_name
                        })
        except Exception as e:
            logger.error(f"Error searching LinkedIn for {staff_name}: {e}")
        
        logger.info(f"Found {len(profiles)} personal profiles for {staff_name}")
        return profiles
        
    except Exception as e:
        logger.error(f"Error searching personal profiles for {staff_name}: {e}")
        return []


# ============================================================================
# ENHANCED STAFF EXTRACTION WITH PERSONAL PROFILES
# ============================================================================

def extract_staff_with_profiles(website_url: str, practice_name: str, location: str) -> List[Dict]:
    """
    Extract staff information and find their personal social profiles
    """
    try:
        soup = scrape_with_scrapingbee_advanced(website_url)
        if not soup:
            return []
        
        staff_members = []
        
        # Look for team/staff/about pages
        text_content = soup.get_text(' ', strip=True).lower()
        
        # Find sections about staff
        staff_sections = soup.find_all(['div', 'section'], class_=re.compile(
            r'(team|staff|provider|doctor|physician|about)', re.I))
        
        processed_names = set()
        
        for section in staff_sections[:5]:  # Limit to first 5 sections
            # Extract names with titles
            name_patterns = [
                r'(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+,\s*MD)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+,\s*DO)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+,\s*NP)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+,\s*PA)',
            ]
            
            section_text = section.get_text(' ', strip=True)
            
            for pattern in name_patterns:
                matches = re.findall(pattern, section_text)
                for name_match in matches:
                    clean_name = name_match.strip()
                    if clean_name not in processed_names and len(clean_name) > 5:
                        processed_names.add(clean_name)
                        
                        # Determine role
                        role = 'Provider'
                        if 'Dr.' in clean_name or 'MD' in clean_name or 'DO' in clean_name:
                            role = 'Physician'
                        elif 'NP' in clean_name:
                            role = 'Nurse Practitioner'
                        elif 'PA' in clean_name:
                            role = 'Physician Assistant'
                        
                        staff_member = {
                            'name': clean_name,
                            'role': role,
                            'context': 'Found on website',
                            'personal_profiles': []
                        }
                        
                        # Search for personal social profiles (limit to first 3 staff)
                        if len(staff_members) < 3:
                            logger.info(f"Searching personal profiles for: {clean_name}")
                            personal_profiles = search_personal_social_profiles(
                                clean_name, location, practice_name
                            )
                            staff_member['personal_profiles'] = personal_profiles
                        
                        staff_members.append(staff_member)
                        
                        if len(staff_members) >= 5:  # Max 5 staff members
                            break
                
                if len(staff_members) >= 5:
                    break
            
            if len(staff_members) >= 5:
                break
        
        logger.info(f"Extracted {len(staff_members)} staff members with profiles")
        return staff_members
        
    except Exception as e:
        logger.error(f"Error extracting staff with profiles: {e}")
        return []


# ============================================================================
# ORIGINAL HELPER FUNCTIONS (Preserved from original deep_dive.py)
# ============================================================================

def scrape_with_scrapingbee_advanced(url: str, wait_time: int = 3000) -> Optional[BeautifulSoup]:
    """
    Advanced ScrapingBee scraping with premium features and retry logic
    """
    if not SCRAPINGBEE_AVAILABLE:
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"🐝 Deep Dive scraping: {url} (Attempt {attempt + 1}/{max_retries})")
            time.sleep(random.uniform(0.5, 1.0))
            
            response = scrapingbee_client.get(
                url,
                params={
                    'render_js': True,
                    'premium_proxy': True,
                    'country_code': 'us',
                    'wait': wait_time,
                    'block_resources': False,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            elif response.status_code >= 500:
                logger.warning(f"ScrapingBee returned status {response.status_code}. Retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                logger.error(f"ScrapingBee error: Status {response.status_code}. Not retrying.")
                return None
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}. Retrying...")
            time.sleep(2 ** attempt)
    
    logger.error(f"Failed to scrape {url} after {max_retries} attempts.")
    return None


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


def detect_technology_stack(website_url: str, soup: Optional[BeautifulSoup] = None) -> Dict:
    """Detect website technology and marketing tools"""
    try:
        if not soup:
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
        elif 'vagaro' in html_content.lower():
            tech_stack['booking_system'] = 'Vagaro'
        
        return tech_stack
        
    except Exception as e:
        logger.error(f"Error detecting tech stack: {e}")
        return {}


# ============================================================================
# MAIN DEEP DIVE FUNCTION (Enhanced)
# ============================================================================

async def perform_deep_dive(prospect: Dict) -> Dict:
    """
    Main deep dive function - comprehensive intelligence gathering with device detection
    
    Args:
        prospect: Basic prospect data from initial search
    
    Returns:
        Enriched data dictionary with comprehensive intelligence
    """
    logger.info(f"🚀 Starting ENHANCED deep dive for: {prospect.get('name', 'Unknown')}")
    
    enriched_data = {
        'status': 'complete',
        'timestamp': time.time(),
        'decision_makers': [],
        'installed_devices': [],
        'device_details': {},
        'icp_classification': {},
        'glp1_opportunity': {},
        'stacking_sophistication': {},
        'social_media_intelligence': {},
        'technology_stack': {},
        'modality_gaps': [],
        'sales_strategy': {}
    }
    
    try:
        website = prospect.get('website', '')
        practice_name = prospect.get('name', '')
        location = prospect.get('address', '')
        
        # Step 1: Scrape website once and reuse soup
        logger.info("🌐 Scraping website...")
        soup = None
        if website:
            soup = scrape_with_scrapingbee_advanced(website)
        
        # Step 2: Device detection (HIGHEST PRIORITY)
        if website and soup:
            logger.info("🔍 Detecting installed aesthetic devices...")
            detected_devices, device_details = detect_aesthetic_devices(website, soup)
            enriched_data['installed_devices'] = detected_devices
            enriched_data['device_details'] = device_details
            
            # ICP Classification
            logger.info("🎯 Classifying ICP based on device portfolio...")
            icp_data = classify_icp(detected_devices)
            enriched_data['icp_classification'] = icp_data
            enriched_data['modality_gaps'] = icp_data.get('modality_gaps', [])
        
        # Step 3: GLP-1 Opportunity Detection
        if website and soup:
            logger.info("💊 Detecting GLP-1 opportunity signals...")
            glp1_data = detect_glp1_opportunity(website, soup)
            enriched_data['glp1_opportunity'] = glp1_data
        
        # Step 4: Treatment Stacking Sophistication
        if website and soup:
            logger.info("🧠 Analyzing treatment stacking sophistication...")
            stacking_data = detect_stacking_sophistication(website, soup)
            enriched_data['stacking_sophistication'] = stacking_data
        
        # Step 5: Decision Maker Intel + Personal Profiles
        if website and practice_name and location:
            logger.info("👥 Extracting decision makers with personal social profiles...")
            staff_with_profiles = extract_staff_with_profiles(website, practice_name, location)
            enriched_data['decision_makers'] = staff_with_profiles
        
        # Step 6: Social media intelligence (practice accounts)
        social_links = prospect.get('socialLinks', []) or prospect.get('social_links', [])
        if social_links:
            logger.info("📱 Analyzing practice social media...")
            enriched_data['social_media_intelligence'] = scrape_social_media_deep(social_links)
        
        # Step 7: Technology stack
        if website and soup:
            logger.info("💻 Detecting technology stack...")
            enriched_data['technology_stack'] = detect_technology_stack(website, soup)
        
        # Step 8: Generate sales strategy
        logger.info("📋 Generating sales strategy...")
        enriched_data['sales_strategy'] = generate_sales_strategy(enriched_data)
        
        logger.info(f"✅ Enhanced deep dive complete for {practice_name}")
        enriched_data['status'] = 'complete'
        
    except Exception as e:
        logger.error(f"Error in deep dive: {e}", exc_info=True)
        enriched_data['status'] = 'failed'
        enriched_data['error'] = str(e)
    
    return enriched_data


def generate_sales_strategy(enriched_data: Dict) -> Dict:
    """
    Generate comprehensive sales strategy based on intelligence gathered
    """
    strategy = {
        'primary_approach': 'Phone/Email',
        'talking_points': [],
        'recommended_devices': [],
        'urgency_level': 'Low'
    }
    
    # Determine primary approach
    decision_makers = enriched_data.get('decision_makers', [])
    if decision_makers:
        has_personal_profiles = any(
            dm.get('personal_profiles', []) for dm in decision_makers
        )
        if has_personal_profiles:
            strategy['primary_approach'] = 'Personal Social Media DM'
            strategy['talking_points'].append(
                'Engage with their personal social media content before reaching out'
            )
    
    # Add talking points based on ICP
    icp_data = enriched_data.get('icp_classification', {})
    if icp_data.get('sales_opportunity'):
        strategy['talking_points'].append(icp_data['sales_opportunity'])
    
    # Add GLP-1 talking point
    glp1_data = enriched_data.get('glp1_opportunity', {})
    if glp1_data.get('detected'):
        strategy['talking_points'].append(glp1_data.get('sales_message', ''))
        strategy['urgency_level'] = 'High'
    
    # Add stacking talking point
    stacking_data = enriched_data.get('stacking_sophistication', {})
    if stacking_data.get('detected'):
        strategy['talking_points'].append(stacking_data.get('sales_message', ''))
    
    # Recommend devices based on gaps
    modality_gaps = enriched_data.get('modality_gaps', [])
    if 'No skin tightening device' in ' '.join(modality_gaps):
        strategy['recommended_devices'].append('Venus Legacy')
        strategy['recommended_devices'].append('Venus Nova')
    if 'No fat reduction device' in ' '.join(modality_gaps):
        strategy['recommended_devices'].append('Venus Bliss MAX')
        strategy['recommended_devices'].append('Venus Nova')
    if 'No muscle building device' in ' '.join(modality_gaps):
        strategy['recommended_devices'].append('Venus Nova')
    
    return strategy


def is_deep_dive_available() -> bool:
    """Check if deep dive functionality is available"""
    return SCRAPINGBEE_AVAILABLE
