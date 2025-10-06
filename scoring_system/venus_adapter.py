
"""
Venus-specific adapter for the lead scoring system.
Enhanced Version 2.0 - Production Rebuild

New Features:
- Review text preprocessing
- Service extraction from descriptions
- Data quality indicators
- Better feature engineering for limited data scenarios
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VenusScoringAdapter:
    """Adapts Venus practice data to the scoring engine format"""
    
    def __init__(self):
        """Initialize the adapter with enhanced feature extractors"""
        
        # Service keyword mappings for better detection
        self.service_mappings = {
            'injectables': ['botox', 'dysport', 'xeomin', 'filler', 'juvederm', 
                           'restylane', 'sculptra', 'radiesse', 'belotero'],
            'body_contouring': ['coolsculpting', 'emsculpt', 'body contouring',
                               'fat reduction', 'body sculpting', 'liposuction'],
            'laser': ['laser hair removal', 'laser resurfacing', 'ipl', 
                     'laser treatment', 'laser therapy'],
            'skin_rejuvenation': ['microneedling', 'chemical peel', 'hydrafacial',
                                 'dermaplaning', 'prp', 'vampire facial', 'facials'],
            'advanced': ['vaginal rejuvenation', 'hormone therapy', 'iv therapy',
                        'platelet rich plasma', 'stem cell']
        }
        
        # Specialty indicators
        self.specialty_indicators = {
            'dermatology': ['dermatology', 'dermatologist', 'skin care', 'skin clinic',
                           'medical dermatology', 'cosmetic dermatology'],
            'plastic_surgery': ['plastic surgery', 'plastic surgeon', 'cosmetic surgery',
                               'cosmetic surgeon', 'reconstructive surgery'],
            'obgyn': ['obgyn', 'ob-gyn', 'ob/gyn', 'obstetrics', 'gynecology', 
                     'women\'s health', 'womens health', 'gynecologist'],
            'medspa': ['med spa', 'medical spa', 'medspa', 'aesthetic center',
                      'cosmetic center', 'rejuvenation center'],
            'family_practice': ['family medicine', 'family practice', 'family physician',
                               'primary care', 'general practice', 'internal medicine']
        }
        
        logger.info("✅ VenusScoringAdapter v2.0 initialized")
    
    def convert_practice_to_features(self, practice_data: Dict) -> Dict:
        """
        Convert a single practice dict to feature dict for scoring
        
        Args:
            practice_data: Dictionary containing practice information
            
        Returns:
            Dictionary of features for scoring engine
        """
        features = {}
        
        # Extract and normalize text data
        practice_name = practice_data.get('name', '').lower()
        practice_desc = practice_data.get('description', '').lower()
        services = practice_data.get('services', [])
        service_text = ' '.join([s.lower() for s in services if s])
        all_text = f"{practice_name} {practice_desc} {service_text}".lower()
        
        # ═══════════════════════════════════════════════════════════════════
        # SPECIALTY MATCH FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        features['dermatology'] = self._detect_specialty(all_text, 'dermatology')
        features['plastic_surgery'] = self._detect_specialty(all_text, 'plastic_surgery')
        features['obgyn_practice'] = self._detect_specialty(all_text, 'obgyn')
        features['med_spa'] = self._detect_specialty(all_text, 'medspa')
        features['family_practice'] = self._detect_specialty(all_text, 'family_practice')
        features['cosmetic_clinic'] = 1.0 if any(term in all_text for term in ['cosmetic clinic', 'cosmetic center', 'aesthetic clinic']) else 0.0
        features['aesthetic_clinic'] = 1.0 if any(term in all_text for term in ['aesthetic', 'beauty', 'rejuvenation']) else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # OWNERSHIP & AUTONOMY FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        # Solo practitioner indicators
        solo_indicators = ['solo', 'private', 'independent', 'boutique', 'exclusive', 'owner']
        features['solo_practitioner'] = 1.0 if any(ind in all_text for ind in solo_indicators) else 0.0
        
        # Hospital system indicators (negative)
        hospital_indicators = ['hospital', 'medical center', 'health system', 'healthcare system']
        features['hospital_affiliated'] = 1.0 if any(ind in all_text for ind in hospital_indicators) else 0.0
        
        # Independent ownership indicators
        ownership_indicators = ['physician-owned', 'doctor-owned', 'owner-operated', 'privately owned']
        features['independent_ownership'] = 1.0 if any(ind in all_text for ind in ownership_indicators) else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # SERVICE CATEGORY FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        for category, keywords in self.service_mappings.items():
            feature_name = f'{category}_services'
            match_count = sum(1 for keyword in keywords if keyword in all_text)
            # Normalize to 0-1 scale
            features[feature_name] = min(match_count / 3.0, 1.0)
        
        # ═══════════════════════════════════════════════════════════════════
        # DIGITAL PRESENCE FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        features['has_website'] = 1.0 if practice_data.get('website') else 0.0
        
        social_links = practice_data.get('social_links', [])
        features['social_media_count'] = min(len(social_links) / 3.0, 1.0)
        features['has_instagram'] = 1.0 if 'Instagram' in social_links else 0.0
        features['has_facebook'] = 1.0 if 'Facebook' in social_links else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # PRACTICE SIZE & SOPHISTICATION
        # ═══════════════════════════════════════════════════════════════════
        
        staff_count = practice_data.get('staff_count', 0)
        features['staff_size'] = min(staff_count / 10.0, 1.0)
        features['small_practice'] = 1.0 if staff_count <= 5 else 0.0
        features['medium_practice'] = 1.0 if 6 <= staff_count <= 15 else 0.0
        features['large_practice'] = 1.0 if staff_count > 15 else 0.0
        
        # Service variety (indicates sophistication)
        service_count = len(services) if services else 0
        features['service_variety'] = min(service_count / 10.0, 1.0)
        features['multi_service'] = 1.0 if service_count >= 5 else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # REPUTATION & SOCIAL PROOF
        # ═══════════════════════════════════════════════════════════════════
        
        rating = practice_data.get('rating', 0)
        review_count = practice_data.get('review_count', 0)
        
        features['high_rating'] = 1.0 if rating >= 4.5 else 0.0
        features['good_rating'] = 1.0 if 4.0 <= rating < 4.5 else 0.0
        features['low_rating'] = 1.0 if rating < 4.0 else 0.0
        
        features['high_review_volume'] = 1.0 if review_count >= 50 else 0.0
        features['medium_review_volume'] = 1.0 if 20 <= review_count < 50 else 0.0
        features['low_review_volume'] = 1.0 if review_count < 20 else 0.0
        
        # Combined reputation score
        features['reputation_score'] = (rating / 5.0) * 0.6 + min(review_count / 100.0, 1.0) * 0.4
        
        # ═══════════════════════════════════════════════════════════════════
        # FINANCIAL & BUSINESS MODEL INDICATORS
        # ═══════════════════════════════════════════════════════════════════
        
        # Cash-pay service indicators
        cashpay_keywords = ['cash', 'elective', 'cosmetic', 'aesthetic', 'spa', 
                           'membership', 'concierge', 'boutique']
        features['cashpay_focus'] = min(
            sum(1 for kw in cashpay_keywords if kw in all_text) / 3.0, 1.0
        )
        
        # Premium service indicators
        premium_keywords = ['luxury', 'premium', 'exclusive', 'vip', 'elite', 
                           'boutique', 'upscale', 'high-end']
        features['premium_positioning'] = 1.0 if any(kw in all_text for kw in premium_keywords) else 0.0
        
        # Membership/subscription model
        membership_keywords = ['membership', 'subscription', 'club', 'concierge']
        features['membership_model'] = 1.0 if any(kw in all_text for kw in membership_keywords) else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # DATA QUALITY INDICATORS (NEW)
        # ═══════════════════════════════════════════════════════════════════
        
        # Measure how much data we have about this practice
        data_quality_score = 0.0
        
        if practice_data.get('website'):
            data_quality_score += 0.25
        if service_count > 0:
            data_quality_score += 0.25
        if practice_desc and practice_desc != 'not available':
            data_quality_score += 0.25
        if rating > 0 and review_count > 0:
            data_quality_score += 0.25
        
        features['data_quality'] = data_quality_score
        features['has_rich_data'] = 1.0 if data_quality_score >= 0.75 else 0.0
        features['has_limited_data'] = 1.0 if data_quality_score < 0.5 else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # ENHANCED SERVICE DETECTION (from reviews/descriptions)
        # ═══════════════════════════════════════════════════════════════════
        
        # If we have description but no services detected, try to extract
        if service_count == 0 and practice_desc and practice_desc != 'not available':
            extracted_services = self._extract_services_from_description(practice_desc)
            if extracted_services:
                features['services_from_description'] = 1.0
                # Update service category features based on extracted services
                for category, keywords in self.service_mappings.items():
                    if any(kw in extracted_services for kw in keywords):
                        feature_name = f'{category}_services'
                        features[feature_name] = max(features.get(feature_name, 0), 0.5)
        
        # ═══════════════════════════════════════════════════════════════════
        # SPECIALTY-SPECIFIC FEATURES (Enhanced)
        # ═══════════════════════════════════════════════════════════════════
        
        # OB/GYN specific
        if features['obgyn_practice'] > 0:
            features['offers_aesthetic_gyn'] = 1.0 if any(
                term in all_text for term in ['vaginal rejuvenation', 'cosmetic gynecology', 
                                               'feminine wellness', 'vaginal health']
            ) else 0.0
            
            features['womens_wellness_focus'] = 1.0 if any(
                term in all_text for term in ['hormone', 'menopause', 'wellness', 
                                               'anti-aging', 'functional medicine']
            ) else 0.0
        
        # MedSpa specific
        if features['med_spa'] > 0:
            features['medical_director'] = 1.0 if any(
                term in all_text for term in ['medical director', 'physician-owned', 'doctor-owned']
            ) else 0.0
            
            features['advanced_technology'] = 1.0 if any(
                term in all_text for term in ['laser', 'coolsculpt', 'emsculpt', 
                                               'technology', 'device', 'equipment']
            ) else 0.0
        
        # Plastic Surgery specific
        if features['plastic_surgery'] > 0:
            features['board_certified'] = 1.0 if any(
                term in all_text for term in ['board certified', 'board-certified', 'abps', 
                                               'american board']
            ) else 0.0
            
            features['non_surgical_focus'] = 1.0 if any(
                term in all_text for term in ['non-surgical', 'nonsurgical', 'non-invasive', 
                                               'noninvasive', 'aesthetic']
            ) else 0.0
        
        # Family Practice specific
        if features['family_practice'] > 0:
            features['dpc_model'] = 1.0 if any(
                term in all_text for term in ['direct primary care', 'dpc', 'membership medicine', 
                                               'concierge']
            ) else 0.0
            
            features['functional_medicine'] = 1.0 if any(
                term in all_text for term in ['functional medicine', 'integrative', 
                                               'holistic', 'wellness']
            ) else 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # WEIGHT ADJUSTMENT FOR LIMITED DATA
        # ═══════════════════════════════════════════════════════════════════
        
        # When data is limited, boost high-confidence signals
        if features['has_limited_data'] > 0:
            # Boost specialty match if clearly indicated in name
            if any(ind in practice_name for ind in ['dermatology', 'derm', 'skin']):
                features['dermatology'] = 1.0
            if any(ind in practice_name for ind in ['plastic surgery', 'cosmetic']):
                features['plastic_surgery'] = 1.0
            if any(ind in practice_name for ind in ['obgyn', 'ob-gyn', 'women']):
                features['obgyn_practice'] = 1.0
            if any(ind in practice_name for ind in ['med spa', 'medspa', 'spa']):
                features['med_spa'] = 1.0
            
            # Boost reputation if we have good reviews
            if features['high_rating'] > 0 and features['medium_review_volume'] > 0:
                features['reputation_score'] = min(features['reputation_score'] * 1.3, 1.0)
        
        logger.debug(f"Converted practice '{practice_name[:30]}' to {len(features)} features")
        
        return features
    
    def _detect_specialty(self, text: str, specialty: str) -> float:
        """
        Detect if text indicates a specific specialty
        Returns confidence score 0.0 to 1.0
        """
        if specialty not in self.specialty_indicators:
            return 0.0
        
        indicators = self.specialty_indicators[specialty]
        matches = sum(1 for ind in indicators if ind in text)
        
        # Return normalized confidence
        return min(matches / 2.0, 1.0)
    
    def _extract_services_from_description(self, description: str) -> List[str]:
        """
        Extract service keywords from practice description
        Returns list of detected services
        """
        description_lower = description.lower()
        extracted_services = []
        
        # Check all service mappings
        for category, keywords in self.service_mappings.items():
            for keyword in keywords:
                if keyword in description_lower:
                    extracted_services.append(keyword)
        
        return list(set(extracted_services))
    
    def batch_convert_practices(self, practices: List[Dict]) -> pd.DataFrame:
        """
        Convert multiple practices to feature DataFrame
        
        Args:
            practices: List of practice dictionaries
            
        Returns:
            DataFrame with one row per practice, columns are features
        """
        feature_dicts = []
        
        for practice in practices:
            try:
                features = self.convert_practice_to_features(practice)
                # Add practice identifier
                features['practice_name'] = practice.get('name', 'Unknown')
                features['practice_id'] = practice.get('id', '')
                feature_dicts.append(features)
            except Exception as e:
                logger.error(f"Error converting practice {practice.get('name', 'Unknown')}: {str(e)}")
                continue
        
        if not feature_dicts:
            logger.warning("No practices successfully converted")
            return pd.DataFrame()
        
        df = pd.DataFrame(feature_dicts)
        logger.info(f"✅ Converted {len(df)} practices to feature DataFrame")
        
        return df
    
    def enrich_with_external_data(self, practice_data: Dict, external_data: Dict) -> Dict:
        """
        Enrich practice data with external information
        (e.g., from review analysis, Google Maps enrichment)
        
        Args:
            practice_data: Base practice dictionary
            external_data: Additional data to merge
            
        Returns:
            Enriched practice dictionary
        """
        enriched = practice_data.copy()
        
        # Merge services
        if 'services' in external_data:
            existing_services = set(enriched.get('services', []))
            new_services = set(external_data['services'])
            enriched['services'] = list(existing_services | new_services)
        
        # Merge social proof
        if 'social_proof' in external_data:
            enriched['social_proof'] = external_data['social_proof']
        
        # Merge staff mentions
        if 'staff_mentions' in external_data:
            enriched['staff_count'] = len(external_data['staff_mentions'])
        
        # Update description if available
        if 'description' in external_data and external_data['description']:
            if not enriched.get('description') or enriched['description'] == 'Not Available':
                enriched['description'] = external_data['description']
        
        logger.debug(f"Enriched practice data with external information")
        
        return enriched
    
    def get_feature_importance_for_specialty(self, specialty: str) -> Dict[str, float]:
        """
        Get feature importance weights for a specific specialty
        
        Args:
            specialty: Specialty name (dermatology, plastic_surgery, etc.)
            
        Returns:
            Dictionary of feature names to importance weights
        """
        # Default weights
        base_weights = {
            'specialty_match': 1.0,
            'service_variety': 0.8,
            'reputation_score': 0.9,
            'independent_ownership': 0.7,
            'cashpay_focus': 0.8,
            'digital_presence': 0.6
        }
        
        # Specialty-specific adjustments
        if specialty == 'dermatology':
            base_weights['dermatology'] = 1.0
            base_weights['injectables_services'] = 0.9
            base_weights['laser_services'] = 0.9
            
        elif specialty == 'plastic_surgery':
            base_weights['plastic_surgery'] = 1.0
            base_weights['board_certified'] = 0.9
            base_weights['non_surgical_focus'] = 0.8
            
        elif specialty == 'obgyn':
            base_weights['obgyn_practice'] = 1.0
            base_weights['offers_aesthetic_gyn'] = 0.9
            base_weights['womens_wellness_focus'] = 0.8
            base_weights['solo_practitioner'] = 0.9
            
        elif specialty == 'medspa':
            base_weights['med_spa'] = 1.0
            base_weights['medical_director'] = 0.9
            base_weights['advanced_technology'] = 0.8
            base_weights['independent_ownership'] = 0.9
            
        elif specialty == 'family_practice':
            base_weights['family_practice'] = 1.0
            base_weights['dpc_model'] = 0.9
            base_weights['functional_medicine'] = 0.8
        
        return base_weights


# Export convenience function
def create_adapter() -> VenusScoringAdapter:
    """Create and return a Venus adapter instance"""
    return VenusScoringAdapter()
