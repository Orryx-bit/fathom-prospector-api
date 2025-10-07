
"""
Venus-specific adapter for the lead scoring system.
Converts Venus practice data to the format expected by the scoring engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VenusScoringAdapter:
    """Adapts Venus practice data to the scoring engine format"""
    
    def __init__(self):
        """Initialize the adapter"""
        pass
    
    def convert_practice_to_features(self, practice_data: Dict) -> Dict:
        """
        Convert a single practice dict to feature dict for scoring
        
        Args:
            practice_data: Dictionary containing practice information
            
        Returns:
            Dictionary of features for scoring engine
        """
        features = {}
        
        # Extract basic info
        practice_name = practice_data.get('name', '').lower()
        practice_desc = practice_data.get('description', '').lower()
        all_text = f"{practice_name} {practice_desc}".lower()
        services = practice_data.get('services', [])
        service_text = ' '.join([s.lower() for s in services])
        
        # SPECIALTY MATCH FEATURES
        # Dermatology & Plastic Surgery (original)
        features['dermatology'] = 1.0 if any(term in all_text for term in ['dermatology', 'dermatologist', 'skin care', 'skin clinic']) else 0.0
        features['plastic_surgery'] = 1.0 if any(term in all_text for term in ['plastic surgery', 'plastic surgeon', 'cosmetic surgery']) else 0.0
        features['cosmetic_clinic'] = 1.0 if any(term in all_text for term in ['cosmetic clinic', 'cosmetic center']) else 0.0
        features['med_spa'] = 1.0 if any(term in all_text for term in ['med spa', 'medical spa', 'medspa']) else 0.0
        features['aesthetic_clinic'] = 1.0 if any(term in all_text for term in ['aesthetic', 'beauty', 'rejuvenation']) else 0.0
        
        # OB/GYN Specialty Features
        features['obgyn_practice'] = 1.0 if any(term in all_text for term in ['obgyn', 'ob-gyn', 'ob/gyn', 'obstetrics', 'gynecology']) else 0.0
        features['womens_health_clinic'] = 1.0 if any(term in all_text for term in ["women's health", 'womens health', 'female health']) else 0.0
        features['cosmetic_gynecology'] = 1.0 if any(term in all_text for term in ['cosmetic gynecology', 'aesthetic gynecology', 'vaginal rejuvenation']) else 0.0
        
        # Family Practice Specialty Features
        features['family_practice_aesthetic'] = 1.0 if any(term in all_text for term in ['family medicine', 'family practice']) and any(term in all_text for term in ['aesthetic', 'cosmetic', 'beauty']) else 0.0
        features['integrative_medicine'] = 1.0 if any(term in all_text for term in ['integrative medicine', 'holistic medicine', 'functional medicine']) else 0.0
        features['functional_medicine'] = 1.0 if any(term in all_text for term in ['functional medicine', 'anti-aging', 'longevity', 'regenerative medicine']) else 0.0
        features['internal_medicine'] = 1.0 if any(term in all_text for term in ['internal medicine', 'internist']) else 0.0
        features['general_practice'] = 1.0 if any(term in all_text for term in ['general practice', 'general practitioner', 'primary care']) else 0.0
        
        # AESTHETIC SERVICES FEATURES
        features['body_contouring'] = 1.0 if any(term in service_text for term in ['body contouring', 'body sculpting', 'coolsculpting', 'fat reduction']) else 0.0
        features['laser_hair_removal'] = 1.0 if any(term in service_text for term in ['laser hair removal', 'hair removal', 'laser']) else 0.0
        features['injectables'] = 1.0 if any(term in service_text for term in ['botox', 'filler', 'injectable', 'juvederm', 'restylane']) else 0.0
        features['skin_tightening'] = 1.0 if any(term in service_text for term in ['skin tightening', 'tightening', 'firming']) else 0.0
        features['photorejuvenation'] = 1.0 if any(term in service_text for term in ['photorejuvenation', 'ipl', 'photo facial', 'rejuvenation']) else 0.0
        
        # COMPETING DEVICES FEATURES
        multi_platform_competitors = ['lumenis m22', 'alma harmony', 'cutera xeo', 'syneron etwo', 'cynosure elite', 'vydence', 'btl exilis']
        single_devices = ['coolsculpting', 'thermage', 'ultherapy', 'sculptra', 'morpheus8', 'potenza']
        
        has_multi_platform = any(device in all_text for device in multi_platform_competitors)
        has_single_device = any(device in all_text for device in single_devices)
        
        features['multi_platform_present'] = 1.0 if has_multi_platform else 0.0
        features['single_device_present'] = 1.0 if has_single_device and not has_multi_platform else 0.0
        features['no_devices'] = 1.0 if not has_multi_platform and not has_single_device else 0.0
        
        # SOCIAL ACTIVITY FEATURES
        social_links = practice_data.get('social_links', [])
        features['instagram_present'] = 1.0 if any('instagram' in link.lower() for link in social_links) else 0.0
        features['facebook_present'] = 1.0 if any('facebook' in link.lower() for link in social_links) else 0.0
        features['linkedin_present'] = 1.0 if any('linkedin' in link.lower() for link in social_links) else 0.0
        features['social_engagement'] = min(1.0, len(social_links) / 3.0)  # Normalized by 3 platforms
        
        # PRACTICE SIZE FEATURES (inverted - smaller is better)
        staff_count = practice_data.get('staff_count', 0)
        
        # Check for startup indicators
        startup_indicators = ['new', 'recently opened', 'grand opening', 'now open', 'established 202', 'founded 202']
        is_startup = any(indicator in all_text for indicator in startup_indicators)
        
        if is_startup or staff_count <= 2:
            features['startup_very_small'] = 1.0
            features['small_practice'] = 0.0
            features['medium_practice'] = 0.0
            features['large_practice'] = 0.0
        elif staff_count <= 4:
            features['startup_very_small'] = 0.0
            features['small_practice'] = 1.0
            features['medium_practice'] = 0.0
            features['large_practice'] = 0.0
        elif staff_count <= 8:
            features['startup_very_small'] = 0.0
            features['small_practice'] = 0.0
            features['medium_practice'] = 1.0
            features['large_practice'] = 0.0
        else:
            features['startup_very_small'] = 0.0
            features['small_practice'] = 0.0
            features['medium_practice'] = 0.0
            features['large_practice'] = 1.0
        
        # REVIEWS & RATING FEATURES
        rating = practice_data.get('rating', 0)
        review_count = practice_data.get('review_count', 0)
        
        if rating >= 4.5 and review_count >= 50:
            features['high_rating_high_volume'] = 1.0
            features['high_rating_medium_volume'] = 0.0
            features['medium_rating'] = 0.0
            features['low_rating'] = 0.0
        elif rating >= 4.0 and review_count >= 20:
            features['high_rating_high_volume'] = 0.0
            features['high_rating_medium_volume'] = 1.0
            features['medium_rating'] = 0.0
            features['low_rating'] = 0.0
        elif rating >= 3.5:
            features['high_rating_high_volume'] = 0.0
            features['high_rating_medium_volume'] = 0.0
            features['medium_rating'] = 1.0
            features['low_rating'] = 0.0
        else:
            features['high_rating_high_volume'] = 0.0
            features['high_rating_medium_volume'] = 0.0
            features['medium_rating'] = 0.0
            features['low_rating'] = 1.0
        
        # SEARCH VISIBILITY FEATURES
        features['has_website'] = 1.0 if practice_data.get('website') else 0.0
        features['has_phone'] = 1.0 if practice_data.get('phone') else 0.0
        features['has_email'] = 1.0 if '@' in practice_desc else 0.0  # Rough check
        
        # GEOGRAPHY FIT FEATURES
        address = practice_data.get('address', '').lower()
        
        small_market_indicators = ['rd', 'drive', 'country', 'rural', 'main street', 'main st', 'town', 'village']
        over_serviced_areas = ['beverly hills', 'manhattan', 'miami beach', 'scottsdale', 'malibu', 'newport beach', 'la jolla']
        
        if any(indicator in address for indicator in small_market_indicators):
            features['small_market'] = 1.0
            features['suburban'] = 0.0
            features['urban'] = 0.0
            features['over_serviced'] = 0.0
        elif any(area in address for area in over_serviced_areas):
            features['small_market'] = 0.0
            features['suburban'] = 0.0
            features['urban'] = 0.0
            features['over_serviced'] = 1.0
        elif any(term in address for term in ['downtown', 'center city', 'midtown']):
            features['small_market'] = 0.0
            features['suburban'] = 0.0
            features['urban'] = 1.0
            features['over_serviced'] = 0.0
        else:
            features['small_market'] = 0.0
            features['suburban'] = 1.0
            features['urban'] = 0.0
            features['over_serviced'] = 0.0
        
        # WEIGHT LOSS / GLP-1 FEATURES
        glp1_keywords = ['semaglutide', 'tirzepatide', 'ozempic', 'wegovy', 'mounjaro', 'compounded glp-1']
        weight_keywords = ['weight loss', 'weight management', 'medical weight', 'metabolic medicine']
        hormone_keywords = ['biote', 'hormone pellets', 'hormone optimization', 'hormone therapy']
        iv_keywords = ['iv therapy', 'iv drip', 'vitamin infusion']
        nutrition_keywords = ['nutrition', 'nutritional counseling', 'dietitian', 'diet program']
        
        features['glp1_services'] = 1.0 if any(kw in all_text for kw in glp1_keywords) else 0.0
        features['weight_management'] = 1.0 if any(kw in all_text for kw in weight_keywords) else 0.0
        features['hormone_therapy'] = 1.0 if any(kw in all_text for kw in hormone_keywords) else 0.0
        features['iv_therapy'] = 1.0 if any(kw in all_text for kw in iv_keywords) else 0.0
        features['nutrition_counseling'] = 1.0 if any(kw in all_text for kw in nutrition_keywords) else 0.0
        features['weight_loss_aesthetics'] = 1.0 if any(kw in all_text for kw in weight_keywords) and any(kw in all_text for kw in ['body contouring', 'skin tightening', 'aesthetic']) else 0.0
        
        # POSTPARTUM SERVICES FEATURES (for OB/GYN)
        mommy_makeover_keywords = ['mommy makeover', 'post-pregnancy', 'postpartum body', 'after baby']
        postpartum_contouring_keywords = ['postpartum body contouring', 'post-pregnancy body sculpting', 'after pregnancy body']
        diastasis_keywords = ['diastasis recti', 'abdominal separation', 'post-pregnancy core']
        postpartum_tightening_keywords = ['postpartum skin tightening', 'post-pregnancy skin', 'after baby skin']
        
        features['mommy_makeover'] = 1.0 if any(kw in all_text for kw in mommy_makeover_keywords) else 0.0
        features['postpartum_body_contouring'] = 1.0 if any(kw in all_text for kw in postpartum_contouring_keywords) or (features['body_contouring'] and features['obgyn_practice']) else 0.0
        features['diastasis_recti_treatment'] = 1.0 if any(kw in all_text for kw in diastasis_keywords) else 0.0
        features['postpartum_skin_tightening'] = 1.0 if any(kw in all_text for kw in postpartum_tightening_keywords) or (features['skin_tightening'] and features['obgyn_practice']) else 0.0
        
        # WELLNESS PROGRAMS FEATURES (for Family Practice)
        longevity_keywords = ['longevity', 'healthspan', 'lifespan optimization', 'age management']
        preventive_keywords = ['preventive care', 'preventative medicine', 'wellness program', 'health optimization']
        concierge_keywords = ['concierge', 'membership medicine', 'direct primary care', 'dpc']
        
        features['longevity_programs'] = 1.0 if any(kw in all_text for kw in longevity_keywords) else 0.0
        features['preventive_care_focus'] = 1.0 if any(kw in all_text for kw in preventive_keywords) else 0.0
        features['concierge_medicine'] = 1.0 if any(kw in all_text for kw in concierge_keywords) else 0.0
        
        # SKIN LAXITY TRIGGERS FEATURES
        post_surgical_keywords = ['post-surgical', 'facelift', 'body contouring surgery', 'blepharoplasty', 'mommy makeover', 'post-cosmetic surgery']
        bariatric_keywords = ['bariatric surgery', 'gastric bypass', 'significant weight loss', 'post-bariatric']
        postpartum_keywords = ['post-pregnancy', 'postpartum', 'post-natal', 'after pregnancy']
        
        features['post_surgical'] = 1.0 if any(kw in all_text for kw in post_surgical_keywords) else 0.0
        features['bariatric_patients'] = 1.0 if any(kw in all_text for kw in bariatric_keywords) else 0.0
        features['postpartum'] = 1.0 if any(kw in all_text for kw in postpartum_keywords) else 0.0
        
        # GROWTH INDICATORS FEATURES
        recently_opened_keywords = ['new', 'recently opened', 'grand opening', 'now open', 'opening soon', 'established 202', 'founded 202', 'new location']
        expanding_keywords = ['expanding', 'adding services', 'new treatments', 'introducing', 'upgraded', 'state-of-the-art', 'newly renovated']
        active_marketing_keywords = ['follow us', 'like us', '@', 'social media', 'check out our']
        professional_branding_keywords = ['award-winning', 'certified', 'accredited', 'board-certified', 'premier', 'leading', 'elite']
        
        features['recently_opened'] = 1.0 if any(kw in all_text for kw in recently_opened_keywords) else 0.0
        features['expanding_services'] = 1.0 if any(kw in all_text for kw in expanding_keywords) else 0.0
        features['active_marketing'] = 1.0 if (len(social_links) >= 2 or any(kw in all_text for kw in active_marketing_keywords)) else 0.0
        features['professional_branding'] = 1.0 if any(kw in all_text for kw in professional_branding_keywords) else 0.0
        
        # HIGH VALUE SPECIALTIES FEATURES
        family_practice_keywords = ['family medicine', 'family practice', 'family physician', 'primary care', 'general practice']
        obgyn_keywords = ['obgyn', 'ob-gyn', 'obstetrics', 'gynecology', "women's health", 'womens health']
        concierge_keywords = ['concierge medicine', 'concierge practice', 'membership practice', 'direct primary care', 'dpc']
        functional_medicine_keywords = ['functional medicine', 'integrative medicine', 'holistic', 'wellness center', 'anti-aging']
        
        features['family_practice'] = 1.0 if any(kw in all_text for kw in family_practice_keywords) else 0.0
        features['obgyn'] = 1.0 if any(kw in all_text for kw in obgyn_keywords) else 0.0
        features['concierge_medicine'] = 1.0 if any(kw in all_text for kw in concierge_keywords) else 0.0
        features['functional_medicine'] = 1.0 if any(kw in all_text for kw in functional_medicine_keywords) else 0.0
        
        return features
    
    def create_dataframe_from_practices(self, practices: List[Dict]) -> pd.DataFrame:
        """
        Convert list of practices to DataFrame for scoring
        
        Args:
            practices: List of practice dictionaries
            
        Returns:
            DataFrame ready for scoring engine
        """
        rows = []
        for i, practice in enumerate(practices):
            features = self.convert_practice_to_features(practice)
            # Add identifier
            features['business_id'] = practice.get('name', f'practice_{i}')
            features['name'] = practice.get('name', f'Practice {i}')
            rows.append(features)
        
        df = pd.DataFrame(rows)
        
        # Ensure business_id is first column
        if 'business_id' in df.columns:
            cols = ['business_id'] + [col for col in df.columns if col != 'business_id']
            df = df[cols]
        
        return df
    
    def map_score_back_to_practice(self, practice_data: Dict, score_result: Dict) -> Dict:
        """
        Map scoring results back to practice format
        
        Args:
            practice_data: Original practice data
            score_result: Scoring engine results
            
        Returns:
            Enhanced practice data with scores
        """
        # Create enhanced practice record
        enhanced = practice_data.copy()
        
        # Add scoring data
        enhanced['ai_score'] = int(score_result.get('score', 0))
        enhanced['confidence_level'] = score_result.get('confidence_flag', 'Low')
        enhanced['data_completeness'] = score_result.get('data_completeness', 0.0)
        
        # Parse top contributors
        top_contributors = score_result.get('top_3_contributors', '')
        enhanced['top_contributors'] = top_contributors
        
        # Create score breakdown dict
        score_breakdown = {}
        if 'top_1_feature' in score_result:
            score_breakdown['top_1'] = {
                'feature': score_result.get('top_1_feature', ''),
                'contribution': score_result.get('top_1_contribution_pct', 0.0)
            }
        if 'top_2_feature' in score_result:
            score_breakdown['top_2'] = {
                'feature': score_result.get('top_2_feature', ''),
                'contribution': score_result.get('top_2_contribution_pct', 0.0)
            }
        if 'top_3_feature' in score_result:
            score_breakdown['top_3'] = {
                'feature': score_result.get('top_3_feature', ''),
                'contribution': score_result.get('top_3_contribution_pct', 0.0)
            }
        
        enhanced['score_breakdown'] = score_breakdown
        
        return enhanced
