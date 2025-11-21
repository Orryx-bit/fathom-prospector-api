import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class UniversalScorer:
    """
    Canonical Scoring System for Fathom Prospector.
    Implements specialty-aware, readiness-based scoring with strict 'Golden Rule' enforcement.
    """

    def __init__(self, vendor: str = "venus"):
        self.vendor = vendor.lower()
        self.config_dir = Path(__file__).parent.parent / 'scoring_configs' / self.vendor
        self.configs = {}
        self._load_all_configs()

    def _load_all_configs(self):
        """Pre-load all specialty configurations"""
        if not self.config_dir.exists():
            logger.error(f"Configuration directory not found: {self.config_dir}")
            return

        for config_file in self.config_dir.glob('*.yaml'):
            try:
                with open(config_file, 'r') as f:
                    specialty = config_file.stem.replace(f"{self.vendor}-", "")
                    self.configs[specialty] = yaml.safe_load(f)
                    logger.info(f"Loaded scoring config for: {specialty}")
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")

    def detect_specialty(self, practice_data: Dict) -> str:
        """
        Detect practice specialty based on hierarchy and explicit signals.
        Hierarchy: Medspa > Wellness > OB/GYN > Family Practice > Plastic Surgery > Dermatology > ENT > Urology > Other
        
        CRITICAL: Do NOT reclassify Family Practice as Medspa just because they have injectables.
        Only reclassify if the entire identity is Medspa-focused.
        """
        
        # 1. Gather Signals
        google_types = practice_data.get('types', [])
        name = practice_data.get('name', '').lower()
        text = practice_data.get('text', '').lower()
        
        # Helper to check for strong medspa identity
        def is_strong_medspa_identity():
            # Google explicitly says Medspa/Spa/Beauty Salon
            if 'spa' in google_types or 'beauty_salon' in google_types:
                return True
            # Name explicitly says Medspa/Aesthetics
            if 'medspa' in name or 'medical spa' in name or 'aesthetics' in name or 'laser center' in name:
                return True
            return False

        # 2. Hierarchy Checks
        
        # Medspa / Aesthetics
        # Only classify as Medspa if strong identity signals exist.
        # Mere presence of "botox" in text is NOT enough.
        if is_strong_medspa_identity():
             return 'medspa'

        # Wellness / Weight Loss
        if 'weight loss' in name or 'wellness' in name:
            return 'wellness'
        if 'weight loss clinic' in text or 'wellness center' in text:
            return 'wellness'

        # OB/GYN
        if 'obstetrician' in name or 'gynecologist' in name or 'obgyn' in name or 'women' in name:
            return 'obgyn'
        if 'obstetrics' in text or 'gynecology' in text:
            return 'obgyn'

        # Family Practice / Internal Med
        if 'family' in name or 'primary care' in name or 'internal medicine' in name or 'pediatrics' in name:
            if 'family' in name or 'pediatrics' in name: return 'family_practice'
            return 'internal_med'
        if 'family medicine' in text or 'family practice' in text or 'pediatrics' in text:
            return 'family_practice'
        if 'internal medicine' in text or 'internist' in text:
            return 'internal_med'

        # Plastic Surgery
        if 'plastic' in name or 'cosmetic surgery' in name:
            return 'plastic_surgery'
        if 'plastic surgeon' in text:
            return 'plastic_surgery'

        # Dermatology
        if 'derm' in name or 'skin' in name:
             return 'dermatology'
        if 'dermatology' in text or 'dermatologist' in text:
            return 'dermatology'
            
        # ENT
        if 'ent' in name or 'ear' in name or 'nose' in name or 'throat' in name:
            return 'ent'
        if 'otolaryngology' in text:
            return 'ent'
            
        # Urology
        if 'urology' in name or 'men' in name:
            return 'urology'
        if 'urologist' in text:
            return 'urology'

        # Default / Fallback
        # If no specific medical specialty is found, and we have some aesthetic signals, 
        # we might default to Medspa, but be careful.
        # For now, default to Medspa as catch-all for aesthetic businesses not otherwise classified.
        return 'medspa'

    def score_practice(self, practice_data: Dict) -> Dict:
        """
        Main scoring entry point.
        """
        # 1. Detect Specialty
        specialty = self.detect_specialty(practice_data)
        config = self.configs.get(specialty, self.configs.get('medspa')) # Fallback
        
        if not config:
            logger.error(f"No configuration found for specialty: {specialty}")
            return {'score': 0, 'error': 'Missing configuration'}

        # 2. Extract/Normalize Keywords from Practice Data
        full_text = (
            practice_data.get('title', '') + ' ' +
            practice_data.get('description', '') + ' ' +
            practice_data.get('text', '') + ' ' + 
            ' '.join(practice_data.get('headers', [])) + ' ' +
            practice_data.get('footer_text', '')
        ).lower()

        # Helper to check for keywords in text
        def find_signals(signal_list):
            found = []
            for signal in signal_list:
                if signal in full_text:
                    found.append(signal)
            return found

        # 3. Device Owner Check (Priority 1)
        found_devices = find_signals(config.get('device_ownership_signals', []))
        is_device_owner = len(found_devices) > 0
        
        # 4. Calculate Readiness Score (Always calculated for context)
        # Merge synonyms into high signals for broader detection
        expansion_config = config.get('intelligence_expansion_phase', {})
        synonyms = expansion_config.get('synonyms', [])
        adjacent = expansion_config.get('adjacent_services', [])
        
        high_signals_list = config['readiness_signals']['high'] + synonyms
        medium_signals_list = config['readiness_signals']['medium']
        low_signals_list = config['readiness_signals']['low']
        
        high_signals = find_signals(high_signals_list)
        medium_signals = find_signals(medium_signals_list)
        low_signals = find_signals(low_signals_list)
        
        weights = config['scoring_weights']
        
        # High Score: Base + Bonus (capped)
        high_score = 0
        if high_signals:
            high_score = weights['readiness_high_base'] + (len(high_signals) - 1) * weights['readiness_high_bonus']
            high_score = min(high_score, weights['readiness_high_cap'])
            
        # Medium Score
        readiness_score = 0
        readiness_tier = "Low"
        
        if high_signals:
            readiness_score = high_score
            readiness_tier = "High"
        elif medium_signals:
            readiness_score = weights['readiness_medium']
            readiness_tier = "Medium"
        elif low_signals:
            readiness_score = weights['readiness_low']
            readiness_tier = "Low"
        else:
            readiness_score = 0
            readiness_tier = "None"

        # 5. Practice Match & Expansion
        practice_signals = find_signals(config.get('practice_type_signals', {}).get('helpful', []))
        
        # Merge adjacent services into expansion signals
        expansion_signals_list = config.get('expansion_signals', []) + adjacent
        expansion_signals = find_signals(expansion_signals_list)
        
        practice_score = weights['practice_match'] if practice_signals else 0
        expansion_score = weights['expansion'] if expansion_signals else 0
        
        # 6. Disqualifiers
        found_disqualifiers = find_signals(config.get('disqualifiers', []))
        disqualifier_penalty = len(found_disqualifiers) * weights['disqualifier']
        
        # 7. Device Owner Opportunity Scoring
        opportunity_breakdown = {}
        if is_device_owner:
            readiness_tier = "Device Owner"
            # Calculate opportunity score based on gaps
            # This is a simplified version; real logic would check specific categories
            # For now, we give points for missing categories that are high value
            
            # Check for key categories
            has_body = any(s in full_text for s in ['body', 'fat', 'sculpt', 'muscle'])
            has_hair = any(s in full_text for s in ['hair removal', 'laser hair'])
            has_skin = any(s in full_text for s in ['skin', 'resurfacing', 'ipl'])
            
            opportunity_score = 0
            if not has_body: opportunity_score += 20
            if not has_hair: opportunity_score += 15
            if not has_skin: opportunity_score += 15
            
            # Add expansion signals to opportunity
            opportunity_score += expansion_score
            
            # Use opportunity score as the primary score for Device Owners?
            # User said: "Return 'readiness_tier: Device Owner' and provide a structured opportunity breakdown"
            # And "Do not zero out readiness completely."
            # So we keep readiness_score in breakdown, but maybe total_score reflects opportunity?
            # Let's keep total_score as the "Lead Score" - for a device owner, a high lead score means high opportunity to sell MORE.
            # So opportunity_score makes sense here.
            
            # Base score for being a device owner (proven buyer)
            base_owner_score = 50 
            total_score = base_owner_score + opportunity_score + disqualifier_penalty
            
            opportunity_breakdown = {
                'missing_body_contouring': not has_body,
                'missing_hair_removal': not has_hair,
                'missing_skin_resurfacing': not has_skin,
                'opportunity_score': opportunity_score
            }
            
        else:
            # Standard New Prospect Scoring
            total_score = readiness_score + practice_score + expansion_score + disqualifier_penalty

        total_score = max(0, min(100, total_score))

        # 8. Construct Output
        track = "device_owner" if is_device_owner else "new_prospect"
        
        result = {
            'specialty': specialty,
            'track': track,
            'readiness_tier': readiness_tier,
            'score': total_score,
            'breakdown': {
                'readiness_score': readiness_score,
                'practice_match_score': practice_score,
                'expansion_score': expansion_score,
                'disqualifier_penalty': disqualifier_penalty,
                'opportunity_breakdown': opportunity_breakdown,
                'found_high_signals': high_signals,
                'found_medium_signals': medium_signals,
                'found_low_signals': low_signals,
                'found_devices': found_devices,
                'found_practice_signals': practice_signals,
                'found_expansion_signals': expansion_signals,
                'found_disqualifiers': found_disqualifiers
            }
        }
        
        return result
