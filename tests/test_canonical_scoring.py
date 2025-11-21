import unittest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scoring_system.universal_scorer import UniversalScorer

class TestCanonicalScoring(unittest.TestCase):
    def setUp(self):
        self.scorer = UniversalScorer(vendor="venus")

    def test_medspa_high_readiness(self):
        """Scenario 1: Medspa with Injectables -> High Readiness"""
        data = {
            'name': 'Luxe Medspa',
            'types': ['spa', 'beauty_salon'],
            'text': 'We offer botox, juvederm, and pdo threads for anti-aging.',
            'headers': ['Services', 'Injectables'],
            'footer_text': 'Contact us today.'
        }
        result = self.scorer.score_practice(data)
        self.assertEqual(result['specialty'], 'medspa')
        self.assertEqual(result['track'], 'new_prospect')
        self.assertEqual(result['readiness_tier'], 'High')
        self.assertGreaterEqual(result['score'], 40)
        print(f"\n✅ Scenario 1 (Medspa High): Score {result['score']} - Tier {result['readiness_tier']}")

    def test_fp_medium_readiness(self):
        """Scenario 2: FP with no aesthetics -> Medium Readiness"""
        data = {
            'name': 'Smith Family Practice',
            'types': ['doctor', 'health'],
            'text': 'Serving the community for 20 years. Comprehensive care for the whole family.',
            'headers': ['About Us', 'Services'],
            'footer_text': 'Copyright 2023.'
        }
        result = self.scorer.score_practice(data)
        self.assertEqual(result['specialty'], 'family_practice')
        self.assertEqual(result['track'], 'new_prospect')
        self.assertEqual(result['readiness_tier'], 'Medium') # "established practice" logic might need refinement, but "comprehensive care" is in medium list
        print(f"✅ Scenario 2 (FP Medium): Score {result['score']} - Tier {result['readiness_tier']}")

    def test_pediatrics_low_readiness(self):
        """Scenario 3: Pediatrics -> Low Readiness"""
        data = {
            'name': 'Little Ones Pediatrics',
            'types': ['doctor'],
            'text': 'Specializing in sick visits, immunizations, and newborn care.',
            'headers': ['Pediatric Care'],
            'footer_text': ''
        }
        result = self.scorer.score_practice(data)
        # Should detect as FP or Internal Med (or maybe Medspa fallback if not caught), 
        # but score should be Low due to disqualifiers/low signals
        # "pediatrics" is in Low signals for FP
        self.assertIn(result['specialty'], ['family_practice', 'internal_med', 'medspa']) 
        self.assertEqual(result['readiness_tier'], 'Low')
        print(f"✅ Scenario 3 (Pediatrics Low): Score {result['score']} - Tier {result['readiness_tier']}")

    def test_device_owner_tribella(self):
        """Scenario 4: Device Owner (TriBella) -> Device Owner Track"""
        data = {
            'name': 'Radiant Skin',
            'types': ['spa'],
            'text': 'Experience the power of TriBella for total skin rejuvenation.',
            'headers': ['Treatments'],
            'footer_text': ''
        }
        result = self.scorer.score_practice(data)
        self.assertEqual(result['track'], 'device_owner')
        self.assertEqual(result['readiness_tier'], 'Device Owner')
        self.assertTrue('tribella' in result['breakdown']['found_devices'])
        self.assertIn('opportunity_breakdown', result['breakdown'])
        print(f"✅ Scenario 4 (Device Owner): Track {result['track']} - Tier {result['readiness_tier']} - Opp Score {result['breakdown']['opportunity_breakdown'].get('opportunity_score')}")

    def test_obgyn_high_readiness(self):
        """Scenario 5: OBGYN with Postpartum/Pelvic -> High Readiness"""
        data = {
            'name': 'Center for Women\'s Health',
            'types': ['doctor'],
            'text': 'Specializing in postpartum recovery, pelvic floor therapy, and sexual wellness.',
            'headers': ['Our Services'],
            'footer_text': ''
        }
        result = self.scorer.score_practice(data)
        self.assertEqual(result['specialty'], 'obgyn')
        self.assertEqual(result['readiness_tier'], 'High')
        print(f"✅ Scenario 5 (OBGYN High): Score {result['score']} - Tier {result['readiness_tier']}")

    def test_wellness_high_readiness(self):
        """Scenario 6: Wellness with GLP-1 -> High Readiness"""
        data = {
            'name': 'Vitality Wellness',
            'types': ['health'],
            'text': 'Medical weight loss programs including Semaglutide and HRT.',
            'headers': ['Weight Loss'],
            'footer_text': ''
        }
        result = self.scorer.score_practice(data)
        self.assertEqual(result['specialty'], 'wellness')
        self.assertEqual(result['readiness_tier'], 'High')
        print(f"✅ Scenario 6 (Wellness High): Score {result['score']} - Tier {result['readiness_tier']}")

    def test_plastic_surgery_low_readiness(self):
        """Scenario 7: Plastic Surgery (Reconstructive) -> Low Readiness"""
        data = {
            'name': 'City Plastic & Reconstructive Surgery',
            'types': ['doctor'],
            'text': 'Specializing in trauma reconstruction, hand surgery, and burn care.',
            'headers': ['Procedures'],
            'footer_text': ''
        }
        result = self.scorer.score_practice(data)
        self.assertEqual(result['specialty'], 'plastic_surgery')
        self.assertEqual(result['readiness_tier'], 'Low')
        print(f"✅ Scenario 7 (Plastic Low): Score {result['score']} - Tier {result['readiness_tier']}")

    def test_competitor_device_owner(self):
        """Scenario 8: Competitor (CoolSculpting) -> Device Owner Track"""
        data = {
            'name': 'Body Contour Clinic',
            'types': ['spa'],
            'text': 'We offer CoolSculpting Elite for fat reduction.',
            'headers': ['Body'],
            'footer_text': ''
        }
        result = self.scorer.score_practice(data)
        self.assertEqual(result['track'], 'device_owner')
        self.assertTrue('coolsculpting' in result['breakdown']['found_devices'])
        print(f"✅ Scenario 8 (Competitor): Track {result['track']} - Found {result['breakdown']['found_devices']}")

if __name__ == '__main__':
    unittest.main()
