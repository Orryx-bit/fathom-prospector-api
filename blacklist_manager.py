import re
import os
from urllib.parse import urlparse
from typing import Set, List, Optional
import logging

logger = logging.getLogger(__name__)

class BlacklistManager:
    """
    Manages domain and URL pattern blacklists to prevent
    wasting time and money on problematic sites.
    """
    
    def __init__(self, 
                 domain_blacklist_path: str = 'domain_blacklist.txt',
                 pattern_blacklist_path: str = 'url_pattern_blacklist.txt'):
        """
        Initialize the blacklist manager.
        
        Args:
            domain_blacklist_path: Path to domain blacklist file
            pattern_blacklist_path: Path to URL pattern blacklist file
        """
        self.domain_blacklist: Set[str] = set()
        self.pattern_blacklist: List[re.Pattern] = []
        self.blacklist_hits = {}  # Track which rules are being hit
        
        self._load_domain_blacklist(domain_blacklist_path)
        self._load_pattern_blacklist(pattern_blacklist_path)
        
        logger.info(f"Loaded {len(self.domain_blacklist)} blacklisted domains")
        logger.info(f"Loaded {len(self.pattern_blacklist)} blacklisted patterns")
    
    def _load_domain_blacklist(self, filepath: str):
        """Load domain blacklist from file"""
        if not os.path.exists(filepath):
            logger.warning(f"Domain blacklist file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    self.domain_blacklist.add(line.lower())
    
    def _load_pattern_blacklist(self, filepath: str):
        """Load URL pattern blacklist from file"""
        if not os.path.exists(filepath):
            logger.warning(f"Pattern blacklist file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    try:
                        pattern = re.compile(line, re.IGNORECASE)
                        self.pattern_blacklist.append(pattern)
                    except re.error as e:
                        logger.error(f"Invalid regex pattern '{line}': {e}")
    
    def is_blacklisted(self, url: str) -> tuple[bool, Optional[str]]:
        """
        Check if a URL is blacklisted.
        
        Args:
            url: The URL to check
        
        Returns:
            tuple: (is_blacklisted: bool, reason: str or None)
        """
        if not url:
            return False, None
        
        # Parse the URL
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc or parsed.path
            
            # Remove www. prefix for matching
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check domain blacklist (exact match)
            if domain in self.domain_blacklist:
                self._record_hit(f"domain:{domain}")
                return True, f"Blacklisted domain: {domain}"
            
            # Check domain blacklist (subdomain match)
            for blacklisted_domain in self.domain_blacklist:
                if domain.endswith('.' + blacklisted_domain) or domain == blacklisted_domain:
                    self._record_hit(f"domain:{blacklisted_domain}")
                    return True, f"Blacklisted domain: {blacklisted_domain}"
            
            # Check URL pattern blacklist
            full_url = url.lower()
            for pattern in self.pattern_blacklist:
                if pattern.search(full_url):
                    self._record_hit(f"pattern:{pattern.pattern}")
                    return True, f"Blacklisted pattern: {pattern.pattern}"
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking blacklist for {url}: {e}")
            return False, None
    
    def _record_hit(self, rule: str):
        """Record when a blacklist rule is triggered"""
        self.blacklist_hits[rule] = self.blacklist_hits.get(rule, 0) + 1
    
    def get_stats(self) -> dict:
        """Get statistics on blacklist usage"""
        return {
            'total_domains': len(self.domain_blacklist),
            'total_patterns': len(self.pattern_blacklist),
            'top_hits': sorted(
                self.blacklist_hits.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def add_domain(self, domain: str):
        """Dynamically add a domain to the blacklist"""
        domain = domain.lower().strip()
        if domain.startswith('www.'):
            domain = domain[4:]
        self.domain_blacklist.add(domain)
        logger.info(f"Added domain to blacklist: {domain}")
    
    def add_pattern(self, pattern: str):
        """Dynamically add a URL pattern to the blacklist"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self.pattern_blacklist.append(compiled_pattern)
            logger.info(f"Added pattern to blacklist: {pattern}")
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")


# Global blacklist manager instance
_blacklist_manager = None

def get_blacklist_manager() -> BlacklistManager:
    """Get or create the global blacklist manager instance"""
    global _blacklist_manager
    if _blacklist_manager is None:
        _blacklist_manager = BlacklistManager()
    return _blacklist_manager
