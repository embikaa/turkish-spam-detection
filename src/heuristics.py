import re
import numpy as np

class SpamHeuristics:
    def __init__(self, generic_keywords):
        self.generic_keywords = generic_keywords

    def extract_structural_features(self, text):
        text_str = str(text)
        text_lower = text_str.lower()
        words = re.findall(r'\w+', text_lower)
        word_count = len(words)
        
        if word_count == 0:
            return [1, 0, 0, 0, 0]

        is_short = 1 if word_count <= 4 else 0
        
        generic_count = sum(1 for w in words if w in self.generic_keywords)
        is_generic = 1 if (word_count > 0 and generic_count / word_count > 0.5) else 0
        
        has_digits = 1 if re.search(r'\d+', text_str) else 0
        is_long = 1 if word_count > 20 else 0
        
        caps_ratio = sum(1 for c in text_str if c.isupper()) / len(text_str) if len(text_str) > 0 else 0
        is_caps_lock = 1 if caps_ratio > 0.6 and len(text_str) > 5 else 0
        
        return [is_short, is_generic, has_digits, is_long, is_caps_lock]

    def generate_weak_label(self, structural_features):
        is_short, is_generic, has_digits, is_long, is_caps_lock = structural_features
        
        
        if (is_short and is_generic) or is_caps_lock:
            return 1
        
       
        if is_long or has_digits:
            return 0
            
       
        if is_short:
            return 1
            
        return 0
