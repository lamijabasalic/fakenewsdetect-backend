"""
Bosnian language preprocessing module for fake news detection.
Handles tokenization, stopwords removal, and special character handling (čćšđž).
"""

import re
import string
from typing import List

# Bosnian stopwords list
BOSNIAN_STOPWORDS = {
    'a', 'ali', 'baš', 'bez', 'bi', 'bih', 'bila', 'bili', 'bilo', 'bio', 'biti',
    'cijela', 'cijeli', 'cijelo', 'će', 'ćemo', 'ćete', 'ću', 'da', 'dakle', 'danas',
    'dok', 'do', 'duž', 'ga', 'gdje', 'gde', 'gore', 'i', 'ili', 'ima', 'imaju',
    'iz', 'između', 'između', 'ja', 'je', 'jedan', 'jedna', 'jedno', 'jer', 'jesam',
    'jesi', 'jesmo', 'jeste', 'jesu', 'još', 'ju', 'kada', 'kako', 'kao', 'koja',
    'koje', 'koji', 'kojima', 'koju', 'kroz', 'li', 'me', 'mene', 'mi', 'mimo',
    'može', 'možemo', 'možete', 'možeš', 'na', 'nad', 'nakon', 'nam', 'nama', 'nas',
    'naš', 'naša', 'naše', 'našeg', 'našem', 'naših', 'našim', 'našoj', 'našu',
    'ne', 'nego', 'neka', 'neke', 'neki', 'nekim', 'neko', 'nekog', 'nekoj', 'nekome',
    'neku', 'nešto', 'ni', 'nije', 'nijedan', 'nijedna', 'nijedno', 'nikad', 'nikada',
    'nikoga', 'niko', 'nikoga', 'nisam', 'nisi', 'nismo', 'niste', 'nisu', 'njega',
    'njegov', 'njegova', 'njegove', 'njegovo', 'njemu', 'njezin', 'njezina', 'njezine',
    'njezino', 'njih', 'njihov', 'njihova', 'njihove', 'njihovo', 'njima', 'njoj',
    'nju', 'no', 'o', 'od', 'odmah', 'oko', 'on', 'ona', 'onaj', 'oni', 'ono',
    'osim', 'ova', 'ovaj', 'ovako', 'ovamo', 'ove', 'ovo', 'pa', 'pak', 'po', 'pod',
    'pored', 'pre', 'preko', 'prema', 'prije', 'prije', 's', 'sa', 'sada', 'sam',
    'samo', 'se', 'sebe', 'sebi', 'si', 'smo', 'ste', 'su', 'sve', 'svi', 'svih',
    'svim', 'svima', 'svog', 'svoj', 'svoja', 'svoje', 'svojeg', 'svojem', 'svojih',
    'svojim', 'svojoj', 'svoju', 'svom', 'svome', 'ta', 'tada', 'taj', 'tako', 'takođe',
    'također', 'tamo', 'te', 'tebe', 'tebi', 'ti', 'tim', 'tima', 'to', 'toj', 'tome',
    'tu', 'tvoj', 'tvoja', 'tvoje', 'tvog', 'tvom', 'tvome', 'u', 'umjesto', 'umesto',
    'uz', 'vam', 'vama', 'vas', 'vaš', 'vaša', 'vaše', 'vašeg', 'vašem', 'vaših',
    'vašim', 'vašoj', 'vašu', 'već', 'većina', 'vi', 'vjerojatno', 'vjerovatno',
    'za', 'zar', 'zbog', 'će', 'ćemo', 'ćete', 'ću', 'čak', 'čega', 'čemu', 'čija',
    'čije', 'čiji', 'čijim', 'čijima', 'čiju', 'čim', 'šta', 'što', 'što', 'šta'
}


def normalize_bosnian_text(text: str) -> str:
    """
    Normalize Bosnian text: lowercase and preserve special characters (čćšđž).
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    return text.lower().strip()


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation while preserving Bosnian special characters.
    
    Args:
        text: Input text string
        
    Returns:
        Text without punctuation
    """
    # Keep Bosnian characters: čćšđžČĆŠĐŽ
    bosnian_chars = 'čćšđžČĆŠĐŽ'
    # Remove all punctuation except Bosnian characters
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization by whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    # Split by whitespace and filter empty strings
    tokens = [token.strip() for token in text.split() if token.strip()]
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove Bosnian stopwords from token list.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of tokens without stopwords
    """
    return [token for token in tokens if token not in BOSNIAN_STOPWORDS]


def simple_stem(word: str) -> str:
    """
    Simple stemming for Bosnian language (basic suffix removal).
    This is a simplified version - for production, use a proper stemmer.
    
    Args:
        word: Input word
        
    Returns:
        Stemmed word
    """
    if len(word) < 4:
        return word
    
    # Common Bosnian suffixes to remove
    suffixes = ['ima', 'ama', 'ima', 'ama', 'om', 'em', 'om', 'em', 'ih', 'ah', 'ih', 'ah']
    
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    
    return word


def preprocess_bosnian_text(text: str, use_stemming: bool = False) -> str:
    """
    Complete preprocessing pipeline for Bosnian text.
    
    Steps:
    1. Normalize (lowercase)
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    5. (Optional) Stemming
    6. Join back to string
    
    Args:
        text: Input text string
        use_stemming: Whether to apply stemming (default: False)
        
    Returns:
        Preprocessed text string
    """
    if not text:
        return ""
    
    # Step 1: Normalize
    text = normalize_bosnian_text(text)
    
    # Step 2: Remove punctuation
    text = remove_punctuation(text)
    
    # Step 3: Tokenize
    tokens = tokenize(text)
    
    # Step 4: Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Step 5: Optional stemming
    if use_stemming:
        tokens = [simple_stem(token) for token in tokens]
    
    # Step 6: Join back
    return " ".join(tokens)


