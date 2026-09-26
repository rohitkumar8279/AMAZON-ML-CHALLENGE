import re
import unicodedata

def normalize_text_basic(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Unicode normalization to ascii
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove punctuation, but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_business_name(name):
    """
    Normalizes business names, handling common legal suffixes.
    """
    name = normalize_text_basic(name)
    
    # Common legal suffixes to remove or normalize
    # Using word boundaries to ensure we only replace full words
    legal_suffixes = [
        r'\binc\b', r'\bincorporated\b', r'\bllc\b', r'\bcorp\b', r'\bcorporation\b',
        r'\bltd\b', r'\blimited\b', r'\bco\b', r'\bcompany\b', r'\bplc\b', r'\bpvt\b', r'\bprivate\b'
    ]
    
    # We might want to keep the legal suffix separately if it's a feature, 
    # but for pure normalization blocking, removing them helps matching.
    # We will remove them for the normalized field.
    for suffix in legal_suffixes:
        name = re.sub(suffix, '', name)
        
    return re.sub(r'\s+', ' ', name).strip()

def normalize_business_address(address):
    """
    Normalizes business addresses, expanding common abbreviations and extracting numbers.
    """
    address = normalize_text_basic(address)
    
    # Common address abbreviation normalization
    address_replacements = {
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bsq\b': 'square',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment',
        r'\bdept\b': 'department',
        r'\bhwy\b': 'highway'
    }
    
    for pattern, replacement in address_replacements.items():
        address = re.sub(pattern, replacement, address)
        
    return re.sub(r'\s+', ' ', address).strip()

def extract_numbers_from_address(address):
    """
    Extracts numerical tokens from addresses (useful for ZIP codes, street numbers, PO Boxes).
    """
    if not isinstance(address, str):
        return ""
    # Extract consecutive digits
    numbers = re.findall(r'\d+', address)
    return " ".join(numbers)

def tokenize_string(text):
    """
    Returns a sorted tuple of tokens for consistent comparison.
    """
    if not text:
        return ()
    return tuple(sorted(text.split()))

if __name__ == '__main__':
    # Test cases
    print(normalize_business_name("Amazon.com, Inc."))
    print(normalize_business_name("  Apple Computer Corporation  "))
    print(normalize_business_address("123 Main St. Apt 4B"))
    print(extract_numbers_from_address("123 Main St. Apt 4B"))
