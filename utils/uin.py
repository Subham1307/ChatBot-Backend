import re
from typing import Optional

def extract_uin(text: str) -> Optional[str]:
    """
    Extract UIN number from policy document text.
    
    Args:
        text (str): The policy document text to search in
        
    Returns:
        Optional[str]: The found UIN number, or None if no UIN is found
    """
    pattern = r'(?:UIN[:\s.]*)([A-Za-z0-9]+[Vv]\d+)'
    
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    
    alt_pattern = r'([A-Za-z0-9]+[Vv]\d+)'
    match = re.search(alt_pattern, text)
    if match:
        return match.group(1)
    
    return None
