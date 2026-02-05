import hashlib
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup, Tag, NavigableString

def get_xpath(element: Tag) -> str:
    """Generate a simple XPath for a BS4 tag."""
    components = []
    child = element
    for parent in child.parents:
        siblings = parent.find_all(child.name, recursive=False)
        index = 0
        if len(siblings) > 1:
            for i, sib in enumerate(siblings):
                if sib is child:
                    index = i + 1
                    break
            components.append(f"{child.name}[{index}]")
        else:
            components.append(child.name)
        child = parent
    components.reverse()
    return "/" + "/".join(components)

def get_css_selector(element: Tag) -> str:
    """Generate a simple CSS selector."""
    # This is a simplified version.
    path = []
    for parent in element.parents:
        if parent.name == '[document]':
            break
        selector = parent.name
        if parent.has_attr('id'):
            selector += f"#{parent['id']}"
            path.append(selector)
            break # ID is unique
        path.append(selector)
    
    path.reverse()
    
    # Add self
    selector = element.name
    if element.has_attr('id'):
        selector += f"#{element['id']}"
    elif element.has_attr('class'):
        selector += "." + ".".join(element['class'])
        
    path.append(selector)
    return " > ".join(path)

def is_likely_visible(tag: Tag) -> bool:
    if tag.name in ['script', 'style', 'meta', 'head', 'link', 'noscript', 'template']:
        return False
    if tag.has_attr('hidden'):
        return False
    if tag.has_attr('aria-hidden') and tag['aria-hidden'] == 'true':
        return False
    style = tag.get('style', '')
    if 'display:none' in style.replace(' ', '') or 'visibility:hidden' in style.replace(' ', ''):
        return False
    # Heuristic: inputs with type=hidden
    if tag.name == 'input' and tag.get('type') == 'hidden':
        return False
    return True

def extract_html_only(html_content: str) -> List[Dict[str, Any]]:
    """
    Parses HTML content and returns normalized intermediate records.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    normalized = []
    
    # Iterate all tags
    for tag in soup.find_all(True):
        if not is_likely_visible(tag):
            continue
            
        # Attributes
        attrs = {k: (v if isinstance(v, str) else " ".join(v)) for k, v in tag.attrs.items()}
        
        # Text
        # Get direct text
        direct_text = "".join([t for t in tag.contents if isinstance(t, NavigableString)]).strip()
        # Fallback to full text if no direct text but has children
        full_text = tag.get_text(" ", strip=True)[:500] # Truncate
        
        xpath = get_xpath(tag)
        node_id = hashlib.md5(xpath.encode('utf-8')).hexdigest()[:8]
        
        # Simple Role Inference
        role = attrs.get('role', tag.name)
        
        normalized.append({
            "id": node_id,
            "tag": tag.name,
            "role": role,
            "text": direct_text if direct_text else full_text,
            "attrs": attrs,
            "rect": None, # No geometry in HTML-only mode
            "locator": f"xpath={xpath}", # XPath is safer for static HTML parsing
            "xpath": xpath
        })
        
    return normalized
