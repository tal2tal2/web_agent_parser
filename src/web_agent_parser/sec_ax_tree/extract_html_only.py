import hashlib
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup, Tag, NavigableString

from .locator_simplify import simplify_locator

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

def get_smart_css_selector(element: Tag, soup: BeautifulSoup) -> str:
    """
    Generate a concise, informative CSS selector for a BS4 tag.
    Prioritizes IDs, unique attributes, and unique classes.
    """
    # 1. IDs
    if element.has_attr('id'):
        # Check uniqueness in the whole tree
        id_val = element['id']
        if len(soup.select(f"#{id_val}")) == 1:
            return f"#{id_val}"

    # 2. Unique Attributes
    # We can't easily check uniqueness efficiently for everything in BS4 without re-querying.
    # But we can try the most common ones.
    for attr in ['data-testid', 'data-test', 'name', 'title', 'aria-label']:
        if element.has_attr(attr):
            val = element[attr]
            if isinstance(val, list): val = " ".join(val) # handle multi-value attrs
            if val and len(val) < 50:
                sel = f'[{attr}="{val}"]'
                if len(soup.select(sel)) == 1:
                    return sel
                # Try tag + attr
                sel = f'{element.name}[{attr}="{val}"]'
                if len(soup.select(sel)) == 1:
                    return sel

    # 3. Path Generation
    # Anchor to nearest ID
    path = []
    current = element
    
    while current and current.name != '[document]':
        selector = current.name
        
        # If we hit a unique ID, we can stop and anchor here
        if current.has_attr('id'):
            id_val = current['id']
            # Only anchor if unique
            if len(soup.select(f"#{id_val}")) == 1:
                selector = f"#{id_val}"
                path.append(selector)
                break
        
        # Try to distinguish with class
        distinguished = False
        if current.has_attr('class'):
            for cls in current['class']:
                # Check siblings
                siblings = current.find_previous_siblings(current.name) + current.find_next_siblings(current.name)
                sibling_has_class = any(cls in (s.get('class', []) or []) for s in siblings)
                
                if not sibling_has_class:
                    selector += f".{cls}"
                    distinguished = True
                    break
        
        if not distinguished:
            # nth-of-type equivalent
            # BS4 doesn't have a direct nth-of-type property, so we count previous siblings of same name
            prev_siblings = current.find_previous_siblings(current.name)
            nth = len(prev_siblings) + 1
            if nth > 1:
                selector += f":nth-of-type({nth})"
        
        path.append(selector)
        current = current.parent
        
    path.reverse()
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
        direct_text = "".join([t for t in tag.contents if isinstance(t, NavigableString)]).strip()
        full_text = tag.get_text(" ", strip=True)[:500] 
        
        xpath = get_xpath(tag)
        css_selector = get_smart_css_selector(tag, soup)
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
            "locator": simplify_locator(f"css={css_selector}", max_parts=4),
            "xpath": xpath # Kept for hierarchy
        })
        
    return normalized
