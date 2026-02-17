import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from playwright.async_api import Page, ElementHandle

from .locator_simplify import simplify_locator

# We'll use a script to extract everything in one go to minimize round-trips.
EXTRACT_SCRIPT = """
(() => {
    function getXPath(element) {
        if (element.id !== '')
            return 'id("' + element.id + '")';
        if (element === document.body)
            return element.tagName;
        var ix = 0;
        var siblings = element.parentNode.childNodes;
        for (var i = 0; i < siblings.length; i++) {
            var sibling = siblings[i];
            if (sibling === element)
                return getXPath(element.parentNode) + '/' + element.tagName + '[' + (ix + 1) + ']';
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName)
                ix++;
        }
    }

    function getSmartSelector(el) {
        if (!(el instanceof Element)) return;
        
        // 1. IDs (Global uniqueness assumed for valid HTML, or close enough)
        if (el.id) {
             const sel = '#' + CSS.escape(el.id);
             // Verify uniqueness just in case (some pages have duplicate IDs)
             if (document.querySelectorAll(sel).length === 1) return sel;
        }
        
        // 2. Attributes usually used for testing or specific identification
        const uniqueAttrs = ['data-testid', 'data-test', 'data-cy', 'data-qa', 'name', 'title', 'placeholder', 'aria-label'];
        for (const attr of uniqueAttrs) {
            if (el.hasAttribute(attr)) {
                const val = el.getAttribute(attr);
                if (val && val.trim().length > 0 && val.length < 50) { // Skip overly long attributes
                    const sel = `[${attr}="${CSS.escape(val)}"]`;
                    if (document.querySelectorAll(sel).length === 1) return sel;
                    // Also try combined with tag
                    const tagSel = `${el.tagName.toLowerCase()}${sel}`;
                    if (document.querySelectorAll(tagSel).length === 1) return tagSel;
                }
            }
        }
        
        // 3. Unique Alt (Images)
        if (el.tagName === 'IMG' && el.hasAttribute('alt')) {
            const val = el.getAttribute('alt');
            if (val && val.trim().length > 0 && val.length < 50) {
                const sel = `img[alt="${CSS.escape(val)}"]`;
                if (document.querySelectorAll(sel).length === 1) return sel;
            }
        }
        
        // 4. Unique Class? (Heuristic: complex classes are often unique)
        if (el.className && typeof el.className === 'string') {
             const classes = el.className.split(/\s+/).filter(c => c);
             for (const cls of classes) {
                 const sel = '.' + CSS.escape(cls);
                 if (document.querySelectorAll(sel).length === 1) return sel;
             }
        }

        // 5. Fallback: Path relative to nearest ID
        // Improvement: Use classes in the path where possible to be specific and avoid nth-of-type
        var path = [];
        var current = el;
        
        while (current.nodeType === Node.ELEMENT_NODE) {
            let selector = current.nodeName.toLowerCase();
            
            if (current.id) {
                const idSel = '#' + CSS.escape(current.id);
                // Only use ID if it's unique, otherwise treat as normal node
                if (document.querySelectorAll(idSel).length === 1) {
                    path.unshift(idSel);
                    break; 
                }
            }
            
            // Try to make this level specific using a class
            // Only if the class distinguishes it from siblings!
            let distinguished = false;
            if (current.className && typeof current.className === 'string') {
                const classes = current.className.split(/\s+/).filter(c => c);
                for (const cls of classes) {
                    // Check if any sibling has this class
                    let siblingHasClass = false;
                    let sib = current.parentNode ? current.parentNode.firstElementChild : null;
                    while (sib) {
                        if (sib !== current && sib.nodeType === Node.ELEMENT_NODE && sib.classList.contains(cls)) {
                            siblingHasClass = true;
                            break;
                        }
                        sib = sib.nextElementSibling;
                    }
                    
                    if (!siblingHasClass) {
                        selector = '.' + CSS.escape(cls); // Just the class is enough!
                        distinguished = true;
                        break; // Found a distinguishing class
                    }
                }
            }
            
            if (!distinguished) {
                // nth-of-type fallback
                var nth = 1;
                var sib = current;
                while (sib = sib.previousElementSibling) {
                    if (sib.nodeName.toLowerCase() == current.nodeName.toLowerCase())
                        nth++;
                }
                if (nth != 1)
                    selector += ":nth-of-type(" + nth + ")";
            }
            
            path.unshift(selector);
            current = current.parentNode;
            if (!current) break;
        }
        
        return path.join(" > ");
    }

    function isVisible(elem) {
        if (!elem) return false;
        const style = window.getComputedStyle(elem);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
        const rect = elem.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    function getRole(el) {
        return el.getAttribute('role') || el.tagName.toLowerCase(); 
    }

    const elements = document.querySelectorAll('*');
    const result = [];

    elements.forEach((el, index) => {
        if (!isVisible(el)) return;

        const rect = el.getBoundingClientRect();
        const attrs = {};
        for (let i = 0; i < el.attributes.length; i++) {
            attrs[el.attributes[i].name] = el.attributes[i].value;
        }

        // Text content (direct text nodes only to avoid duplication)
        let directText = "";
        el.childNodes.forEach(node => {
            if (node.nodeType === 3) directText += node.textContent.trim() + " ";
        });

        result.push({
            tagName: el.tagName.toLowerCase(),
            xpath: getXPath(el), // Kept for internal hierarchy building
            css: getSmartSelector(el),
            attributes: attrs,
            text: directText.trim(),
            innerText: el.innerText ? el.innerText.substring(0, 500) : "",
            rect: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                in_viewport: (
                    rect.top >= 0 &&
                    rect.left >= 0 &&
                    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
                )
            },
            role: getRole(el)
        });
    });
    return result;
})()
"""

async def extract_playwright(page: Page) -> List[Dict[str, Any]]:
    """
    Extracts raw DOM elements with geometry and attributes.
    Returns a list of intermediate dicts.
    """
    # 1. Run the JS extraction script
    raw_nodes = await page.evaluate(EXTRACT_SCRIPT)
    
    # 2. Normalize and Add Stable IDs
    
    normalized = []
    for node in raw_nodes:
        # Stable ID based on the XPath (structural)
        node_id = hashlib.md5(node['xpath'].encode('utf-8')).hexdigest()[:8]
        
        # Determine locator (Prefer CSS)
        css_sel = node.get("css")
        if css_sel:
            locator_str = simplify_locator(f"css={css_sel}", max_parts=4)
        else:
            locator_str = simplify_locator(f"xpath={node['xpath']}", max_parts=4)
        
        normalized.append({
            "id": node_id,
            "tag": node['tagName'],
            "role": node['role'],
            "text": node['text'] or node['innerText'], 
            "attrs": node['attributes'],
            "rect": node['rect'],
            "locator": locator_str,
            "xpath": node['xpath'] # Required for hierarchy
        })
        
    return normalized
