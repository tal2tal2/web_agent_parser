import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from playwright.async_api import Page, ElementHandle

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

    function getCssSelector(el) {
        if (!(el instanceof Element)) return;
        var path = [];
        while (el.nodeType === Node.ELEMENT_NODE) {
            var selector = el.nodeName.toLowerCase();
            if (el.id) {
                selector += '#' + el.id;
                path.unshift(selector);
                break;
            } else {
                var sib = el, nth = 1;
                while (sib = sib.previousElementSibling) {
                    if (sib.nodeName.toLowerCase() == selector)
                        nth++;
                }
                if (nth != 1)
                    selector += ":nth-of-type(" + nth + ")";
            }
            path.unshift(selector);
            el = el.parentNode;
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
        // A real impl would use the AX tree, but this is a heuristic fallback
        // for the JS-side traversal if we don't map back to the full AX snapshot.
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
            xpath: getXPath(el),
            css: getCssSelector(el),
            attributes: attrs,
            text: directText.trim(),
            innerText: el.innerText ? el.innerText.substring(0, 500) : "", // Truncate early
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
    # In a real impl, we might merge this with page.accessibility.snapshot()
    # But for this PoC, the JS traversal gives us better control over geometry and locators.
    
    normalized = []
    for node in raw_nodes:
        # Create a stable ID based on the xpath or unique attributes
        # Xpath is structural, so it changes if structure changes, but stable for a static snapshot.
        node_id = hashlib.md5(node['xpath'].encode('utf-8')).hexdigest()[:8]
        
        # Determine locator (prefer CSS, fallback to XPath)
        # For Playwright, we want something robust.
        # We can construct a Playwright locator string.
        locator_str = f"css={node['css']}" if node.get('css') else f"xpath={node['xpath']}"
        
        normalized.append({
            "id": node_id,
            "tag": node['tagName'],
            "role": node['role'],
            "text": node['text'] or node['innerText'], # Prefer direct text, fallback to inner
            "attrs": node['attributes'],
            "rect": node['rect'],
            "locator": locator_str,
            "xpath": node['xpath']
        })
        
    return normalized
