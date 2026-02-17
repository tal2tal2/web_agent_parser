from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


_CSS_SPLIT_RE = re.compile(r"\s*>\s*")
_NTH_RE = re.compile(r":nth-of-type\(\s*\d+\s*\)")
_ID_RE = re.compile(r"#(?:\\.|[A-Za-z0-9_-])+")
_CLASS_RE = re.compile(r"\.([A-Za-z0-9_-]+)")
_ATTR_RE = re.compile(r"\[([^\]]+)\]")

# Prefer attributes that are usually stable / intentional.
_PREFERRED_ATTR_KEYS = (
    "data-testid",
    "data-test",
    "data-cy",
    "data-qa",
    "aria-label",
    "name",
    "title",
    "placeholder",
    "role",
)


@dataclass(frozen=True)
class _CssPart:
    idx: int
    raw: str
    simplified: str
    score: int


def simplify_locator(locator: str, *, max_parts: int = 4) -> str:
    """
    Reduce token-heavy locators to something compact and still useful.

    Today this is primarily aimed at Playwright-style `css=...` locators that
    contain long ` > ` chains with lots of `table/tbody/tr/td/div` and `:nth-of-type`.
    """
    if not locator:
        return locator

    locator = locator.strip()
    if locator.startswith("css="):
        css = locator[len("css=") :].strip()
        return "css=" + _simplify_css_selector(css, max_parts=max_parts)
    if locator.startswith("xpath="):
        xpath = locator[len("xpath=") :].strip()
        return "xpath=" + _simplify_xpath(xpath, max_parts=max_parts)
    # Unknown engine; leave as-is.
    return locator


def _simplify_css_selector(css: str, *, max_parts: int) -> str:
    css = css.strip()
    if not css:
        return css

    # Already short / atomic.
    if ">" not in css:
        return _cleanup_css_token(css)

    raw_parts = [p.strip() for p in _CSS_SPLIT_RE.split(css) if p.strip()]
    if len(raw_parts) <= max_parts:
        return " ".join(_cleanup_css_token(p) for p in raw_parts)

    parts: List[_CssPart] = []
    for i, p in enumerate(raw_parts):
        simplified, score = _simplify_css_part(p)
        parts.append(_CssPart(idx=i, raw=p, simplified=simplified, score=score))

    # Always keep a good "anchor" if present (early #id / stable attribute),
    # and always keep the leaf.
    leaf = parts[-1]
    anchor_idx = _find_anchor_idx(parts)
    anchor = parts[anchor_idx]

    keep: List[_CssPart] = []
    keep.append(anchor)
    if leaf.idx != anchor.idx:
        keep.append(leaf)

    remaining_budget = max(0, max_parts - len(keep))
    if remaining_budget:
        candidates = [
            p
            for p in parts
            if p.idx not in {anchor.idx, leaf.idx} and p.score >= 2
        ]
        # Prefer high score, and prefer closer to leaf for specificity.
        candidates.sort(key=lambda p: (p.score, p.idx), reverse=True)
        chosen = candidates[:remaining_budget]
        keep.extend(chosen)

    # Restore original order, de-dupe by idx.
    keep_map = {p.idx: p for p in keep}
    ordered = [keep_map[i] for i in sorted(keep_map.keys())]

    # Use descendant combinator (space) so skipping levels remains valid CSS.
    return " ".join(_cleanup_css_token(p.simplified) for p in ordered)


def _find_anchor_idx(parts: List[_CssPart]) -> int:
    for p in parts:
        if p.score >= 4:
            return p.idx
    # Fallback: keep the first part.
    return 0


def _simplify_css_part(part: str) -> Tuple[str, int]:
    """
    Return (simplified_token, score).

    Score heuristic:
    - 5: #id
    - 4: preferred attribute selector
    - 3: any attribute selector
    - 2: a "good" class selector
    - 1: tag / nth-of-type / generic
    """
    part = part.strip()
    if not part:
        return part, 1

    # Drop nth-of-type as it's noisy and brittle.
    cleaned = _NTH_RE.sub("", part).strip()

    m_id = _ID_RE.search(cleaned)
    if m_id:
        return m_id.group(0), 5

    # Prefer a stable attribute if present.
    attrs = _ATTR_RE.findall(cleaned)
    if attrs:
        best_attr = _pick_best_attr(attrs)
        if best_attr:
            key = best_attr.split("=", 1)[0].strip().strip('"').strip("'")
            if key in _PREFERRED_ATTR_KEYS:
                return f"[{best_attr}]", 4
            return f"[{best_attr}]", 3

    # Prefer a meaningful class (longest tends to be most specific).
    classes = _CLASS_RE.findall(cleaned)
    if classes:
        cls = max(classes, key=len)
        if len(cls) >= 3:
            return f".{cls}", 2

    # Fallback: keep a compact tag-ish token if any.
    token = _cleanup_css_token(cleaned)
    return token, 1


def _pick_best_attr(attrs: List[str]) -> Optional[str]:
    # attrs entries look like: key="value" or key='value' or key=value
    def key_of(a: str) -> str:
        return a.split("=", 1)[0].strip().strip('"').strip("'")

    preferred = [a for a in attrs if key_of(a) in _PREFERRED_ATTR_KEYS]
    if preferred:
        # Prefer shortest value (less token-y) among preferred keys.
        return min(preferred, key=len)
    return min(attrs, key=len) if attrs else None


def _cleanup_css_token(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    # Collapse internal whitespace.
    token = re.sub(r"\s+", " ", token)
    # If it's still a long compound, keep it as-is (we already simplified parts above).
    return token


def _simplify_xpath(xpath: str, *, max_parts: int) -> str:
    """
    Keep only the last `max_parts` segments of an XPath-ish string.
    This is a best-effort fallback for cases where CSS is unavailable.
    """
    xpath = xpath.strip()
    if not xpath:
        return xpath

    # Handle the `id("...")/...` style produced by our JS getXPath().
    if xpath.startswith('id("'):
        # Keep id("...") + last few steps.
        try:
            after = xpath.split(")", 1)[1]
            tail = [p for p in after.split("/") if p]
            if len(tail) > max_parts:
                tail = tail[-max_parts:]
            return xpath.split(")", 1)[0] + ")" + ("/" + "/".join(tail) if tail else "")
        except Exception:
            return xpath

    parts = [p for p in xpath.split("/") if p]
    if len(parts) <= max_parts:
        return "/" + "/".join(parts) if xpath.startswith("/") else "/".join(parts)
    tail = parts[-max_parts:]
    return ("/" if xpath.startswith("/") else "") + "/".join(tail)
