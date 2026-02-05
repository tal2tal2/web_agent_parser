"""
Security-aware accessibility tree parser (prototype)

Takes a simplified accessibility-tree JSON snapshot (Playwright/Puppeteer-style),
treats it as untrusted input, and outputs:

1) Action Catalog: interactive elements with stable IDs
2) Evidence Pack: minimal contextual text (nearby headings, form labels)
3) Risk Metadata: per-element prompt/indirect-injection (PI) flags

Important caveat: we DO NOT redact/remove text.
We keep text visible but attach structured risk tags + a risk_score.
"""

from __future__ import annotations
import argparse

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable

# -----------------------------
# Config / lightweight heuristics
# -----------------------------

# Interactive roles we consider "actions"
INTERACTIVE_ROLES = {
    "button", "link", "textbox", "searchbox", "combobox", "listbox", "menuitem",
    "checkbox", "radio", "switch", "slider", "spinbutton",
    "option", "tab", "treeitem",
}

# Roles used to infer regions
REGION_ROOT_ROLES = {
    "main", "navigation", "banner", "contentinfo", "complementary",
    "form", "dialog", "alertdialog", "region",
}

ROLE_TO_REGION = {
    "main": "main",
    "navigation": "nav",
    "banner": "header",
    "contentinfo": "footer",
    "complementary": "sidebar",
    "dialog": "modal",
    "alertdialog": "modal",
    # "region" and "form" are inferred by hierarchy/name hints
}

# Headings
HEADING_ROLES = {"heading"}

# Basic PI / exfil indicators (names + local context only)
# Keep these small and explainable; expand later.
PI_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("prompt_injection_phrase", re.compile(r"\b(ignore|disregard)\b.*\b(instruction|previous)\b", re.I)),
    ("prompt_injection_phrase", re.compile(r"\b(system prompt|developer message|hidden instruction)\b", re.I)),
    ("prompt_injection_phrase", re.compile(r"\byou are (chatgpt|an ai)\b", re.I)),
    ("exfiltration_keyword",
     re.compile(r"\b(export|dump|reveal|print|show)\b.*\b(token|session|cookie|secret|password|key)\b", re.I)),
    ("exfiltration_keyword", re.compile(r"\b(api key|access token|session token|cookie|csrf)\b", re.I)),
    ("credential_keyword", re.compile(r"\b(password|passcode|otp|2fa|mfa|authenticator)\b", re.I)),
    ("data_leak_keyword", re.compile(r"\b(exfiltrate|leak|steal|send to)\b", re.I)),
]

# Region penalties: actions in odd regions for typical tasks are riskier.
REGION_RISK_BONUS = {
    "footer": 0.12,
    "sidebar": 0.08,
    "nav": 0.06,
    "modal": 0.04,  # not always risky; modals are often important
}

# Default top-K actions kept after relevance pruning
DEFAULT_TOPK = 25

# How much local context we inspect for PI indicators (characters)
LOCAL_CONTEXT_CHARS = 240


# -----------------------------
# Data shapes
# -----------------------------

@dataclass
class ActionItem:
    id: str
    role: str
    name: str
    region: str
    risk_score: float
    risk_reasons: List[str]
    provenance: str
    path: str  # helps stability across runs + debugging (can be removed later)


@dataclass
class EvidencePack:
    headings: List[str]
    form_labels: List[str]


@dataclass
class ParsedOutput:
    actions: List[Dict[str, Any]]
    evidence: Dict[str, Any]


# -----------------------------
# Utility: safe string handling
# -----------------------------

def _safe_str(x: Any, max_len: int = 400) -> str:
    """Convert untrusted input to a bounded string (avoid huge payloads)."""
    if x is None:
        return ""
    s = str(x)
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _hash_id(stable_key: str, prefix: str = "A") -> str:
    """
    Produce a stable-ish short ID from a key.
    This avoids leaking raw DOM ids while staying stable across snapshots
    as long as the same path/name/role remains.
    """
    h = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:6].upper()
    return f"{prefix}{h}"


# -----------------------------
# Tree walking / extraction
# -----------------------------

def iter_nodes(tree: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], List[int]]]:
    """
    DFS over the accessibility tree.
    Yields (node, index_path) where index_path is list of child indexes.
    """
    stack: List[Tuple[Dict[str, Any], List[int]]] = [(tree, [])]
    while stack:
        node, path = stack.pop()
        yield node, path
        children = node.get("children") or []
        if isinstance(children, list):
            # reverse so DFS visits in original order
            for i in range(len(children) - 1, -1, -1):
                child = children[i]
                if isinstance(child, dict):
                    stack.append((child, path + [i]))


def node_text_sources(node: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract common accessible name sources from a simplified snapshot.
    We keep provenance as a single label for now; later you can store multiple.
    """
    # Common shapes across tools:
    # - {"name": "..."} might be computed accessible name (could come from various sources)
    # - {"aria_label": "..."} or {"ariaLabel": "..."}
    # - {"value": "..."} for inputs
    # - {"text": "..."} for visible text nodes (depends on exporter)
    name = _safe_str(node.get("name"))
    aria_label = _safe_str(node.get("aria_label") or node.get("ariaLabel"))
    value = _safe_str(node.get("value"))
    text = _safe_str(node.get("text"))

    return {
        "name": _norm_space(name),
        "aria-label": _norm_space(aria_label),
        "value": _norm_space(value),
        "text": _norm_space(text),
    }


def choose_accessible_name(node: Dict[str, Any]) -> Tuple[str, str]:
    """
    Pick an accessible name + provenance label.
    Minimal and explainable priority:
      aria-label > name > text > value
    """
    src = node_text_sources(node)
    if src["aria-label"]:
        return src["aria-label"], "aria-label"
    if src["name"]:
        return src["name"], "name"
    if src["text"]:
        return src["text"], "visible_text"
    if src["value"]:
        return src["value"], "value"
    return "", "unknown"


def get_role(node: Dict[str, Any]) -> str:
    return _safe_str(node.get("role") or node.get("type") or "").lower()


def get_disabled(node: Dict[str, Any]) -> bool:
    # Different snapshots encode disabled in different ways.
    return bool(node.get("disabled") or node.get("aria_disabled") or node.get("ariaDisabled"))


# -----------------------------
# Region inference (lightweight)
# -----------------------------

def infer_region_for_path(
        path_roles: List[str],
        path_names: List[str],
) -> str:
    """
    Infer region based on the nearest ancestor with a region-ish role,
    plus small name hints.
    """
    # Walk from leaf ancestor -> root ancestor
    for role, name in reversed(list(zip(path_roles, path_names))):
        if role in ROLE_TO_REGION:
            return ROLE_TO_REGION[role]
        if role == "form":
            return "main"  # forms typically belong to main content
        if role == "region":
            # Named regions are often sections within main
            if re.search(r"\b(side|aside|related)\b", name, re.I):
                return "sidebar"
            return "main"

    # If nothing, default main
    return "main"


# -----------------------------
# Evidence extraction
# -----------------------------

def collect_evidence(tree: Dict[str, Any], max_headings: int = 12, max_labels: int = 16) -> EvidencePack:
    headings: List[str] = []
    labels: List[str] = []

    for node, _path in iter_nodes(tree):
        role = get_role(node)
        name, prov = choose_accessible_name(node)
        if not name:
            continue

        # Headings
        if role in HEADING_ROLES:
            headings.append(name)

        # Form-ish labels: this is heuristic; you can tighten later.
        # We treat "label" role and also "text" nodes near inputs in later versions.
        if role == "label" or (
                role == "statictext" and re.search(r"\b(email|password|username|phone|search)\b", name, re.I)):
            labels.append(name)

    # De-dup while preserving order
    def dedup(xs: List[str], limit: int) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
            if len(out) >= limit:
                break
        return out

    return EvidencePack(
        headings=dedup(headings, max_headings),
        form_labels=dedup(labels, max_labels),
    )


# -----------------------------
# PI risk scoring (text is preserved)
# -----------------------------

def pi_risk_from_text(text: str) -> Tuple[float, List[str]]:
    """
    Returns (score_add, reasons).
    Small, transparent scoring. Extend later with ML or richer signals.
    """
    reasons: List[str] = []
    score = 0.0

    if not text:
        return score, reasons

    hits = []
    for reason, pat in PI_PATTERNS:
        if pat.search(text):
            hits.append(reason)

    # De-dup reasons
    for r in dict.fromkeys(hits).keys():
        reasons.append(r)

    # Score contributions
    if "prompt_injection_phrase" in reasons:
        score += 0.55
    if "exfiltration_keyword" in reasons:
        score += 0.50
    if "credential_keyword" in reasons:
        score += 0.25
    if "data_leak_keyword" in reasons:
        score += 0.35

    # Cap soft (we'll clamp later)
    return score, reasons


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# -----------------------------
# Action relevance pruning
# -----------------------------

def relevance_score(name: str, role: str, region: str, task_keywords: List[str]) -> float:
    """
    Simple lexical relevance:
      - keyword overlap with action name
      - modest boost for main/modal regions
    """
    if not name:
        return 0.0

    name_l = name.lower()
    hits = 0
    for kw in task_keywords:
        kw_l = kw.lower().strip()
        if kw_l and kw_l in name_l:
            hits += 1

    base = min(1.0, hits / max(1, len(task_keywords))) if task_keywords else 0.15

    # Role bias: inputs + buttons tend to be higher value than generic links
    if role in {"textbox", "searchbox", "combobox"}:
        base += 0.10
    if role == "button":
        base += 0.06

    # Region bias: main/modal usually most relevant for accomplishing tasks
    if region in {"main", "modal"}:
        base += 0.08
    elif region in {"nav", "sidebar"}:
        base += 0.02

    return clamp01(base)


# -----------------------------
# Main compile function
# -----------------------------

def compile_accessibility_snapshot(
        tree: Dict[str, Any],
        task_keywords: Optional[List[str]] = None,
        topk_actions: int = DEFAULT_TOPK,
) -> ParsedOutput:
    """
    Core pipeline:
      1) Walk nodes (untrusted)
      2) Infer region per node from ancestor roles
      3) Extract interactive actions with stable IDs
      4) Compute PI risk from (name + local context), keep text intact
      5) Prune to top-K by relevance, but keep risk metadata
      6) Build evidence pack (headings, form labels)
    """
    task_keywords = task_keywords or []

    evidence = collect_evidence(tree)

    # First pass: gather nodes with their ancestor info so we can infer region
    # and get local context (nearby ancestor names).
    candidates: List[Tuple[ActionItem, float]] = []

    # Build a quick lookup from index_path -> (role, name) for ancestor context
    # during DFS.
    path_stack_roles: List[str] = []
    path_stack_names: List[str] = []

    def dfs(node: Dict[str, Any], index_path: List[int]):
        role = get_role(node)
        name, prov = choose_accessible_name(node)

        # Push current node onto "ancestor stacks"
        path_stack_roles.append(role)
        path_stack_names.append(name)

        # Region inference uses the current ancestor stacks
        region = infer_region_for_path(path_stack_roles, path_stack_names)

        # Local context: ancestor names (closest first), bounded
        local_bits = []
        # take a few closest ancestor names (including self) as "local context"
        for nm in reversed(path_stack_names[-8:]):
            if nm:
                local_bits.append(nm)
        local_context = _norm_space(" | ".join(local_bits))
        if len(local_context) > LOCAL_CONTEXT_CHARS:
            local_context = local_context[: LOCAL_CONTEXT_CHARS - 1] + "…"

        # Identify interactive action nodes
        if role in INTERACTIVE_ROLES:
            # Stable key: role + inferred region + index_path + name
            # (index path is stable if the tree structure stays stable; name helps)
            stable_key = f"{role}|{region}|{'.'.join(map(str, index_path))}|{name}"
            action_id = _hash_id(stable_key, prefix="A")

            # Risk score from accessible name + local context only
            risk_score = 0.0
            risk_reasons: List[str] = []

            add, reasons = pi_risk_from_text(name)
            risk_score += add
            risk_reasons.extend(reasons)

            add2, reasons2 = pi_risk_from_text(local_context)
            # local context is a weaker signal than the action name itself
            risk_score += 0.35 * add2
            for r in reasons2:
                if r not in risk_reasons:
                    risk_reasons.append(r)

            # Region-based risk bonus (e.g., suspicious "export token" in footer)
            region_bonus = REGION_RISK_BONUS.get(region, 0.0)
            if region_bonus > 0:
                risk_score += region_bonus
                # Only add this reason if there is *some* other signal or region is clearly odd.
                risk_reasons.append("out_of_task_region")

            # Disabled elements are usually not actionable; we keep them but mark.
            if get_disabled(node):
                risk_reasons.append("disabled")
                risk_score += 0.05

            risk_score = clamp01(risk_score)

            item = ActionItem(
                id=action_id,
                role=role,
                name=name,
                region=region,
                risk_score=risk_score,
                risk_reasons=risk_reasons,
                provenance=prov,
                path=".".join(map(str, index_path)),
            )

            rel = relevance_score(name=name, role=role, region=region, task_keywords=task_keywords)
            candidates.append((item, rel))

        # Recurse
        children = node.get("children") or []
        if isinstance(children, list):
            for i, child in enumerate(children):
                if isinstance(child, dict):
                    dfs(child, index_path + [i])

        # Pop ancestor stacks
        path_stack_roles.pop()
        path_stack_names.pop()

    dfs(tree, [])

    # Prune: rank by relevance (desc), keep top-K
    candidates.sort(key=lambda x: x[1], reverse=True)
    kept = [asdict(item) for item, _rel in candidates[:topk_actions]]

    # Strip debugging path if you want a smaller payload.
    # For prototyping, it's useful; for production, remove it.
    # for a in kept: a.pop("path", None)

    return ParsedOutput(
        actions=kept,
        evidence=asdict(evidence),
    )


# -----------------------------
# Example CLI usage
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compile an untrusted a11y snapshot into LLM-friendly JSON.")
    ap.add_argument("snapshot_json", help="Path to simplified accessibility-tree JSON file.")
    ap.add_argument("--keywords", nargs="*", default=[], help="Task keywords for action relevance pruning.")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Keep top-K actions by relevance.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = ap.parse_args()

    with open(args.snapshot_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    out = compile_accessibility_snapshot(tree, task_keywords=args.keywords, topk_actions=args.topk)
    payload = {"actions": out.actions, "evidence": out.evidence}

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
