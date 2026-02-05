import re
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from .types import (
    NodeCore, SecAXTree, OverviewNode, RiskMetadata,
    ActionType, ContentKind, RegionType, Provenance, FieldSensitivity,
    ActionType
)

# --- Heuristic Constants ---

INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"you are a (chat)?gpt",
    r"step by step",
    r"reasoning:",
    r"Example \d+:",
    r"dev(eloper)? mode",
    r"unrestricted",
]

IMPERATIVE_VERBS = {
    "click", "tap", "select", "choose", "enter", "type", "navigate", "go",
    "ignore", "forget", "accept", "reject", "confirm", "submit", "login",
    "sign", "buy", "order", "pay"
}

MODAL_VERBS = {"must", "required", "need", "should", "ought"}

SENSITIVE_KEYWORDS = {
    "password", "credit", "card", "cvv", "social security", "ssn", "bank", "account number"
}

# --- Risk Analysis ---

class RiskAnalyzer:
    def compute_risk(self, node: NodeCore) -> RiskMetadata:
        text = (node.text + " " + node.name).lower()
        risk = RiskMetadata()
        
        # 1. Prompt Injection Signals
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text):
                risk.triggers.append(f"injection_pattern: {pattern}")
                risk.risk_score += 0.4
                risk.risk_reasons.append("Contains prompt injection phrasing")

        # 2. Attack Surface Flags
        if node.tag == "a" or node.attrs.get("href"):
            risk.has_url = True
        if node.tag in ["button", "a", "input", "select"]:
            risk.has_button_or_link = True
        
        # Instruction-like detection
        verb_count = sum(1 for word in text.split() if word in IMPERATIVE_VERBS)
        if verb_count > 2 and len(text.split()) < 20:
             risk.instruction_like = True
             risk.triggers.append(f"high_imperative_density: {verb_count}")

        # 3. Context-Fit Heuristics (The "Does this belong here?" check)
        fit_score = self._compute_context_fit(node, text)
        risk.context_fit_score = fit_score
        
        if fit_score < 0.5:
            risk.risk_score += (0.5 - fit_score) * 2  # Penalize low fit
            risk.risk_reasons.append(f"Low context fit ({fit_score:.2f})")
            
        # Clamp score
        risk.risk_score = min(1.0, risk.risk_score)
        return risk

    def _compute_context_fit(self, node: NodeCore, text: str) -> float:
        """
        Returns a score 0.0 (anomalous) to 1.0 (normal).
        """
        score = 1.0
        words = text.split()
        word_count = len(words)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        
        # Heuristic 1: Nav Items shouldn't be paragraphs
        if ContentKind.NAV_ITEM in node.content_kinds:
            if word_count > 15:
                score -= 0.3
            if sentence_count > 1:
                score -= 0.4
                
        # Heuristic 2: Buttons shouldn't contain essays
        if node.tag == "button" or node.role == "button":
            if word_count > 30:
                score -= 0.5
            if sentence_count > 2:
                score -= 0.5

        # Heuristic 3: Email subjects shouldn't be massive
        if ContentKind.EMAIL_SUBJECT in node.content_kinds:
            if word_count > 50:
                score -= 0.3

        # Heuristic 4: URLs in weird places (e.g. inside a button text that isn't a link)
        if "http" in text and node.tag not in ["a"]:
             # This is suspicious in some contexts
             pass

        # Heuristic 5: "About" sections with heavy imperative directives
        if ContentKind.PROFILE_ABOUT in node.content_kinds:
            imperative_ratio = sum(1 for w in words if w in IMPERATIVE_VERBS) / max(1, word_count)
            if imperative_ratio > 0.2: # > 20% verbs is weird for a bio
                score -= 0.4
        
        return max(0.0, score)

# --- Tag Inference ---

class TagInferencer:
    def infer(self, node: NodeCore):
        self._infer_region(node)
        self._infer_content_kind(node)
        self._infer_sensitivity(node)
        self._infer_provenance(node)
        self._infer_action_type(node)

    def _infer_region(self, node: NodeCore):
        # Infer region based on HTML5 tags and Roles
        if node.tag == "header" or node.role == "banner":
            node.region = RegionType.HEADER
        elif node.tag == "nav" or node.role == "navigation":
            node.region = RegionType.NAV
        elif node.tag == "main" or node.role == "main":
            node.region = RegionType.MAIN
        elif node.tag == "footer" or node.role == "contentinfo":
            node.region = RegionType.FOOTER
        elif node.tag == "aside" or node.role == "complementary":
            node.region = RegionType.ASIDE
        elif node.role == "dialog" or "modal" in node.attrs.get("class", ""):
            node.region = RegionType.MODAL
        
        # Propagate? (Usually handled by parent-child logic, but for now purely local)

    def _infer_content_kind(self, node: NodeCore):
        # Heuristic keyword matching
        classes = node.attrs.get("class", "")
        if isinstance(classes, list): classes = " ".join(classes)
        text = (node.name + " " + classes).lower()
        
        if "email" in text or "inbox" in text:
            node.content_kinds.add(ContentKind.EMAIL)
        if "subject" in text:
            node.content_kinds.add(ContentKind.EMAIL_SUBJECT)
            
        if "post" in text or "comment" in text:
            node.content_kinds.add(ContentKind.POST)
            
        if "product" in text or "price" in text:
             node.content_kinds.add(ContentKind.PRODUCT_LISTING)
             
        if "profile" in text or "bio" in text:
            node.content_kinds.add(ContentKind.PROFILE_ABOUT)
            
        if node.tag == "nav" or "nav" in classes:
            node.content_kinds.add(ContentKind.NAV_ITEM)

    def _infer_sensitivity(self, node: NodeCore):
        if node.tag == "input":
            type_attr = node.attrs.get("type", "")
            if type_attr == "password":
                node.sensitivity = FieldSensitivity.CREDENTIAL
            elif type_attr in ["text", "email"]:
                 # Check labels/names
                 name = (node.name or node.attrs.get("name", "")).lower()
                 if any(k in name for k in SENSITIVE_KEYWORDS):
                     node.sensitivity = FieldSensitivity.PII
                     if "card" in name:
                         node.sensitivity = FieldSensitivity.PAYMENT

    def _infer_provenance(self, node: NodeCore):
        # Hard to do locally, assume First Party unless marked otherwise
        node.provenance = Provenance.UNKNOWN

    def _infer_action_type(self, node: NodeCore):
        if node.tag == "a" and node.attrs.get("href"):
            node.action_type = ActionType.NAVIGATE
        elif node.tag == "button" or node.role == "button":
            node.action_type = ActionType.CLICK
        elif node.tag in ["input", "textarea"]:
            t = node.attrs.get("type", "")
            if t in ["checkbox", "radio"]:
                node.action_type = ActionType.TOGGLE
            elif t in ["file"]:
                node.action_type = ActionType.UPLOAD
            elif t in ["submit", "button", "image"]:
                 node.action_type = ActionType.CLICK
            else:
                node.action_type = ActionType.INPUT_TEXT
        elif node.tag == "select":
             node.action_type = ActionType.SELECT
        else:
            node.action_type = ActionType.READ_ONLY

# --- Builder ---

class SecAXTreeBuilder:
    def __init__(self, enable_domain_safety: bool = False):
        self.risk_analyzer = RiskAnalyzer()
        self.tag_inferencer = TagInferencer()
        self.enable_domain_safety = enable_domain_safety

    def build(self, raw_nodes: List[Dict[str, Any]], url: str = "") -> SecAXTree:
        # 1. Convert to NodeCore
        nodes_map: Dict[str, NodeCore] = {}
        xpath_map: Dict[str, str] = {} # xpath -> node_id
        
        for r in raw_nodes:
            # Basic Node
            n = NodeCore(
                id=r['id'],
                tag=r['tag'],
                role=r['role'],
                text=r['text'],
                attrs=r['attrs'],
                geom=r['rect'],
                locator=r['locator']
            )
            # Infer Name (accessible name calculation is complex, simplified here)
            n.name = r['attrs'].get('aria-label') or r['attrs'].get('title') or r['attrs'].get('placeholder') or ""
            if not n.name and len(n.text) < 50:
                n.name = n.text
                
            # Run Inference
            self.tag_inferencer.infer(n)
            n.risk = self.risk_analyzer.compute_risk(n)
            
            nodes_map[n.id] = n
            xpath_map[r['xpath']] = n.id

        # 2. Reconstruct Hierarchy
        # We assume the parent's xpath is the prefix of the child's xpath
        # Xpath: /html/body/div[1]/p
        # Parent: /html/body/div[1]
        
        for nid, node in nodes_map.items():
            current_xpath = next(k for k, v in xpath_map.items() if v == nid)
            # Find parent xpath
            if "/" in current_xpath:
                # Basic string manipulation to find parent path
                # Remove last segment /[...]([...])
                parent_xpath_candidates = []
                
                # Heuristic: strip last component
                parts = current_xpath.rsplit('/', 1)
                if len(parts) == 2:
                    parent_xpath = parts[0]
                    if parent_xpath in xpath_map:
                        parent_id = xpath_map[parent_xpath]
                        node.parent_id = parent_id
                        nodes_map[parent_id].child_ids.append(nid)
                        
                        # Inherit region from parent if unknown
                        if node.region == RegionType.UNKNOWN and nodes_map[parent_id].region != RegionType.UNKNOWN:
                            node.region = nodes_map[parent_id].region

        # 3. Create Overview Tree
        # Filter: Only Interactive or High Risk or Region Containers
        roots = []
        overview_map = {}
        
        # Identify "Overview Worthy" nodes
        # - Interactive elements
        # - Region roots
        # - Nodes with high risk
        # - Nodes with text content (maybe? we want to minimize tokens)
        
        def is_overview_worthy(n: NodeCore) -> bool:
            if n.action_type != ActionType.READ_ONLY: return True
            if n.risk.risk_score > 0.3: return True
            if n.region != RegionType.UNKNOWN and n.parent_id and nodes_map[n.parent_id].region != n.region: return True # Region change
            if n.tag in ["h1", "h2", "h3"]: return True # Headings
            return False

        # Build recursive overview structure
        # This is tricky because we are pruning the tree. 
        # If a parent is pruned, its children need to attach to the nearest ancestor that is kept.
        
        # Simple approach for PoC: 
        # Keep the full hierarchy but only emit OverviewNode fields, 
        # and maybe collapse "div > div > div" chains that add no value.
        
        # Better approach: 
        # Just emit the "significant" nodes and structure them by Region.
        # But we need a tree.
        
        # Let's stick to: Map every NodeCore to an OverviewNode, but prune the `child_ids` list 
        # to skip boring nodes?
        # Actually, let's keep it simple: Overview = Full Tree but stripped down.
        # Just aggressive text truncation.
        
        for nid, node in nodes_map.items():
            onode = OverviewNode(
                id=node.id,
                role=node.role,
                action_type=node.action_type,
                name_preview=(node.name or node.text)[:30], # Aggressive truncation
                region=node.region,
                risk_score=node.risk.risk_score,
                locator=node.locator,
                child_ids=node.child_ids
            )
            overview_map[nid] = onode
            if not node.parent_id:
                roots.append(onode)

        tree = SecAXTree(
            roots=roots,
            detail_store=nodes_map,
            url=url,
            overview_map=overview_map
        )
        return tree

def annotate_and_build(raw_nodes: List[Dict[str, Any]], url: str = "", enable_domain_safety: bool = False) -> SecAXTree:
    builder = SecAXTreeBuilder(enable_domain_safety=enable_domain_safety)
    return builder.build(raw_nodes, url)
