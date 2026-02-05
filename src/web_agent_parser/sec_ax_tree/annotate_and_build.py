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

# Ported from pi_html_parser.py
RE_OVERRIDE = re.compile(
    r"\b(ignore (all )?(previous|prior|earlier) (instructions|messages|context)|"
    r"disregard (the )?(above|previous)|"
    r"forget (the )?(above|previous)|"
    r"override (the )?(instructions|rules)|"
    r"highest priority|"
    r"must comply|"
    r"do this before anything else)\b",
    re.IGNORECASE,
)

RE_ROLE = re.compile(
    r"\b(act as|you are now|from now on|pretend to be|as (a|an) (support|assistant|agent|auditor|collector))\b",
    re.IGNORECASE,
)

RE_SYS_EXFIL = re.compile(
    r"\b(system prompt|system instructions|developer message|hidden prompt|internal policy|confidential prompt)\b",
    re.IGNORECASE,
)

RE_EXFIL_VERB = re.compile(
    r"\b(send|email|forward|upload|post|submit|share|report|export|dump|leak)\b",
    re.IGNORECASE,
)

RE_DATA_OBJECT = re.compile(
    r"\b(cookies?|session|token|api key|password|credentials?|transaction logs?|chat history|history|"
    r"personal data|user data|all data|system prompt|instructions)\b",
    re.IGNORECASE,
)

RE_HYPOTHETICAL = re.compile(
    r"\b(how would you|how could you|what steps would you take|please demonstrate|demonstrate how|"
    r"for testing purposes|hypothetically|in a hypothetical scenario|as an example)\b",
    re.IGNORECASE,
)

RE_DELIMS = re.compile(
    r"(?m)^\s*(---\s*END\s*CONTEXT\s*---|END CONTEXT|BEGIN SYSTEM|END SYSTEM|NEW INSTRUCTION|"
    r"###\s*NEW\s*INSTRUCTIONS\s*###)\s*$|```",
    re.IGNORECASE,
)

RE_IMPORTANT_MSG = re.compile(
    r"\b(important message|this is an important message|before you can solve the task|"
    r"<\s*information\s*>|</\s*information\s*>)\b",
    re.IGNORECASE,
)

RE_TODO = re.compile(r"\b(todo\s*:|important!!!\s*todo)\b", re.IGNORECASE)
RE_INJECAGENT = re.compile(r"\binstruction\s*:\s*", re.IGNORECASE)

RE_URGENCY = re.compile(
    r"\b(urgent|urgently|asap|immediately|important|critical|attention|required now|time[- ]sensitive)\b",
    re.IGNORECASE,
)

RE_INTENT_ACTION = re.compile(
    r"\b("
    r"your response (must|should)|"
    r"answer (only )?with|"
    r"reply (only )?with|"
    r"output (only )?|"
    r"return (only )?|"
    r"respond (only )?with|"
    r"begin (your )?response|"
    r"include (the )?following|"
    r"do not mention|don't mention|"
    r"do not explain|don't explain"
    r")\b",
    re.IGNORECASE,
)

IMPERATIVE_VERBS = {
    "click", "tap", "select", "choose", "enter", "type", "navigate", "go",
    "ignore", "forget", "accept", "reject", "confirm", "submit", "login",
    "sign", "buy", "order", "pay"
}

SENSITIVE_KEYWORDS = {
    "password", "credit", "card", "cvv", "social security", "ssn", "bank", "account number"
}

# --- Risk Analysis ---

class RiskAnalyzer:
    def compute_risk(self, node: NodeCore) -> RiskMetadata:
        text = (node.text + " " + node.name).lower()
        risk = RiskMetadata()
        
        # 1. Prompt Injection Signals (Regex-based from pi_html_parser)
        
        # High Severity Checks
        if RE_SYS_EXFIL.search(text):
            risk.risk_score += 10.0
            risk.triggers.append("system_prompt_exfiltration")
            risk.risk_reasons.append("Attempts to exfiltrate system prompt")
            
        if RE_OVERRIDE.search(text) or RE_ROLE.search(text) or RE_DELIMS.search(text):
            risk.risk_score += 6.0
            risk.triggers.append("intent_override")
            risk.risk_reasons.append("Attempts to override instructions or context")

        # Exfiltration Intent (Verb + Object)
        if RE_EXFIL_VERB.search(text) and RE_DATA_OBJECT.search(text):
            risk.risk_score += 10.0
            risk.triggers.append("intent_exfiltration")
            risk.risk_reasons.append("Explicit data exfiltration intent")

        # Action Constraints
        if RE_INTENT_ACTION.search(text):
            risk.risk_score += 5.0
            risk.triggers.append("intent_action")
            risk.risk_reasons.append("Attempts to constrain agent output format")

        # Medium/Lower Severity Checks
        if RE_HYPOTHETICAL.search(text):
            risk.risk_score += 2.0
            risk.triggers.append("indirect_hypothetical")
        
        if RE_URGENCY.search(text):
            risk.risk_score += 1.0
            risk.triggers.append("urgency")

        if RE_IMPORTANT_MSG.search(text) or RE_TODO.search(text) or RE_INJECAGENT.search(text):
            risk.risk_score += 4.0
            risk.triggers.append("dataset_specific_pattern")

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
             risk.risk_score += 2.0

        # 3. Context-Fit Heuristics (The "Does this belong here?" check)
        fit_score = self._compute_context_fit(node, text)
        risk.context_fit_score = fit_score
        
        if fit_score < 0.5:
            risk.risk_score += (0.5 - fit_score) * 4  # Penalize low fit more heavily
            risk.risk_reasons.append(f"Low context fit ({fit_score:.2f})")
            
        # CLAMPING REMOVED per user request
        # risk.risk_score = min(1.0, risk.risk_score)
        
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
        
        def is_redundant_wrapper(n: NodeCore) -> bool:
            """Checks if a node is redundant."""
            # 1. Always skip HTML/BODY if they have children (they are just roots)
            if n.tag in {"html", "body"} and n.child_ids:
                return True

            # 2. Check for single-child identical-text wrappers
            if len(n.child_ids) == 1:
                child = nodes_map[n.child_ids[0]]
                
                # Normalize text comparison
                if " ".join(n.text.split()) == " ".join(child.text.split()):
                    # If parent is interactive and child isn't, KEEP parent (e.g. <button><span>text</span></button>)
                    if n.action_type != ActionType.READ_ONLY and child.action_type == ActionType.READ_ONLY:
                        return False
                        
                    # Otherwise, the child is likely the more specific element (or equally specific).
                    # We prefer the inner element as it's closer to the leaf content.
                    return True
                    
            return False

        def is_overview_worthy(n: NodeCore) -> bool:
            # 1. Prune redundant wrappers immediately
            if is_redundant_wrapper(n):
                return False

            # 2. Prune children of Interactive Elements (The Parent is the Atom)
            # If the parent is interactive (e.g. <button>), its text content is already
            # summarized in the parent. We don't need to see the <span> inside.
            # Exception: If the child itself is interactive (rare/invalid but possible).
            if n.parent_id:
                parent = nodes_map[n.parent_id]
                if parent.action_type != ActionType.READ_ONLY and n.action_type == ActionType.READ_ONLY:
                    return False
                
            # 3. Keep Interactive Elements
            if n.action_type != ActionType.READ_ONLY: 
                return True
            
            # 4. Keep High Risk
            if n.risk.risk_score > 0.0: 
                return True
            
            # 5. Keep Region Roots (if they signal a region change)
            if n.region != RegionType.UNKNOWN and n.parent_id and nodes_map[n.parent_id].region != n.region: 
                return True 
                
            # 6. Keep Headings
            if n.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}: 
                return True 
            
            # 7. Keep nodes with text content...
            # BUT only if they aren't just purely containers for other worthy nodes.
            # If a READ_ONLY node has children, we generally trust the children to carry the signal,
            # UNLESS it's a "mixed content" node (text + links).
            if n.text.strip():
                # Existing single-child check (covered by is_redundant_wrapper, but logic here was slightly different)
                # Let's rely on is_redundant_wrapper for the strict single-child case.
                
                # Heuristic: If it has multiple children, and it's not a Region/Heading/Risk (checked above),
                # and it's just a DIV/UL/SECTION/LI, it's likely just aggregating text.
                # We'll skip it to reduce "DOM Soup" noise.
                if n.child_ids and n.tag in {"div", "ul", "ol", "li", "section", "article", "nav", "header", "footer"}:
                     # Be careful: <p> is not in this list, so we keep <p>Text <a>Link</a></p>
                     return False
                     
                return True
                
            return False

        # Build recursive overview structure
        # We need to rebuild the tree connecting only "worthy" nodes.
        
        for nid, node in nodes_map.items():
            if not is_overview_worthy(node):
                continue

            onode = OverviewNode(
                id=node.id,
                role=node.role,
                action_type=node.action_type,
                name_preview=(node.name or node.text)[:30], # Aggressive truncation
                region=node.region,
                risk_score=node.risk.risk_score,
                locator=node.locator,
                child_ids=[] # Will populate below
            )
            overview_map[nid] = onode
            
        # Link up the worthy nodes
        for nid, onode in overview_map.items():
            original_node = nodes_map[nid]
            
            # Find nearest worthy ancestor
            curr_parent_id = original_node.parent_id
            worthy_parent_id = None
            
            while curr_parent_id:
                if curr_parent_id in overview_map:
                    worthy_parent_id = curr_parent_id
                    break
                curr_parent_id = nodes_map[curr_parent_id].parent_id
            
            if worthy_parent_id:
                overview_map[worthy_parent_id].child_ids.append(nid)
            else:
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
