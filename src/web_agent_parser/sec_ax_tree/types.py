from enum import Enum
from typing import List, Dict, Optional, Any, Set, Union
from dataclasses import dataclass, field

# --- Enums ---

class ActionType(str, Enum):
    NAVIGATE = "NAVIGATE"
    CLICK = "CLICK"
    INPUT_TEXT = "INPUT_TEXT"
    SELECT = "SELECT"
    TOGGLE = "TOGGLE"
    UPLOAD = "UPLOAD"
    SCROLL_TARGET = "SCROLL_TARGET"
    READ_ONLY = "READ_ONLY"
    COMPOSITE = "COMPOSITE"

class ContentKind(str, Enum):
    # Communication
    EMAIL = "EMAIL"
    EMAIL_SUBJECT = "EMAIL_SUBJECT"
    EMAIL_BODY = "EMAIL_BODY"
    POST = "POST"
    COMMENT = "COMMENT"
    REVIEW = "REVIEW"
    
    # User Profile
    PROFILE_INTRO = "PROFILE_INTRO"
    PROFILE_ABOUT = "PROFILE_ABOUT"
    PROFILE_RECOMMENDATION = "PROFILE_RECOMMENDATION"
    
    # Calendar / Events
    CALENDAR_EVENT = "CALENDAR_EVENT"
    EVENT_TITLE = "EVENT_TITLE"
    EVENT_LOCATION = "EVENT_LOCATION"
    EVENT_DESCRIPTION = "EVENT_DESCRIPTION"
    
    # Commerce
    PRODUCT_LISTING = "PRODUCT_LISTING"
    PRODUCT_PRICE = "PRODUCT_PRICE"
    CART = "CART"
    CHECKOUT = "CHECKOUT"
    
    # Navigation / General
    NAV_ITEM = "NAV_ITEM"
    SEARCH = "SEARCH"
    FILTER = "FILTER"
    UNKNOWN = "UNKNOWN"

class RegionType(str, Enum):
    HEADER = "HEADER"
    NAV = "NAV"
    MAIN = "MAIN"
    ASIDE = "ASIDE"
    FOOTER = "FOOTER"
    MODAL = "MODAL"
    BANNER = "BANNER"
    TOAST = "TOAST"
    UNKNOWN = "UNKNOWN"

class Provenance(str, Enum):
    FIRST_PARTY_UI = "FIRST_PARTY_UI"
    USER_GENERATED = "USER_GENERATED"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"

class FieldSensitivity(str, Enum):
    NONE = "NONE"
    CREDENTIAL = "CREDENTIAL"
    PAYMENT = "PAYMENT"
    PII = "PII"

# --- Dataclasses ---

@dataclass
class RiskMetadata:
    risk_score: float = 0.0  # 0.0 to 1.0
    risk_reasons: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)  # Excerpts that caused flags
    # Attack surface flags
    has_url: bool = False
    has_button_or_link: bool = False
    instruction_like: bool = False
    asks_credentials: bool = False
    asks_payment: bool = False
    
    # Context-fit specific
    context_fit_score: float = 1.0  # 1.0 = fits well, 0.0 = completely anomalous

@dataclass
class NodeCore:
    """Base representation of a page element."""
    id: str  # Stable hash
    tag: str
    role: str = ""
    name: str = ""
    text: str = ""
    attrs: Dict[str, str] = field(default_factory=dict)
    state: Dict[str, bool] = field(default_factory=dict)  # disabled, checked, etc.
    geom: Optional[Dict[str, float]] = None  # bbox + in_viewport
    locator: str = ""  # Playwright locator or CSS/XPath
    
    # Computed Semantic Tags
    content_kinds: Set[ContentKind] = field(default_factory=set)
    region: RegionType = RegionType.UNKNOWN
    provenance: Provenance = Provenance.UNKNOWN
    sensitivity: FieldSensitivity = FieldSensitivity.NONE
    
    # Computed Action / Risk
    action_type: ActionType = ActionType.READ_ONLY
    risk: RiskMetadata = field(default_factory=RiskMetadata)
    
    # Hierarchy (IDs only for lightweight storage)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)

@dataclass
class OverviewNode:
    """Tier 1: Lightweight node for the overview tree."""
    id: str
    role: str
    action_type: ActionType
    name_preview: str  # Aggressively truncated
    region: RegionType
    risk_score: float
    locator: str
    child_ids: List[str] = field(default_factory=list)

@dataclass
class SecAXTree:
    """
    The main output object. 
    Contains the lightweight overview tree and an on-demand detail store.
    """
    # Tier 1: The Overview
    # Root IDs of the overview tree
    roots: List[OverviewNode] 
    
    # Tier 2: The Detail Store (Reference only, usually not sent to LLM initially)
    # Map of node_id -> Full NodeCore
    detail_store: Dict[str, NodeCore] = field(default_factory=dict)
    
    # Global metadata
    url: str = ""
    domain_safety: Optional[Dict[str, Any]] = None  # If enabled
    
    def get_node_detail(self, node_id: str) -> Optional[NodeCore]:
        return self.detail_store.get(node_id)
    
    def get_region_detail(self, region_type: RegionType) -> List[NodeCore]:
        return [
            node for node in self.detail_store.values() 
            if node.region == region_type
        ]

    def to_json(self, include_details: bool = False) -> Dict[str, Any]:
        """Helper to serialize for the LLM."""
        data = {
            "url": self.url,
            "domain_safety": self.domain_safety,
            "overview": [self._serialize_overview(r) for r in self.roots]
        }
        if include_details:
            data["details"] = {
                nid: self._serialize_detail(n) 
                for nid, n in self.detail_store.items()
            }
        return data

    def _serialize_overview(self, node: OverviewNode) -> Dict[str, Any]:
        # Recursive serialization for overview tree
        # In a real impl, we might flatten this to save tokens even more,
        # but a tree structure is often easier for LLMs to reason about relationships.
        return {
            "id": node.id,
            "role": node.role,
            "action": node.action_type.value,
            "preview": node.name_preview,
            "region": node.region.value,
            "risk": node.risk_score,
            "loc": node.locator,
            "children": [
                self._serialize_overview(self._find_overview_node(cid)) 
                for cid in node.child_ids 
                if self._find_overview_node(cid)
            ]
        }

    def _find_overview_node(self, node_id: str) -> Optional[OverviewNode]:
        # Helper to find overview node in the tree structure.
        # This is a bit inefficient if not indexed, but fine for PoC.
        # Ideally, we'd store overview nodes in a flat dict too or search recursively.
        # For simplicity in this method, let's assume we can traverse or we have a map.
        # We'll traverse from roots for now or expect the caller to handle structure.
        # Actually, let's just make roots a list of top-level nodes, and children are nested.
        # The overview structure in memory is a tree.
        
        # Wait, the dataclass defines `roots: List[OverviewNode]`. 
        # But `OverviewNode` has `child_ids`. If we want the JSON to be nested, 
        # we need to be able to resolve those IDs to `OverviewNode` objects.
        # For the PoC, let's assume `roots` contains the top-level nodes, 
        # and we might need an `overview_map` to resolve children if they aren't directly linked.
        # Let's add an overview_map to the class for easier serialization.
        return self.overview_map.get(node_id)

    overview_map: Dict[str, OverviewNode] = field(default_factory=dict)

    def _serialize_detail(self, node: NodeCore) -> Dict[str, Any]:
        return {
            "id": node.id,
            "tag": node.tag,
            "role": node.role,
            "name": node.name,
            "text": node.text,
            "attrs": node.attrs,
            "state": node.state,
            "kinds": [k.value for k in node.content_kinds],
            "region": node.region.value,
            "risk": {
                "score": node.risk.risk_score,
                "reasons": node.risk.risk_reasons,
                "triggers": node.risk.triggers,
                "context_fit": node.risk.context_fit_score
            }
        }
