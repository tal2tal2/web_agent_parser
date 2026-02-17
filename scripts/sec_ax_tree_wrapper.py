import json
import sys
import os

# Ensure src is in pythonpath
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.web_agent_parser.sec_ax_tree.extract_html_only import extract_html_only
    from src.web_agent_parser.sec_ax_tree.annotate_and_build import annotate_and_build
except ImportError:
    # Fallback: try adding src directly if package resolution fails
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)
    try:
        from web_agent_parser.sec_ax_tree.extract_html_only import extract_html_only
        from web_agent_parser.sec_ax_tree.annotate_and_build import annotate_and_build
    except ImportError:
         # Last resort for structure variants
        from src.web_agent_parser.sec_ax_tree.extract_html_only import extract_html_only
        from src.web_agent_parser.sec_ax_tree.annotate_and_build import annotate_and_build

# Set this to True to filter output to only include potentially harmful nodes
FILTER_HARMFUL_ONLY = True

def _filter_overview_recursive(nodes: list) -> list:
    """Recursively filter overview nodes to keep only those with risk > 0 or having risky descendants."""
    filtered = []
    for node in nodes:
        # Process children first
        node['children'] = _filter_overview_recursive(node.get('children', []))
        
        # Keep node if it has risk OR if it has preserved children
        if node.get('risk', 0) > 0 or node['children']:
            filtered.append(node)
    return filtered


def sec_ax_tree_overview(html: str) -> str:
    """
    Parses HTML string into SecAXTree and returns the Overview JSON as a string.
    Usage in browsesafe_infer.py: parser_fn=sec_ax_tree_overview
    """
    if not html:
        return "{}"
        
    try:
        raw_nodes = extract_html_only(html)
        # We don't have a real URL or domain safety context here, using defaults
        tree = annotate_and_build(raw_nodes, url="about:blank", enable_domain_safety=False)
        
        full_data = tree.to_json(include_details=False)
        # Exclude details if present (though include_details=False should handle it)
        overview_data = {k: v for k, v in full_data.items() if k != 'details'}
        
        if FILTER_HARMFUL_ONLY:
            overview_data['overview'] = _filter_overview_recursive(overview_data.get('overview', []))

        return json.dumps(overview_data, ensure_ascii=False)
    except Exception as e:
        sys.stderr.write(f"Error in sec_ax_tree_overview: {e}\n")
        return "{}"


def sec_ax_tree_details(html: str) -> str:
    """
    Parses HTML string into SecAXTree and returns the Details JSON as a string.
    Usage in browsesafe_infer.py: parser_fn=sec_ax_tree_details
    """
    if not html:
        return "{}"
        
    try:
        raw_nodes = extract_html_only(html)
        tree = annotate_and_build(raw_nodes, url="about:blank", enable_domain_safety=False)
        
        full_data = tree.to_json(include_details=True)
        details_data = full_data.get('details', {})
        
        if FILTER_HARMFUL_ONLY:
            details_data = {
                k: v for k, v in details_data.items()
                if v.get('risk', {}).get('score', 0) > 0
            }

        return json.dumps(details_data, ensure_ascii=False)
    except Exception as e:
        sys.stderr.write(f"Error in sec_ax_tree_details: {e}\n")
        return "{}"

# For easy testing/usage similar to pi_html_parser main block
if __name__ == "__main__":
    sample_html = """
    <html><body>
      <button>Click me</button>
      <div><p>Some text</p></div>
    </body></html>
    """
    print("--- OVERVIEW ---")
    print(sec_ax_tree_overview(sample_html))
    print("\n--- DETAILS ---")
    print(sec_ax_tree_details(sample_html))
