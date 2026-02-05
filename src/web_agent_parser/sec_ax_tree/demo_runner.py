import argparse
import asyncio
import json
import sys
import os

from playwright.async_api import async_playwright

# Fix path to allow importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.web_agent_parser.sec_ax_tree.extract_playwright import extract_playwright
from src.web_agent_parser.sec_ax_tree.extract_html_only import extract_html_only
from src.web_agent_parser.sec_ax_tree.annotate_and_build import annotate_and_build

def save_tree_outputs(tree, base_name: str):
    """Saves the tree into two files: one for overview and one for full details."""
    
    # 1. Text Overview (Compact, 1 line per element)
    text_overview = tree.to_compact_text()
    overview_file = f"{base_name}_overview.txt"
    with open(overview_file, 'w', encoding='utf-8') as f:
        f.write(text_overview)
    print(f"Saved compact overview tree to {overview_file}", file=sys.stderr)

    # 2. JSON Overview (Optional, for reference)
    full_data = tree.to_json(include_details=True)
    json_overview = {k: v for k, v in full_data.items() if k != 'details'}
    json_file = f"{base_name}_overview.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_overview, f, indent=2) # Indented JSON for readability
    
    # 3. Details (only the 'details' part)
    details_data = full_data.get('details', {})
    details_file = f"{base_name}_details.json"
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(details_data, f, indent=2)
    print(f"Saved details tree to {details_file}", file=sys.stderr)


async def run_browser_mode(url: str, enable_safety: bool):
    print(f"Starting Browser Mode for {url}...", file=sys.stderr)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url)
            # await page.wait_for_load_state("networkidle") # Optional
            
            raw_nodes = await extract_playwright(page)
            tree = annotate_and_build(raw_nodes, url=url, enable_domain_safety=enable_safety)
            
            save_tree_outputs(tree, "browser_output")
            
        finally:
            await browser.close()

def run_html_mode(file_path: str, enable_safety: bool):
    print(f"Starting HTML Mode for {file_path}...", file=sys.stderr)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        raw_nodes = extract_html_only(html_content)
        tree = annotate_and_build(raw_nodes, url="file://" + file_path, enable_domain_safety=enable_safety)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_tree_outputs(tree, base_name)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SecAXTree Parser PoC")
    parser.add_argument("--mode", choices=["browser", "html"], required=True, help="Input mode")
    parser.add_argument("--target", required=True, help="URL (browser mode) or File Path (html mode)")
    parser.add_argument("--enable-domain-safety", action="store_true", help="Enable optional domain safety features")
    parser.add_argument("--show-details", action="store_true", help="Include full details in output (usually on-demand)")
    
    args = parser.parse_args()
    
    if args.mode == "browser":
        asyncio.run(run_browser_mode(args.target, args.enable_domain_safety))
    else:
        run_html_mode(args.target, args.enable_domain_safety)

if __name__ == "__main__":
    main()
