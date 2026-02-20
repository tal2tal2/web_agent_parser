"""
run_agisdk_harness.py
---------------------
Runner for the SecAwareAgent on AGI SDK (REAL bench) tasks.

Two ways to configure:
  1. Edit the MANUAL CONFIG block below and run: python tests/run_agisdk_harness.py
  2. Pass CLI flags (they override the manual config values):
       python tests/run_agisdk_harness.py --task-type omnizon --security-aware
       python tests/run_agisdk_harness.py --task-name v2.omnizon-1 --model gpt-4o
       python tests/run_agisdk_harness.py --samples-json tests/sample_tasks.json

Task sources (pick one; priority: CLI > manual config):
  --task-name      : single named task, e.g. "v2.omnizon-1"
  --task-type      : all tasks of a type, e.g. "omnizon"
  --task-id        : specific ID within a type (used together with --task-type)
  --samples-json   : path to JSON file with task names or task-dict list
  (none)           : run all available tasks

See tests/sample_tasks.json for the expected JSON format.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Force UTF-8 stdout/stderr so box-drawing characters print correctly on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# ── MANUAL CONFIG ────────────────────────────────────────────────────────────
# Edit these values to run without CLI flags.
# CLI arguments (if supplied) override these.
# ---------------------------------------------------------------------------
MODEL_NAME     = "gpt-4.1-nano"   # any gpt-* model, e.g. "gpt-4o", "gpt-4o-mini"
SECURITY_AWARE = True             # True  → SecAXTree replaces DOM/AXTree
TASK_TYPE      = None              # e.g. "omnizon"  |  None = all tasks
TASK_NAME      = None              # e.g. "v2.omnizon-1"  (overrides TASK_TYPE)
TASK_ID        = None              # e.g. 1  (used together with TASK_TYPE)
SAMPLES_JSON   = "tests/sample_tasks.json"    # e.g. "tests/sample_tasks.json"  (use / not \)
HEADLESS       = False              # False  → opens a visible browser window
MAX_STEPS      = 5                # max actions per task episode
RESULTS_DIR    = "./tests/results" # directory to write result JSON files
FORCE_REFRESH  = True             # True  → ignore cached results and re-run
# ---------------------------------------------------------------------------

# Ensure project root is on sys.path so custom_nano_agent resolves
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.custom_nano_agent import SecAwareAgentArgs  # noqa: E402
from agisdk import REAL                                 # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(cfg: dict) -> None:
    """Print a startup banner showing the resolved configuration."""
    w = 52
    border = "═" * w
    print(f"\n╔{border}╗")
    print(f"║{'  SecAware Harness — Config':^{w}}║")
    print(f"╠{border}╣")
    for key, val in cfg.items():
        line = f"  {key:<18}: {val}"
        print(f"║{line:<{w}}║")
    print(f"╚{border}╝\n")


def load_tasks_from_json(path: str) -> List[str]:
    """
    Load a list of AGI SDK task-name strings from a JSON file.

    Accepted formats:
      - List of strings:
          ["webclones.omnizon-1", "webclones.dashdish-1"]

      - List of dicts (tries keys: "task_name", "id", "name"):
          [{"task_name": "webclones.omnizon-1", ...}, ...]

    Returns a list of task-name strings ready to pass to harness.run(tasks=...).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"samples-json file not found: {path}")

    with path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"samples-json must contain a JSON array, got {type(data).__name__}"
        )
    if len(data) == 0:
        return []

    first = data[0]
    if isinstance(first, str):
        return data

    if isinstance(first, dict):
        for key in ("task_name", "id", "name"):
            if key in first:
                return [item[key] for item in data]
        raise ValueError(
            f"Cannot detect task-name key in dict entries. "
            f"Available keys: {list(first.keys())}. "
            f"Expected one of: 'task_name', 'id', 'name'."
        )

    raise ValueError(
        f"Unexpected item type in samples-json array: {type(first).__name__}. "
        "Expected str or dict."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_agisdk_harness.py",
        description=(
            "Run the SecAwareAgent on AGI SDK / REAL Bench tasks. "
            "Edit the MANUAL CONFIG block at the top of this file, "
            "or pass CLI flags to override."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help=f"OpenAI model name (default from config: {MODEL_NAME!r})",
    )

    # Security mode
    p.add_argument(
        "--security-aware",
        action="store_true",
        default=False,
        help=(
            "Enable SecAXTree mode: replaces DOM/AXTree with the Security-Aware "
            "Accessibility Tree + gives the model a query_element_detail() tool."
        ),
    )

    # Task selection (mutually exclusive in effect, but not enforced strictly
    # so the user can combine --task-type and --task-id)
    task_group = p.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task-name",
        default=None,
        metavar="NAME",
        help="Run a single specific task, e.g. 'v2.omnizon-1'.",
    )
    task_group.add_argument(
        "--task-type",
        default=None,
        metavar="TYPE",
        help="Run all tasks of a type, e.g. 'omnizon'.",
    )
    task_group.add_argument(
        "--samples-json",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file containing task names to run. "
            "See tests/sample_tasks.json for format."
        ),
    )

    p.add_argument(
        "--task-id",
        type=int,
        default=None,
        metavar="ID",
        help="Specific task ID within --task-type, e.g. 1 → runs 'TYPE-1' only.",
    )

    # Execution options
    p.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run the browser headlessly (no visible window). Overrides HEADLESS=False in config.",
    )
    p.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Show the browser window (default if not specified via CLI).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help=f"Maximum actions per task (default from config: {MAX_STEPS}).",
    )
    p.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help=f"Directory to write results (default from config: {RESULTS_DIR!r}).",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Ignore cached results and re-run every task.",
    )
    p.add_argument(
        "--openai-api-key",
        default=None,
        metavar="KEY",
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable).",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    # Parse known args so unrecognised flags don't crash if user appends extras
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Warning] Unrecognised CLI arguments (ignored): {unknown}")

    # ── Merge CLI args with manual config ──────────────────────────────
    # CLI takes priority; fall back to the MANUAL CONFIG block values.
    model          = args.model          or MODEL_NAME
    security_aware = args.security_aware or SECURITY_AWARE
    task_name      = args.task_name      or TASK_NAME
    task_type      = args.task_type      or TASK_TYPE
    task_id        = args.task_id        if args.task_id is not None else TASK_ID
    samples_json   = args.samples_json   or SAMPLES_JSON
    # For headless: CLI flag wins if explicitly passed; otherwise use config.
    # argparse default is False, so we check if any headless-related flag was
    # explicitly given by comparing against the manual config default.
    headless_set_by_cli = "--headless" in sys.argv or "--no-headless" in sys.argv
    headless       = args.headless if headless_set_by_cli else HEADLESS
    max_steps      = args.max_steps      if args.max_steps is not None else MAX_STEPS
    results_dir    = args.results_dir    or RESULTS_DIR
    force_refresh  = args.force_refresh  or FORCE_REFRESH
    openai_api_key = args.openai_api_key

    # Apply API key to environment if provided (OpenAI client picks it up)
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # ── Startup banner ─────────────────────────────────────────────────
    cfg = {
        "model":          model,
        "security_aware": security_aware,
        "task_name":      task_name      or "(not set)",
        "task_type":      task_type      or "(not set)",
        "task_id":        task_id        or "(not set)",
        "samples_json":   samples_json   or "(not set)",
        "headless":       headless,
        "max_steps":      max_steps,
        "results_dir":    results_dir,
        "force_refresh":  force_refresh,
    }
    _banner(cfg)

    # ── Build agent args ───────────────────────────────────────────────
    agent_args = SecAwareAgentArgs(
        model_name=model,
        security_aware=security_aware,
        openai_api_key=openai_api_key,
    )

    # ── Determine task list ────────────────────────────────────────────
    # When samples_json is set we load the task list ourselves and pass it
    # directly to harness.run(tasks=...), bypassing the SDK's task resolver.
    custom_tasks: Optional[List[str]] = None
    if samples_json:
        print(f"Loading tasks from JSON: {samples_json}", flush=True)
        custom_tasks = load_tasks_from_json(samples_json)
        print(f"  → {len(custom_tasks)} task(s) loaded: {custom_tasks}", flush=True)
        # Clear task_name/task_type so harness does not apply additional filters
        task_name = None
        task_type = None
        task_id   = None

    # ── Build harness ──────────────────────────────────────────────────
    h = REAL.harness(
        agentargs=agent_args,
        task_name=task_name,
        task_type=task_type,
        task_id=task_id,
        headless=headless,
        max_steps=max_steps,
        use_html=True,          # needed for pruned_html in obs_preprocessor
        use_axtree=True,        # keep axtree for non-secure fallback & goal resolution
        use_screenshot=False,   # screenshots not required; saves bandwidth
        results_dir=results_dir,
        use_cache=not force_refresh,
        force_refresh=force_refresh,
    )

    # ── Run ────────────────────────────────────────────────────────────
    print("Starting harness run…\n", flush=True)
    if custom_tasks is not None:
        results = h.run(tasks=custom_tasks)
    else:
        results = h.run()

    # ── Summary ────────────────────────────────────────────────────────
    total   = len(results)
    success = sum(1 for r in results.values() if r.get("cum_reward", 0) == 1)
    print(f"\n{'═'*52}")
    print(f"  Run complete: {success}/{total} tasks succeeded ({success/total*100:.1f}%)" if total else "  No results.")
    print(f"  Results dir : {results_dir}")
    print(f"{'═'*52}\n")

    return results


if __name__ == "__main__":
    main()

