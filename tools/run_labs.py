"""
Run every lab in the repo with the mock model and report which ones fail.

  python tools/run_labs.py                 # all labs
  python tools/run_labs.py 05 08           # only chapters whose folder starts with 05 or 08

Each script runs from its own folder, with a time limit, and with the API-key
environment variables removed, so nothing can reach a real model by accident.
Exit code 1 if any lab fails.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 300                                   # seconds per script
NOT_LABS = {"team.py", "bridge_server.py"}      # helper modules / long-running servers, not labs
NEEDS_MCP = {"wire_demo.py"}                    # needs `pip install "mcp[cli]>=1.20,<2"`
KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "AGENT_LAB_PROVIDER"]


def labs(prefixes):
    folders = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())
    folders += sorted(p for p in (ROOT / "side-quests").iterdir() if p.is_dir())
    for folder in folders:
        if prefixes and not any(folder.name.startswith(x) for x in prefixes):
            continue
        for script in sorted(folder.glob("*.py")):
            if script.name not in NOT_LABS:
                yield script


def has_mcp():
    """Side quest 2 is written for the 1.x MCP SDK (2.x renamed FastMCP)."""
    try:
        import mcp.server.fastmcp  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    prefixes = sys.argv[1:]
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env["PYTHONIOENCODING"] = "utf-8"
    mcp_ok = has_mcp()
    failed = []
    for script in labs(prefixes):
        name = script.relative_to(ROOT)
        if script.name in NEEDS_MCP and not mcp_ok:
            print(f"  skip  {name}  (needs: pip install \"mcp[cli]>=1.20,<2\")")
            continue
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, script.name], cwd=script.parent, env=env,
                                  capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT)
            ok, detail = proc.returncode == 0, (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            ok, detail = False, [f"timed out after {TIMEOUT} s"]
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}  ({time.perf_counter() - t0:.1f} s)")
        if not ok:
            failed.append(name)
            for line in detail:
                print(f"          {line}")
    print(f"\n{len(failed)} failed" if failed else "\nall labs passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
