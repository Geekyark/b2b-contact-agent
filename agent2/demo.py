"""Run in offline mode without API keys:  python -m agent2.demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent2.config import Settings
from agent2.inputs import load_companies
from agent2.pipeline import process_batch
from agent2.storage import Database


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = Settings.from_env()
    settings.search_provider = "mock"   # force offline mode
    settings.tavily_api_key = ""
    settings.output_dir = str(REPO_ROOT / "data" / "output")

    companies = load_companies(REPO_ROOT / "data" / "sample_input" / "agent1_companies.json")
    if not companies.companies:
        print("No companies loaded - check data/sample_input/agent1_companies.json")
        return 1

    offline = settings.search_provider == "mock"
    print(f"Running offline demo over {len(companies.companies)} companies (mock search)...")
    with Database(settings.database_path) as db:
        results, run_id = process_batch(
            companies.companies, settings, db,
            do_verify=True, do_export=True,
            search_provider=None,
            offline=offline,
        )
    print(f"Run {run_id} finished.")
    for r in results:
        print(f"  {r.company_name}: {len(r.contacts)} contacts" + (f" (error: {r.error})" if r.error else ""))
        for c in r.contacts:
            print(f"     - {c.name} | {c.title} | {c.work_email} [{c.email_status}] | {c.linkedin_url}")
    print(f"\nExports written to {settings.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())