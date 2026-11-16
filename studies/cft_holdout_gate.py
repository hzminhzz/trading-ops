"""Read-only prospective freeze integrity/maturity gate. Never reads OOS outcomes."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pit_guardrails import verify_forward_freeze

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "reports/2026-11-16__cft__10k__pit-evidence-v4__forward_freeze.json"
REGISTRY = Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-preflight", action="store_true")
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    args = parser.parse_args()
    doc = json.loads(args.freeze.read_text())
    result = verify_forward_freeze(
        doc, REGISTRY, datetime.now(UTC),
        evaluation_requested=args.evaluation_preflight,
    )
    if args.evaluation_preflight:
        # Original real-time source-receipt coverage for these new members
        # requires independent evidence. Do not treat maturity alone as proof.
        result["status"] = "BLOCKED_SOURCE_RECEIPTS_NOT_ENABLED"
        result["evaluate_allowed"] = False
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
