"""Commercial prop-firm challenge fee schedule keyed against RulebookCatalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FEE_SCHEDULE_PATH: Path = Path(__file__).resolve().with_name("prop_firm_fees.json")


@dataclass(frozen=True)
class FeeQuote:
    """Verified per-tier challenge fee and payout metadata (no risk rules)."""

    challenge_key: str
    account_usd: int
    fee_usd: float
    refundable_on_pass: bool
    payout_frequency: str | None
    verified_at: str | None
    source_urls: tuple[str, ...]
    notes: str | None


def load_fee_schedule(path: Path = DEFAULT_FEE_SCHEDULE_PATH) -> dict[str, Any]:
    """Load the research prop-firm fee schedule JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"INVALID_FEE_SCHEDULE:{path}")
    if not isinstance(data.get("challenges"), dict):
        raise TypeError(f"INVALID_FEE_SCHEDULE:{path}")
    return data


def lookup_tier_fee(
    challenge_key: str,
    account_usd: float,
    *,
    path: Path = DEFAULT_FEE_SCHEDULE_PATH,
) -> FeeQuote | None:
    """Return the exact sourced fee quote for ``<firm>:<program>`` and ``account_usd``.

    Never scales prices linearly across account sizes; returns ``None`` when the
    challenge or exact tier is absent.
    """
    if isinstance(account_usd, bool):
        raise TypeError("ACCOUNT_USD_CANNOT_BE_BOOL")
    account_int = int(account_usd)
    if float(account_usd) != float(account_int) or account_int <= 0:
        return None

    schedule = load_fee_schedule(path)
    entry = schedule["challenges"].get(challenge_key)
    if not isinstance(entry, dict):
        return None

    tiers = entry.get("tiers")
    if not isinstance(tiers, dict):
        return None

    raw_tier = tiers.get(str(account_int))
    fee_usd: float | None = None
    if isinstance(raw_tier, (int, float)) and not isinstance(raw_tier, bool):
        fee_usd = float(raw_tier)
    elif isinstance(raw_tier, dict):
        raw_fee = raw_tier.get("fee")
        if isinstance(raw_fee, (int, float)) and not isinstance(raw_fee, bool):
            fee_usd = float(raw_fee)

    if fee_usd is None:
        return None

    urls = entry.get("source_urls") or ()
    return FeeQuote(
        challenge_key=challenge_key,
        account_usd=account_int,
        fee_usd=fee_usd,
        refundable_on_pass=bool(entry.get("refundableOnPass", False)),
        payout_frequency=entry.get("payoutFrequency"),
        verified_at=entry.get("verified_at"),
        source_urls=tuple(str(u) for u in urls),
        notes=entry.get("notes"),
    )
