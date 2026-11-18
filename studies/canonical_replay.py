"""Canonical research replay, universe discovery, Guardian rules, and PM03 reconciliation.

Provides shared, import-safe building blocks for ``trading-ops`` studies so individual
reports reuse one audited implementation rather than duplicating ad-hoc simulators:

1. ``load_registry_universe`` — dynamic SQLite screening over all configured wallets.
2. ``assess_bybit_symbol_screen`` — materiality-aware symbol mapping and scale alias flags.
3. ``reconstruct_positions_on_grid`` — causal ``new`` vs anchored ``full`` replay with zero
   pre-first-fill backward lookahead.
4. ``replay_unit_coins_to_5m`` — execution-aligned 5m frames directly consumable by
   ``pod_constructor.build_tensor_from_unit_replays``, with pair-local gap termination.
5. ``simulate_guardian_challenge`` — causal Challenge Lifecycle simulator distinguishing
   ``static_floor_dd_usd`` from ``peak_to_trough_dd_usd`` and separating terminal challenge
   verdicts from continuous empirical daily MC blocks.
6. ``reconcile_portfolio_with_pm03`` — Tier-2 ``copytrade_pm.pm03_account`` Decimal check.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
import tomllib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal as D
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from copytrade_pm import wallet_registry as pm_registry
from copytrade_pm.pm03_account import Account, Instrument, Rulebook
from copytrade_pm.pm03_execution import CostModel, MarketObservation, execute_targets
from copytrade_pm.pm_research_contracts import RetrospectiveEvidence

PolicyType = Literal["new", "full"]
DEFAULT_COST_BPS = 6.5
DEFAULT_FEE_BPS = 4.5
DEFAULT_SLIPPAGE_BPS = 2.0
GAP_EXIT_HAIRCUT_BPS = 50.0


@dataclass(frozen=True)
class WalletRecord:
    """Read-only wallet metadata from ``wallet_registry.sqlite``."""

    wallet_id: int
    address: str
    alias: str | None
    configured: bool
    excluded: bool

    @property
    def label(self) -> str:
        """Sanitized identifier safe for public reports and JSON outputs."""
        return self.alias if self.alias else f"W#{self.wallet_id}"


@dataclass(frozen=True)
class SymbolScreenResult:
    """Materiality-aware venue symbol screen for one candidate wallet."""

    label: str
    total_fills: int
    mapped_fills: int
    unmapped_coins: tuple[str, ...]
    mapped_coins: tuple[str, ...]
    unmapped_fill_frac: float
    unmapped_notional_frac: float | None
    scale_alias_candidates: dict[str, str]
    status: Literal[
        "EXACT_MAPPED",
        "MAPPED_WITH_EXCLUDED_TAIL",
        "REJECT_UNMAPPED_MATERIAL",
        "REJECT_NO_FILLS",
    ]


@dataclass(frozen=True)
class ReconstructedGrid:
    """Causal 5-minute signed position vector with provenance flags."""

    timestamps_ms: np.ndarray
    pos: np.ndarray
    unanchored_pre_inventory: bool
    first_fill_ms: int | None
    first_observed_pre_pos: float


@dataclass(frozen=True)
class AsOfUniverseResult:
    """Point-in-time or explicitly labeled retrospective universe snapshot."""

    records: list[WalletRecord]
    roster_sha256: str
    registry_inventory_sha256: str
    asof_utc: str
    mode: Literal["PIT_STRICT_ASOF", "RETROSPECTIVE_ASOF_MASKED"]
    retrospective_evidence: dict[str, Any] | None


def _parse_reset_hhmm(value: str | None) -> tuple[int, int]:
    """Parse canonical ``HH:MM`` UTC reset schedule, defaulting UNKNOWN to 00:00."""
    if value is None:
        return 0, 0
    raw = str(value).strip()
    if not raw or raw.upper() == "UNKNOWN":
        return 0, 0
    if ":" not in raw:
        raise ValueError(f"INVALID_RESET_TIME:{value}")
    hh_s, mm_s = raw.split(":", 1)
    if not (hh_s.isdigit() and mm_s.isdigit() and len(hh_s) == 2 and len(mm_s) == 2):
        raise ValueError(f"INVALID_RESET_TIME:{value}")
    hh, mm = int(hh_s), int(mm_s)
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"INVALID_RESET_TIME:{value}")
    return hh, mm


def check_cross_repo_pin_status(
    trading_ops_root: Path | str,
    prop_production_root: Path | str | None = None,
    copytrade_pm_root: Path | str | None = None,
) -> dict[str, Any]:
    """Compare ``trading-ops`` copytrade-pm pin against ``prop-production/compatibility.json``."""
    ops_toml = Path(trading_ops_root) / "pyproject.toml"
    pyproject = tomllib.loads(ops_toml.read_text(encoding="utf-8"))
    trading_ops_rev = str(pyproject["tool"]["uv"]["sources"]["copytrade-pm"]["rev"])

    compat_commit: str | None = None
    contracts: dict[str, Any] = {}
    if prop_production_root is not None:
        compat_path = Path(prop_production_root) / "src" / "prop_production" / "compatibility.json"
        compat = json.loads(compat_path.read_text(encoding="utf-8"))
        compat_commit = str(compat["repositories"]["copytrade-pm"]["commit"])
        contracts = dict(compat.get("contracts", {}))

    if compat_commit is None:
        return {
            "trading_ops_rev": trading_ops_rev,
            "prop_production_pin": None,
            "ancestry_valid": None,
            "promotion_ready": False,
            "status": "NO_COMPATIBILITY_MANIFEST",
            "contracts": contracts,
        }

    if trading_ops_rev == compat_commit:
        ancestry_valid = True
        status = "SYNCED"
    elif copytrade_pm_root is not None and Path(copytrade_pm_root).exists():
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(copytrade_pm_root),
                "merge-base",
                "--is-ancestor",
                compat_commit,
                trading_ops_rev,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        ancestry_valid = proc.returncode == 0
        status = "PRODUCTION_PIN_PROMOTION_PENDING" if ancestry_valid else "DIVERGED_PIN_ERROR"
    else:
        ancestry_valid = False
        status = "DIVERGED_PIN_ERROR"

    return {
        "trading_ops_rev": trading_ops_rev,
        "prop_production_pin": compat_commit,
        "ancestry_valid": ancestry_valid,
        "promotion_ready": trading_ops_rev == compat_commit,
        "status": status,
        "contracts": contracts,
    }


@dataclass(frozen=True)
class GuardianSpec:
    """Prop-firm challenge and internal Guardian risk specification."""

    initial_equity: float = 10_000.0
    profit_target: float = 800.0
    max_static_dd: float = 1_000.0
    daily_loss_limit: float = 500.0
    guardian_sweep_loss: float = 400.0
    cost_bps: float = DEFAULT_COST_BPS
    cost_multiplier: float = 1.0
    funding_bps_per_8h: float = 0.0
    funding_multiplier: float = 1.0
    reset_utc_hour: int = 0
    reset_utc_minute: int = 5
    daily_loss_basis: Literal["balance", "equity"] = "balance"
    daily_loss_rate: float | None = None
    guardian_sweep_rate: float | None = None
    reenter_after_sweep: Literal["new_only", "immediate_target"] = "new_only"
    min_trade_notional_usd: float = 0.0
    amount_step: float = 0.0
    min_amount: float = 0.0
    amount_step_by_coin: Mapping[str, float] | None = None
    min_amount_by_coin: Mapping[str, float] | None = None
    max_gross_leverage: float | None = None

    def __post_init__(self) -> None:
        if self.daily_loss_basis not in ("balance", "equity"):
            raise ValueError(f"INVALID_DAILY_LOSS_BASIS:{self.daily_loss_basis}")
        if self.daily_loss_rate is not None and not (0.0 < float(self.daily_loss_rate) < 1.0):
            raise ValueError(f"INVALID_DAILY_LOSS_RATE:{self.daily_loss_rate}")
        if self.guardian_sweep_rate is not None and not (
            0.0 < float(self.guardian_sweep_rate) < 1.0
        ):
            raise ValueError(f"INVALID_GUARDIAN_SWEEP_RATE:{self.guardian_sweep_rate}")

    @classmethod
    def from_challenge_spec(
        cls,
        spec: Any,
        *,
        scale: float = 1.0,
        use_balance_rate: bool = True,
        **overrides: Any,
    ) -> GuardianSpec:
        """Build ``GuardianSpec`` from a canonical ``ChallengeRulebookSpec`` or mapping."""
        if scale <= 0.0:
            raise ValueError(f"RULEBOOK_SCALE_POSITIVE:{scale}")
        get = spec.get if isinstance(spec, Mapping) else lambda k, d=None: getattr(spec, k, d)
        trailing_basis = str(get("trailing_basis", "")).upper()
        if trailing_basis != "STATIC":
            raise ValueError(f"UNSUPPORTED_TRAILING_BASIS:{trailing_basis}")
        raw_daily_basis = str(get("daily_basis", "")).upper()
        if raw_daily_basis == "BALANCE":
            daily_loss_basis: Literal["balance", "equity"] = "balance"
        elif raw_daily_basis == "EQUITY":
            daily_loss_basis = "equity"
        else:
            raise ValueError(f"UNSUPPORTED_DAILY_BASIS:{raw_daily_basis}")

        initial_base = float(get("initial"))
        daily_base = float(get("daily_loss"))
        static_base = float(get("trailing_loss"))
        raw_target = get("target")
        if "profit_target" in overrides:
            target_usd = float(overrides["profit_target"])
        elif raw_target is None or str(raw_target).upper() in ("NONE", "N/A", "UNKNOWN"):
            raise ValueError(f"CHALLENGE_PROFIT_TARGET_UNSPECIFIED:{raw_target}")
        else:
            target_usd = float(raw_target) * float(scale)

        initial_usd = initial_base * float(scale)
        daily_usd = daily_base * float(scale)
        static_usd = static_base * float(scale)
        sweep_usd = 0.8 * daily_usd
        reset_h, reset_m = _parse_reset_hhmm(get("reset_time_hhmm", "00:00"))
        if use_balance_rate and daily_loss_basis == "balance":
            daily_rate: float | None = daily_base / initial_base
            sweep_rate: float | None = 0.8 * daily_rate
        else:
            daily_rate = None
            sweep_rate = None

        kwargs: dict[str, Any] = {
            "initial_equity": initial_usd,
            "profit_target": target_usd,
            "max_static_dd": static_usd,
            "daily_loss_limit": daily_usd,
            "guardian_sweep_loss": sweep_usd,
            "reset_utc_hour": reset_h,
            "reset_utc_minute": reset_m,
            "daily_loss_basis": daily_loss_basis,
            "daily_loss_rate": daily_rate,
            "guardian_sweep_rate": sweep_rate,
        }
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class ChallengeRunResult:
    """Separated terminal challenge metrics and continuous empirical MC path."""

    verdict: Literal["PASS", "FAIL_STATIC", "FAIL_DAILY", "TIMEOUT"]
    terminal_step: int | None
    terminal_ts_ms: int | None
    terminal_net_usd: float
    min_equity_usd: float
    static_floor_dd_usd: float
    peak_to_trough_dd_usd: float
    worst_daily_usd: float
    sweeps: int
    forced_gap_exits: int
    unpriced_open_steps: int
    deferred_notional_usd: float
    full_window_net_usd: float
    full_window_peak_to_trough_dd_usd: float
    full_window_static_floor_dd_usd: float
    full_window_sweeps: int
    step_pnl: np.ndarray
    equity_curve: np.ndarray
    peak_gross_leverage: float = 0.0
    full_window_peak_gross_leverage: float = 0.0
    total_funding_usd: float = 0.0
    full_window_funding_usd: float = 0.0
    executed_deltas: np.ndarray | None = None


def roster_digest(records: Sequence[WalletRecord]) -> str:
    """Compute a deterministic SHA-256 hash of roster composition without raw addresses."""
    canonical = ";".join(
        f"{r.wallet_id}:{r.label}:{int(r.configured)}:{int(r.excluded)}"
        for r in sorted(records, key=lambda rec: rec.wallet_id)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def is_hl_perp_coin(coin: str) -> bool:
    """Return True for main-DEX perpetual symbols (exclude spot ``@/#`` and HIP-3 ``dex:``)."""
    return bool(coin) and not coin.startswith(("@", "#")) and "/" not in coin and ":" not in coin


def _parse_utc_us(value: str | datetime) -> int:
    """Parse an ISO-8601 UTC timestamp or tz-aware datetime into integer UTC microseconds."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("UTC_OFFSET_REQUIRED")
        dt = value.astimezone(UTC)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip())
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("UTC_OFFSET_REQUIRED")
        dt = parsed.astimezone(UTC)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value).__name__}")
    return round(dt.timestamp() * 1_000_000.0)


def _table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _compute_inventory_digest(con: sqlite3.Connection) -> str:
    """Compute canonical SHA-256 over authoritative registry tables present in SQLite."""
    fingerprint = hashlib.sha256()
    for table in pm_registry.AUTHORITATIVE_INVENTORY_TABLES:
        if not _table_exists(con, table):
            continue
        rows = sorted(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for row in con.execute(f"SELECT * FROM {table}").fetchall()
        )
        fingerprint.update(table.encode("ascii") + b"\0")
        for row in rows:
            fingerprint.update(row.encode("utf-8") + b"\n")
    return fingerprint.hexdigest()


def load_universe_asof(
    registry_db: Path | str,
    asof_utc: str | datetime,
    *,
    mode: Literal["PIT_STRICT_ASOF", "RETROSPECTIVE_ASOF_MASKED"] = "PIT_STRICT_ASOF",
    pre_asof_active_addresses: Collection[str] | None = None,
    include_aliases: Sequence[str] = (),
    strict_schema: bool = True,
) -> AsOfUniverseResult:
    """Load candidate universe as of ``asof_utc`` under strict PIT or labeled retrospective mode."""
    if mode not in ("PIT_STRICT_ASOF", "RETROSPECTIVE_ASOF_MASKED"):
        raise ValueError(f"INVALID_UNIVERSE_MODE:{mode}")
    asof_us = _parse_utc_us(asof_utc)
    asof_dt = datetime.fromtimestamp(asof_us / 1_000_000.0, tz=UTC)
    asof_iso = asof_dt.isoformat()

    db_path = Path(registry_db).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        con.execute("BEGIN")
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"REGISTRY_INTEGRITY_FAILED:{integrity}")
        if con.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("REGISTRY_FOREIGN_KEYS_FAILED")
        if strict_schema:
            if not _table_exists(con, "schema_migrations"):
                raise ValueError("REGISTRY_SCHEMA_VERSION_MISMATCH:missing_schema_migrations")
            max_ver = con.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            if max_ver != 8:
                raise ValueError(f"REGISTRY_SCHEMA_VERSION_MISMATCH:{max_ver}")

        inventory_sha = _compute_inventory_digest(con)
        wallet_rows = con.execute(
            "SELECT wallet_id, lower(address) FROM wallets ORDER BY wallet_id"
        ).fetchall()

        cohort_times: dict[int, list[int]] = {}
        if _table_exists(con, "cohort_memberships"):
            for wid, known_at in con.execute(
                "SELECT wallet_id, known_at FROM cohort_memberships"
            ).fetchall():
                cohort_times.setdefault(int(wid), []).append(_parse_utc_us(known_at))

        monitoring_info: dict[int, tuple[bool, int | None]] = {}
        if _table_exists(con, "monitoring"):
            cols = {r[1] for r in con.execute("PRAGMA table_info(monitoring)").fetchall()}
            if "known_at" in cols:
                for wid, cfg, known_at in con.execute(
                    "SELECT wallet_id, configured, known_at FROM monitoring"
                ).fetchall():
                    k_us = _parse_utc_us(known_at) if known_at is not None else None
                    monitoring_info[int(wid)] = (bool(cfg), k_us)
            else:
                for wid, cfg in con.execute("SELECT wallet_id, configured FROM monitoring").fetchall():
                    monitoring_info[int(wid)] = (bool(cfg), None)

        excl_times: dict[int, list[int]] = {}
        if _table_exists(con, "exclusions"):
            cols = {r[1] for r in con.execute("PRAGMA table_info(exclusions)").fetchall()}
            if "known_at" in cols:
                for wid, known_at in con.execute(
                    "SELECT wallet_id, known_at FROM exclusions"
                ).fetchall():
                    excl_times.setdefault(int(wid), []).append(_parse_utc_us(known_at))
            else:
                for (wid,) in con.execute("SELECT wallet_id FROM exclusions").fetchall():
                    excl_times.setdefault(int(wid), []).append(asof_us)

        aliases_by_wid: dict[int, list[tuple[int, int, str]]] = {}
        if _table_exists(con, "wallet_aliases"):
            for wid, alias, known_at, rid in con.execute(
                "SELECT wallet_id, alias, known_at, rowid FROM wallet_aliases"
            ).fetchall():
                aliases_by_wid.setdefault(int(wid), []).append(
                    (_parse_utc_us(known_at), int(rid), str(alias))
                )
    finally:
        con.close()

    wanted_aliases = set(include_aliases)
    active_addr_set = (
        {a.lower() for a in pre_asof_active_addresses}
        if pre_asof_active_addresses is not None
        else None
    )

    records: list[WalletRecord] = []
    for wid_raw, addr_raw in wallet_rows:
        wid = int(wid_raw)
        address = str(addr_raw).lower()
        is_excl = any(t <= asof_us for t in excl_times.get(wid, ()))
        if is_excl:
            continue

        alias_rows = aliases_by_wid.get(wid, ())
        asof_aliases = [row for row in alias_rows if row[0] <= asof_us]
        if asof_aliases:
            alias: str | None = max(asof_aliases, key=lambda r: (r[0], r[1]))[2]
        elif mode == "RETROSPECTIVE_ASOF_MASKED" and alias_rows:
            alias = max(alias_rows, key=lambda r: (r[0], r[1]))[2]
        else:
            alias = None

        mon_cfg, mon_k_us = monitoring_info.get(wid, (False, None))
        c_times = cohort_times.get(wid, ())

        if mode == "PIT_STRICT_ASOF":
            in_cohort = any(t <= asof_us for t in c_times)
            in_mon = mon_cfg and (mon_k_us is not None and mon_k_us <= asof_us)
            keep = in_cohort or in_mon
        else:
            has_membership = bool(c_times) or mon_cfg or (alias is not None and alias in wanted_aliases)
            has_pre_activity = active_addr_set is None or (address in active_addr_set)
            keep = has_membership and has_pre_activity

        if keep:
            records.append(
                WalletRecord(
                    wallet_id=wid,
                    address=address,
                    alias=alias,
                    configured=bool(mon_cfg or c_times),
                    excluded=False,
                )
            )

    if mode == "PIT_STRICT_ASOF" and not records:
        raise ValueError(f"EMPTY_PIT_UNIVERSE_BEFORE_RECEIPT_EPOCH:{asof_iso}")

    retro_wire = RetrospectiveEvidence().as_wire() if mode == "RETROSPECTIVE_ASOF_MASKED" else None
    return AsOfUniverseResult(
        records=records,
        roster_sha256=roster_digest(records),
        registry_inventory_sha256=inventory_sha,
        asof_utc=asof_iso,
        mode=mode,
        retrospective_evidence=retro_wire,
    )


def load_registry_universe(
    registry_db: Path | str,
    *,
    include_excluded: bool = False,
    include_aliases: Sequence[str] = (),
) -> tuple[list[WalletRecord], str]:
    """Query all configured wallets from native ``wallet_registry.sqlite`` read-only."""
    db_path = Path(registry_db).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        integrity = con.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"REGISTRY_INTEGRITY_FAILED:{integrity}")

        rows = con.execute(
            """
            SELECT w.wallet_id,
                   lower(w.address) AS address,
                   COALESCE(m.configured, 0) AS configured,
                   EXISTS(SELECT 1 FROM exclusions e WHERE e.wallet_id = w.wallet_id) AS excluded,
                   (
                       SELECT wa.alias
                       FROM wallet_aliases wa
                       WHERE wa.wallet_id = w.wallet_id
                       ORDER BY wa.known_at DESC, rowid DESC
                       LIMIT 1
                   ) AS alias
            FROM wallets w
            LEFT JOIN monitoring m USING (wallet_id)
            ORDER BY w.wallet_id
            """
        ).fetchall()
    finally:
        con.close()

    wanted_aliases = set(include_aliases)
    records: list[WalletRecord] = []
    for wallet_id, address, configured, excluded, alias in rows:
        is_cfg = bool(configured)
        is_excl = bool(excluded)
        if is_cfg and (include_excluded or not is_excl):
            keep = True
        elif alias is not None and alias in wanted_aliases:
            keep = include_excluded or not is_excl
        else:
            keep = False
        if keep:
            records.append(
                WalletRecord(
                    wallet_id=int(wallet_id),
                    address=str(address),
                    alias=str(alias) if alias is not None else None,
                    configured=is_cfg,
                    excluded=is_excl,
                )
            )
    return records, roster_digest(records)


def assess_bybit_symbol_screen(
    label: str,
    fills_by_coin: dict[str, pl.DataFrame],
    bybit_usdt_perps: set[str],
    *,
    max_unmapped_fill_frac: float = 0.05,
    max_unmapped_notional_frac: float = 0.05,
) -> SymbolScreenResult:
    """Screen a wallet's Hyperliquid perps against Bybit linear USDT perps by materiality."""
    total_fills = 0
    mapped_fills = 0
    total_notional = 0.0
    unmapped_notional = 0.0
    has_notional = True

    mapped_coins: list[str] = []
    unmapped_coins: list[str] = []
    scale_candidates: dict[str, str] = {}

    for coin in sorted(fills_by_coin):
        if not is_hl_perp_coin(coin):
            continue
        df = fills_by_coin[coin]
        cnt = int(df.height)
        if cnt == 0:
            continue
        total_fills += cnt
        coin_notional = 0.0
        if "px" in df.columns and "sz" in df.columns:
            coin_notional = float((df["px"].abs() * df["sz"].abs()).sum())
            total_notional += coin_notional
        elif "notional" in df.columns:
            coin_notional = float(df["notional"].abs().sum())
            total_notional += coin_notional
        else:
            has_notional = False

        if coin in bybit_usdt_perps:
            mapped_fills += cnt
            mapped_coins.append(coin)
        else:
            unmapped_coins.append(coin)
            unmapped_notional += coin_notional
            if coin.startswith("k"):
                candidate = f"1000{coin[1:]}"
                if candidate in bybit_usdt_perps:
                    scale_candidates[coin] = candidate

    if total_fills == 0:
        return SymbolScreenResult(
            label=label,
            total_fills=0,
            mapped_fills=0,
            unmapped_coins=(),
            mapped_coins=(),
            unmapped_fill_frac=0.0,
            unmapped_notional_frac=None,
            scale_alias_candidates={},
            status="REJECT_NO_FILLS",
        )

    fill_frac = (total_fills - mapped_fills) / total_fills
    notional_frac = (unmapped_notional / total_notional) if has_notional and total_notional > 0 else None

    if not unmapped_coins:
        status: Literal[
            "EXACT_MAPPED",
            "MAPPED_WITH_EXCLUDED_TAIL",
            "REJECT_UNMAPPED_MATERIAL",
            "REJECT_NO_FILLS",
        ] = "EXACT_MAPPED"
    elif fill_frac <= max_unmapped_fill_frac and (
        notional_frac is None or notional_frac <= max_unmapped_notional_frac
    ):
        status = "MAPPED_WITH_EXCLUDED_TAIL"
    else:
        status = "REJECT_UNMAPPED_MATERIAL"

    return SymbolScreenResult(
        label=label,
        total_fills=total_fills,
        mapped_fills=mapped_fills,
        unmapped_coins=tuple(unmapped_coins),
        mapped_coins=tuple(mapped_coins),
        unmapped_fill_frac=fill_frac,
        unmapped_notional_frac=notional_frac,
        scale_alias_candidates=scale_candidates,
        status=status,
    )


def reconstruct_positions_on_grid(
    fills: pl.DataFrame,
    grid_ms: np.ndarray,
    *,
    policy: PolicyType = "new",
    activation_ms: int | None = None,
    initial_anchor_pos: float | None = None,
    eps: float = 1e-9,
) -> ReconstructedGrid:
    """Reconstruct causal 5m positions without backward lookahead from future ``pre[0]``.

    Expects ``fills`` sorted by ``ts`` with columns ``ts`` (epoch ms), ``pre`` (start position
    prior to fill), ``sign`` (+1/-1), and ``sz`` (positive fill size).
    """
    n_grid = int(grid_ms.size)
    if fills.is_empty():
        base = float(initial_anchor_pos) if (policy == "full" and initial_anchor_pos) else 0.0
        return ReconstructedGrid(
            timestamps_ms=grid_ms,
            pos=np.full(n_grid, base, dtype=np.float64),
            unanchored_pre_inventory=False,
            first_fill_ms=None,
            first_observed_pre_pos=0.0,
        )

    times = fills["ts"].to_numpy().astype(np.int64)
    pre = fills["pre"].to_numpy().astype(np.float64)
    signs = fills["sign"].to_numpy().astype(np.float64)
    sz = fills["sz"].to_numpy().astype(np.float64)
    post = pre + signs * sz

    first_fill_ms = int(times[0])
    first_pre = float(pre[0])
    unanchored = initial_anchor_pos is None and abs(first_pre) > eps

    # Strict causal lookup: fill at timestamp t is only reflected at checkpoints > t.
    ix = np.searchsorted(times, grid_ms, side="left") - 1

    if policy == "full":
        if not unanchored:
            pre_first_fill = float(initial_anchor_pos) if initial_anchor_pos is not None else 0.0
            pos = np.where(ix >= 0, post[np.clip(ix, 0, len(times) - 1)], pre_first_fill)
        else:
            eff_post = np.zeros(len(times), dtype=np.float64)
            synced = False
            prev = first_pre
            for i in range(len(times)):
                after = float(post[i])
                if not synced:
                    is_same_side_inc = (prev * after > 1e-12) and (abs(after) > abs(prev) + eps)
                    is_fresh_open = (abs(prev) <= eps) and (abs(after) > eps)
                    is_flip = (prev * after) < -1e-12
                    if is_same_side_inc or is_fresh_open or is_flip:
                        synced = True
                eff_post[i] = after if synced else 0.0
                prev = after
            pos = np.where(ix >= 0, eff_post[np.clip(ix, 0, len(times) - 1)], 0.0)
        return ReconstructedGrid(
            timestamps_ms=grid_ms,
            pos=pos.astype(np.float64),
            unanchored_pre_inventory=unanchored,
            first_fill_ms=first_fill_ms,
            first_observed_pre_pos=first_pre,
        )

    act_ms = int(activation_ms) if activation_ms is not None else int(grid_ms[0])
    n_fills = len(times)
    eligible = np.zeros(n_fills, dtype=bool)
    active = False
    prev = first_pre

    for i in range(n_fills):
        after = float(post[i])
        if times[i] < act_ms:
            prev = after
            continue
        if (abs(prev) < eps and abs(after) >= eps) or (prev * after < -1e-12):
            active = True
        if abs(after) < eps:
            active = False
        eligible[i] = active
        prev = after

    valid_ix = ix >= 0
    clipped = np.clip(ix, 0, n_fills - 1)
    pos = np.where(valid_ix & eligible[clipped], post[clipped], 0.0)
    return ReconstructedGrid(
        timestamps_ms=grid_ms,
        pos=pos.astype(np.float64),
        unanchored_pre_inventory=unanchored,
        first_fill_ms=first_fill_ms,
        first_observed_pre_pos=first_pre,
    )


def _apply_pair_outage_gap_exit(
    cur_pos: float,
    want: float,
    prev_mark: float,
    haircut_rate: float,
    fee_rate: float,
) -> tuple[float, float, float, float, int, int, float | None]:
    if abs(cur_pos) <= 1e-12:
        return cur_pos, 0.0, 0.0, 0.0, 0, 0, None
    snap_val = want if abs(want) > 1e-9 else cur_pos
    if np.isfinite(prev_mark) and prev_mark > 0:
        direction = 1.0 if cur_pos > 0 else -1.0
        exit_px = prev_mark * (1.0 - direction * haircut_rate)
        bar_gross = cur_pos * (exit_px - prev_mark)
        bar_cost = abs(cur_pos) * exit_px * fee_rate
        return 0.0, -cur_pos, bar_gross, bar_cost, 1, 1, snap_val
    return cur_pos, 0.0, 0.0, 0.0, 0, 1, None


def replay_coin_targets_vectorized(
    targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    gap_exit_haircut_bps: float = GAP_EXIT_HAIRCUT_BPS,
    fee_bps: float = DEFAULT_FEE_BPS,
    initial_pos: float = 0.0,
    mode: Literal["legacy_target_chase", "cwsa_one_shot_v3"] = "legacy_target_chase",
    vol_1m_usd: np.ndarray | None = None,
    cwsa_cfg: CWSAReplayConfig | None = None,
    liquidity_cfg: CausalLiquidityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Simulate causal per-bar fills, fees, and pair-local stale-mark exits on a 1-coin target."""
    if mode == "cwsa_one_shot_v3":
        n = int(targets.size)
        grid_ms = np.arange(n, dtype=np.int64) * 300_000
        eff_cwsa = (
            cwsa_cfg
            if cwsa_cfg is not None
            else CWSAReplayConfig(
                fee_bps=fee_bps,
                slippage_bps=max(0.0, cost_bps - fee_bps),
            )
        )
        eff_liq = liquidity_cfg if liquidity_cfg is not None else CausalLiquidityConfig()
        vol_2d = (
            np.asarray(vol_1m_usd, dtype=np.float64).reshape(1, n)
            if vol_1m_usd is not None
            else None
        )
        res = simulate_cwsa_one_shot_replay(
            coins=("COIN_0",),
            source_positions=np.asarray(targets, dtype=np.float64).reshape(1, n),
            trade_open=np.asarray(trade_open, dtype=np.float64).reshape(1, n),
            mark_close=np.asarray(mark_close, dtype=np.float64).reshape(1, n),
            grid_ms=grid_ms,
            k_scale=1.0,
            vol_1m_usd=vol_2d,
            initial_positions=np.array([initial_pos], dtype=np.float64),
            cwsa_cfg=eff_cwsa,
            liquidity_cfg=eff_liq,
        )
        quarantined_bars = int(np.sum(res.dispositions[0] == OrderDisposition.QUARANTINED.value))
        return (
            res.executed_positions[0],
            res.executed_deltas[0],
            res.gross_pnl[0],
            res.costs[0],
            {
                "forced_gap_exits": 0.0,
                "unpriced_open_steps": float(quarantined_bars),
                "deferred_notional_usd": float(res.participation_audit.total_missed_notional_usd),
            },
        )
    n = int(targets.size)
    cost_rate = cost_bps * 1e-4
    haircut_rate = gap_exit_haircut_bps * 1e-4
    fee_rate = fee_bps * 1e-4

    pos = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)
    gross = np.zeros(n, dtype=np.float64)
    cost = np.zeros(n, dtype=np.float64)

    cur_pos = float(initial_pos)
    prev_mark = np.nan
    outage_fenced_snap: float | None = None
    forced_exits = 0
    unpriced_open_steps = 0
    deferred_notional_usd = 0.0

    for j in range(n):
        m_close = float(mark_close[j])
        t_open = float(trade_open[j])
        want = float(targets[j])

        if not np.isfinite(m_close):
            cur_pos, d_gap, g_gap, c_gap, f_inc, u_inc, snap_new = _apply_pair_outage_gap_exit(
                cur_pos, want, prev_mark, haircut_rate, fee_rate
            )
            delta[j] = d_gap
            gross[j] = g_gap
            cost[j] = c_gap
            forced_exits += f_inc
            unpriced_open_steps += u_inc
            if snap_new is not None:
                outage_fenced_snap = snap_new
            pos[j] = cur_pos
            continue

        if outage_fenced_snap is not None:
            if abs(want) <= 1e-9 or (want * outage_fenced_snap < -1e-12):
                outage_fenced_snap = None
            else:
                want = 0.0

        desired = want
        if abs(want - cur_pos) > 1e-9 and not np.isfinite(t_open):
            desired = cur_pos
            deferred_notional_usd += abs(want - cur_pos) * m_close

        exec_px = t_open if np.isfinite(t_open) else m_close
        d_qty = desired - cur_pos
        bar_gross = 0.0
        if np.isfinite(prev_mark):
            bar_gross += cur_pos * (exec_px - prev_mark)
        bar_gross += desired * (m_close - exec_px)
        bar_cost = abs(d_qty) * exec_px * cost_rate

        pos[j] = desired
        delta[j] = d_qty
        gross[j] = bar_gross
        cost[j] = bar_cost
        cur_pos = desired
        prev_mark = m_close

    return (
        pos,
        delta,
        gross,
        cost,
        {
            "forced_gap_exits": float(forced_exits),
            "unpriced_open_steps": float(unpriced_open_steps),
            "deferred_notional_usd": float(deferred_notional_usd),
        },
    )


def build_unit_replay_by_alias(
    grouped_fills: dict[str, dict[str, pl.DataFrame]],
    prices_by_coin: dict[str, tuple[np.ndarray, np.ndarray]],
    aliases: Sequence[str],
    grid_ms: np.ndarray,
    *,
    policy: PolicyType = "new",
    activation_ms: int | None = None,
    initial_anchors: dict[str, dict[str, float]] | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[dict[str, dict[str, pl.DataFrame]], dict[str, dict[str, int]]]:
    """Build Policy-A 5m DataFrames ready for ``pod_constructor.build_tensor_from_unit_replays``.

    Each DataFrame carries ``["timestamp", "open", "close", "pos", "delta", "gross", "cost", "net"]``
    aligned identically across ``grid_ms``.
    """
    ts_col = pl.from_epoch(pl.Series("timestamp", grid_ms, dtype=pl.Int64), time_unit="ms")
    ts_utc = ts_col.dt.replace_time_zone("UTC")
    unit_frames: dict[str, dict[str, pl.DataFrame]] = {}
    diagnostics: dict[str, dict[str, int]] = {}

    for alias in aliases:
        by_coin = grouped_fills.get(alias, {})
        wallet_frames: dict[str, pl.DataFrame] = {}
        diag = {"unanchored_coins": 0, "forced_gap_exits": 0, "unpriced_open_steps": 0}
        anchors = (initial_anchors or {}).get(alias, {})

        for coin, coin_fills in by_coin.items():
            if coin not in prices_by_coin:
                continue
            trade_open, mark_close = prices_by_coin[coin]
            recon = reconstruct_positions_on_grid(
                coin_fills,
                grid_ms,
                policy=policy,
                activation_ms=activation_ms,
                initial_anchor_pos=anchors.get(coin),
            )
            if recon.unanchored_pre_inventory:
                diag["unanchored_coins"] += 1

            anchor_val = anchors.get(coin) if policy == "full" else None
            pos, delta, gross, cost, stats = replay_coin_targets_vectorized(
                recon.pos,
                trade_open,
                mark_close,
                cost_bps=cost_bps,
                initial_pos=float(anchor_val) if anchor_val is not None else 0.0,
            )
            diag["forced_gap_exits"] += int(stats["forced_gap_exits"])
            diag["unpriced_open_steps"] += int(stats["unpriced_open_steps"])

            # pod_constructor treats open=close fallback marks seamlessly for notional weighting.
            eff_open = np.where(np.isfinite(trade_open), trade_open, mark_close)
            eff_open = np.where(np.isfinite(eff_open), eff_open, 0.0)
            eff_close = np.where(np.isfinite(mark_close), mark_close, eff_open)
            net = gross - cost

            wallet_frames[coin] = pl.DataFrame(
                {
                    "timestamp": ts_utc,
                    "open": eff_open,
                    "close": eff_close,
                    "pos": pos,
                    "delta": delta,
                    "gross": gross,
                    "cost": cost,
                    "net": net,
                }
            )
        unit_frames[alias] = wallet_frames
        diagnostics[alias] = diag
    return unit_frames, diagnostics


def _reset_day_index(ts_ms: int, reset_hour: int, reset_minute: int) -> int:
    """Return reset-shifted UTC epoch day index so sparse grids never miss a reset."""
    offset_ms = ((reset_hour * 3600) + (reset_minute * 60)) * 1000
    return (int(ts_ms) - offset_ms) // 86_400_000


def _is_reset_bar(ts_ms: int, reset_hour: int, reset_minute: int) -> bool:
    sec_in_day = (ts_ms // 1000) % 86_400
    target_sec = (reset_hour * 3600) + (reset_minute * 60)
    return target_sec <= sec_in_day < target_sec + 300


def _update_reentry_eligibility(
    desired: np.ndarray,
    sweep_snapshot: np.ndarray,
    coin_eligible: np.ndarray,
    *,
    reenter_mode: Literal["new_only", "immediate_target"],
    eps: float = 1e-9,
) -> np.ndarray:
    """Enforce strict new-position admission after a Guardian daily sweep."""
    if reenter_mode == "immediate_target":
        return desired.copy()

    out = np.zeros_like(desired)
    for c in range(desired.size):
        target = float(desired[c])
        if coin_eligible[c]:
            out[c] = target
            continue
        snap = float(sweep_snapshot[c])
        # Only re-admit after going flat (< eps) or cleanly flipping sign past zero.
        if abs(target) < eps or abs(snap) < eps or (snap * target < -1e-12):
            coin_eligible[c] = True
            out[c] = target
            sweep_snapshot[c] = 0.0
        else:
            out[c] = 0.0
    return out


def _split_exit_first_phase1(cur: float, desired: float, *, eps: float = 1e-12) -> float:
    """Return the exit/reduction-only intermediate position allowed before sweep checks."""
    if abs(desired) <= eps or abs(cur) <= eps or (cur * desired < -eps):
        return 0.0
    if abs(desired) < abs(cur):
        return desired
    return cur


def _conservative_gap_exit_price(
    cur: float,
    prev_m: float,
    m_c: float,
    haircut_rate: float,
) -> tuple[float, float]:
    """Return ``(ref_mark, exit_px)`` for a conservative forced-flatten when open trade price is missing."""
    direction = 1.0 if cur > 0 else -1.0
    if np.isfinite(prev_m) and prev_m > 0:
        ref_px = min(prev_m, m_c) if cur > 0 else max(prev_m, m_c)
        return prev_m, ref_px * (1.0 - direction * haircut_rate)
    return m_c, m_c * (1.0 - direction * haircut_rate)


def _conservative_gap_exit_pnl(
    cur: float,
    prev_m: float,
    m_c: float,
    haircut_rate: float,
    fee_rate: float,
) -> float:
    """Compute conservative forced-flatten PnL when ``trade_open`` is unavailable."""
    base_mark, exit_px = _conservative_gap_exit_price(cur, prev_m, m_c, haircut_rate)
    return cur * (exit_px - base_mark) - abs(cur) * exit_px * fee_rate


def _apply_fill_to_basis(
    cur_pos: float,
    new_pos: float,
    cur_entry: float,
    px: float,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Update weighted-average entry price and return closed realized PnL."""
    delta = new_pos - cur_pos
    if abs(delta) <= eps:
        return 0.0, (0.0 if abs(new_pos) <= eps else cur_entry)
    realized = 0.0
    if abs(cur_pos) > eps and cur_pos * delta < 0.0:
        close_qty = min(abs(cur_pos), abs(delta))
        direction = 1.0 if cur_pos > 0.0 else -1.0
        realized += close_qty * (px - cur_entry) * direction
    if abs(new_pos) <= eps:
        return realized, 0.0
    if abs(cur_pos) <= eps or cur_pos * new_pos < 0.0:
        return realized, px
    if cur_pos * delta > 0.0:
        return realized, (abs(cur_pos) * cur_entry + abs(delta) * px) / abs(new_pos)
    return realized, cur_entry


def _effective_allowance(cashday: float, fixed_limit: float, rate: float | None) -> float:
    """Compute dollar daily/sweep allowance from fixed amount or reset-basis percentage."""
    if rate is not None:
        return max(0.0, cashday * float(rate))
    return float(fixed_limit)


def _quantize_order_target(
    cur: float,
    want: float,
    exec_px: float,
    step: float,
    min_amt: float,
    min_notional: float,
    *,
    eps: float = 1e-12,
) -> float:
    """Apply Bybit lot-step rounding, min-amount, and notional deadband (bypassing full closes)."""
    if abs(want) <= eps:
        return 0.0
    raw_delta = want - cur
    if abs(raw_delta) <= eps:
        return cur
    if step > 0.0:
        sign = -1.0 if raw_delta < 0.0 else 1.0
        quotient = abs(raw_delta) / step
        units = math.floor(quotient + max(1e-12, quotient * 1e-12))
        qty_step = round(sign * units * step, 12)
    else:
        qty_step = raw_delta
    if min_amt > 0.0 and abs(qty_step) + 1e-12 < min_amt:
        return cur
    if min_notional > 0.0 and abs(qty_step) * exec_px + 1e-9 < min_notional:
        return cur
    desired = round(cur + qty_step, 12) if step > 0.0 else (cur + qty_step)
    return 0.0 if abs(desired) <= eps else desired


def _calc_gross_leverage(gross_notional: float, equity: float) -> float:
    if gross_notional <= 1e-12:
        return 0.0
    if equity <= 1e-9:
        return float("inf")
    return gross_notional / equity


def _resolve_funding_matrix(
    funding_rates: np.ndarray | None,
    timestamps_ms: np.ndarray,
    nc: int,
    n: int,
    spec: GuardianSpec,
) -> np.ndarray | None:
    """Resolve optional per-bar signed funding rate matrix ``(nc, n)`` scaled by multiplier."""
    if funding_rates is not None:
        fr = np.asarray(funding_rates, dtype=np.float64)
        if fr.ndim == 1 and fr.shape[0] == n:
            return np.broadcast_to(fr[None, :], (nc, n)) * spec.funding_multiplier
        if fr.ndim == 2 and fr.shape in ((1, n), (nc, n)):
            return np.broadcast_to(fr, (nc, n)) * spec.funding_multiplier
        raise ValueError(f"INVALID_FUNDING_RATES_SHAPE:{fr.shape}")
    if spec.funding_bps_per_8h == 0.0 or spec.funding_multiplier == 0.0:
        return None
    default_dt = int(timestamps_ms[1] - timestamps_ms[0]) if n > 1 else 300_000
    dt_ms = np.empty(n, dtype=np.float64)
    dt_ms[0] = float(default_dt)
    if n > 1:
        dt_ms[1:] = np.diff(timestamps_ms).astype(np.float64)
    per_bar = spec.funding_bps_per_8h * 1e-4 * (dt_ms / 28_800_000.0) * spec.funding_multiplier
    return np.broadcast_to(per_bar[None, :], (nc, n))


def _cap_phase2_leverage(
    pos: np.ndarray,
    prev_mark: np.ndarray,
    exec_px_arr: np.ndarray,
    p1_arr: np.ndarray,
    desired_arr: np.ndarray,
    priced: np.ndarray,
    step_arr: np.ndarray,
    min_amt_arr: np.ndarray,
    eq: float,
    outage_pnl: float,
    cost_rate: float,
    spec: GuardianSpec,
) -> None:
    """Scale Phase-2 entries/increases proportionally so open leverage stays within ``max_gross_leverage``."""
    if spec.max_gross_leverage is None:
        return
    nc = desired_arr.size
    eq_open_p1 = eq + outage_pnl
    used_notional = 0.0
    add_notional = 0.0
    for ci in range(nc):
        if not priced[ci]:
            continue
        cur = float(pos[ci])
        prev_m = float(prev_mark[ci])
        exec_px = exec_px_arr[ci]
        p1_c = p1_arr[ci]
        if np.isfinite(prev_m):
            eq_open_p1 += cur * (exec_px - prev_m)
        eq_open_p1 -= abs(p1_c - cur) * exec_px * cost_rate
        used_notional += abs(p1_c) * exec_px
        add_notional += abs(desired_arr[ci] - p1_c) * exec_px
    if add_notional <= 1e-12:
        return
    cap = max(0.0, float(spec.max_gross_leverage))
    max_allowed = cap * max(0.0, eq_open_p1)
    if used_notional + add_notional * (1.0 + cap * cost_rate) <= max_allowed + 1e-12:
        return
    avail = max(0.0, max_allowed - used_notional)
    scale = min(1.0, avail / (add_notional * (1.0 + cap * cost_rate)))
    for ci in range(nc):
        if not priced[ci]:
            continue
        p1_c = p1_arr[ci]
        delta2 = desired_arr[ci] - p1_c
        if abs(delta2) <= 1e-12:
            continue
        if scale <= 1e-12:
            desired_arr[ci] = p1_c
        else:
            desired_arr[ci] = _quantize_order_target(
                p1_c,
                p1_c + delta2 * scale,
                exec_px_arr[ci],
                float(step_arr[ci]),
                float(min_amt_arr[ci]),
                spec.min_trade_notional_usd,
            )


def _record_sweep_halt(
    pos: np.ndarray,
    sweep_snapshot: np.ndarray,
    raw_want: np.ndarray,
    coin_eligible: np.ndarray,
    swept_flags: np.ndarray,
    step_idx: int,
    reenter_mode: str,
    sweeps: int,
) -> tuple[int, bool]:
    """Flatten residual positions, snapshot targets, and lock re-entry after a sweep breach."""
    pos[:] = 0.0
    sweep_snapshot[:] = raw_want
    if reenter_mode == "new_only":
        coin_eligible[:] = False
    swept_flags[step_idx] = True
    return sweeps + 1, True


def _execute_guardian_pass(
    coins: Sequence[str],
    net_targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    timestamps_ms: np.ndarray,
    spec: GuardianSpec,
    *,
    stop_on_terminal: bool,
    funding_rates: np.ndarray | None = None,
) -> dict:
    """Shared causal Guardian execution kernel used for both terminal and MC views."""
    nc, n = net_targets.shape
    cost_rate = spec.cost_bps * spec.cost_multiplier * 1e-4
    haircut_rate = GAP_EXIT_HAIRCUT_BPS * 1e-4
    fee_rate = DEFAULT_FEE_BPS * spec.cost_multiplier * 1e-4
    funding_mat = _resolve_funding_matrix(funding_rates, timestamps_ms, nc, n, spec)
    step_arr = np.array(
        [
            float(spec.amount_step_by_coin.get(c, spec.amount_step))
            if spec.amount_step_by_coin is not None
            else float(spec.amount_step)
            for c in coins
        ],
        dtype=np.float64,
    )
    min_amt_arr = np.array(
        [
            float(spec.min_amount_by_coin.get(c, spec.min_amount))
            if spec.min_amount_by_coin is not None
            else float(spec.min_amount)
            for c in coins
        ],
        dtype=np.float64,
    )

    pos = np.zeros(nc, dtype=np.float64)
    entry_px = np.zeros(nc, dtype=np.float64)
    prev_mark = np.full(nc, np.nan, dtype=np.float64)
    sweep_snapshot = np.zeros(nc, dtype=np.float64)
    coin_eligible = np.ones(nc, dtype=bool)

    eq = spec.initial_equity
    balance = spec.initial_equity
    cashday = spec.initial_equity
    eff_daily_limit = _effective_allowance(cashday, spec.daily_loss_limit, spec.daily_loss_rate)
    eff_sweep_loss = _effective_allowance(cashday, spec.guardian_sweep_loss, spec.guardian_sweep_rate)
    peak = spec.initial_equity
    min_eq = spec.initial_equity
    peak_to_trough_dd = 0.0
    worst_daily = 0.0
    sweeps = 0
    blocked_until_reset = False
    forced_gap_exits = 0
    unpriced_open_steps = 0
    deferred_notional_usd = 0.0
    peak_gross_leverage = 0.0
    total_funding_usd = 0.0

    verdict: Literal["PASS", "FAIL_STATIC", "FAIL_DAILY", "TIMEOUT"] = "TIMEOUT"
    terminal_step: int | None = None
    terminal_ts_ms: int | None = None
    terminal_net = 0.0

    step_pnl = np.zeros(n, dtype=np.float64)
    eq_curve = np.zeros(n, dtype=np.float64)
    bal_curve = np.zeros(n, dtype=np.float64)
    cashday_curve = np.zeros(n, dtype=np.float64)
    swept_flags = np.zeros(n, dtype=bool)
    executed_deltas = np.zeros((nc, n), dtype=np.float64)
    cur_day_idx = (
        _reset_day_index(int(timestamps_ms[0]), spec.reset_utc_hour, spec.reset_utc_minute)
        if n > 0
        else 0
    )

    for j in range(n):
        prev_pos = pos.copy()
        ts_ms = int(timestamps_ms[j])
        new_day_idx = _reset_day_index(ts_ms, spec.reset_utc_hour, spec.reset_utc_minute)
        if j > 0 and new_day_idx > cur_day_idx:
            cur_day_idx = new_day_idx
            cashday = balance if spec.daily_loss_basis == "balance" else eq
            eff_daily_limit = _effective_allowance(
                cashday, spec.daily_loss_limit, spec.daily_loss_rate
            )
            eff_sweep_loss = _effective_allowance(
                cashday, spec.guardian_sweep_loss, spec.guardian_sweep_rate
            )
            blocked_until_reset = False
            worst_daily = min(worst_daily, eq - cashday)
            if terminal_step is None and (eq - cashday) <= -eff_daily_limit:
                verdict = "FAIL_DAILY"
                terminal_step = j
                terminal_ts_ms = ts_ms
                terminal_net = eq - spec.initial_equity
                if stop_on_terminal:
                    eq_curve[j:] = eq
                    bal_curve[j:] = balance
                    cashday_curve[j:] = cashday
                    break

        p = trade_open[:, j]
        m = mark_close[:, j]
        raw_want = net_targets[:, j]

        if not blocked_until_reset:
            admitted = _update_reentry_eligibility(
                raw_want,
                sweep_snapshot,
                coin_eligible,
                reenter_mode=spec.reenter_after_sweep,
            )
        else:
            admitted = np.zeros(nc, dtype=np.float64)

        priced = np.zeros(nc, dtype=bool)
        exec_px_arr = np.zeros(nc, dtype=np.float64)
        desired_arr = np.zeros(nc, dtype=np.float64)
        p1_arr = np.zeros(nc, dtype=np.float64)
        outage_pnl = 0.0
        open_mtm = 0.0

        for ci in range(nc):
            m_c = float(m[ci])
            t_c = float(p[ci])
            prev_m = float(prev_mark[ci])
            cur = float(pos[ci])

            if not np.isfinite(m_c):
                if abs(cur) > 1e-12:
                    unpriced_open_steps += 1
                    adm_c = float(admitted[ci])
                    snap_c = adm_c if abs(adm_c) > 1e-9 else (
                        float(raw_want[ci]) if abs(float(raw_want[ci])) > 1e-9 else cur
                    )
                    if np.isfinite(prev_m) and prev_m > 0:
                        direction = 1.0 if cur > 0 else -1.0
                        exit_px = prev_m * (1.0 - direction * haircut_rate)
                        outage_pnl += cur * (exit_px - prev_m) - abs(cur) * exit_px * fee_rate
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), exit_px)
                        balance += realized - abs(cur) * exit_px * fee_rate
                        entry_px[ci] = 0.0
                        pos[ci] = 0.0
                        forced_gap_exits += 1
                        if spec.reenter_after_sweep == "new_only":
                            coin_eligible[ci] = False
                            sweep_snapshot[ci] = snap_c
                continue

            priced[ci] = True
            exec_px = t_c if np.isfinite(t_c) else m_c
            exec_px_arr[ci] = exec_px
            if np.isfinite(t_c) and np.isfinite(prev_m) and abs(cur) > 1e-12:
                open_mtm += cur * (t_c - prev_m)

            want_c = float(admitted[ci])
            if blocked_until_reset:
                desired_c = 0.0
            elif abs(want_c - cur) > 1e-9 and not np.isfinite(t_c):
                desired_c = cur
                deferred_notional_usd += abs(want_c - cur) * m_c
            else:
                desired_c = _quantize_order_target(
                    cur,
                    want_c,
                    exec_px,
                    float(step_arr[ci]),
                    float(min_amt_arr[ci]),
                    spec.min_trade_notional_usd,
                )

            desired_arr[ci] = desired_c
            p1_arr[ci] = 0.0 if blocked_until_reset else _split_exit_first_phase1(cur, desired_c)

        bar_pnl = outage_pnl
        eq_open = eq + outage_pnl + open_mtm
        swept_at_open = (not blocked_until_reset) and ((eq_open - cashday) <= -eff_sweep_loss)
        if not swept_at_open:
            min_eq = min(min_eq, eq_open)
            peak_to_trough_dd = min(peak_to_trough_dd, eq_open - peak)
            worst_daily = min(worst_daily, eq_open - cashday)
        hit_static_open = (not swept_at_open) and (
            eq_open <= (spec.initial_equity - spec.max_static_dd)
        )
        hit_daily_open = (not swept_at_open) and ((eq_open - cashday) <= -eff_daily_limit)
        if swept_at_open:
            # Open-time sweep breach: flatten carried positions at open without admitting entries.
            gross_open = 0.0
            for ci in range(nc):
                if not priced[ci]:
                    continue
                cur = float(pos[ci])
                prev_m = float(prev_mark[ci])
                m_c = float(m[ci])
                t_c = float(p[ci])
                if abs(cur) > 1e-12:
                    gross_open += abs(cur) * (t_c if np.isfinite(t_c) else m_c)
                    if np.isfinite(t_c):
                        if np.isfinite(prev_m):
                            bar_pnl += cur * (t_c - prev_m)
                        bar_pnl -= abs(cur) * t_c * cost_rate
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), t_c)
                        balance += realized - abs(cur) * t_c * cost_rate
                    else:
                        _, exit_px = _conservative_gap_exit_price(cur, prev_m, m_c, haircut_rate)
                        bar_pnl += _conservative_gap_exit_pnl(
                            cur, prev_m, m_c, haircut_rate, fee_rate
                        )
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), exit_px)
                        balance += realized - abs(cur) * exit_px * fee_rate
                        unpriced_open_steps += 1
                        forced_gap_exits += 1
                entry_px[ci] = 0.0
                pos[ci] = 0.0
                prev_mark[ci] = m_c
            peak_gross_leverage = max(
                peak_gross_leverage, _calc_gross_leverage(gross_open, eq + outage_pnl + open_mtm)
            )
            eq += bar_pnl
            sweeps, blocked_until_reset = _record_sweep_halt(
                pos,
                sweep_snapshot,
                raw_want,
                coin_eligible,
                swept_flags,
                j,
                spec.reenter_after_sweep,
                sweeps,
            )
        else:
            p1_pnl = outage_pnl
            p1_funding = 0.0
            if not blocked_until_reset:
                for ci in range(nc):
                    if not priced[ci]:
                        continue
                    cur = float(pos[ci])
                    prev_m = float(prev_mark[ci])
                    exec_px = exec_px_arr[ci]
                    m_c = float(m[ci])
                    p1_c = p1_arr[ci]
                    if np.isfinite(prev_m):
                        p1_pnl += cur * (exec_px - prev_m)
                    p1_pnl += p1_c * (m_c - exec_px) - abs(p1_c - cur) * exec_px * cost_rate
                    if funding_mat is not None and abs(p1_c) > 1e-12:
                        f_c = -p1_c * m_c * float(funding_mat[ci, j])
                        p1_pnl += f_c
                        p1_funding += f_c

            if not blocked_until_reset and (eq + p1_pnl - cashday) <= -eff_sweep_loss:
                # Exit-first sweep breach: exits/reductions settled, new opens/increases blocked.
                flatten_cost = 0.0
                gross_p1 = 0.0
                for ci in range(nc):
                    if priced[ci]:
                        cur = float(pos[ci])
                        exec_px = exec_px_arr[ci]
                        m_c = float(m[ci])
                        t_c = float(p[ci])
                        prev_m = float(prev_mark[ci])
                        p1_c = p1_arr[ci]
                        realized1, new_entry1 = _apply_fill_to_basis(
                            cur, p1_c, float(entry_px[ci]), exec_px
                        )
                        balance += realized1 - abs(p1_c - cur) * exec_px * cost_rate
                        if funding_mat is not None and abs(p1_c) > 1e-12:
                            balance += -p1_c * m_c * float(funding_mat[ci, j])
                        if np.isfinite(t_c) or abs(p1_c) <= 1e-12:
                            realized2, _ = _apply_fill_to_basis(p1_c, 0.0, new_entry1, m_c)
                            balance += realized2 - abs(p1_c) * m_c * cost_rate
                            flatten_cost += abs(p1_c) * m_c * cost_rate
                        else:
                            _, exit_px = _conservative_gap_exit_price(
                                p1_c, prev_m, m_c, haircut_rate
                            )
                            realized2, _ = _apply_fill_to_basis(p1_c, 0.0, new_entry1, exit_px)
                            balance += realized2 - abs(p1_c) * exit_px * fee_rate
                            p1_pnl += p1_c * (exit_px - m_c)
                            flatten_cost += abs(p1_c) * exit_px * fee_rate
                            unpriced_open_steps += 1
                            forced_gap_exits += 1
                        entry_px[ci] = 0.0
                        gross_p1 += abs(p1_c) * m_c
                        pos[ci] = 0.0
                        prev_mark[ci] = m_c
                peak_gross_leverage = max(
                    peak_gross_leverage, _calc_gross_leverage(gross_p1, eq + p1_pnl)
                )
                bar_pnl = p1_pnl - flatten_cost
                total_funding_usd += p1_funding
                eq += bar_pnl
                sweeps, blocked_until_reset = _record_sweep_halt(
                    pos,
                    sweep_snapshot,
                    raw_want,
                    coin_eligible,
                    swept_flags,
                    j,
                    spec.reenter_after_sweep,
                    sweeps,
                )
            else:
                if not blocked_until_reset:
                    _cap_phase2_leverage(
                        pos,
                        prev_mark,
                        exec_px_arr,
                        p1_arr,
                        desired_arr,
                        priced,
                        step_arr,
                        min_amt_arr,
                        eq,
                        outage_pnl,
                        cost_rate,
                        spec,
                    )

                bar_pnl = outage_pnl
                gross_close = 0.0
                bar_funding = 0.0
                for ci in range(nc):
                    if not priced[ci]:
                        continue
                    cur = float(pos[ci])
                    prev_m = float(prev_mark[ci])
                    exec_px = exec_px_arr[ci]
                    m_c = float(m[ci])
                    p1_c = p1_arr[ci]
                    desired_c = desired_arr[ci]
                    if blocked_until_reset and abs(cur) > 1e-12 and not np.isfinite(float(p[ci])):
                        _, exit_px = _conservative_gap_exit_price(cur, prev_m, m_c, haircut_rate)
                        bar_pnl += _conservative_gap_exit_pnl(
                            cur, prev_m, m_c, haircut_rate, fee_rate
                        )
                        realized, _ = _apply_fill_to_basis(cur, 0.0, float(entry_px[ci]), exit_px)
                        balance += realized - abs(cur) * exit_px * fee_rate
                        entry_px[ci] = 0.0
                        unpriced_open_steps += 1
                        forced_gap_exits += 1
                        pos[ci] = 0.0
                        prev_mark[ci] = m_c
                        continue
                    if np.isfinite(prev_m):
                        bar_pnl += cur * (exec_px - prev_m)
                    trade_cost = (abs(p1_c - cur) + abs(desired_c - p1_c)) * exec_px * cost_rate
                    bar_pnl += desired_c * (m_c - exec_px) - trade_cost
                    realized, new_entry = _apply_fill_to_basis(
                        cur, desired_c, float(entry_px[ci]), exec_px
                    )
                    balance += realized - trade_cost
                    if funding_mat is not None and abs(desired_c) > 1e-12:
                        f_c = -desired_c * m_c * float(funding_mat[ci, j])
                        bar_pnl += f_c
                        bar_funding += f_c
                        balance += f_c
                    entry_px[ci] = new_entry
                    pos[ci] = desired_c
                    prev_mark[ci] = m_c
                    gross_close += abs(desired_c) * m_c

                total_funding_usd += bar_funding
                eq += bar_pnl
                peak_gross_leverage = max(
                    peak_gross_leverage, _calc_gross_leverage(gross_close, eq)
                )
                if not blocked_until_reset and (eq - cashday) <= -eff_sweep_loss:
                    valid_open = priced & (np.abs(pos) > 1e-12)
                    flatten_cost = float(np.sum(np.abs(pos[valid_open]) * m[valid_open])) * cost_rate
                    for ci in np.flatnonzero(valid_open):
                        m_c = float(m[ci])
                        cur_open = float(pos[ci])
                        realized, _ = _apply_fill_to_basis(
                            cur_open, 0.0, float(entry_px[ci]), m_c
                        )
                        balance += realized - abs(cur_open) * m_c * cost_rate
                        entry_px[ci] = 0.0
                    bar_pnl -= flatten_cost
                    eq -= flatten_cost
                    sweeps, blocked_until_reset = _record_sweep_halt(
                        pos,
                        sweep_snapshot,
                        raw_want,
                        coin_eligible,
                        swept_flags,
                        j,
                        spec.reenter_after_sweep,
                        sweeps,
                    )

        executed_deltas[:, j] = pos - prev_pos
        step_pnl[j] = bar_pnl
        eq_curve[j] = eq
        bal_curve[j] = balance
        cashday_curve[j] = cashday
        peak = max(peak, eq)
        min_eq = min(min_eq, eq)
        peak_to_trough_dd = min(peak_to_trough_dd, eq - peak)
        worst_daily = min(worst_daily, eq - cashday)

        hit_pass = (eq - spec.initial_equity) >= spec.profit_target
        hit_static = hit_static_open or (eq <= (spec.initial_equity - spec.max_static_dd))
        hit_daily = hit_daily_open or ((eq - cashday) <= -eff_daily_limit)

        if terminal_step is None and (hit_pass or hit_static or hit_daily):
            if hit_static:
                verdict = "FAIL_STATIC"
            elif hit_daily:
                verdict = "FAIL_DAILY"
            else:
                verdict = "PASS"
            terminal_step = j
            terminal_ts_ms = ts_ms
            terminal_net = eq - spec.initial_equity
            if stop_on_terminal:
                if j + 1 < n:
                    eq_curve[j + 1 :] = eq
                    bal_curve[j + 1 :] = balance
                    cashday_curve[j + 1 :] = cashday
                break

    if terminal_step is None:
        terminal_net = eq - spec.initial_equity

    return {
        "verdict": verdict,
        "terminal_step": terminal_step,
        "terminal_ts_ms": terminal_ts_ms,
        "terminal_net_usd": round(terminal_net, 2),
        "min_equity_usd": round(min_eq, 2),
        "static_floor_dd_usd": round(min(0.0, min_eq - spec.initial_equity), 2),
        "peak_to_trough_dd_usd": round(peak_to_trough_dd, 2),
        "worst_daily_usd": round(worst_daily, 2),
        "sweeps": sweeps,
        "forced_gap_exits": forced_gap_exits,
        "unpriced_open_steps": unpriced_open_steps,
        "deferred_notional_usd": round(deferred_notional_usd, 2),
        "full_window_net_usd": round(eq - spec.initial_equity, 2),
        "peak_gross_leverage": peak_gross_leverage,
        "total_funding_usd": round(total_funding_usd, 4),
        "step_pnl": step_pnl,
        "eq_curve": eq_curve,
        "balance_curve": bal_curve,
        "cashday_curve": cashday_curve,
        "swept_flags": swept_flags,
        "executed_deltas": executed_deltas,
    }


def extract_mc_daily_blocks(
    step_pnl: np.ndarray,
    swept_flags: np.ndarray,
    timestamps_ms: np.ndarray,
    spec: GuardianSpec,
    *,
    chop_mask: np.ndarray | None = None,
    balance_curve: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition continuous 5m PnL into reset-aligned daily blocks for ``pod_constructor`` MC."""
    n = int(timestamps_ms.size)
    if n == 0:
        raise ValueError("NO_DAILY_RESET_BOUNDARIES_FOR_MC")

    day_indices = np.array(
        [_reset_day_index(int(timestamps_ms[j]), spec.reset_utc_hour, spec.reset_utc_minute) for j in range(n)],
        dtype=np.int64,
    )
    starts = [
        j
        for j in range(n)
        if (j == 0 and _is_reset_bar(int(timestamps_ms[0]), spec.reset_utc_hour, spec.reset_utc_minute))
        or (j > 0 and day_indices[j] > day_indices[j - 1])
    ]
    if not starts:
        starts = [0]
    elif starts[0] > 0:
        s0 = starts[0]
        if (
            s0 >= 12
            or bool(np.any(np.abs(step_pnl[:s0]) > 1e-12))
            or bool(np.any(swept_flags[:s0]))
        ):
            starts = [0, *starts]

    day_chop: list[bool] = []
    day_end: list[float] = []
    day_max: list[float] = []
    day_min: list[float] = []
    day_swp: list[int] = []
    day_mf: list[bool] = []
    day_min_vs_bal: list[float] = []

    step_bars = 288
    cum_all = np.cumsum(step_pnl)
    for idx_pos, s_idx in enumerate(starts):
        next_start = starts[idx_pos + 1] if idx_pos + 1 < len(starts) else n
        e_idx = min(n, next_start)
        if e_idx <= s_idx:
            continue

        eq_prev = spec.initial_equity + (float(cum_all[s_idx - 1]) if s_idx > 0 else 0.0)
        if spec.daily_loss_basis == "balance" and balance_curve is not None:
            cashday_s = float(balance_curve[s_idx - 1]) if s_idx > 0 else spec.initial_equity
            unrl_carry = eq_prev - cashday_s
        else:
            cashday_s = eq_prev
            unrl_carry = 0.0

        eff_daily_s = _effective_allowance(cashday_s, spec.daily_loss_limit, spec.daily_loss_rate)
        seg = step_pnl[s_idx:e_idx]
        cum = np.cumsum(seg)
        d_min_eq = float(np.min(cum))
        d_min_bal_raw = float(min(unrl_carry, float(np.min(unrl_carry + cum))))
        if (
            spec.daily_loss_rate is not None
            and spec.daily_loss_basis == "balance"
            and eff_daily_s > 0.0
            and abs(eff_daily_s - spec.daily_loss_limit) > 1e-12
        ):
            d_min_bal = d_min_bal_raw * (spec.daily_loss_limit / eff_daily_s)
        else:
            d_min_bal = d_min_bal_raw

        hit_breach_partial = (
            (d_min_bal <= -spec.daily_loss_limit)
            or ((eq_prev + d_min_eq) <= (spec.initial_equity - spec.max_static_dd))
            or bool(np.any(swept_flags[s_idx:e_idx]))
        )
        is_short_sliver = (e_idx - s_idx) < (step_bars - 12)
        if is_short_sliver and len(day_end) > 0 and not hit_breach_partial:
            continue

        day_end.append(float(cum[-1]))
        day_max.append(float(np.max(cum)))
        day_min.append(d_min_eq)
        day_swp.append(int(np.any(swept_flags[s_idx:e_idx])))
        day_mf.append(bool(np.argmax(cum) < np.argmin(cum)))
        day_min_vs_bal.append(d_min_bal)
        if chop_mask is not None:
            day_chop.append(bool(np.mean(chop_mask[s_idx:e_idx]) >= 0.5))
        else:
            day_chop.append(False)

    return (
        np.asarray(day_chop, dtype=bool),
        np.asarray(day_end, dtype=np.float64),
        np.asarray(day_max, dtype=np.float64),
        np.asarray(day_min, dtype=np.float64),
        np.asarray(day_swp, dtype=np.int32),
        np.asarray(day_mf, dtype=bool),
        np.asarray(day_min_vs_bal, dtype=np.float64),
    )


def simulate_guardian_challenge(
    coins: Sequence[str],
    net_targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    timestamps_ms: np.ndarray,
    spec: GuardianSpec = GuardianSpec(),
    *,
    chop_mask: np.ndarray | None = None,
    funding_rates: np.ndarray | None = None,
) -> tuple[ChallengeRunResult, tuple]:
    """Run causal Guardian simulation returning both frozen terminal verdict and MC blocks."""
    term = _execute_guardian_pass(
        coins,
        net_targets,
        trade_open,
        mark_close,
        timestamps_ms,
        spec,
        stop_on_terminal=True,
        funding_rates=funding_rates,
    )
    if term["terminal_step"] is None:
        cont = term
    else:
        cont = _execute_guardian_pass(
            coins,
            net_targets,
            trade_open,
            mark_close,
            timestamps_ms,
            spec,
            stop_on_terminal=False,
            funding_rates=funding_rates,
        )
    blocks = extract_mc_daily_blocks(
        cont["step_pnl"],
        cont["swept_flags"],
        timestamps_ms,
        spec,
        chop_mask=chop_mask,
        balance_curve=cont["balance_curve"],
    )
    result = ChallengeRunResult(
        verdict=term["verdict"],
        terminal_step=term["terminal_step"],
        terminal_ts_ms=term["terminal_ts_ms"],
        terminal_net_usd=term["terminal_net_usd"],
        min_equity_usd=term["min_equity_usd"],
        static_floor_dd_usd=term["static_floor_dd_usd"],
        peak_to_trough_dd_usd=term["peak_to_trough_dd_usd"],
        worst_daily_usd=term["worst_daily_usd"],
        sweeps=term["sweeps"],
        forced_gap_exits=term["forced_gap_exits"],
        unpriced_open_steps=term["unpriced_open_steps"],
        deferred_notional_usd=term["deferred_notional_usd"],
        full_window_net_usd=cont["full_window_net_usd"],
        full_window_peak_to_trough_dd_usd=cont["peak_to_trough_dd_usd"],
        full_window_static_floor_dd_usd=cont["static_floor_dd_usd"],
        full_window_sweeps=cont["sweeps"],
        step_pnl=cont["step_pnl"],
        equity_curve=cont["eq_curve"],
        peak_gross_leverage=term["peak_gross_leverage"],
        full_window_peak_gross_leverage=cont["peak_gross_leverage"],
        total_funding_usd=term["total_funding_usd"],
        full_window_funding_usd=cont["total_funding_usd"],
        executed_deltas=cont["executed_deltas"],
    )
    return result, blocks


def net_member_targets(member_pos: np.ndarray, k_weights: Sequence[float]) -> np.ndarray:
    """Net member target positions across coins (M, C, T) weighted by ``k_weights``."""
    w = np.asarray(k_weights, dtype=np.float64)[:, None, None]
    return np.sum(member_pos * w, axis=0)


def reconcile_portfolio_with_pm03(
    coins: Sequence[str],
    timestamps_ms: np.ndarray,
    net_targets: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    pm_name: str = "pod",
) -> dict:
    """Validate Tier-1 netted portfolio execution against canonical Decimal ``PM03 Account``."""
    instruments = tuple(Instrument.parse(f"hyperliquid/mainnet:perp:default/{c}") for c in coins)
    start_iso = datetime.fromtimestamp(int(timestamps_ms[0]) / 1000.0, tz=UTC).isoformat()
    rb = Rulebook(
        program="CANONICAL_REPLAY_RECONCILIATION",
        stage="research_tier2",
        source="LOCAL_1M_OHLCV_PROXY",
        observed_at=start_iso,
        version="reconcile-v1",
        timezone="UTC",
        initial_balance=D(str(initial_equity)),
        profit_target=None,
        daily_loss=None,
        static_loss=None,
        trailing_loss=None,
        permitted=instruments,
        confirmed=True,
        account_terms_confirmed=False,
    )
    account = Account.start(rb, start_iso)
    costs = CostModel(
        fee_bps=D(str(fee_bps)),
        full_spread_bps=D(0),
        slippage_bps=D(str(slippage_bps)),
        impact_bps_at_full_participation=D(0),
        max_participation=D(1),
        max_age_seconds=300,
        latency_seconds=0,
    )

    fill_count = 0
    for j in range(int(timestamps_ms.size)):
        ts_ms = int(timestamps_ms[j])
        decision_iso = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).isoformat()
        close_iso = datetime.fromtimestamp((ts_ms + 240_000) / 1000.0, tz=UTC).isoformat()

        observations: dict[Instrument, MarketObservation] = {}
        close_marks: dict[Instrument, D] = {}
        desired_book: dict[Instrument, D] = {}

        for ci, inst in enumerate(instruments):
            m_c = float(mark_close[ci, j])
            t_c = float(trade_open[ci, j])
            if not np.isfinite(m_c):
                continue
            exec_px = t_c if np.isfinite(t_c) else m_c
            desired_qty = D(str(round(float(net_targets[ci, j]), 12)))
            cur_qty = account.quantities.get(inst, D(0))
            if abs(float(desired_qty - cur_qty)) > 1e-9 and not np.isfinite(t_c):
                desired_qty = cur_qty
            desired_book[inst] = desired_qty
            observations[inst] = MarketObservation(
                instrument=inst,
                event_at=decision_iso,
                first_known_at=decision_iso,
                mark=D(str(exec_px)),
                volume_base=D(1_000_000_000),
                quality="OHLCV_1M_PROXY",
                supported=True,
            )
            close_marks[inst] = D(str(m_c))

        report = execute_targets(
            account,
            {pm_name: desired_book},
            observations,
            costs,
            decision_at=decision_iso,
            hypothetical_fill_at=decision_iso,
            permitted=True,
        )
        fill_count += len(report["fills"])
        account.mark(close_marks, close_iso)

    account.verify()
    return {
        "final_equity": float(account.equity),
        "net_pnl": float(account.equity - D(str(initial_equity))),
        "realized_pnl": float(account.realized_pnl),
        "total_costs": float(account.costs),
        "fill_count": fill_count,
        "quantities": {inst.symbol: float(qty) for inst, qty in account.quantities.items()},
        "halted": account.breached,
        "account_verified": True,
    }


class CheckpointClass(StrEnum):
    """CWSA 5-minute checkpoint classification."""

    BASELINE = "BASELINE"
    UNCHANGED = "UNCHANGED"
    CHANGED = "CHANGED"
    FENCED = "FENCED"
    RECOVERY_BASELINE = "RECOVERY_BASELINE"


class OrderDisposition(StrEnum):
    """Per-seat market checkpoint execution disposition."""

    NOOP = "NOOP"
    EXECUTED = "EXECUTED"
    CLIPPED = "CLIPPED"
    MISSED = "MISSED"
    QUARANTINED = "QUARANTINED"
    STATIC_HELD = "STATIC_HELD"


class MarketProjectionMode(StrEnum):
    """Q23=A single-order checkpoint market projection mode."""

    DIRECT = "DIRECT"
    CROSS_ZERO_SINGLE_ORDER = "CROSS_ZERO_SINGLE_ORDER"
    REDUCE_ONLY_FALLBACK = "REDUCE_ONLY_FALLBACK"


DEFAULT_SCALE_ALIASES: dict[str, tuple[str, float]] = {
    "kPEPE": ("1000PEPE", 1.0),
    "kSHIB": ("1000SHIB", 1.0),
    "kFLOKI": ("1000FLOKI", 1.0),
    "kBONK": ("1000BONK", 1.0),
    "kLUNC": ("1000LUNC", 1.0),
    "kNEIRO": ("1000NEIRO", 1.0),
}


def resolve_scale_alias(
    coin: str,
    scale_aliases: Mapping[str, tuple[str, float]] | None = None,
) -> tuple[str, float]:
    """Resolve source coin symbol to ``(follower_symbol, qty_multiplier)``."""
    mapping = scale_aliases if scale_aliases is not None else DEFAULT_SCALE_ALIASES
    if coin in mapping:
        sym, mult = mapping[coin]
        return str(sym), float(mult)
    return str(coin), 1.0


@dataclass(frozen=True)
class CWSAReplayConfig:
    """Parameters governing CWSA one-shot 5-minute checkpoint projection."""

    source_stale_ms: int = 600_000
    mark_stale_ms: int = 300_000
    recovery_clean_bars: int = 2
    cross_zero_permitted: bool = True
    allow_q15_late_increase: bool = True
    half_spread_bps: float = 0.0
    fee_bps: float = DEFAULT_FEE_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    qty_eps: float = 1e-12
    scale_aliases: Mapping[str, tuple[str, float]] = field(
        default_factory=lambda: dict(DEFAULT_SCALE_ALIASES)
    )


@dataclass(frozen=True)
class CausalLiquidityConfig:
    """Parameters governing 3-layer causal follower liquidity capacity."""

    l1_window_days: int = 14
    l1_min_median_daily_usd: float = 1_000_000.0
    l1_min_1m_completeness: float = 0.95
    l2_lookback_bars: int = 12
    l2_quantile: float = 0.50
    l2_min_valid_bars: int = 3
    max_participation_1m: float = 0.05
    protective_exit_haircut_bps: float = 15.0


@dataclass
class PairCWSAState:
    """Mutable per-``(seat, coin)`` CWSA checkpoint tracking and quarantine state."""

    initialized: bool = False
    prev_source_qty: float = 0.0
    held_follower_qty: float = 0.0
    fenced: bool = False
    recovery_clean_count: int = 0
    quarantined: bool = False
    quarantine_fresh_marks: int = 0
    quarantine_awaiting_change: bool = False


@dataclass(frozen=True)
class L1ScreenResult:
    """Layer-1 causal market eligibility screen summary."""

    coins: tuple[str, ...]
    allowed_mask: np.ndarray
    allowed_coins: tuple[str, ...]
    median_daily_usd: np.ndarray
    completeness_1m: np.ndarray
    rolling_allowed_mask: np.ndarray


@dataclass(frozen=True)
class SeatMarketIntent:
    """Per-seat target proposal on one market prior to Q23=A and L3 allocation."""

    seat_id: str
    held_qty: float
    raw_desired_qty: float
    seat_blocks_cross_zero: bool = False
    initial_disposition: OrderDisposition = OrderDisposition.NOOP


@dataclass(frozen=True)
class MarketStepAllocation:
    """Layer-3 Q23=A market projection, internal crosses, and pro-rata fill result."""

    coin_idx: int
    projection_mode: MarketProjectionMode
    seat_projected_qty: dict[str, float]
    seat_post_qty: dict[str, float]
    seat_dispositions: dict[str, OrderDisposition]
    internal_crosses: tuple[tuple[str, str, float], ...]
    external_allocations: dict[str, float]
    external_delta_qty: float
    external_exec_price: float
    external_notional_usd: float
    missed_notional_usd: float
    extra_haircut_bps: float
    l2_cap_usd: float
    l2_prior_vol_valid: bool
    realized_participation_1m: float


@dataclass(frozen=True)
class RealizedParticipationAudit:
    """Ex-post 1-minute bar volume participation and missed-notional diagnostics."""

    lookahead_bars_used: int
    trade_count: int
    volume_covered_trades: int
    unpriced_or_missing_vol_trades: int
    total_external_notional_usd: float
    total_missed_notional_usd: float
    max_participation_1m: float
    p95_participation_1m: float
    median_participation_1m: float
    bars_gt_5pct_count: int
    clipped_order_count: int
    missed_order_count: int
    protective_haircut_count: int


@dataclass(frozen=True)
class CWSAReplayResult:
    """Single-seat CWSA one-shot replay output aligned to ``grid_ms``."""

    mode: Literal["cwsa_one_shot_v3"]
    coins: tuple[str, ...]
    follower_coins: tuple[str, ...]
    grid_ms: np.ndarray
    executed_positions: np.ndarray
    proposed_positions: np.ndarray
    executed_deltas: np.ndarray
    gross_pnl: np.ndarray
    costs: np.ndarray
    net_pnl: np.ndarray
    step_pnl: np.ndarray
    checkpoint_classes: np.ndarray
    dispositions: np.ndarray
    realized_participation_1m: np.ndarray
    l2_cap_usd: np.ndarray
    l1_screen: L1ScreenResult | None
    participation_audit: RealizedParticipationAudit


def _opposes(a: float, b: float, eps: float = 1e-12) -> bool:
    return (a > eps and b < -eps) or (a < -eps and b > eps)


def _same_sign(a: float, b: float, eps: float = 1e-12) -> bool:
    return (a > eps and b > eps) or (a < -eps and b < -eps)


def compute_notional_mirror_qty(
    *,
    k_scale: float,
    source_qty: float,
    source_mark: float,
    follower_exec_quote: float,
    qty_multiplier: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Convert ``k * source_qty * source_mark`` USD notional into follower contracts."""
    if not math.isfinite(k_scale) or not math.isfinite(qty_multiplier):
        raise ValueError("INVALID_MIRROR_SCALE")
    if not math.isfinite(source_qty) or abs(source_qty) <= eps or abs(k_scale) <= eps:
        return 0.0
    if not (math.isfinite(source_mark) and source_mark > 0.0):
        raise ValueError("INVALID_SOURCE_MARK")
    if not (math.isfinite(follower_exec_quote) and follower_exec_quote > 0.0):
        raise ValueError("INVALID_FOLLOWER_EXEC_QUOTE")
    requested_notional = float(k_scale) * float(source_qty) * float(source_mark) * float(
        qty_multiplier
    )
    return requested_notional / float(follower_exec_quote)


def _select_directional_mirror_qty(
    *,
    held_follower_qty: float,
    k_scale: float,
    source_qty: float,
    source_mark: float,
    exec_quote_buy: float,
    exec_quote_sell: float,
    qty_multiplier: float,
    eps: float,
) -> float:
    if abs(source_qty) <= eps or abs(k_scale) <= eps:
        return 0.0
    q_buy_tgt = compute_notional_mirror_qty(
        k_scale=k_scale,
        source_qty=source_qty,
        source_mark=source_mark,
        follower_exec_quote=exec_quote_buy,
        qty_multiplier=qty_multiplier,
        eps=eps,
    )
    q_sell_tgt = compute_notional_mirror_qty(
        k_scale=k_scale,
        source_qty=source_qty,
        source_mark=source_mark,
        follower_exec_quote=exec_quote_sell,
        qty_multiplier=qty_multiplier,
        eps=eps,
    )
    if q_buy_tgt > held_follower_qty + eps:
        return q_buy_tgt
    if q_sell_tgt < held_follower_qty - eps:
        return q_sell_tgt
    return held_follower_qty


def _protective_recovery_target(
    *,
    held_follower_qty: float,
    k_scale: float,
    source_qty: float,
    source_mark: float,
    exec_quote_buy: float,
    exec_quote_sell: float,
    qty_multiplier: float,
    eps: float,
) -> float | None:
    if abs(held_follower_qty) <= eps:
        return None
    if abs(source_qty) <= eps or _opposes(source_qty, held_follower_qty, eps):
        return 0.0
    rec_tgt = _select_directional_mirror_qty(
        held_follower_qty=held_follower_qty,
        k_scale=k_scale,
        source_qty=source_qty,
        source_mark=source_mark,
        exec_quote_buy=exec_quote_buy,
        exec_quote_sell=exec_quote_sell,
        qty_multiplier=qty_multiplier,
        eps=eps,
    )
    if _same_sign(rec_tgt, held_follower_qty, eps) and abs(rec_tgt) < abs(held_follower_qty) - eps:
        return float(rec_tgt)
    if abs(rec_tgt) <= eps:
        return 0.0
    return None


def classify_and_propose_cwsa_pair(
    state: PairCWSAState,
    *,
    source_qty: float,
    source_mark: float,
    mark_price: float,
    exec_quote_buy: float,
    exec_quote_sell: float,
    k_scale: float,
    source_age_ms: int = 0,
    mark_age_ms: int = 0,
    qty_multiplier: float = 1.0,
    initial_follower_qty: float = 0.0,
    cfg: CWSAReplayConfig = CWSAReplayConfig(),
) -> tuple[CheckpointClass, float, bool]:
    """Classify one ``(seat, coin)`` checkpoint and return ``(cls, desired_qty, quote_ok)``."""
    quote_ok = (
        0 <= mark_age_ms <= cfg.mark_stale_ms
        and math.isfinite(mark_price)
        and mark_price > 0.0
        and math.isfinite(exec_quote_buy)
        and exec_quote_buy > 0.0
        and math.isfinite(exec_quote_sell)
        and exec_quote_sell > 0.0
    )
    source_ok = math.isfinite(source_qty) and 0 <= source_age_ms <= cfg.source_stale_ms
    source_mark_ok = (not math.isfinite(source_qty)) or (abs(source_qty) <= cfg.qty_eps) or (
        math.isfinite(source_mark) and source_mark > 0.0
    )

    if not state.initialized:
        state.initialized = True
        state.prev_source_qty = float(source_qty) if math.isfinite(source_qty) else 0.0
        state.held_follower_qty = float(initial_follower_qty)
        if not quote_ok or not source_mark_ok:
            state.fenced = True
            state.recovery_clean_count = 0
            state.quarantined = True
            state.quarantine_fresh_marks = 0
            state.quarantine_awaiting_change = True
            return CheckpointClass.FENCED, state.held_follower_qty, False
        if not source_ok:
            state.fenced = True
            state.recovery_clean_count = 0
            return CheckpointClass.FENCED, state.held_follower_qty, True
        return CheckpointClass.BASELINE, state.held_follower_qty, True

    if not quote_ok or not source_mark_ok:
        state.fenced = True
        state.recovery_clean_count = 0
        state.quarantined = True
        state.quarantine_fresh_marks = 0
        state.quarantine_awaiting_change = True
        return CheckpointClass.FENCED, state.held_follower_qty, False

    if not source_ok:
        state.fenced = True
        state.recovery_clean_count = 0
        return CheckpointClass.FENCED, state.held_follower_qty, True

    if state.quarantined:
        state.quarantine_fresh_marks += 1
        if state.quarantine_fresh_marks < cfg.recovery_clean_bars:
            state.prev_source_qty = float(source_qty)
            return CheckpointClass.RECOVERY_BASELINE, state.held_follower_qty, False

        if abs(source_qty - state.prev_source_qty) <= cfg.qty_eps:
            prot_tgt = _protective_recovery_target(
                held_follower_qty=state.held_follower_qty,
                k_scale=k_scale,
                source_qty=source_qty,
                source_mark=source_mark,
                exec_quote_buy=exec_quote_buy,
                exec_quote_sell=exec_quote_sell,
                qty_multiplier=qty_multiplier,
                eps=cfg.qty_eps,
            )
            if prot_tgt is not None:
                state.prev_source_qty = float(source_qty)
                state.fenced = False
                state.quarantined = False
                state.quarantine_awaiting_change = False
                return CheckpointClass.CHANGED, prot_tgt, True
            cls = CheckpointClass.RECOVERY_BASELINE if state.fenced else CheckpointClass.UNCHANGED
            state.fenced = False
            state.quarantine_awaiting_change = True
            return cls, state.held_follower_qty, True

        state.fenced = False
        state.quarantined = False
        state.quarantine_awaiting_change = False
        state.recovery_clean_count = 0

    elif state.fenced:
        state.recovery_clean_count += 1
        prot_tgt = _protective_recovery_target(
            held_follower_qty=state.held_follower_qty,
            k_scale=k_scale,
            source_qty=source_qty,
            source_mark=source_mark,
            exec_quote_buy=exec_quote_buy,
            exec_quote_sell=exec_quote_sell,
            qty_multiplier=qty_multiplier,
            eps=cfg.qty_eps,
        )
        if state.recovery_clean_count < cfg.recovery_clean_bars:
            state.prev_source_qty = float(source_qty)
            if prot_tgt is not None:
                return CheckpointClass.CHANGED, prot_tgt, True
            return CheckpointClass.RECOVERY_BASELINE, state.held_follower_qty, True

        state.fenced = False
        state.recovery_clean_count = 0
        if abs(source_qty - state.prev_source_qty) <= cfg.qty_eps:
            state.prev_source_qty = float(source_qty)
            if prot_tgt is not None:
                return CheckpointClass.CHANGED, prot_tgt, True
            return CheckpointClass.RECOVERY_BASELINE, state.held_follower_qty, True

    if abs(source_qty - state.prev_source_qty) <= cfg.qty_eps:
        return CheckpointClass.UNCHANGED, state.held_follower_qty, True

    prev_old = state.prev_source_qty
    state.prev_source_qty = float(source_qty)
    if abs(source_qty) <= cfg.qty_eps:
        return CheckpointClass.CHANGED, 0.0, True

    full_target = _select_directional_mirror_qty(
        held_follower_qty=state.held_follower_qty,
        k_scale=k_scale,
        source_qty=source_qty,
        source_mark=source_mark,
        exec_quote_buy=exec_quote_buy,
        exec_quote_sell=exec_quote_sell,
        qty_multiplier=qty_multiplier,
        eps=cfg.qty_eps,
    )

    if abs(prev_old) <= cfg.qty_eps or _opposes(source_qty, prev_old, cfg.qty_eps):
        desired = full_target
    elif abs(source_qty) > abs(prev_old) + cfg.qty_eps:
        if cfg.allow_q15_late_increase or abs(state.held_follower_qty) > cfg.qty_eps:
            desired = full_target
        else:
            desired = 0.0
    else:
        if abs(state.held_follower_qty) <= cfg.qty_eps:
            desired = 0.0
        elif _same_sign(state.held_follower_qty, full_target, cfg.qty_eps):
            sign = 1.0 if state.held_follower_qty > 0.0 else -1.0
            desired = sign * min(abs(state.held_follower_qty), abs(full_target))
        else:
            desired = 0.0

    return CheckpointClass.CHANGED, float(desired), True


def screen_l1_allowed_markets(
    coins: Sequence[str],
    vol_1m_usd: np.ndarray,
    grid_ms: np.ndarray,
    *,
    freeze_ts_ms: int | None = None,
    bars_per_day: int = 288,
    cfg: CausalLiquidityConfig = CausalLiquidityConfig(),
) -> L1ScreenResult:
    """Screen markets using trailing completed ``l1_window_days`` median volume & completeness."""
    coin_tuple = tuple(str(c) for c in coins)
    vol_arr = np.asarray(vol_1m_usd, dtype=np.float64)
    grid_arr = np.asarray(grid_ms, dtype=np.int64)
    n_coins = len(coin_tuple)
    n_bars = int(grid_arr.size)
    if vol_arr.shape != (n_coins, n_bars):
        raise ValueError(f"INVALID_VOL_SHAPE:{vol_arr.shape}!={(n_coins, n_bars)}")
    if n_coins == 0 or n_bars == 0 or bars_per_day <= 0:
        return L1ScreenResult(
            coins=coin_tuple,
            allowed_mask=np.zeros(n_coins, dtype=bool),
            allowed_coins=(),
            median_daily_usd=np.zeros(n_coins, dtype=np.float64),
            completeness_1m=np.zeros(n_coins, dtype=np.float64),
            rolling_allowed_mask=np.zeros((n_coins, n_bars), dtype=bool),
        )

    step_ms = int(np.median(np.diff(grid_arr))) if n_bars > 1 else 300_000
    utc_calendar_mode = (bars_per_day * step_ms) == 86_400_000 and step_ms >= 60_000
    minutes_per_sample = float(step_ms) / 60_000.0 if utc_calendar_mode else 1.0

    if utc_calendar_mode:
        day_ids = grid_arr // 86_400_000
    else:
        day_ids = np.arange(n_bars, dtype=np.int64) // int(bars_per_day)

    unique_days = np.unique(day_ids)
    day_slices: list[tuple[int, np.ndarray, int, np.ndarray, np.ndarray]] = []
    for d_id in unique_days:
        idx = np.flatnonzero(day_ids == d_id)
        if idx.size < bars_per_day:
            continue
        end_ts_ms = (
            int((d_id + 1) * 86_400_000)
            if utc_calendar_mode
            else int(grid_arr[int(idx[-1])]) + step_ms
        )
        sub = vol_arr[:, idx[:bars_per_day]]
        valid = np.isfinite(sub) & (sub >= 0.0)
        d_vol = np.where(valid, sub, 0.0).sum(axis=1) * minutes_per_sample
        d_valid_cnt = valid.sum(axis=1).astype(np.float64)
        day_slices.append((int(d_id), idx, end_ts_ms, d_vol, d_valid_cnt))

    w_days = max(1, int(cfg.l1_window_days))
    rolling_mask = np.zeros((n_coins, n_bars), dtype=bool)
    for d_id in unique_days:
        bar_idx = np.flatnonzero(day_ids == d_id)
        prior = [entry for entry in day_slices if entry[0] < int(d_id)]
        if len(prior) < w_days:
            continue
        win = prior[-w_days:]
        vols_mat = np.stack([x[3] for x in win], axis=1)
        cnts_sum = np.sum([x[4] for x in win], axis=0)
        med = np.median(vols_mat, axis=1)
        comp = cnts_sum / float(w_days * bars_per_day)
        ok = (med >= cfg.l1_min_median_daily_usd) & (comp >= cfg.l1_min_1m_completeness)
        rolling_mask[:, bar_idx] = ok[:, None]

    cutoff_ms = int(freeze_ts_ms) if freeze_ts_ms is not None else int(grid_arr[-1]) + step_ms
    completed = [entry for entry in day_slices if entry[2] <= cutoff_ms]
    if len(completed) < w_days:
        return L1ScreenResult(
            coins=coin_tuple,
            allowed_mask=np.zeros(n_coins, dtype=bool),
            allowed_coins=(),
            median_daily_usd=np.zeros(n_coins, dtype=np.float64),
            completeness_1m=np.zeros(n_coins, dtype=np.float64),
            rolling_allowed_mask=rolling_mask,
        )

    win_f = completed[-w_days:]
    vols_f = np.stack([x[3] for x in win_f], axis=1)
    cnts_f = np.sum([x[4] for x in win_f], axis=0)
    median_daily = np.median(vols_f, axis=1)
    completeness = cnts_f / float(w_days * bars_per_day)
    allowed_mask = (median_daily >= cfg.l1_min_median_daily_usd) & (
        completeness >= cfg.l1_min_1m_completeness
    )
    allowed_coins = tuple(c for c, flag in zip(coin_tuple, allowed_mask, strict=True) if flag)
    return L1ScreenResult(
        coins=coin_tuple,
        allowed_mask=allowed_mask,
        allowed_coins=allowed_coins,
        median_daily_usd=median_daily,
        completeness_1m=completeness,
        rolling_allowed_mask=rolling_mask,
    )


def estimate_causal_bar_capacity(
    vol_usd_strictly_prior: np.ndarray,
    cfg: CausalLiquidityConfig = CausalLiquidityConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Layer-2 ``(cap_usd, prior_vol_valid)`` from strictly prior 1m bar volumes."""
    arr = np.asarray(vol_usd_strictly_prior, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"EXPECTED_2D_PRIOR_VOLUME_SLICE:{arr.shape}")
    if arr.shape[1] > cfg.l2_lookback_bars:
        arr = arr[:, -cfg.l2_lookback_bars :]

    n_coins, n_win = arr.shape
    cap_usd = np.zeros(n_coins, dtype=np.float64)
    prior_vol_valid = np.zeros(n_coins, dtype=bool)
    if n_win < cfg.l2_min_valid_bars or cfg.l2_lookback_bars <= 0:
        return cap_usd, prior_vol_valid

    valid_mask = np.isfinite(arr) & (arr >= 0.0)
    valid_counts = np.sum(valid_mask, axis=1)
    for ci in range(n_coins):
        if int(valid_counts[ci]) < cfg.l2_min_valid_bars:
            continue
        vals = arr[ci, valid_mask[ci]]
        v_hat = float(np.quantile(vals, cfg.l2_quantile))
        if math.isfinite(v_hat) and v_hat > 0.0:
            prior_vol_valid[ci] = True
            cap_usd[ci] = cfg.max_participation_1m * v_hat
    return cap_usd, prior_vol_valid


def _cross_pro_rata(
    buy_map: dict[str, float],
    sell_map: dict[str, float],
    crossed_delta: dict[str, float],
    cross_records: list[tuple[str, str, float]],
    eps: float,
) -> None:
    tot_b = sum(q for q in buy_map.values() if q > eps)
    tot_s = sum(q for q in sell_map.values() if q > eps)
    ct = min(tot_b, tot_s)
    if ct <= eps:
        return
    b_take = {k: q * (ct / tot_b) for k, q in sorted(buy_map.items()) if q > eps}
    s_take = {k: q * (ct / tot_s) for k, q in sorted(sell_map.items()) if q > eps}
    for bk, dq in b_take.items():
        buy_map[bk] = max(0.0, buy_map[bk] - dq)
    for sk, dq in s_take.items():
        sell_map[sk] = max(0.0, sell_map[sk] - dq)
    for bk in sorted(b_take):
        for sk in sorted(s_take):
            qty = min(b_take[bk], s_take[sk])
            if qty <= eps:
                continue
            cross_records.append((bk, sk, float(qty)))
            b_take[bk] -= qty
            s_take[sk] -= qty
            crossed_delta[bk] += qty
            crossed_delta[sk] -= qty


def allocate_l3_market_step(
    coin_idx: int,
    intents: Sequence[SeatMarketIntent],
    *,
    exec_price_buy: float,
    exec_price_sell: float,
    mark_price: float,
    l2_cap_usd: float,
    l2_prior_vol_valid: bool,
    actual_bar_vol_1m_usd: float,
    cross_zero_permitted: bool = True,
    protective_exit_haircut_bps: float = 15.0,
    eps: float = 1e-12,
) -> MarketStepAllocation:
    """Project Q23=A market orders, cross opposing seats at mark, and allocate L3 fills."""
    ordered = sorted(intents, key=lambda x: x.seat_id)
    old_net = sum(float(i.held_qty) for i in ordered)
    raw_net = sum(float(i.raw_desired_qty) for i in ordered)

    locked_opposes_active = any(
        _opposes(float(locked.held_qty), float(act.raw_desired_qty), eps)
        for locked in ordered
        if locked.seat_blocks_cross_zero and abs(locked.held_qty) > eps
        for act in ordered
        if not act.seat_blocks_cross_zero and act.seat_id != locked.seat_id
    )
    fallback = (_opposes(old_net, raw_net, eps) and not cross_zero_permitted) or (
        locked_opposes_active
    )

    projected: dict[str, float] = {}
    dispositions: dict[str, OrderDisposition] = {}
    q23_clamped: dict[str, bool] = {}
    q23_partial_clipped: set[str] = set()
    missed_notional_usd = 0.0

    if fallback:
        projection_mode = MarketProjectionMode.REDUCE_ONLY_FALLBACK
        for item in ordered:
            sid = item.seat_id
            if item.seat_blocks_cross_zero:
                proj = float(item.raw_desired_qty)
            elif _same_sign(float(item.held_qty), float(item.raw_desired_qty), eps):
                sign = 1.0 if item.held_qty > 0.0 else -1.0
                proj = sign * min(abs(item.held_qty), abs(item.raw_desired_qty))
            else:
                proj = 0.0
            projected[sid] = proj
            if abs(proj - item.raw_desired_qty) > eps:
                q23_clamped[sid] = True
                dispositions[sid] = OrderDisposition.MISSED
                if math.isfinite(mark_price) and mark_price > 0.0:
                    missed_notional_usd += abs(item.raw_desired_qty - proj) * mark_price
            else:
                q23_clamped[sid] = False
                dispositions[sid] = item.initial_disposition

        proj_net = sum(projected.values())
        if abs(old_net) <= eps or _opposes(old_net, proj_net, eps):
            allowed_net = 0.0
        elif abs(proj_net) > abs(old_net) + eps:
            allowed_net = old_net
        else:
            allowed_net = proj_net

        if abs(proj_net - allowed_net) > eps:
            need_trim = abs(proj_net - allowed_net)
            move_dir = 1.0 if (proj_net - allowed_net) > 0.0 else -1.0
            movers = [
                i
                for i in ordered
                if not i.seat_blocks_cross_zero
                and _same_sign(projected[i.seat_id] - float(i.held_qty), move_dir, eps)
            ]
            mover_sum = sum(abs(projected[i.seat_id] - float(i.held_qty)) for i in movers)
            if mover_sum > eps:
                trim_amt = min(need_trim, mover_sum)
                keep_ratio = max(0.0, (mover_sum - trim_amt) / mover_sum)
                for item in movers:
                    sid = item.seat_id
                    old_proj = projected[sid]
                    d_orig = old_proj - float(item.held_qty)
                    new_proj = float(item.held_qty) + d_orig * keep_ratio
                    projected[sid] = new_proj
                    trimmed_q = abs(old_proj - new_proj)
                    if trimmed_q > eps:
                        if math.isfinite(mark_price) and mark_price > 0.0:
                            missed_notional_usd += trimmed_q * mark_price
                        if abs(new_proj - float(item.held_qty)) <= eps:
                            q23_clamped[sid] = True
                            dispositions[sid] = OrderDisposition.MISSED
                        else:
                            q23_partial_clipped.add(sid)
    else:
        projection_mode = (
            MarketProjectionMode.CROSS_ZERO_SINGLE_ORDER
            if _opposes(old_net, raw_net, eps)
            else MarketProjectionMode.DIRECT
        )
        for item in ordered:
            projected[item.seat_id] = float(item.raw_desired_qty)
            q23_clamped[item.seat_id] = False
            dispositions[item.seat_id] = item.initial_disposition

    deltas = {i.seat_id: projected[i.seat_id] - float(i.held_qty) for i in ordered}
    red_deltas: dict[str, float] = {}
    inc_deltas: dict[str, float] = {}
    for item in ordered:
        sid = item.seat_id
        h_q = float(item.held_qty)
        d_q = deltas[sid]
        if abs(d_q) <= eps:
            red_deltas[sid] = 0.0
            inc_deltas[sid] = 0.0
        elif _opposes(h_q, d_q, eps):
            r_mag = min(abs(h_q), abs(d_q))
            r_val = math.copysign(r_mag, d_q)
            red_deltas[sid] = r_val
            inc_deltas[sid] = d_q - r_val
        else:
            red_deltas[sid] = 0.0
            inc_deltas[sid] = d_q

    red_buys = {sid: q for sid, q in red_deltas.items() if q > eps}
    red_sells = {sid: -q for sid, q in red_deltas.items() if q < -eps}
    inc_buys = {sid: q for sid, q in inc_deltas.items() if q > eps}
    inc_sells = {sid: -q for sid, q in inc_deltas.items() if q < -eps}

    cross_records: list[tuple[str, str, float]] = []
    crossed_delta: dict[str, float] = {i.seat_id: 0.0 for i in ordered}

    _cross_pro_rata(red_buys, red_sells, crossed_delta, cross_records, eps)
    _cross_pro_rata(red_buys, inc_sells, crossed_delta, cross_records, eps)
    _cross_pro_rata(inc_buys, red_sells, crossed_delta, cross_records, eps)
    _cross_pro_rata(inc_buys, inc_sells, crossed_delta, cross_records, eps)

    pos_after_cross: dict[str, float] = {}
    reduce_parts: dict[str, float] = {}
    increase_parts: dict[str, float] = {}
    for item in ordered:
        sid = item.seat_id
        pos_after_cross[sid] = float(item.held_qty) + crossed_delta[sid]
        rb = red_buys.get(sid, 0.0)
        rs = red_sells.get(sid, 0.0)
        ib = inc_buys.get(sid, 0.0)
        is_q = inc_sells.get(sid, 0.0)
        reduce_parts[sid] = rb if rb > eps else (-rs if rs > eps else 0.0)
        increase_parts[sid] = ib if ib > eps else (-is_q if is_q > eps else 0.0)

    clipped_seats: set[str] = set(q23_partial_clipped)
    missed_inc_seats: set[str] = set()

    tot_red_signed = sum(reduce_parts.values())
    tot_inc_signed = sum(increase_parts.values())
    ext_probe = tot_red_signed if abs(tot_red_signed) > eps else tot_inc_signed
    ext_sign = 1.0 if ext_probe > eps else (-1.0 if ext_probe < -eps else 0.0)
    exec_px = (
        float(exec_price_buy)
        if ext_sign > 0.0
        else (float(exec_price_sell) if ext_sign < 0.0 else float(mark_price))
    )

    if not l2_prior_vol_valid and abs(tot_red_signed) > eps:
        max_red_mag = abs(old_net) if _opposes(old_net, tot_red_signed, eps) else 0.0
        cur_red_mag = sum(abs(v) for v in reduce_parts.values())
        if cur_red_mag > max_red_mag + eps:
            r_scale = max_red_mag / cur_red_mag if max_red_mag > eps else 0.0
            missed_notional_usd += (cur_red_mag - max_red_mag) * exec_px
            for item in ordered:
                sid = item.seat_id
                if abs(reduce_parts[sid]) > eps:
                    reduce_parts[sid] = reduce_parts[sid] * r_scale
                    if abs(reduce_parts[sid]) > eps or abs(crossed_delta[sid]) > eps:
                        clipped_seats.add(sid)
                    else:
                        missed_inc_seats.add(sid)

    total_reduce_qty = sum(abs(v) for v in reduce_parts.values())
    extra_haircut_bps = (
        float(protective_exit_haircut_bps)
        if (not l2_prior_vol_valid and total_reduce_qty > eps)
        else 0.0
    )

    rem_cap_usd = (
        0.0
        if not l2_prior_vol_valid
        else max(0.0, float(l2_cap_usd) - total_reduce_qty * exec_px)
    )
    req_inc_qty = sum(abs(v) for v in increase_parts.values())
    req_inc_usd = req_inc_qty * exec_px

    inc_fills: dict[str, float] = {i.seat_id: 0.0 for i in ordered}

    if req_inc_qty > eps:
        if rem_cap_usd <= eps:
            missed_notional_usd += req_inc_usd
            for item in ordered:
                sid = item.seat_id
                if abs(increase_parts[sid]) > eps:
                    has_prior_fill = (
                        abs(crossed_delta[sid]) > eps or abs(reduce_parts[sid]) > eps
                    )
                    if has_prior_fill:
                        clipped_seats.add(sid)
                    else:
                        missed_inc_seats.add(sid)
        elif req_inc_usd <= rem_cap_usd + 1e-9:
            for item in ordered:
                sid = item.seat_id
                inc_fills[sid] = increase_parts[sid]
        else:
            scale = rem_cap_usd / req_inc_usd
            missed_notional_usd += req_inc_usd - rem_cap_usd
            for item in ordered:
                sid = item.seat_id
                if abs(increase_parts[sid]) > eps:
                    inc_fills[sid] = increase_parts[sid] * scale
                    clipped_seats.add(sid)

    seat_post_qty: dict[str, float] = {}
    external_allocations: dict[str, float] = {}
    for item in ordered:
        sid = item.seat_id
        ext_i = reduce_parts[sid] + inc_fills[sid]
        post_i = pos_after_cross[sid] + ext_i
        if abs(post_i) <= eps:
            post_i = 0.0
        seat_post_qty[sid] = post_i
        if abs(ext_i) > eps:
            external_allocations[sid] = ext_i

        if q23_clamped[sid]:
            dispositions[sid] = OrderDisposition.MISSED
        elif sid in clipped_seats:
            dispositions[sid] = OrderDisposition.CLIPPED
        elif sid in missed_inc_seats:
            dispositions[sid] = OrderDisposition.MISSED
        elif abs(deltas[sid]) > eps:
            dispositions[sid] = OrderDisposition.EXECUTED

    external_delta_qty = sum(external_allocations.values())
    if abs(external_delta_qty) <= eps:
        external_delta_qty = 0.0
        external_notional_usd = 0.0
    else:
        external_notional_usd = abs(external_delta_qty) * exec_px

    if math.isfinite(actual_bar_vol_1m_usd) and actual_bar_vol_1m_usd > 0.0:
        realized_part = external_notional_usd / float(actual_bar_vol_1m_usd)
    else:
        realized_part = float("nan")

    return MarketStepAllocation(
        coin_idx=coin_idx,
        projection_mode=projection_mode,
        seat_projected_qty=projected,
        seat_post_qty=seat_post_qty,
        seat_dispositions=dispositions,
        internal_crosses=tuple(cross_records),
        external_allocations=external_allocations,
        external_delta_qty=external_delta_qty,
        external_exec_price=exec_px,
        external_notional_usd=external_notional_usd,
        missed_notional_usd=missed_notional_usd,
        extra_haircut_bps=extra_haircut_bps,
        l2_cap_usd=float(l2_cap_usd),
        l2_prior_vol_valid=bool(l2_prior_vol_valid),
        realized_participation_1m=realized_part,
    )


def simulate_cwsa_one_shot_replay(
    coins: Sequence[str],
    source_positions: np.ndarray,
    trade_open: np.ndarray,
    mark_close: np.ndarray,
    grid_ms: np.ndarray,
    *,
    source_marks: np.ndarray | None = None,
    k_scale: float = 1.0,
    vol_1m_usd: np.ndarray | None = None,
    source_ages_ms: np.ndarray | None = None,
    mark_ages_ms: np.ndarray | None = None,
    allowed_markets_mask: np.ndarray | None = None,
    initial_positions: np.ndarray | None = None,
    seat_id: str = "SEAT_0",
    mode: Literal["cwsa_one_shot_v3"] = "cwsa_one_shot_v3",
    cwsa_cfg: CWSAReplayConfig = CWSAReplayConfig(),
    liquidity_cfg: CausalLiquidityConfig = CausalLiquidityConfig(),
) -> CWSAReplayResult:
    """Run single-seat CWSA one-shot replay with 3-layer causal follower liquidity."""
    if mode != "cwsa_one_shot_v3":
        raise ValueError(f"UNSUPPORTED_CWSA_REPLAY_MODE:{mode}")

    coin_tuple = tuple(str(c) for c in coins)
    n_coins = len(coin_tuple)
    grid_arr = np.asarray(grid_ms, dtype=np.int64)
    n_bars = int(grid_arr.size)

    src_pos = np.asarray(source_positions, dtype=np.float64)
    t_open = np.asarray(trade_open, dtype=np.float64)
    m_close = np.asarray(mark_close, dtype=np.float64)
    if src_pos.shape != (n_coins, n_bars) or t_open.shape != (n_coins, n_bars) or (
        m_close.shape != (n_coins, n_bars)
    ):
        raise ValueError("CWSA_REPLAY_SHAPE_MISMATCH")

    src_marks = (
        np.asarray(source_marks, dtype=np.float64)
        if source_marks is not None
        else np.where(np.isfinite(t_open), t_open, m_close)
    )
    vol_mat = (
        np.asarray(vol_1m_usd, dtype=np.float64)
        if vol_1m_usd is not None
        else np.full((n_coins, n_bars), np.nan, dtype=np.float64)
    )
    src_ages = (
        np.asarray(source_ages_ms, dtype=np.int64)
        if source_ages_ms is not None
        else np.zeros((n_coins, n_bars), dtype=np.int64)
    )
    mk_ages = (
        np.asarray(mark_ages_ms, dtype=np.int64)
        if mark_ages_ms is not None
        else np.zeros((n_coins, n_bars), dtype=np.int64)
    )
    init_pos = (
        np.asarray(initial_positions, dtype=np.float64)
        if initial_positions is not None
        else np.zeros(n_coins, dtype=np.float64)
    )

    resolved_pairs = [resolve_scale_alias(c, cwsa_cfg.scale_aliases) for c in coin_tuple]
    follower_coins = tuple(pair[0] for pair in resolved_pairs)
    qty_mults = np.array([pair[1] for pair in resolved_pairs], dtype=np.float64)

    l1_res: L1ScreenResult | None = None
    if allowed_markets_mask is not None:
        allow_arr = np.asarray(allowed_markets_mask, dtype=bool)
        if allow_arr.ndim == 1:
            allow_2d = np.broadcast_to(allow_arr[:, None], (n_coins, n_bars))
        else:
            allow_2d = allow_arr
    else:
        allow_2d = np.ones((n_coins, n_bars), dtype=bool)

    exec_pos = np.zeros((n_coins, n_bars), dtype=np.float64)
    prop_pos = np.zeros((n_coins, n_bars), dtype=np.float64)
    exec_deltas = np.zeros((n_coins, n_bars), dtype=np.float64)
    gross_pnl = np.zeros((n_coins, n_bars), dtype=np.float64)
    costs = np.zeros((n_coins, n_bars), dtype=np.float64)
    ckpt_classes = np.empty((n_coins, n_bars), dtype=object)
    disp_arr = np.empty((n_coins, n_bars), dtype=object)
    real_part = np.full((n_coins, n_bars), np.nan, dtype=np.float64)
    l2_caps = np.zeros((n_coins, n_bars), dtype=np.float64)

    pair_states = [PairCWSAState() for _ in range(n_coins)]
    prev_marks = np.full(n_coins, np.nan, dtype=np.float64)
    spread_frac = cwsa_cfg.half_spread_bps * 1e-4

    trade_ratios: list[float] = []
    trade_count = 0
    missing_vol_trades = 0
    total_ext_notional = 0.0
    total_missed_notional = 0.0
    clipped_cnt = 0
    missed_cnt = 0
    haircut_cnt = 0

    for j in range(n_bars):
        prior_start = max(0, j - liquidity_cfg.l2_lookback_bars)
        prior_vol_slice = vol_mat[:, prior_start:j]
        bar_caps, bar_vol_valid = estimate_causal_bar_capacity(prior_vol_slice, liquidity_cfg)
        l2_caps[:, j] = bar_caps

        for ci in range(n_coins):
            st = pair_states[ci]
            to_px = float(t_open[ci, j])
            mc_px = float(m_close[ci, j])
            if math.isfinite(to_px) and to_px > 0.0:
                q_buy = to_px * (1.0 + spread_frac)
                q_sell = to_px * max(1e-9, 1.0 - spread_frac)
            else:
                q_buy = float("nan")
                q_sell = float("nan")

            cls, raw_want, executable = classify_and_propose_cwsa_pair(
                st,
                source_qty=float(src_pos[ci, j]),
                source_mark=float(src_marks[ci, j]),
                mark_price=mc_px,
                exec_quote_buy=q_buy,
                exec_quote_sell=q_sell,
                k_scale=k_scale,
                source_age_ms=int(src_ages[ci, j]),
                mark_age_ms=int(mk_ages[ci, j]),
                qty_multiplier=float(qty_mults[ci]),
                initial_follower_qty=float(init_pos[ci]),
                cfg=cwsa_cfg,
            )
            ckpt_classes[ci, j] = cls.value

            if not allow_2d[ci, j]:
                if _same_sign(st.held_follower_qty, raw_want, cwsa_cfg.qty_eps):
                    sgn = 1.0 if st.held_follower_qty > 0.0 else -1.0
                    raw_want = sgn * min(abs(st.held_follower_qty), abs(raw_want))
                else:
                    raw_want = 0.0

            prop_pos[ci, j] = raw_want
            old_qty = st.held_follower_qty

            if not executable:
                disp_arr[ci, j] = OrderDisposition.QUARANTINED.value
                exec_pos[ci, j] = old_qty
                exec_deltas[ci, j] = 0.0
                if (
                    math.isfinite(mc_px)
                    and mc_px > 0.0
                    and 0 <= int(mk_ages[ci, j]) <= cwsa_cfg.mark_stale_ms
                ):
                    if math.isfinite(prev_marks[ci]):
                        gross_pnl[ci, j] = old_qty * (mc_px - float(prev_marks[ci]))
                    prev_marks[ci] = mc_px
                continue

            alloc = allocate_l3_market_step(
                ci,
                (
                    SeatMarketIntent(
                        seat_id=seat_id,
                        held_qty=old_qty,
                        raw_desired_qty=raw_want,
                        seat_blocks_cross_zero=False,
                        initial_disposition=OrderDisposition.NOOP,
                    ),
                ),
                exec_price_buy=q_buy,
                exec_price_sell=q_sell,
                mark_price=mc_px,
                l2_cap_usd=float(bar_caps[ci]),
                l2_prior_vol_valid=bool(bar_vol_valid[ci]),
                actual_bar_vol_1m_usd=float(vol_mat[ci, j]),
                cross_zero_permitted=cwsa_cfg.cross_zero_permitted,
                protective_exit_haircut_bps=liquidity_cfg.protective_exit_haircut_bps,
                eps=cwsa_cfg.qty_eps,
            )

            new_qty = alloc.seat_post_qty[seat_id]
            d_qty = new_qty - old_qty
            disp = alloc.seat_dispositions[seat_id]
            st.held_follower_qty = new_qty

            exec_pos[ci, j] = new_qty
            exec_deltas[ci, j] = d_qty
            disp_arr[ci, j] = disp.value
            real_part[ci, j] = alloc.realized_participation_1m
            total_missed_notional += alloc.missed_notional_usd
            if disp == OrderDisposition.CLIPPED:
                clipped_cnt += 1
            elif disp == OrderDisposition.MISSED:
                missed_cnt += 1
            if alloc.extra_haircut_bps > 0.0:
                haircut_cnt += 1

            bar_gross = 0.0
            if math.isfinite(prev_marks[ci]):
                bar_gross += old_qty * (to_px - float(prev_marks[ci]))
            bar_gross += new_qty * (mc_px - to_px)

            bar_cost = 0.0
            if abs(d_qty) > cwsa_cfg.qty_eps:
                trade_count += 1
                total_ext_notional += alloc.external_notional_usd
                if math.isfinite(alloc.realized_participation_1m):
                    trade_ratios.append(alloc.realized_participation_1m)
                else:
                    missing_vol_trades += 1
                ext_sgn = 1.0 if d_qty > 0.0 else -1.0
                slip_rate = (cwsa_cfg.slippage_bps + alloc.extra_haircut_bps) * 1e-4
                fill_px = alloc.external_exec_price * (1.0 + ext_sgn * slip_rate)
                drag_usd = d_qty * (fill_px - to_px)
                fee_usd = abs(d_qty) * fill_px * (cwsa_cfg.fee_bps * 1e-4)
                bar_cost = drag_usd + fee_usd

            gross_pnl[ci, j] = bar_gross
            costs[ci, j] = bar_cost
            prev_marks[ci] = mc_px

    net_pnl = gross_pnl - costs
    step_pnl = np.sum(net_pnl, axis=0)
    ratios_np = np.asarray(trade_ratios, dtype=np.float64)
    audit = RealizedParticipationAudit(
        lookahead_bars_used=0,
        trade_count=trade_count,
        volume_covered_trades=int(ratios_np.size),
        unpriced_or_missing_vol_trades=missing_vol_trades,
        total_external_notional_usd=total_ext_notional,
        total_missed_notional_usd=total_missed_notional,
        max_participation_1m=float(np.max(ratios_np)) if ratios_np.size else 0.0,
        p95_participation_1m=float(np.quantile(ratios_np, 0.95)) if ratios_np.size else 0.0,
        median_participation_1m=float(np.median(ratios_np)) if ratios_np.size else 0.0,
        bars_gt_5pct_count=int(np.count_nonzero(ratios_np > 0.05)) if ratios_np.size else 0,
        clipped_order_count=clipped_cnt,
        missed_order_count=missed_cnt,
        protective_haircut_count=haircut_cnt,
    )
    return CWSAReplayResult(
        mode=mode,
        coins=coin_tuple,
        follower_coins=follower_coins,
        grid_ms=grid_arr,
        executed_positions=exec_pos,
        proposed_positions=prop_pos,
        executed_deltas=exec_deltas,
        gross_pnl=gross_pnl,
        costs=costs,
        net_pnl=net_pnl,
        step_pnl=step_pnl,
        checkpoint_classes=ckpt_classes,
        dispositions=disp_arr,
        realized_participation_1m=real_part,
        l2_cap_usd=l2_caps,
        l1_screen=l1_res,
        participation_audit=audit,
    )
