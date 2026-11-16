"""Reproducible May-Oct 2026 CFT study orchestrator: canonical replay only.

All wallet and broker state read-only. Files only under trading-ops/reports.
No bespoke copy engine, Guardian kernel, optimizer or Monte Carlo implementation.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_replay as cr
import pod_constructor as pc

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite"
ARCHIVE = Path.home() / ".local/share/copytrade/research"
STORE = Path("/home/quant/dev/quant/ml4t-data/.data/store")
VIKE = Path("/home/quant/dev/copytrade-pm/data/private/backtest_20261112/bars_vike")
HL_PREFIX = ROOT / "reports/2026-11-16__cft__10k__hyperliquid-only-v5"
LEGACY_PREFIX = ROOT / "reports/2026-11-16__cft__10k__exhaustive-v3"
PREFIX = HL_PREFIX
VENUE_MODE = "hyperliquid"
START = 1_777_593_600_000  # 2026-05-01T00:00Z
SPLIT = START + 123 * 86_400_000  # 2026-09-01T00:00Z
END = SPLIT + 61 * 86_400_000  # 2026-11-01T00:00Z
GRID = np.arange(START, END, 300_000, dtype=np.int64)
CUT = int(np.searchsorted(GRID, SPLIT))
TRAIN_GRID, OOS_GRID = GRID[:CUT], GRID[CUT:]
TRIAD = ("ACT04", "ACT08", "ACT09")
TRIAD_K = (0.00045, 0.00356, 0.00099)
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"
MIN_TRAIN_FILLS = 15
MAX_TRAIN_SHORTLIST = 30


def dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=json_float) + "\n")


def json_float(obj: object) -> float:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj).__name__)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def paths_inventory() -> tuple[list[str], dict]:
    full = glob.glob(str(ARCHIVE / "reservoir-full-history-20261115/hyperliquid/fills/day=2026-0[56]-*/data.parquet"))
    monitored = glob.glob(str(ARCHIVE / "reservoir-monitored-plus-six-hyperliquid-*/fills/day=2026-*/data.parquet"))
    monitored = [p for p in monitored if "2026-07-01" <= p.split("day=")[-1][:10] < "2026-11-01"]
    by_day = {}
    duplicates = []
    for path in sorted(full + monitored):
        day = path.split("day=")[-1][:10]
        if day in by_day:
            duplicates.append(day)
            continue
        by_day[day] = path
    paths = [by_day[day] for day in sorted(by_day)]
    required = {datetime.fromtimestamp(ms / 1000, UTC).date().isoformat() for ms in GRID[::288]}
    missing = sorted(required - set(by_day))
    manifest = {
        "count_daily_files": len(paths),
        "duplicate_date_sources": duplicates,
        "missing_dates": missing,
        "files": [{"day": p.split("day=")[-1][:10], "path": p, "bytes": os.stat(p).st_size, "sha256": sha256_file(p)} for p in paths],
    }
    manifest["archive_sha256"] = hashlib.sha256(
        "".join(x["sha256"] for x in manifest["files"]).encode()
    ).hexdigest()
    return paths, manifest


def load_fills(records: list[cr.WalletRecord], paths: list[str]) -> dict[str, dict[str, pl.DataFrame]]:
    conn = duckdb.connect(":memory:")
    conn.execute("SET threads=4")
    rows = conn.execute(
        """
        SELECT lower(wallet) wallet, coin, epoch_ms(exchange_ts) AS ts,
               IF(side='buy',1,-1) AS sign, CAST(size AS DOUBLE) sz,
               CAST(start_position AS DOUBLE) pre, CAST(trade_id AS VARCHAR) tid
        FROM read_parquet(?, hive_partitioning=false)
        WHERE lower(wallet) IN (SELECT unnest(?::VARCHAR[]))
          AND exchange_ts >= TIMESTAMPTZ '2026-05-01'
          AND exchange_ts < TIMESTAMPTZ '2026-11-01'
          AND dex='hyperliquid'
        ORDER BY wallet,coin,ts,tid
        """,
        [paths, [r.address for r in records]],
    ).fetchall()
    conn.close()
    print("FILLS_ROWS",len(rows),flush=True)
    df = pl.DataFrame(
        rows,
        schema=["wallet", "coin", "ts", "sign", "sz", "pre", "tid"],
        orient="row",
    )
    del rows
    if df.is_empty():
        raise RuntimeError("EMPTY_ARCHIVE")
    records_by_addr = {r.address: r.label for r in records}
    grouped: dict[str, dict[str, pl.DataFrame]] = defaultdict(dict)
    for (address, coin), part in df.group_by(["wallet", "coin"]):
        if address in records_by_addr and cr.is_hl_perp_coin(coin):
            grouped[records_by_addr[address]][coin] = (
                part.select("ts", "sign", "sz", "pre", "tid")
                .sort(["ts", "tid"])
            )
    return grouped


def bybit_instruments() -> tuple[set[str], dict]:
    coins: set[str] = set()
    cursor = ""
    pages = 0
    for _ in range(12):
        uri = "https://api.bybit.com/v5/market/instruments-info?category=linear&status=Trading&limit=1000"
        if cursor:
            uri += "&cursor=" + urllib.parse.quote(cursor)
        req = urllib.request.Request(uri, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=18) as resp:
            obj = json.load(resp)
        if int(obj.get("retCode", -1)) != 0:
            raise RuntimeError("BYBIT_INSTRUMENTS_ERROR")
        data = obj["result"]
        pages += 1
        coins.update(
            row["baseCoin"] for row in data["list"]
            if row.get("quoteCoin") == "USDT"
            and row.get("contractType") == "LinearPerpetual"
        )
        cursor = data.get("nextPageCursor") or ""
        if not cursor:
            break
    if not coins or cursor:
        raise RuntimeError("INCOMPLETE_BYBIT_PAGINATION")
    return coins, {"pages": pages, "base_coins": len(coins), "lookup_utc": datetime.now(UTC).isoformat()}


def request_wallet_record(rec: cr.WalletRecord) -> tuple[float | None, dict | None, str | None]:
    address = rec.address
    try:
        req = urllib.request.Request(
            "https://api.hyperliquid.xyz/info",
            data=json.dumps({"type": "clearinghouseState", "user": address}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            state = json.load(response)
        account_value = float(state["marginSummary"]["accountValue"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, None, type(exc).__name__
    return account_value, None, None


def hyperx_metrics(rec: cr.WalletRecord) -> dict:
    try:
        req = urllib.request.Request(
            f"https://data-api.hyperx.trade/wallet_metrics/{rec.address}?period=30d",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            metrics = json.load(response).get("metrics", {})
        return {
            "pnl_30d": round(float(metrics.get("total_realized_pnl", 0)), 2),
            "trades_30d": int(metrics.get("total_complete_trade_count", 0)),
            "win_rate_30d": round(float(metrics.get("win_rate", 0)), 4),
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {"status": "UNPROVEN", "error": type(exc).__name__}


def summarize_wallet(
    rec: cr.WalletRecord,
    fills: dict[str, pl.DataFrame],
    bybit: set[str] | None = None,
) -> dict:
    """Screen source activity using Train only.

    ``bybit=None`` is the Hyperliquid-only research contract. It accepts
    every main-DEX perp with a Train fill, without reading Bybit listings,
    present-day source account value, or OOS activity for selection.
    """
    train_coins = {
        coin: df.filter(pl.col("ts") < SPLIT)
        for coin, df in fills.items()
    }
    if bybit is None:
        mapped_coins = sorted(
            coin for coin, df in train_coins.items() if not df.is_empty()
        )
        symbol_status = (
            "HYPERLIQUID_TRAIN_AVAILABLE" if mapped_coins else "REJECT_NO_FILLS"
        )
        unmapped_coins: list[str] = []
        unmapped_fill_frac = 0.0
        unmapped_notional_frac = 0.0
        scale_alias_candidates: list[str] = []
    else:
        legacy = cr.assess_bybit_symbol_screen(rec.label, train_coins, bybit)
        mapped_coins = list(legacy.mapped_coins)
        symbol_status = legacy.status
        unmapped_coins = list(legacy.unmapped_coins)
        unmapped_fill_frac = round(legacy.unmapped_fill_frac, 5)
        unmapped_notional_frac = legacy.unmapped_notional_frac
        scale_alias_candidates = legacy.scale_alias_candidates
    all_t = (
        np.concatenate([x["ts"].to_numpy() for x in fills.values()])
        if fills else np.array([], dtype=np.int64)
    )
    train_count = int(np.count_nonzero(all_t < SPLIT))
    oos_count = int(np.count_nonzero(all_t >= SPLIT))
    first_ms = int(all_t.min()) if len(all_t) else None
    last_train_ms = int(all_t[all_t < SPLIT].max()) if train_count else None
    last_ms = int(all_t.max()) if len(all_t) else None
    reasons = []
    if symbol_status in ("REJECT_UNMAPPED_MATERIAL", "REJECT_NO_FILLS"):
        reasons.append(symbol_status)
    if train_count < MIN_TRAIN_FILLS:
        reasons.append("TRAIN_FILLS_LT_15")
    if last_train_ms is not None and last_train_ms < SPLIT - 45 * 86_400_000:
        reasons.append("TRAIN_INACTIVE_45D")
    return {
        "alias": rec.label, "configured": rec.configured, "excluded": rec.excluded,
        "train_fills": train_count, "oos_fills": oos_count,
        "total_fills": int(all_t.size), "first_fill_ms": first_ms,
        "last_train_fill_ms": last_train_ms, "last_fill_ms": last_ms,
        "symbol_status": symbol_status, "mapped_coins": mapped_coins,
        "unmapped_coins": unmapped_coins,
        "unmapped_fill_frac": unmapped_fill_frac,
        "unmapped_notional_frac": unmapped_notional_frac,
        "scale_alias_candidates": scale_alias_candidates,
        "account_value_usd": None, "account_value_status": "NOT_FETCHED",
        "hyperx": None, "reasons": reasons,
    }


def prepare_screen() -> tuple[dict, dict[str, dict[str, pl.DataFrame]], list[cr.WalletRecord]]:
    records, roster_hash = cr.load_registry_universe(REGISTRY, include_aliases=TRIAD)
    paths, inv = paths_inventory()
    print("ROSTER",len(records),roster_hash,"ARCHIVE_DAYS",len(paths),"MISSING",len(inv["missing_dates"]),flush=True)
    if inv["missing_dates"]:
        raise RuntimeError(f"MISSING_SOURCE_DAYS {inv['missing_dates'][:10]}")
    grouped = load_fills(records, paths)
    if VENUE_MODE == "hyperliquid":
        # Zero Bybit or current-state API requests in historical research.
        exchange = {"venue": "HYPERLIQUID_MAIN_DEX", "bybit_required": False}
        screen = [summarize_wallet(r, grouped.get(r.label, {})) for r in records]
        print("HL_TRAIN_ELIGIBLE", sum(not r["reasons"] for r in screen), flush=True)
    else:
        bybit, exchange = bybit_instruments()
        screen = [
            summarize_wallet(r, grouped.get(r.label, {}), bybit) for r in records
        ]
        requests = [
            r for r in records
            if not next(x for x in screen if x["alias"] == r.label)["reasons"]
        ]
        print("SYMBOL_AND_TRAIN_ELIGIBLE", len(requests), "BYBIT", len(bybit), flush=True)
        fetched: dict[str, tuple] = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(request_wallet_record, r): r.label for r in requests
            }
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    fetched[label] = fut.result()
                except (OSError, TypeError, ValueError):
                    fetched[label] = (None, None, "REQUEST_FAILED")
        for row in screen:
            if row["alias"] in fetched:
                av, _, err = fetched[row["alias"]]
                row["account_value_usd"] = round(av, 2) if av is not None else None
                row["account_value_status"] = (
                    "VERIFIED" if av is not None else "UNPROVEN:" + str(err)
                )
                if av is not None and av < 1000:
                    row["reasons"].append("WITHDRAWN_LT_1000")
                if av is None:
                    row["reasons"].append("ACCOUNT_VALUE_UNPROVEN")
    survivors = [x["alias"] for x in screen if not x["reasons"]]
    print("STATUS_ELIGIBLE", len(survivors), flush=True)
    result = {
        "study":"May-Nov 2026 CFT $10k canonical",
        "status":"RETROSPECTIVE_SCENARIO",
        "window_start":"2026-05-01T00:00Z",
        "train_end":"2026-09-01T00:00Z",
        "window_end":"2026-11-01T00:00Z",
        "roster_sha256":roster_hash,
        "roster_count":len(records),
        "source":inv,
        "bybit": exchange if VENUE_MODE == "legacy-bybit" else None,
        "backtest_venue": VENUE_MODE,
        "historical_broker_screen": "NONE" if VENUE_MODE == "hyperliquid" else "CURRENT_BYBIT",
        "market_source": "HYPERLIQUID_1M_OHLCV",
        "cost_bps_each_execution":6.5,
        "screen":screen,
        "screen_survivors":survivors,
        "symbol_screen_basis": ("HYPERLIQUID_MAIN_DEX_TRAIN_ONLY" if VENUE_MODE == "hyperliquid"
                                else "TRAIN_ONLY_FILLS_BEFORE_2026_09_01"),
        "pit_roster_provenance":"UNPROVEN_PRESENT_DAY_REGISTRY",
        "source_cwsa_checkpoint_provenance":"UNPROVEN_HISTORICAL_FILLS_ONLY",
    }
    dump(PREFIX.with_name(PREFIX.name+"__inventory.json"),result)
    return result,grouped,records


class Prices:
    """Historical HL 1m price adapter; no execution or Guardian calculations."""
    def __init__(self, grid: np.ndarray):
        self.grid=grid
        self.cache: dict[str, tuple[np.ndarray,np.ndarray]]={}
        self.stats: dict[str,dict]={}

    def get(self, coin: str) -> tuple[np.ndarray,np.ndarray]:
        if coin in self.cache:
            return self.cache[coin]
        paths=[]
        for m in range(5,11):
            paths.extend(glob.glob(str(STORE / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{coin}_PERP" / "year=2026" / f"month={m:02d}" / "day=*" / "data.parquet")))
        if not paths and not (VIKE/f"{coin}.parquet").exists():
            trade=np.full(self.grid.size,np.nan)
            mark=trade.copy()
            self.cache[coin]=(trade,mark)
            self.stats[coin]={"status":"NO_LOCAL_OR_VIKE_PRICE","mark_valid_frac":0.0,"trade_valid_frac":0.0,"store_days":0}
            return trade,mark
        parts=[]
        for f in sorted(paths):
            try:
                parts.append(pl.read_parquet(f).select("timestamp","open","close").with_columns(pl.col("timestamp").cast(pl.Datetime("ms","UTC"))))
            except (OSError,ValueError,pl.exceptions.PolarsError):
                continue
        vike=VIKE/f"{coin}.parquet"
        if vike.exists():
            parts.append(pl.read_parquet(vike).select(
                pl.from_epoch("ts_ms",time_unit="ms").dt.replace_time_zone("UTC").cast(pl.Datetime("ms","UTC")).alias("timestamp"),
                "open","close",
            ))
        if not parts:
            trade=np.full(self.grid.size,np.nan);mark=trade.copy()
            self.cache[coin]=(trade,mark)
            self.stats[coin]={"status":"INVALID_PRICE_INPUT","mark_valid_frac":0.0,"trade_valid_frac":0.0,"store_days":0}
            return trade,mark
        bars=pl.concat(parts).unique("timestamp",keep="first").sort("timestamp")
        ts=bars["timestamp"].dt.epoch("ms").to_numpy()
        opens=bars["open"].to_numpy()
        closes=bars["close"].to_numpy()
        # Causal trade: 1m open after checkpoint decision, valuation: last close <= bar T+4.
        ix=np.searchsorted(ts,self.grid+60_000)
        ok=(ix<ts.size)&(ts[np.clip(ix,0,ts.size-1)]==self.grid+60_000)
        trade=np.where(ok,opens[np.clip(ix,0,ts.size-1)],np.nan)
        e=self.grid+240_000
        last=np.searchsorted(ts,e,side="right")-1
        age=np.where(last>=0,e-ts[np.clip(last,0,ts.size-1)],10**15)
        valid=(last>=0)&(age<=180*60_000)
        mark=np.where(valid,closes[np.clip(last,0,ts.size-1)],np.nan)
        self.cache[coin]=(trade,mark)
        self.stats[coin]={
            "status":"STORE_PLUS_VIKE" if vike.exists() else "STORE_ONLY",
            "mark_valid_frac":round(float(np.mean(np.isfinite(mark))),5),
            "trade_valid_frac":round(float(np.mean(np.isfinite(trade))),5),
            "train_mark_valid_frac":round(float(np.mean(np.isfinite(mark[:CUT]))),5),
            "oos_mark_valid_frac":round(float(np.mean(np.isfinite(mark[CUT:]))),5),
            "store_days":len(paths),
        }
        return trade,mark


def eligible_fills(
    grouped: dict[str,dict[str,pl.DataFrame]],
    rows: list[dict],
    prices: Prices,
) -> tuple[dict[str,dict[str,pl.DataFrame]], list[dict]]:
    chosen={}
    status_by_alias={r["alias"]:r for r in rows}
    for alias in [r["alias"] for r in rows if not r.get("historical_reasons", r["reasons"])]:
        row=status_by_alias[alias]
        mapped=row["mapped_coins"]
        usable={}
        no_price=0
        for coin in mapped:
            _tr,mk=prices.get(coin)
            if np.count_nonzero(np.isfinite(mk[:CUT]))==0:
                no_price+=int((grouped[alias][coin]["ts"]<SPLIT).sum())
                continue
            usable[coin]=grouped[alias][coin]
        row["price_missing_train_fills"]=no_price
        row["price_missing_train_frac"]=round(no_price/max(1,row["train_fills"]),5)
        if row["price_missing_train_frac"]>0.05 or not usable:
            row["reasons"].append("PRICE_MISSING_MATERIAL")
            row.setdefault("historical_reasons", []).append("PRICE_MISSING_MATERIAL")
        else:
            chosen[alias]=usable
    return chosen,rows


def classify_wallet_eligibility(row: dict) -> dict:
    """Separate historical Train gating from present-day operational admission."""
    operational_flags = {"WITHDRAWN_LT_1000", "ACCOUNT_VALUE_UNPROVEN"}
    historical = [x for x in row["reasons"] if x not in operational_flags]
    operational = [x for x in row["reasons"] if x in operational_flags]
    return {
        "historical_reasons": historical,
        "operational_reasons": operational,
        "historical_eligible": not historical,
        "operational_eligible": not operational,
    }


def cap_primary_multipliers(optimized: list[float], baseline: list[float]) -> list[float]:
    """Never let a Train optimizer override the individual half-risk budget."""
    if len(optimized) != len(baseline):
        raise ValueError("SIZING_LENGTH_MISMATCH")
    if any(not np.isfinite(x) or x < 0 for x in optimized + baseline):
        raise ValueError("INVALID_SIZING_PARAMETER")
    return [min(a, b) for a, b in zip(optimized, baseline)]


def train_passable(
    financial_rows: list[dict], *, venue_mode: str
) -> list[dict]:
    """Exclude current account-value/survival filters from HL Train ranking."""
    if venue_mode not in ("hyperliquid", "legacy-bybit"):
        raise ValueError("UNKNOWN_BACKTEST_VENUE")
    return [
        row for row in financial_rows
        if row["net_usd"] > 0 and row["maxdd_usd"] > 0
        and (venue_mode == "hyperliquid" or row["operational_eligible"])
    ]


def train_coverage_and_shortlist(
    inventory: dict, grouped: dict[str, dict[str, pl.DataFrame]],
    records: list[cr.WalletRecord],
) -> tuple[list[str], dict, Prices, dict[str, dict[str, pl.DataFrame]]]:
    """Replay all historical Train-eligible wallets, excluding OOS from selection."""
    prices = Prices(GRID)
    historical_rows = []
    for row in inventory["screen"]:
        row.update(classify_wallet_eligibility(row))
        if row["historical_eligible"]:
            historical_rows.append(row)
    print("HISTORICAL_ELIGIBLE_BEFORE_PRICE", len(historical_rows), flush=True)
    eligible, updated = eligible_fills(grouped, historical_rows, prices)
    checkpoint = PREFIX.with_name(PREFIX.name + "__train_progress.json")
    progress = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    if progress and (progress.get("roster_sha256") != inventory["roster_sha256"]
                     or progress.get("archive_sha256") != inventory["source"]["archive_sha256"]):
        raise RuntimeError("CHECKPOINT_INPUT_DRIFT")
    completed = progress.get("wallets", {})
    results = []
    for i, (alias, coins) in enumerate(sorted(eligible.items())):
        if alias in completed:
            results.append(completed[alias])
            continue
        if i % 5 == 0:
            print("TRAIN_FINANCIAL_REPLAY", i, "of", len(eligible), flush=True)
        frames, diag = cr.build_unit_replay_by_alias(
            {alias: coins},
            {c: tuple(x[:CUT] for x in prices.get(c)) for c in coins},
            [alias], TRAIN_GRID, policy="new", activation_ms=START,
        )
        totals = np.zeros(TRAIN_GRID.size, dtype=np.float64)
        active = 0
        for frame in frames[alias].values():
            totals += frame["net"].to_numpy()
            active += int(np.count_nonzero(np.abs(frame["pos"].to_numpy()) > 1e-12))
        eq = np.cumsum(totals)
        peak = np.maximum.accumulate(np.maximum(eq, 0.0))
        dd = float(np.max(peak - eq)) if eq.size else 0.0
        net = float(totals.sum())
        august = float(totals[92 * 288:].sum())
        record = next(r for r in updated if r["alias"] == alias)
        results.append({
            "alias": alias, "net_usd": round(net, 4), "maxdd_usd": round(dd, 4),
            "net_over_dd": round(net / max(dd, 0.01), 6),
            "last_train_month_net": round(august, 4),
            "active_coin_bars": active,
            "gap": diag[alias], "available_coins": len(coins),
            "train_fills": record["train_fills"],
            "historical_eligible": True,
            "operational_eligible": record["operational_eligible"],
        })
        completed[alias] = results[-1]
        dump(checkpoint, {
            "roster_sha256": inventory["roster_sha256"],
            "archive_sha256": inventory["source"]["archive_sha256"],
            "wallets": completed,
        })
        del frames
    passable = train_passable(results, venue_mode=VENUE_MODE)
    ranking = sorted(passable, key=lambda r: (r["net_over_dd"], r["net_usd"]), reverse=True)
    resilience = sorted(passable, key=lambda r: (r["last_train_month_net"], r["net_usd"]), reverse=True)
    frequency = sorted(passable, key=lambda r: (r["train_fills"], r["net_over_dd"]), reverse=True)
    low_turnover = sorted(passable, key=lambda r: (r["train_fills"], -r["net_over_dd"]))
    shortlist = []
    for pool, quota in ((ranking, 15), (resilience, 10), (frequency, 5), (low_turnover, 5)):
        for row in pool:
            if len(shortlist) >= MAX_TRAIN_SHORTLIST or quota <= 0:
                break
            if row["alias"] not in shortlist:
                shortlist.append(row["alias"])
                quota -= 1
    for row in ranking:
        if len(shortlist) >= MAX_TRAIN_SHORTLIST:
            break
        if row["alias"] not in shortlist:
            shortlist.append(row["alias"])
    inventory["train_price_replay_policy"] = "ALL_HISTORICALLY_ELIGIBLE_FINANCIAL_REPLAY"
    inventory["shortlist_current_account_value_filter"] = (
        VENUE_MODE != "hyperliquid"
    )
    inventory["train_price_replay_advanced"] = sorted(eligible)
    inventory["train_stats"] = results
    inventory["shortlist"] = shortlist
    inventory["screen"] = [
        next((u for u in updated if u["alias"] == r["alias"]), r)
        for r in inventory["screen"]
    ]
    inventory["price_source_stats"] = prices.stats
    inventory["screen_survivors_after_prices"] = sorted(eligible)
    inventory["historical_screen_eligible_before_prices"] = len(historical_rows)
    inventory["financial_replay_count"] = len(results)
    inventory["financial_coverage"] = {
        "numerator": len(results), "denominator": len(eligible),
        "ratio": len(results) / max(1, len(eligible)),
    }
    inventory["selection_uses_only_train_returns"] = True
    dump(PREFIX.with_name(PREFIX.name + "__inventory.json"), inventory)
    print("FINANCIAL_COVERAGE", len(results), "SHORTLIST", len(shortlist), flush=True)
    return shortlist, inventory, prices, eligible


def build_train_tensor(
    aliases:list[str],eligible:dict[str,dict[str,pl.DataFrame]],prices:Prices
) -> tuple[pc.UnitCopy5mTensor,dict]:
    active={a:eligible[a] for a in aliases}
    symbols=sorted({c for by in active.values() for c in by})
    frames,diagnostics=cr.build_unit_replay_by_alias(
        active,{c:tuple(x[:CUT] for x in prices.get(c)) for c in symbols},aliases,TRAIN_GRID,
        policy="new",activation_ms=START,
    )
    tensor=pc.build_tensor_from_unit_replays(
        frames,TRAIN_GRID,pod_equity_usd=10_000.0,
        chop_split_ms=int(TRAIN_GRID[92*288]),
    )
    del frames
    return tensor,diagnostics


def train_pods(tensor:pc.UnitCopy5mTensor) -> tuple[list[dict],dict]:
    selected=[]; auditing={}
    for size in (3,4):
        conf=dataclasses.replace(
            pc.PodConstructorConfig(),pod_size=size,top_per_cluster=3,max_candidate_pods=10,
            min_active_bar_frac=0.005,
        )
        try:
            sized,metrics,_,components=pc.discover_diverse_pods(tensor,conf)
        except ValueError as exc:
            auditing[str(size)]={"status":"INSUFFICIENT", "error":str(exc)}
            continue
        auditing[str(size)]={
            "passed": [x.alias for x in metrics if x.passed_behavioral_gate],
            "rejected":[{"alias":x.alias,"reason":x.reject_reason,"tilt":round(x.tilt_multiplier,3)} for x in metrics if not x.passed_behavioral_gate],
            "distance_shape":list(components["distance_matrix"].shape),
            "sized":len(sized),
            "candidate_pods": [list(x.aliases) for x in sized],
        }
        for s in sized:
            # Calibrate 50% individual risk target on Train, report divergence from optimized k.
            unit_dd=[metrics[i].standalone_max_dd * tensor.pod_equity_usd for i in s.pod_indices]
            base=[0.5*(1000/size)/max(dd,0.01) for dd in unit_dd]
            selected.append({
                "aliases":list(s.aliases),
                "train_opt_k":[float(x) for x in s.multipliers],
                "train_half_risk_k":[float(x) for x in base],
                "train_unit_dd_usd":[float(x) for x in unit_dd],
                "feasible":s.feasible,
                "train_net_usd":round(s.objective_net_return*10_000,2),
                "train_peak_dd_usd":round(s.max_static_dd*10_000,2),
                "train_daily_cvar_usd":round(s.daily_cvar_99*10_000,2),
                "train_worst_day_usd":round(s.max_daily_loss*10_000,2),
                "train_net_dd":round(s.net_over_max_dd,4),
                "netting_savings_usd":round(s.netting_savings_frac*10_000,2),
                "binding_constraint":s.binding_constraint,
                "min_lot_warnings":list(s.min_lot_warnings),
                "size":size,
                "indices":list(s.pod_indices),
            })
    # Freeze PRIMARY sizes no higher than the individual 50%-risk baseline.
    for row in selected:
        row["aggressive_unconstrained_k"] = row["train_opt_k"][:]
        row["train_opt_k"] = cap_primary_multipliers(
            row["train_opt_k"], row["train_half_risk_k"]
        )
        row["aggressive_exceeds_baseline"] = any(
            a > b * (1 + 1e-9)
            for a, b in zip(row["aggressive_unconstrained_k"], row["train_half_risk_k"])
        )
    selected.sort(key=lambda x:(x["feasible"],x["train_net_dd"],x["train_net_usd"]),reverse=True)
    candidates=[];seen=[]
    for row in selected:
        aset=set(row["aliases"])
        if any(len(aset & s)>=min(len(aset),len(s))-1 for s in seen):
            continue
        seen.append(aset)
        candidates.append(row)
        if len(candidates)>=5:
            break
    if len(candidates)<3:
        # Keep optimizer output (including failures), never fabricate feasible selections.
        for row in selected:
            if row not in candidates:
                candidates.append(row)
            if len(candidates)>=3:
                break
    return candidates,auditing


def oos_source_books(
    selections: list[dict],
    full_source: dict[str, dict[str, pl.DataFrame]],
    train_books: dict[str, dict[str, pl.DataFrame]],
) -> dict[str, dict[str, pl.DataFrame]]:
    """Activate newly traded symbols only *after* Train membership and k freeze.

    Hyperliquid OOS follows every main-DEX perp coin from the selected wallet,
    including one first traded after the split. Such late symbols cannot
    influence Train wallet screening, clustering, sizing, or selection.
    """
    if VENUE_MODE != "hyperliquid":
        return train_books
    members = {alias for row in selections for alias in row["aliases"]}
    if not members <= set(full_source):
        raise ValueError("FROZEN_MEMBER_MISSING_SOURCE")
    return {
        alias: {coin: fills for coin, fills in full_source[alias].items()
                if cr.is_hl_perp_coin(coin)}
        for alias in sorted(members)
    }


def candidate_targets(
    aliases:list[str], k:list[float], grouped:dict[str,dict[str,pl.DataFrame]],
    prices:Prices, grid:np.ndarray, policy:str,
) -> tuple[list[str],np.ndarray,np.ndarray,np.ndarray]:
    coins=sorted({c for a in aliases for c in grouped[a]})
    prices_by={c:prices.get(c) for c in coins}
    start=int(np.searchsorted(GRID,grid[0]))
    stop=start+len(grid)
    member_pos=np.zeros((len(aliases),len(coins),len(grid)))
    for ai,alias in enumerate(aliases):
        for ci,coin in enumerate(coins):
            f=grouped[alias].get(coin)
            if f is None:continue
            reconstruction=cr.reconstruct_positions_on_grid(
                f,grid,policy=policy,activation_ms=int(grid[0]),
            )
            member_pos[ai,ci]=reconstruction.pos
    net=cr.net_member_targets(member_pos,k)
    trade=np.vstack([prices_by[c][0][start:stop] for c in coins])
    mark=np.vstack([prices_by[c][1][start:stop] for c in coins])
    return coins,net,trade,mark


def result_dict(x:cr.ChallengeRunResult) -> dict:
    return {
        "verdict":x.verdict,"terminal_ts_ms":x.terminal_ts_ms,
        "terminal_net_usd":x.terminal_net_usd,"worst_daily_usd":x.worst_daily_usd,
        "static_floor_dd_usd":x.static_floor_dd_usd,
        "peak_to_trough_dd_usd":x.peak_to_trough_dd_usd,
        "sweeps":x.sweeps,"gap_exits":x.forced_gap_exits,
        "unpriced_open_steps":x.unpriced_open_steps,
        "deferred_notional_usd":x.deferred_notional_usd,
        "full_window_net_usd":x.full_window_net_usd,
        "full_window_peak_to_trough_dd_usd":x.full_window_peak_to_trough_dd_usd,
        "full_window_static_floor_dd_usd":x.full_window_static_floor_dd_usd,
        "full_window_sweeps":x.full_window_sweeps,
    }


def evaluate_one(
    row:dict, grouped:dict[str,dict[str,pl.DataFrame]], prices:Prices, grid:np.ndarray,
    *, montecarlo:bool, pm03:bool,
) -> dict:
    aliases=row["aliases"]
    k=row["train_opt_k"]
    out={}
    for policy in ("new","full"):
        coins,tgt,trade,mark=candidate_targets(aliases,k,grouped,prices,grid,policy)
        mask=(grid>=int(datetime(2026,10,1,tzinfo=UTC).timestamp()*1000))
        obj, blocks=cr.simulate_guardian_challenge(coins,tgt,trade,mark,grid,cr.GuardianSpec(),chop_mask=mask)
        data=result_dict(obj)
        if pm03 and policy=="new":
            # Small comparable subset reduces Decimal PM03 runtime; exact window disclosed.
            n=min(288*7,len(grid))
            try:
                pm=cr.reconcile_portfolio_with_pm03(
                    coins[:],grid[:n],tgt[:,:n],trade[:,:n],mark[:,:n],
                )
                check,_=cr.simulate_guardian_challenge(
                    coins,tgt[:,:n],trade[:,:n],mark[:,:n],grid[:n],
                    cr.GuardianSpec(profit_target=1e10,max_static_dd=1e10,
                                    daily_loss_limit=1e10,guardian_sweep_loss=1e10),
                )
                pm["tier1_7d_net"]=check.full_window_net_usd
                pm["drift_usd"]=round(pm["net_pnl"]-check.full_window_net_usd,2)
                pm["window_utc"]="first 7 OOS days; no Guardian crossings"
                data["pm03"]=pm
            except (ValueError,RuntimeError,TypeError) as exc:
                data["pm03"]={"status":"ERROR","detail":type(exc).__name__}
        if montecarlo:
            data["mc"]={}
            for mult in (1.0,1.5,2.0):
                _scaled, mblocks=(
                    (obj,blocks) if mult==1.0
                    else cr.simulate_guardian_challenge(
                        coins,tgt*mult,trade,mark,grid,cr.GuardianSpec(),chop_mask=mask,
                    )
                )
                for regime in ("Blended","75%-Chop"):
                    key=f"{mult:.1f}x/{regime}"
                    data["mc"][key]=pc.run_stationary_block_mc_on_daily_blocks(
                        mblocks,regime,n_paths=5000,horizon_days=90,mean_block_days=6.0,
                        rng=np.random.default_rng(20261116+int(mult*100)+(0 if regime=="Blended" else 1)),
                    )
        out[policy]=data
    return out


def conflict(reference:tuple, candidate:tuple) -> dict:
    rc,rq,_,_=reference
    cc,cq,_,_=candidate
    common=set(rc)&set(cc)
    count=0;hours=0.0;coins=[]
    for coin in sorted(common):
        opp=rq[rc.index(coin)]*cq[cc.index(coin)]< -1e-12
        n=int(np.count_nonzero(opp))
        if n:
            count+=int(np.sum(np.diff(opp.astype(np.int8),prepend=0)==1))
            hours+=5*n/60
            coins.append(coin)
    return {"events":count,"coin_hours":round(hours,2),"symbols":coins}


def main() -> None:
    start=time.monotonic()
    parser=argparse.ArgumentParser()
    parser.add_argument("--through",choices=("screen","train","all","evaluate"),default="all")
    parser.add_argument(
        "--venue-mode", choices=("hyperliquid", "legacy-bybit"),
        default="hyperliquid",
        help="Hyperliquid is research default; legacy Bybit mode reproduces v3 only",
    )
    args=parser.parse_args()
    global VENUE_MODE, PREFIX
    VENUE_MODE = args.venue_mode
    PREFIX = HL_PREFIX if VENUE_MODE == "hyperliquid" else LEGACY_PREFIX
    if args.through=="evaluate":
        inventory=json.loads(PREFIX.with_name(PREFIX.name+"__inventory.json").read_text())
        freeze=json.loads(PREFIX.with_name(PREFIX.name+"__freeze.json").read_text())
        records,roster_sha=cr.load_registry_universe(REGISTRY,include_aliases=TRIAD)
        if roster_sha != inventory["roster_sha256"] or roster_sha != freeze["roster_sha256"]:
            raise RuntimeError("FROZEN_ROSTER_SHA_CHANGED")
        if freeze["cutoff_ms"] != SPLIT or freeze["archive_sha256"] != inventory["source"]["archive_sha256"]:
            raise RuntimeError("FROZEN_ARCHIVE_OR_TRAIN_SPLIT_CHANGED")
        expected = "hyperliquid" if VENUE_MODE == "hyperliquid" else None
        if inventory.get("backtest_venue") != expected or freeze.get("backtest_venue") != expected:
            raise RuntimeError("FROZEN_RESEARCH_VENUE_MISMATCH")
        grouped=load_fills(records,[f["path"] for f in inventory["source"]["files"]])
        shortlist=freeze["shortlist"]
        selections=freeze["frozen_portfolios"]
        prices=Prices(GRID)
        rows={r["alias"]:r for r in inventory["screen"]}
        eligible={a:{c:grouped[a][c] for c in rows[a]["mapped_coins"] if c in grouped[a]} for a in shortlist}
        print("LOADED_FROZEN",len(selections),"ROSTER",roster_sha,flush=True)
    else:
        cached = PREFIX.with_name(PREFIX.name + "__inventory.json")
        if cached.exists():
            inventory = json.loads(cached.read_text())
            records, roster_sha = cr.load_registry_universe(REGISTRY, include_aliases=TRIAD)
            if roster_sha != inventory["roster_sha256"]:
                raise RuntimeError("CACHED_ROSTER_SHA_CHANGED")
            if inventory.get("backtest_venue") != (
                "hyperliquid" if VENUE_MODE == "hyperliquid" else None
            ):
                raise RuntimeError("CACHED_VENUE_MODE_MISMATCH")
            grouped = load_fills(records, [x["path"] for x in inventory["source"]["files"]])
        else:
            inventory, grouped, records = prepare_screen()
        if args.through == "screen":
            return
        shortlist,inventory,prices,eligible=train_coverage_and_shortlist(inventory,grouped,records)
        if len(shortlist)<4:
            raise RuntimeError("INSUFFICIENT_TRAIN_POSITIVE_WALLETS")
        tensor,diags=build_train_tensor(shortlist,eligible,prices)
        selections,audit=train_pods(tensor)
        if not selections:
            raise RuntimeError("NO_OPTIMIZABLE_PODS")
        print("FROZEN_CANDIDATES",[(p["aliases"],p["train_opt_k"],p["train_net_usd"]) for p in selections],flush=True)
        freeze={
            "freeze_time_utc":datetime.now(UTC).isoformat(),
            "cutoff_ms":SPLIT,"roster_sha256":inventory["roster_sha256"],
            "archive_sha256":inventory["source"]["archive_sha256"],
            "bybit":inventory["bybit"],
            "backtest_venue": ("hyperliquid" if VENUE_MODE == "hyperliquid" else None),
            "shortlist":shortlist,"train_audit":audit,
            "train_diagnostics":diags,
            "frozen_portfolios":selections,
            "cost_bps":6.5,
            "scientific_status":"RETROSPECTIVE_NO_PIT_ROSTER",
        }
        freeze_path = PREFIX.with_name(PREFIX.name + "__freeze.json")
        if freeze_path.exists():
            raise FileExistsError(
                "RESEARCH_ALREADY_FROZEN_DO_NOT_RESELECT_USE_THROUGH_EVALUATE"
            )
        dump(freeze_path, freeze)
        if args.through=="train":return
    # All OOS work below this point; never adjust frozen weights based on OOS outcome.
    # Newly traded Hyperliquid OOS symbols enter this OOS-only book now,
    # strictly AFTER Train shortlist and frozen k are committed.
    evaluation_books = oos_source_books(selections, grouped, eligible)
    reference=None
    if all(a in grouped for a in TRIAD):
        reference=candidate_targets(
            list(TRIAD),list(TRIAD_K),grouped,prices,OOS_GRID,"full",
        )
    progress_path = PREFIX.with_name(PREFIX.name + "__oos_progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if progress and (progress.get("roster_sha256") != inventory["roster_sha256"]
                     or progress.get("archive_sha256") != inventory["source"]["archive_sha256"]):
        raise RuntimeError("OOS_CHECKPOINT_INPUT_DRIFT")
    previous = {r["index"]: r for r in progress.get("results", [])}
    results=[]
    for idx,row in enumerate(selections):
        if idx + 1 in previous:
            print("OOS_REUSED_FROZEN", idx + 1, flush=True)
            results.append(previous[idx + 1])
            continue
        print("OOS_CANDIDATE",idx+1,row["aliases"],flush=True)
        train_result=evaluate_one(row,eligible,prices,TRAIN_GRID,montecarlo=False,pm03=False)
        oos_result=evaluate_one(row,evaluation_books,prices,OOS_GRID,montecarlo=True,pm03=idx<3)
        ctx=candidate_targets(row["aliases"],row["train_opt_k"],evaluation_books,prices,OOS_GRID,"new")
        result={"index":idx+1,"aliases":row["aliases"],"k":row["train_opt_k"],"baseline_k":row["train_half_risk_k"],
                    "train":train_result,"oos":oos_result,"optimization":{k:v for k,v in row.items() if k not in ("train_opt_k","train_half_risk_k","indices")}}
        if reference:
            result["triada_conflicts"]=conflict(reference,ctx)
        results.append(result)
        dump(progress_path, {
            "roster_sha256": inventory["roster_sha256"],
            "archive_sha256": inventory["source"]["archive_sha256"],
            "frozen_candidates": len(selections),
            "results": results,
        })
    cost_sensitivity = {}
    for row in selections:
        name = " + ".join(row["aliases"])
        coins, targets, opens, marks = candidate_targets(
            row["aliases"], row["train_opt_k"], evaluation_books, prices, OOS_GRID, "new"
        )
        cost_sensitivity[name] = {}
        for bps in (6.5, 10.0, 15.0):
            run, _ = cr.simulate_guardian_challenge(
                coins, targets, opens, marks, OOS_GRID,
                cr.GuardianSpec(cost_bps=bps),
            )
            cost_sensitivity[name][str(bps)] = {
                "net_usd": round(run.full_window_net_usd, 2),
                "first_barrier": run.verdict,
                "peak_dd_usd": round(run.peak_to_trough_dd_usd, 2),
                "worst_daily_usd": round(run.worst_daily_usd, 2),
            }
    ref_stats={}
    if reference:
        trow={"aliases":list(TRIAD),"train_opt_k":list(TRIAD_K)}
        ref_stats=evaluate_one(trow,grouped,prices,OOS_GRID,montecarlo=True,pm03=False)
        ref_stats["k_source"]="prop-production/bybit_executor.py static defaults, live runtime unverified"
    output={
        "status":"RETROSPECTIVE_SCENARIO",
        "backtest_venue":VENUE_MODE,
        "broker_execution_model": "NOT_MODELED_HYPERLIQUID_PROXY" if VENUE_MODE == "hyperliquid" else "RETROSPECTIVE_BYBIT_SCREEN",
        "train_symbols_from_train_only":True,
        "oos_new_symbols_loaded_after_freeze":VENUE_MODE == "hyperliquid",
        "calibration_frozen_before_oos":True,
        "roster_sha256":inventory["roster_sha256"],
        "source_archive_sha256":inventory["source"]["archive_sha256"],
        "train_window":["2026-05-01","2026-09-01"],
        "oos_window":["2026-09-01","2026-11-01"],
        "historical_oos_already_inspected": True,
        "prospective_oos_qualified": False,
        "screened_wallets":len(inventory["screen"]),
        "screen_eligible":len(inventory.get("screen_survivors_after_prices",[])),
        "shortlist_count":len(shortlist),
        "financial_replay_count":inventory.get("financial_replay_count"),
        "frozen_candidates":len(selections),
        "results":results,
        "cost_sensitivity":cost_sensitivity,
        "triada":ref_stats,
        "limitations":{
            "historic_known_at_roster":"UNPROVEN",
            "follower_bybit_fills":"UNPROVEN",
            "market_impact_funding":"UNPROVEN",
            "CWSA_historical_5m_snapshots":"UNPROVEN",
            "historical_bybit_symbol_eligibility": (
                "OUT_OF_SCOPE_HYPERLIQUID_RESEARCH" if VENUE_MODE == "hyperliquid" else "UNPROVEN"
            ),
            "hyperx_discovery_only":True,
            "pm03_subset":"first 7 OOS days; independent from Guardian",
        },
        "elapsed_s":round(time.monotonic()-start,2),
    }
    dump(PREFIX.with_name(PREFIX.name+"__results.json"),output)
    report=[
        "---","firm: crypto-fund-trader","account_usd: 10000","status: RESEARCH",
        "top_candidate: NOT_SCIENTIFICALLY_QUALIFIED",
        "script: studies/cft_20261116_exhaustive.py",
        "---","",
        f"# CFT-risk 5m research / {VENUE_MODE} market — May–November 2026",
        "",f"Full roster: {len(records)}; SHA256 {inventory['roster_sha256']}. Train May 1–Sep 1; OOS Sep 1–Nov 1 (exclusive).",
        "","All figures are retrospective modeled Hyperliquid-source follower net returns, not live broker fills or PIT portfolio selection.",
        f"Backtest venue = {VENUE_MODE}; challenge budget is a risk scenario only. Bybit listings and minimum order rules are NOT historical selection inputs when Hyperliquid-only.",
        "",
        "| Train rank | Members | Frozen k | Train new net/static floor DD | OOS new net/static floor DD | OOS full net/static floor DD | OOS worst daily | OOS sweeps | 90d MC 1x blended pass | Triad A conflict coin-h |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        train=r["train"]["new"]; new=r["oos"]["new"];full=r["oos"]["full"]
        report.append("| {idx} | {members} | {ks} | {tn:+.0f} / {td:.0f} | {on:+.0f} / {od:.0f} | {fn:+.0f} / {fd:.0f} | {wd:.0f} | {sw} | {mc:.2f}% | {conf:.1f} |".format(
            idx=r["index"],members=", ".join(r["aliases"]),
            ks=", ".join(f"{a}={v:.7g}" for a,v in zip(r["aliases"],r["k"])),
            tn=train["full_window_net_usd"],td=train["full_window_static_floor_dd_usd"],
            on=new["full_window_net_usd"],od=new["full_window_static_floor_dd_usd"],
            fn=full["full_window_net_usd"],fd=full["full_window_static_floor_dd_usd"],
            wd=new["worst_daily_usd"],sw=new["sweeps"],
            mc=new["mc"]["1.0x/Blended"]["p_pass"],conf=r.get("triada_conflicts",{}).get("coin_hours",0),
        ))
    report+=["","## Frozen sizing audit","The optimizer was fit exclusively to Train. Primary frozen k is capped at 50% individual wallet risk baselines; uncapped optimizer feasibility is diagnostic only."]
    for i,s in enumerate(selections,1):
        kcmp=", ".join(
            f"{alias} baseline={base:.8g}, optimized={opt:.8g} ({opt/max(base,1e-12):.1f}x)"
            for alias,base,opt in zip(s["aliases"],s["train_half_risk_k"],s["train_opt_k"])
        )
        report.append(f"- **Pod {i}**: {kcmp}; binding={s['binding_constraint']}; uncapped-optimizer-feasible={s['feasible']}; min-lot warnings: {', '.join(s['min_lot_warnings']) or 'none'}.")
    report+=["","## Monte Carlo sensitivity: strict new-position-only","Stationary 6-day blocks; 5,000 resamples per cell, 90-day horizon. These figures measure path-resampling sensitivity, not independently executable challenge success.","","| Pod | Multiplier | Blended pass / static fail / daily fail | 75%-chop pass / static fail / daily fail | Median pass days (blended) | Avg Guardian sweeps (blended) |","|---|---:|---:|---:|---:|---:|"]
    for r in results:
        for mult in ("1.0x","1.5x","2.0x"):
            b=r["oos"]["new"]["mc"][mult+"/Blended"]
            c=r["oos"]["new"]["mc"][mult+"/75%-Chop"]
            report.append(f'| {r["index"]} | {mult} | {b["p_pass"]:.2f}% / {b["p_fail_static"]:.2f}% / {b["p_fail_daily"]:.2f}% | {c["p_pass"]:.2f}% / {c["p_fail_static"]:.2f}% / {c["p_fail_daily"]:.2f}% | {b["med_days_pass"]} | {b["avg_sweeps"]:.2f} |')
    report += ["", "## OOS cost sensitivity (frozen new-only)",
               "| Pod | 6.5 bps net | 10 bps net | 15 bps net | 15 bps verdict |",
               "|---|---:|---:|---:|---|"]
    for row in selections:
        cost = cost_sensitivity[" + ".join(row["aliases"])]
        report.append(
            f"| {' + '.join(row['aliases'])} | {cost['6.5']['net_usd']:+.0f} "
            f"| {cost['10.0']['net_usd']:+.0f} "
            f"| {cost['15.0']['net_usd']:+.0f} "
            f"| {cost['15.0']['first_barrier']} |"
        )
    report+=["","## Model and coverage"]
    report.append(f"- {len(inventory['source']['files'])} complete daily Reservoir archives; {inventory['source']['archive_sha256']} content digest.")
    screen_label = (
        "Train activity / Hyperliquid-only symbol checks"
        if VENUE_MODE == "hyperliquid"
        else "Train activity / symbol / present account-value checks"
    )
    report.append(f"- {len(inventory['screen'])} registry wallets screened; {len(inventory.get('screen_survivors',[]))} passed {screen_label}; {len(inventory.get('train_price_replay_advanced',[]))} historically eligible for price screening, {len(inventory.get('screen_survivors_after_prices',[]))} price-replayed; {len(shortlist)} Train-shortlisted. Every price-eligible wallet was financially replayed on Train.")
    if VENUE_MODE == "hyperliquid":
        report.append("- Historical wallet and symbol Train eligibility uses only Hyperliquid main-DEX perp fills before cutoff; no Bybit listing/account-value lookup is performed. OOS-only coins are admitted only AFTER freeze.")
    else:
        report.append("- Legacy cross-venue symbol suitability uses Train-only fills and current Bybit availability; this is NOT a PIT Bybit listing.")
    report.append("- Prices: local Hyperliquid 1m OHLCV, local cached Vike fallback; trade at 1m open after 5m checkpoint and mark by checkpoint T+4 minutes, stale <=180m.")
    report.append("- Source: reconstructed Reservoir start_position + fills; pair-local stale-mark termination via canonical_replay.")
    report.append("- Test: Train-only multi-style clustering, dual-barrier sizing and frozen k; MC 5000 paths, stationary 6-day blocks, 90-day horizon.")
    report+=["","## Guardian, PM03 and conflicts"]
    for r in results:
        new=r["oos"]["new"];pm=new.get("pm03",{})
        report.append(f"- Candidate {r['index']}: first-barrier {new['verdict']}, peak-DD {new['peak_to_trough_dd_usd']}, worst daily {new['worst_daily_usd']}, sweep count {new['sweeps']}; PM03 drift {pm.get('drift_usd','UNPROVEN')}; Triad A reverse intervals {r.get('triada_conflicts',{}).get('events','UNPROVEN')}.")
    if ref_stats:
        report.append(f"- Triad A reference: new-only full-window {ref_stats['new']['full_window_net_usd']}, full-target {ref_stats['full']['full_window_net_usd']}; static k source unverified live.")
    explored_count = sum(
        group.get("sized", 0) for group in freeze["train_audit"].values()
        if isinstance(group, dict)
    )
    observed_best = max(
        results, key=lambda row: row["oos"]["new"]["full_window_net_usd"]
    )
    shared_members = sorted(set.intersection(
        *(set(p["aliases"]) for p in selections)
    )) if selections else []
    report += [
        "", "## Qualification and Train rank",
        "- The numbered portfolios were ranked and frozen using TRAIN ONLY; the post-hoc OOS winner is not promoted to a new pre-cutoff decision.",
        f"- Largest already-opened historical OOS new-only net: candidate {observed_best['index']} (USD {observed_best['oos']['new']['full_window_net_usd']:+.2f}); descriptive only.",
        f"- {len(selections)} frozen Train candidates, {explored_count} explored Train combinations. Bounded search, not global portfolio optimality.",
        "- All candidates have historical opposite-position overlaps with Triad A. Never combine them blindly on a reverse-position-restricted account.",
        "- Each historical eligible wallet with Train pricing underwent financial replay; the 30-wallet shortlist is Train-only.",
        "- OOS returns did not tune any frozen candidate k or membership.",
"- The optimizer's optional $10 notional warnings are diagnostics, not Hyperliquid execution constraints. Primary k respects individual 50%-risk calibration. Actual broker order rejection and follower fill economics remain UNPROVEN.","- If no candidate both survives OOS and has positive OOS net, no portfolio is economically qualified.","- The present-day wallet roster has no pre-September discovery proof, source first-known receipt clocks are unavailable historically, and September–October has already been inspected. This is RETROSPECTIVE_SCENARIO, not an independent prospective OOS.","- Full MC matrix, each wallet's eligibility/rejection reasons, trace hashes and all frozen parameters are in the companion JSON files.","","## Unmet qualification gates",
f"- CPCV/PBO was not executed; the bounded Train optimizer explored {explored_count} combinations, below the 20–40 target.",
"- Venue-specific lot quantities, step sizes, and funding were not integrated into follower replay; none is a selection gate for Hyperliquid-only backtests.",
"- Cost sensitivities at 10 and 15 bps were tested on frozen memberships and k after OOS; they are stress tests, not new independent holdouts.",
"- Entry-delay, leave-one-out, coin-removal and independent PIT roster stress checks remain incomplete.",
f"- Wallets shared by every frozen candidate: {', '.join(shared_members) or 'none'}. Shared-member risk is not proof of independence or broad style diversification.",
"- Formal statuses: ENGINEERING_VERIFIED for the scoped research code; RETROSPECTIVE_SCENARIO for returns; SCIENTIFICALLY_QUALIFIED=false; ECONOMICALLY_QUALIFIED=false; TRADING_AUTHORIZED=false.",
"",
"### Three-line caveat footer","- CFT trading restrictions and account framing require independent compliance confirmation.","- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.","- No deployments, allocations, orders, service or registry writes occurred.",""]
    PREFIX.with_name(PREFIX.name+".md").write_text("\n".join(report))
    print("SAVED_RESULT",str(PREFIX.with_name(PREFIX.name+"__results.json")),flush=True)


if __name__=="__main__":
    main()
