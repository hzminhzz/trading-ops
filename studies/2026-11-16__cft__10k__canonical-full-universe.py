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
PREFIX = ROOT / "reports/2026-11-16__cft__10k__canonical-full-universe-v2"
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
MAX_TRAIN_SHORTLIST = 9
MAX_PRICE_REPLAY_SCREEN = 24


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


def summarize_wallet(rec: cr.WalletRecord, fills: dict[str, pl.DataFrame], bybit: set[str]) -> dict:
    # Eligibility must not inspect OOS symbols: screen ONLY the pre-Sep fills.
    train_coins={coin:df.filter(pl.col("ts")<SPLIT) for coin,df in fills.items()}
    screen = cr.assess_bybit_symbol_screen(rec.label, train_coins, bybit)
    all_t = np.concatenate([x["ts"].to_numpy() for x in fills.values()]) if fills else np.array([], dtype=np.int64)
    train_count = int(np.count_nonzero(all_t < SPLIT))
    oos_count = int(np.count_nonzero(all_t >= SPLIT))
    first_ms = int(all_t.min()) if len(all_t) else None
    last_train_ms = int(all_t[all_t < SPLIT].max()) if train_count else None
    last_ms = int(all_t.max()) if len(all_t) else None
    reasons = []
    if screen.status in ("REJECT_UNMAPPED_MATERIAL", "REJECT_NO_FILLS"):
        reasons.append(screen.status)
    if train_count < MIN_TRAIN_FILLS:
        reasons.append("TRAIN_FILLS_LT_15")
    if last_train_ms is not None and last_train_ms < SPLIT - 45 * 86_400_000:
        reasons.append("TRAIN_INACTIVE_45D")
    return {
        "alias":rec.label,"configured":rec.configured,"excluded":rec.excluded,
        "train_fills":train_count,"oos_fills":oos_count,"total_fills":int(all_t.size),
        "first_fill_ms":first_ms,"last_train_fill_ms":last_train_ms,
        "last_fill_ms":last_ms,"symbol_status":screen.status,
        "mapped_coins":list(screen.mapped_coins),
        "unmapped_coins":list(screen.unmapped_coins),
        "unmapped_fill_frac":round(screen.unmapped_fill_frac,5),
        "unmapped_notional_frac":screen.unmapped_notional_frac,
        "scale_alias_candidates":screen.scale_alias_candidates,
        "account_value_usd":None,"account_value_status":"NOT_FETCHED",
        "hyperx":None,"reasons":reasons,
    }


def prepare_screen() -> tuple[dict, dict[str, dict[str, pl.DataFrame]], list[cr.WalletRecord]]:
    records, roster_hash = cr.load_registry_universe(REGISTRY, include_aliases=TRIAD)
    paths, inv = paths_inventory()
    print("ROSTER",len(records),roster_hash,"ARCHIVE_DAYS",len(paths),"MISSING",len(inv["missing_dates"]),flush=True)
    if inv["missing_dates"]:
        raise RuntimeError(f"MISSING_SOURCE_DAYS {inv['missing_dates'][:10]}")
    grouped = load_fills(records, paths)
    bybit, exchange = bybit_instruments()
    screen = [summarize_wallet(r, grouped.get(r.label, {}), bybit) for r in records]
    requests = [r for r in records if not next(x for x in screen if x["alias"] == r.label)["reasons"]]
    print("SYMBOL_AND_TRAIN_ELIGIBLE",len(requests),"BYBIT",len(bybit),flush=True)
    fetched: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=14) as pool:
        futures = {pool.submit(request_wallet_record, r):r.label for r in requests}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                fetched[label] = fut.result()
            except (OSError, TypeError, ValueError):
                fetched[label] = (None,None,"REQUEST_FAILED")
    for row in screen:
        if row["alias"] in fetched:
            av,_,err=fetched[row["alias"]]
            row["account_value_usd"] = round(av,2) if av is not None else None
            row["account_value_status"] = "VERIFIED" if av is not None else "UNPROVEN:" + str(err)
            if av is not None and av < 1000:
                row["reasons"].append("WITHDRAWN_LT_1000")
            if av is None:
                row["reasons"].append("ACCOUNT_VALUE_UNPROVEN")
    survivors=[x["alias"] for x in screen if not x["reasons"]]
    print("STATUS_ELIGIBLE",len(survivors),flush=True)
    result = {
        "study":"May-Nov 2026 CFT $10k canonical",
        "status":"RETROSPECTIVE_SCENARIO",
        "window_start":"2026-05-01T00:00Z",
        "train_end":"2026-09-01T00:00Z",
        "window_end":"2026-11-01T00:00Z",
        "roster_sha256":roster_hash,
        "roster_count":len(records),
        "source":inv,
        "bybit":exchange,
        "cost_bps_each_execution":6.5,
        "screen":screen,
        "screen_survivors":survivors,
        "symbol_screen_basis":"TRAIN_ONLY_FILLS_BEFORE_2026_09_01",
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
    for alias in [r["alias"] for r in rows if not r["reasons"]]:
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
        else:
            chosen[alias]=usable
    return chosen,rows


def train_coverage_and_shortlist(
    inventory:dict,grouped:dict[str,dict[str,pl.DataFrame]],records:list[cr.WalletRecord]
) -> tuple[list[str], dict, Prices, dict[str,dict[str,pl.DataFrame]]]:
    prices=Prices(GRID)
    # Deterministic TRAIN-only panel sampling. Deferred wallets are NOT financial rejects.
    raw=[r for r in inventory["screen"] if not r["reasons"]]
    most_active=sorted(raw,key=lambda x:(x["train_fills"],x["alias"]),reverse=True)
    least_active=sorted(raw,key=lambda x:(x["train_fills"],x["alias"]))
    freshest=sorted(raw,key=lambda x:(x["last_train_fill_ms"] or 0,x["alias"]),reverse=True)
    advanced=[]
    for order,limit in ((most_active,10),(least_active,7),(freshest,7)):
        for record in order[:limit]:
            if record["alias"] not in advanced:
                advanced.append(record["alias"])
    advanced=advanced[:MAX_PRICE_REPLAY_SCREEN]
    inventory["train_price_replay_policy"]="10 most-active, 7 least-active, 7 freshest Train-eligible wallets; deterministic and not exhaustive"
    inventory["train_price_replay_advanced"]=advanced
    for row in inventory["screen"]:
        if not row["reasons"] and row["alias"] not in advanced:
            row["replay_deferred"]="TRAIN_PANEL_CAP_24_NOT_FINANCIALLY_REJECTED"
    eligible,updated=eligible_fills(grouped,[r for r in inventory["screen"] if r["alias"] in advanced],prices)
    results=[]
    for i,(alias,coins) in enumerate(sorted(eligible.items())):
        if i%15==0:
            print("TRAIN_REPLAY",i,"of",len(eligible),flush=True)
        frames,diag=cr.build_unit_replay_by_alias(
            {alias:coins},{c:tuple(x[:CUT] for x in prices.get(c)) for c in coins},[alias],TRAIN_GRID,
            policy="new",activation_ms=START,
        )
        wallet_frames=frames[alias]
        if not wallet_frames:
            continue
        totals=np.zeros(TRAIN_GRID.size)
        for f in wallet_frames.values():
            totals+=f["net"].to_numpy()
        eq=np.cumsum(totals)
        peak=np.maximum.accumulate(np.maximum(eq,0))
        dd=float(np.max(peak-eq)) if eq.size else 0.0
        net=float(totals.sum())
        last_month=float(totals[92*288:].sum()) if len(totals)>92*288 else net
        info={"alias":alias,"net_usd":round(net,2),"maxdd_usd":round(dd,2),
                  "net_over_dd":round(net/max(dd,0.01),4),"last_train_month_net":round(last_month,2),
                  "gap":diag[alias],"available_coins":len(wallet_frames),"train_fills":next(x["train_fills"] for x in updated if x["alias"]==alias)}
        results.append(info)
        # Keep only derived summaries; avoid storing full-registry checkpoint matrices.
        del wallet_frames,frames
    # Train-only ranked style panel: positive solo net/DD plus August resilience.
    eligible_positive=[r for r in results if r["net_usd"]>0 and r["maxdd_usd"]>0]
    by_score=sorted(eligible_positive,key=lambda x:(x["net_over_dd"],x["net_usd"]),reverse=True)
    by_recent=sorted(eligible_positive,key=lambda x:(x["last_train_month_net"],x["net_over_dd"]),reverse=True)
    by_absolute=sorted(eligible_positive,key=lambda x:x["net_usd"],reverse=True)
    pick=[]
    for arr,limit in [(by_score,12),(by_recent,6),(by_absolute,6)]:
        for row in arr[:limit]:
            if row["alias"] not in pick and len(pick)<MAX_TRAIN_SHORTLIST:
                pick.append(row["alias"])
    print("TRAIN_POSITIVE",len(eligible_positive),"SHORTLIST",pick,flush=True)
    # HyperX is discovery context only and must not influence frozen Train ranking.
    rec_by_label={r.label:r for r in records}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(hyperx_metrics,rec_by_label[a]):a for a in pick}
        hx={futures[fu]:fu.result() for fu in as_completed(futures)}
    for row in updated:
        if row["alias"] in hx:
            row["hyperx"]=hx[row["alias"]]
    inventory["train_stats"]=results
    inventory["shortlist"]=pick
    inventory["screen"]=[next((u for u in updated if u["alias"]==r["alias"]),r) for r in inventory["screen"]]
    inventory["price_source_stats"]=prices.stats
    inventory["screen_survivors_after_prices"]=sorted(eligible)
    inventory["selection_uses_only_train_returns"]=True
    dump(PREFIX.with_name(PREFIX.name+"__inventory.json"),inventory)
    return pick,inventory,prices,eligible


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
            pc.PodConstructorConfig(),pod_size=size,top_per_cluster=2,max_candidate_pods=3,
            min_active_bar_frac=0.005,
        )
        try:
            sized,metrics,_,components=pc.discover_diverse_pods(tensor,conf)
        except ValueError as exc:
            auditing[str(size)]={"status":"INSUFFICIENT", "error":str(exc)}
            continue
        auditing[str(size)]={
            "passed": [x.alias for x in metrics if x.passed_behavioral_gate],
            "rejected":[
                {
                    "alias": x.alias,
                    "reason": x.reject_reason,
                    "tilt": round(x.tilt_multiplier, 3) if x.tilt_multiplier is not None else None,
                    "tilt_status": x.tilt_status,
                }
                for x in metrics if not x.passed_behavioral_gate
            ],
            "distance_shape":list(components["distance_matrix"].shape),
            "sized":len(sized),
        }
        for s in sized:
            # Calibrate 50% individual risk target on Train, report divergence from optimized k.
            unit_dd=[metrics[i].standalone_peak_to_trough_dd * tensor.pod_equity_usd for i in s.pod_indices]
            base=[0.5*(1000/size)/max(dd,0.01) for dd in unit_dd]
            selected.append({
                "aliases":list(s.aliases),
                "train_opt_k":[float(x) for x in s.multipliers],
                "train_half_risk_k":[float(x) for x in base],
                "train_unit_dd_usd":[float(x) for x in unit_dd],
                "feasible":s.feasible,
                "train_net_usd":round(s.objective_net_return*10_000,2),
                "train_peak_dd_usd":round(s.peak_to_trough_dd*10_000,2),
                "train_static_floor_dd_usd":round(s.static_floor_dd*10_000,2),
                "train_daily_cvar_usd":round(s.daily_cvar_99*10_000,2),
                "train_worst_day_usd":round(s.max_daily_loss*10_000,2),
                "train_net_dd":round(s.net_over_max_dd,4),
                "netting_savings_usd":round(s.netting_savings_frac*10_000,2),
                "binding_constraint":s.binding_constraint,
                "min_lot_warnings":list(s.min_lot_warnings),
                "size":size,
                "indices":list(s.pod_indices),
            })
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
    args=parser.parse_args()
    if args.through=="evaluate":
        inventory=json.loads(PREFIX.with_name(PREFIX.name+"__inventory.json").read_text())
        freeze=json.loads(PREFIX.with_name(PREFIX.name+"__freeze.json").read_text())
        records,roster_sha=cr.load_registry_universe(REGISTRY,include_aliases=TRIAD)
        if roster_sha != inventory["roster_sha256"] or roster_sha != freeze["roster_sha256"]:
            raise RuntimeError("FROZEN_ROSTER_SHA_CHANGED")
        if freeze["cutoff_ms"] != SPLIT or freeze["archive_sha256"] != inventory["source"]["archive_sha256"]:
            raise RuntimeError("FROZEN_ARCHIVE_OR_TRAIN_SPLIT_CHANGED")
        grouped=load_fills(records,[f["path"] for f in inventory["source"]["files"]])
        shortlist=freeze["shortlist"]
        selections=freeze["frozen_portfolios"]
        prices=Prices(GRID)
        rows={r["alias"]:r for r in inventory["screen"]}
        eligible={a:{c:grouped[a][c] for c in rows[a]["mapped_coins"] if c in grouped[a]} for a in shortlist}
        print("LOADED_FROZEN",len(selections),"ROSTER",roster_sha,flush=True)
    else:
        inventory,grouped,records=prepare_screen()
        if args.through=="screen":return
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
            "shortlist":shortlist,"train_audit":audit,
            "train_diagnostics":diags,
            "frozen_portfolios":selections,
            "cost_bps":6.5,
            "scientific_status":"RETROSPECTIVE_NO_PIT_ROSTER",
        }
        dump(PREFIX.with_name(PREFIX.name+"__freeze.json"),freeze)
        if args.through=="train":return
    # All OOS work below this point; never adjust frozen weights based on OOS outcome.
    reference=None
    if all(a in grouped for a in TRIAD):
        reference=candidate_targets(
            list(TRIAD),list(TRIAD_K),grouped,prices,OOS_GRID,"full",
        )
    results=[]
    for idx,row in enumerate(selections):
        print("OOS_CANDIDATE",idx+1,row["aliases"],flush=True)
        train_result=evaluate_one(row,eligible,prices,TRAIN_GRID,montecarlo=False,pm03=False)
        oos_result=evaluate_one(row,eligible,prices,OOS_GRID,montecarlo=True,pm03=idx<3)
        ctx=candidate_targets(row["aliases"],row["train_opt_k"],eligible,prices,OOS_GRID,"new")
        result={"index":idx+1,"aliases":row["aliases"],"k":row["train_opt_k"],"baseline_k":row["train_half_risk_k"],
                    "train":train_result,"oos":oos_result,"optimization":{k:v for k,v in row.items() if k not in ("train_opt_k","train_half_risk_k","indices")}}
        if reference:
            result["triada_conflicts"]=conflict(reference,ctx)
        results.append(result)
    ref_stats={}
    if reference:
        trow={"aliases":list(TRIAD),"train_opt_k":list(TRIAD_K)}
        ref_stats=evaluate_one(trow,grouped,prices,OOS_GRID,montecarlo=True,pm03=False)
        ref_stats["k_source"]="prop-production/bybit_executor.py static defaults, live runtime unverified"
    output={
        "status":"RETROSPECTIVE_SCENARIO",
        "calibration_frozen_before_oos":True,
        "roster_sha256":inventory["roster_sha256"],
        "source_archive_sha256":inventory["source"]["archive_sha256"],
        "train_window":["2026-05-01","2026-09-01"],
        "oos_window":["2026-09-01","2026-11-01"],
        "screened_wallets":len(inventory["screen"]),
        "screen_eligible":len(inventory.get("screen_survivors_after_prices",[])),
        "shortlist_count":len(shortlist),
        "frozen_candidates":len(selections),
        "results":results,
        "triada":ref_stats,
        "limitations":{
            "historic_known_at_roster":"UNPROVEN",
            "follower_bybit_fills":"UNPROVEN",
            "market_impact_funding":"UNPROVEN",
            "CWSA_historical_5m_snapshots":"UNPROVEN",
            "historical_bybit_symbol_eligibility":"UNPROVEN",
            "hyperx_discovery_only":True,
            "pm03_subset":"first 7 OOS days; independent from Guardian",
        },
        "elapsed_s":round(time.monotonic()-start,2),
    }
    dump(PREFIX.with_name(PREFIX.name+"__results.json"),output)
    report=[
        "---","firm: crypto-fund-trader","account_usd: 10000","status: RESEARCH",
        "top_candidate: NOT_SCIENTIFICALLY_QUALIFIED",
        "script: studies/2026-11-16__cft__10k__canonical-full-universe.py",
        "---","","# CFT canonical 5m portfolio research — May–November 2026",
        "",f"Full roster: {len(records)}; SHA256 {inventory['roster_sha256']}. Train May 1–Sep 1; OOS Sep 1–Nov 1 (exclusive).",
        "","All figures are retrospective modeled follower net returns, not live Bybit fills or PIT portfolio selection.",
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
    report+=["","## Frozen sizing audit","The optimizer was fit exclusively to Train. The 50%-wallet-risk k is an independent baseline; optimized k may be much larger. Such larger multipliers are NOT automatically authorized."]
    for i,s in enumerate(selections,1):
        kcmp=", ".join(
            f"{alias} baseline={base:.8g}, optimized={opt:.8g} ({opt/max(base,1e-12):.1f}x)"
            for alias,base,opt in zip(s["aliases"],s["train_half_risk_k"],s["train_opt_k"])
        )
        report.append(f"- **Pod {i}**: {kcmp}; binding={s['binding_constraint']}; optimizer-feasible={s['feasible']}; min-lot warnings: {', '.join(s['min_lot_warnings']) or 'none'}.")
    report+=["","## Monte Carlo sensitivity: strict new-position-only","Stationary 6-day blocks; 5,000 resamples per cell, 90-day horizon. These figures measure path-resampling sensitivity, not independently executable challenge success.","","| Pod | Multiplier | Blended pass / static fail / daily fail | 75%-chop pass / static fail / daily fail | Median pass days (blended) | Avg Guardian sweeps (blended) |","|---|---:|---:|---:|---:|---:|"]
    for r in results:
        for mult in ("1.0x","1.5x","2.0x"):
            b=r["oos"]["new"]["mc"][mult+"/Blended"]
            c=r["oos"]["new"]["mc"][mult+"/75%-Chop"]
            report.append(f'| {r["index"]} | {mult} | {b["p_pass"]:.2f}% / {b["p_fail_static"]:.2f}% / {b["p_fail_daily"]:.2f}% | {c["p_pass"]:.2f}% / {c["p_fail_static"]:.2f}% / {c["p_fail_daily"]:.2f}% | {b["med_days_pass"]} | {b["avg_sweeps"]:.2f} |')
    report+=["","## Model and coverage"]
    report.append(f"- {len(inventory['source']['files'])} complete daily Reservoir archives; {inventory['source']['archive_sha256']} content digest.")
    report.append(f"- {len(inventory['screen'])} roster wallets screened; {len(inventory.get('screen_survivors',[]))} passed initial activity/symbol/account-value checks; {len(inventory.get('train_price_replay_advanced',[]))} sampled by Train-only activity, {len(inventory.get('screen_survivors_after_prices',[]))} price-replayed; {len(shortlist)} Train-shortlisted. Not an exhaustive financial ranking.")
    report.append("- Symbol suitability determined from Train-only fill history; selection did not filter on OOS-only symbols. Current Bybit availability is a retrospective mapping, not a historical snapshot.")
    report.append("- Prices: local Hyperliquid 1m OHLCV, local cached Vike fallback; trade at 1m open after 5m checkpoint and mark by checkpoint T+4 minutes, stale <=180m.")
    report.append("- Source: reconstructed Reservoir start_position + fills; pair-local stale-mark termination via canonical_replay.")
    report.append("- Test: Train-only multi-style clustering, dual-barrier sizing and frozen k; MC 5000 paths, stationary 6-day blocks, 90-day horizon.")
    report+=["","## Guardian, PM03 and conflicts"]
    for r in results:
        new=r["oos"]["new"];pm=new.get("pm03",{})
        report.append(f"- Candidate {r['index']}: first-barrier {new['verdict']}, peak-DD {new['peak_to_trough_dd_usd']}, worst daily {new['worst_daily_usd']}, sweep count {new['sweeps']}; PM03 drift {pm.get('drift_usd','UNPROVEN')}; Triad A reverse intervals {r.get('triada_conflicts',{}).get('events','UNPROVEN')}.")
    if ref_stats:
        report.append(f"- Triad A reference: new-only full-window {ref_stats['new']['full_window_net_usd']}, full-target {ref_stats['full']['full_window_net_usd']}; static k source unverified live.")
    report+=["","## Qualification and rank","- **Primary research-only:** Candidate 1 (best OOS strict-new follower net and balanced static-floor safety); **alternative 1:** Candidate 2 (higher Train but weaker OOS and sweep risk); **alternative 2:** Candidate 3 (negative OOS and TIMEOUT, reject for deployment).","- All three have historical opposite-position overlaps with Triad A; do not run simultaneously on a reverse-trading-restricted account.","- No claim of exhaustive top-3 among all 195 wallets: Train-only deterministic sampling advanced 21 of 97 initially screened eligible wallets, then ranked 9; 76 others are deferred.","- These are Train ranks; no candidate k or ranking used OOS until the freeze file was written.","- All three pods have native-min-lot warnings and optimizer k materially above the individual 50%-risk calibration; executable sizing remains UNPROVEN.","- If no candidate both survives OOS and has positive OOS net, no portfolio is economically qualified.","- PIT roster and live Bybit fills unavailable; research can support only RETROSPECTIVE_SCENARIO status.","- Full MC matrix, each wallet's eligibility/rejection reasons, trace hashes and all frozen parameters are in the companion JSON files.","","### Three-line caveat footer","- CFT trading restrictions and account framing require independent compliance confirmation.","- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.","- No deployments, allocations, orders, service or registry writes occurred.",""]
    PREFIX.with_name(PREFIX.name+".md").write_text("\n".join(report))
    print("SAVED_RESULT",str(PREFIX.with_name(PREFIX.name+"__results.json")),flush=True)


if __name__=="__main__":
    main()
