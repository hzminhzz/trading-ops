"""Tests for the single-pass Parquet-cached May-Nov 2026 CFT v4 study orchestrator."""

from __future__ import annotations

import ast
import importlib.util
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
V4_SCRIPT = ROOT / "studies/2026-11-16__cft__10k__canonical-full-universe-v4.py"


def _load_v4():
    import sys

    spec = importlib.util.spec_from_file_location("cft_20261116_v4", V4_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


FILL_SCHEMA = {
    "wallet": pl.String,
    "exchange_ts": pl.Datetime("us", "UTC"),
    "coin": pl.String,
    "dex": pl.String,
    "side": pl.String,
    "size": pl.Float64,
    "start_position": pl.Float64,
    "trade_id": pl.UInt64,
}


def _write_daily_fills(archive_root: Path, rows_by_day: dict[str, list[dict]]) -> None:
    fills_base = archive_root / "reservoir-full-history-20261115" / "hyperliquid" / "fills"
    for day, rows in rows_by_day.items():
        path = fills_base / f"day={day}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df = (
            pl.DataFrame(rows, schema=FILL_SCHEMA)
            if not rows
            else pl.DataFrame(rows).with_columns(
                pl.col("exchange_ts").cast(pl.Datetime("us", "UTC")),
                pl.col("trade_id").cast(pl.UInt64),
            )
        )
        df.write_parquet(path)


def _write_1m_store(
    store_root: Path,
    coin: str,
    day_str: str,
    start_dt: datetime,
    n_minutes: int,
    *,
    prices: np.ndarray,
    volumes: np.ndarray | None = None,
) -> None:
    y, m, d = day_str.split("-")
    p = (
        store_root
        / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{coin}_PERP"
        / f"year={y}"
        / f"month={m}"
        / f"day={d}"
        / "data.parquet"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = [start_dt + timedelta(minutes=i) for i in range(n_minutes)]
    data: dict[str, object] = {
        "timestamp": ts,
        "symbol": [coin] * n_minutes,
        "open": prices.tolist(),
        "close": prices.tolist(),
    }
    if volumes is not None:
        data["volume"] = volumes.tolist()
        data["quote_volume"] = (volumes * prices).tolist()
    pl.DataFrame(data).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(p)


def test_v4_uses_no_duckdb_and_separates_historical_vs_live_screen(tmp_path: Path) -> None:
    v4 = _load_v4()

    # AST guard: v4 must never import duckdb
    tree = ast.parse(V4_SCRIPT.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "duckdb" not in imported

    assert datetime.fromtimestamp(v4.START / 1000, UTC) == datetime(2026, 5, 1, tzinfo=UTC)
    assert datetime.fromtimestamp(v4.SPLIT / 1000, UTC) == datetime(2026, 9, 1, tzinfo=UTC)
    assert datetime.fromtimestamp(v4.END / 1000, UTC) == datetime(2026, 11, 1, tzinfo=UTC)
    assert v4.TRAIN_GRID.size == 123 * 288
    assert v4.OOS_GRID.size == 61 * 288

    # Synthetic SQLite registry with 2 wallets:
    # W_HIST_ONLY: historically eligible (trades OLDCOIN, AV=$200 < $1000, unmapped on Bybit)
    # W_BOTH: both historically eligible and live deployable (trades BTC, AV=$5000)
    db_path = tmp_path / "registry.sqlite"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE wallets (wallet_id INTEGER PRIMARY KEY, address TEXT);
        CREATE TABLE monitoring (wallet_id INTEGER PRIMARY KEY, configured INTEGER);
        CREATE TABLE exclusions (wallet_id INTEGER PRIMARY KEY, reason TEXT);
        CREATE TABLE wallet_aliases (wallet_id INTEGER, alias TEXT, known_at TEXT);
        INSERT INTO wallets VALUES
            (1, '0x1111111111111111111111111111111111111111'),
            (2, '0x2222222222222222222222222222222222222222');
        INSERT INTO monitoring VALUES (1, 1), (2, 1);
        INSERT INTO wallet_aliases VALUES
            (1, 'W_BOTH', '2026-04-01'),
            (2, 'W_HIST_ONLY', '2026-04-01');
        """
    )
    con.commit()
    con.close()

    # Build 184 daily archive files covering 2026-05-01 .. 2026-10-31
    rows_by_day: dict[str, list[dict]] = {}
    cur = datetime(2026, 5, 1, tzinfo=UTC)
    end_dt = datetime(2026, 11, 1, tzinfo=UTC)
    tid = 1
    while cur < end_dt:
        day_str = cur.date().isoformat()
        day_rows: list[dict] = []
        if cur == datetime(2026, 5, 1, tzinfo=UTC):
            day_rows.append(
                {
                    "wallet": "0x1111111111111111111111111111111111111111",
                    "exchange_ts": datetime(2026, 5, 1, 0, 5, tzinfo=UTC),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": 1.0,
                    "start_position": 0.0,
                    "trade_id": tid,
                }
            )
            tid += 1
        elif cur >= datetime(2026, 8, 10, tzinfo=UTC) and cur < datetime(2026, 8, 28, tzinfo=UTC):
            for addr, coin in (
                ("0x1111111111111111111111111111111111111111", "BTC"),
                ("0x2222222222222222222222222222222222222222", "OLDCOIN"),
            ):
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(hours=1),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "buy",
                        "size": 1.0,
                        "start_position": 0.0,
                        "trade_id": tid,
                    }
                )
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(hours=2),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "sell",
                        "size": 1.0,
                        "start_position": 1.0,
                        "trade_id": tid + 1,
                    }
                )
                tid += 2
        elif cur == datetime(2026, 10, 31, tzinfo=UTC):
            day_rows.append(
                {
                    "wallet": "0x1111111111111111111111111111111111111111",
                    "exchange_ts": datetime(2026, 10, 31, 23, 55, tzinfo=UTC),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "sell",
                    "size": 1.0,
                    "start_position": 1.0,
                    "trade_id": tid,
                }
            )
            tid += 1
        rows_by_day[day_str] = day_rows
        cur += timedelta(days=1)

    archive_root = tmp_path / "archive"
    _write_daily_fills(archive_root, rows_by_day)

    store_root = tmp_path / "store"
    for coin in ("BTC", "OLDCOIN"):
        _write_1m_store(
            store_root,
            coin,
            "2026-08-15",
            datetime(2026, 8, 15, 0, 1, tzinfo=UTC),
            120,
            prices=np.full(120, 100.0),
            volumes=np.full(120, 5000.0),
        )

    paths = v4.StudyPaths(
        registry=db_path,
        archive=archive_root,
        store=store_root,
        vike=tmp_path / "no_vike",
        cache_dir=tmp_path / "cache",
        report_prefix=tmp_path / "reports" / "v4_test",
    )
    av_map = {
        "0x1111111111111111111111111111111111111111": (5000.0, None, None),
        "0x2222222222222222222222222222222222222222": (200.0, None, None),
    }
    inv, _grouped, _records, _prices, hist_elig, live_elig = v4.prepare_screen(
        paths,
        bybit_snapshot=({"BTC"}, {"pages": 1, "base_coins": 1, "lookup_utc": "2026-11-16T00:00Z"}),
        account_value_fetcher=lambda r: av_map[r.address],
    )

    assert sorted(hist_elig) == ["W_BOTH", "W_HIST_ONLY"]
    assert sorted(live_elig) == ["W_BOTH"]
    assert inv["historical_eligible_count"] == 2
    assert inv["live_deployable_count"] == 1
    assert inv["screen_survivors_historical"] == ["W_BOTH", "W_HIST_ONLY"]
    assert inv["screen_survivors_live"] == ["W_BOTH"]
    assert "0x111111" not in paths.inventory_path.read_text()


def test_single_pass_train_replay_exports_parquet_and_avoids_duplicate_simulation(
    tmp_path: Path, monkeypatch
) -> None:
    v4 = _load_v4()
    import canonical_replay as cr
    import pod_constructor as pc

    # Build synthetic 10-day 5m grid (2880 bars) with 3 wallets: 2 positive, 1 negative
    n_days = 10
    n_bars = n_days * 288
    start_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    start_ms = int(start_dt.timestamp() * 1000)
    grid = np.arange(start_ms, start_ms + n_bars * 300_000, 300_000, dtype=np.int64)

    # Synthetic rising BTC & falling ETH 1m store prices
    store_root = tmp_path / "store"
    n_min = n_days * 1440 + 10
    btc_px = np.linspace(100.0, 120.0, n_min)
    eth_px = np.linspace(100.0, 110.0, n_min)
    for d in range(n_days):
        day_dt = start_dt + timedelta(days=d)
        s_idx = d * 1440
        e_idx = s_idx + 1440
        _write_1m_store(
            store_root, "BTC", day_dt.date().isoformat(), day_dt, 1440, prices=btc_px[s_idx:e_idx]
        )
        _write_1m_store(
            store_root, "ETH", day_dt.date().isoformat(), day_dt, 1440, prices=eth_px[s_idx:e_idx]
        )

    import study_data as sd

    prices = sd.LazyPrices(grid, store_root=store_root, vike_root=None)

    def mk_fills(events: list[tuple[int, int, float, float]]) -> pl.DataFrame:
        return pl.DataFrame(
            [{"ts": ts, "sign": s, "sz": sz, "pre": pre, "tid": str(i)} for i, (ts, s, sz, pre) in enumerate(events)]
        )

    eligible_hist = {
        "W_WIN1": {
            "BTC": mk_fills(
                [
                    (start_ms + 300_000, 1, 10.0, 0.0),
                    (start_ms + 200 * 300_000, -1, 10.0, 10.0),
                    (start_ms + 300 * 300_000, 1, 10.0, 0.0),
                    (start_ms + 2000 * 300_000, -1, 10.0, 10.0),
                ]
            )
        },
        "W_WIN2": {
            "ETH": mk_fills(
                [
                    (start_ms + 400_000, 1, 8.0, 0.0),
                    (start_ms + 600 * 300_000, -1, 8.0, 8.0),
                    (start_ms + 800 * 300_000, 1, 8.0, 0.0),
                    (start_ms + 2200 * 300_000, -1, 8.0, 8.0),
                ]
            )
        },
        "W_LOSE": {
            # Short a rising asset -> negative Train net
            "BTC": mk_fills(
                [
                    (start_ms + 300_000, -1, 10.0, 0.0),
                    (start_ms + 2000 * 300_000, 1, 10.0, -10.0),
                ]
            )
        },
    }
    screen_rows = [
        {"alias": "W_WIN1", "train_fills": 4, "live_deployable": True},
        {"alias": "W_WIN2", "train_fills": 4, "live_deployable": False},
        {"alias": "W_LOSE", "train_fills": 2, "live_deployable": True},
    ]

    replay_calls: list[list[str]] = []
    orig_replay = cr.build_unit_replay_by_alias

    def counted_replay(*args, **kwargs):
        aliases_arg = args[2] if len(args) > 2 else kwargs["aliases"]
        replay_calls.append(list(aliases_arg))
        return orig_replay(*args, **kwargs)

    monkeypatch.setattr(cr, "build_unit_replay_by_alias", counted_replay)

    cache_dir = tmp_path / "v4_cache"
    shortlist, train_stats, _diags, tensor, artifacts = v4.run_single_pass_train_replay(
        eligible_hist,
        prices,
        screen_rows,
        cache_dir=cache_dir,
        train_grid=grid,
        start_ms=start_ms,
    )

    # Zero duplicate replay: each eligible wallet simulated strictly once!
    replayed_aliases = [a for batch in replay_calls for a in batch]
    assert sorted(replayed_aliases) == ["W_LOSE", "W_WIN1", "W_WIN2"]
    assert len(replayed_aliases) == 3

    assert sorted(shortlist) == ["W_WIN1", "W_WIN2"]
    assert "W_LOSE" not in tensor.aliases
    assert tensor.aliases == ("W_WIN1", "W_WIN2")
    assert len(train_stats) == 3
    assert next(r for r in train_stats if r["alias"] == "W_WIN2")["live_deployable"] is False

    streams_pq = Path(artifacts["train_unit_streams_parquet"])
    eq_pq = Path(artifacts["train_unit_equity_parquet"])
    summary_pq = Path(artifacts["train_solo_summary_parquet"])
    if not streams_pq.is_absolute():
        streams_pq = v4.ROOT / streams_pq
        eq_pq = v4.ROOT / eq_pq
        summary_pq = v4.ROOT / summary_pq
    assert streams_pq.exists() and eq_pq.exists() and summary_pq.exists()

    # Round-trip Parquet unit streams back into build_tensor_from_unit_replays matches exactly
    pq_df = pl.read_parquet(streams_pq)
    reloaded: dict[str, dict[str, pl.DataFrame]] = {}
    for (alias, coin), part in pq_df.partition_by(
        ["alias", "coin"], as_dict=True, maintain_order=True
    ).items():
        reloaded.setdefault(str(alias), {})[str(coin)] = part.select(
            "timestamp", "open", "close", "pos", "delta", "gross", "cost", "net"
        )
    tensor_rt = pc.build_tensor_from_unit_replays(reloaded, grid, pod_equity_usd=10_000.0)
    assert tensor_rt.aliases == tensor.aliases
    assert tensor_rt.symbols == tensor.symbols
    assert np.allclose(tensor_rt.notional_frac, tensor.notional_frac, atol=1e-6)
    assert np.allclose(tensor_rt.delta_notional_frac, tensor.delta_notional_frac, atol=1e-6)
    assert np.allclose(tensor_rt.net_ret_frac, tensor.net_ret_frac, atol=1e-9)


def _make_synthetic_tensor_with_micro_and_healthy(*, n_healthy: int = 4, n_days: int = 16):
    import pod_constructor as pc

    rng = np.random.default_rng(2026)
    bpd = 288
    n_bars = n_days * bpd
    start_ms = int(datetime(2026, 5, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
    ts = np.arange(start_ms, start_ms + n_bars * 300_000, 300_000, dtype=np.int64)
    chop_mask = np.arange(n_bars) >= (n_bars // 2)

    n_total = n_healthy + 1  # healthy traders + 1 tiny-lot MICRO trader
    notional = np.zeros((n_total, n_total, n_bars), dtype=np.float32)
    d_notional = np.zeros((n_total, n_total, n_bars), dtype=np.float32)
    gross = np.zeros((n_total, n_bars), dtype=np.float64)
    net = np.zeros((n_total, n_bars), dtype=np.float64)

    for ti in range(n_healthy):
        for e_idx in range(10):
            s_b = 40 + e_idx * 360 + ti * 15
            e_b = s_b + 20
            if e_b >= n_bars:
                break
            # $4,000 unit position (q10_frac = 0.40 -> at 2% DD cap, order >> $15)
            notional[ti, ti, s_b:e_b] = 0.40
            d_notional[ti, ti, s_b] = 0.40
            d_notional[ti, ti, e_b] = -0.40
            r = rng.normal(0.0012, 0.0005, size=e_b - s_b - 1)
            gross[ti, s_b + 1 : e_b] = r
            net[ti, s_b + 1 : e_b] = r
        # Give realistic 2% standalone max DD so baseline k is ~0.25..0.33
        net[ti, 25 + ti] = -0.02

    # MICRO trader: positive net, 10 episodes, tiny $5 unit position (q10_frac = 0.0005 -> ~$3.33 < $15)
    m_idx = n_healthy
    for e_idx in range(10):
        s_b = 40 + e_idx * 360
        e_b = s_b + 20
        if e_b >= n_bars:
            break
        notional[m_idx, m_idx, s_b:e_b] = 0.0005
        d_notional[m_idx, m_idx, s_b] = 0.0005
        d_notional[m_idx, m_idx, e_b] = -0.0005
        gross[m_idx, s_b + 1 : e_b] = 0.0008
        net[m_idx, s_b + 1 : e_b] = 0.0008
    net[m_idx, 20] = -0.01

    aliases = tuple([f"HEALTHY_{i}" for i in range(n_healthy)] + ["MICRO_TINY"])
    symbols = tuple(f"COIN_{i}" for i in range(n_total))
    return pc.UnitCopy5mTensor(
        aliases=aliases,
        symbols=symbols,
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=gross,
        net_ret_frac=net,
        chop_mask=chop_mask,
    )


def test_train_pods_filters_sub_15_before_clustering_caps_2pct_dd_and_runs_cpcv() -> None:
    v4 = _load_v4()

    tensor = _make_synthetic_tensor_with_micro_and_healthy(n_healthy=4, n_days=16)
    screen_rows = [
        {"alias": a, "live_deployable": a != "HEALTHY_3", "live_reasons": [] if a != "HEALTHY_3" else ["WITHDRAWN_LT_1000"]}
        for a in tensor.aliases
    ]

    candidates, auditing = v4.train_pods(tensor, screen_rows=screen_rows)

    assert len(candidates) >= 2
    for cand in candidates:
        assert "MICRO_TINY" not in cand["aliases"]
        # Frozen k strictly capped <= 1.0x of 2%-port-DD baseline ($200 / size)
        for opt_k, base_k in zip(cand["train_opt_k"], cand["train_half_risk_k"]):
            assert opt_k <= base_k + 1e-12
        assert cand["strict_min_notional_passed"] is True
        expected_live = all(a != "HEALTHY_3" for a in cand["aliases"])
        assert cand["live_deployable"] is expected_live

    assert "MICRO_TINY" not in auditing["3"]["passed"]
    assert any(
        r["alias"] == "MICRO_TINY" and "below_min_baseline_order" in r["reason"]
        for r in auditing["3"]["rejected"]
    )
    assert auditing["cpcv"]["n_splits"] >= 1
    assert auditing["cpcv"]["pbo"] is not None
    assert 0.0 <= auditing["cpcv"]["pbo"] <= 1.0


def test_train_pods_handles_insufficient_strict_wallets_without_fabricating() -> None:
    v4 = _load_v4()

    # Only 2 healthy wallets + 1 tiny-lot wallet -> strict $15 leaves < 3 wallets for size=3, 4!
    tensor = _make_synthetic_tensor_with_micro_and_healthy(n_healthy=2, n_days=16)
    candidates, auditing = v4.train_pods(tensor)

    assert candidates == []
    assert auditing["3"]["status"] == "INSUFFICIENT"
    assert auditing["4"]["status"] == "INSUFFICIENT"
    assert auditing["strict_min_notional_status"] == "INSUFFICIENT_FOR_POD_SIZES"
    assert auditing["cpcv"]["pbo"] is None
    assert "diagnostic_unfiltered_sensitivity" not in auditing


def _write_monthly_funding(
    store_root: Path,
    coin: str,
    year: int,
    month: int,
    timestamps: list[datetime],
    rates: list[float],
) -> None:
    p = (
        store_root
        / f"crypto_futures_funding_{coin}USDT_PERP"
        / f"year={year}"
        / f"month={month}"
        / "data.parquet"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [f"{coin}USDT"] * len(timestamps),
            "funding_interval_hours": [8] * len(timestamps),
            "funding_rate": rates,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms", "UTC"))).write_parquet(p)


def test_evaluate_one_deadbands_cost_multipliers_leverage_funding_volume_and_parquet(
    tmp_path: Path,
) -> None:
    v4 = _load_v4()
    import study_data as sd

    # 8-day synthetic OOS window (2304 bars) so MC daily blocks & PM03 7d reconciliation both work
    n_days = 8
    n_bars = n_days * 288
    start_dt = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    start_ms = int(start_dt.timestamp() * 1000)
    oos_grid = np.arange(start_ms, start_ms + n_bars * 300_000, 300_000, dtype=np.int64)

    store_root = tmp_path / "store"
    n_min = n_days * 1440 + 10
    # Steady upward drift from 100 -> 108 across 8 days with active 1m volume
    px_all = np.linspace(100.0, 108.0, n_min)
    vol_all = np.full(n_min, 100.0)  # ~$10,000/min dollar volume
    for d in range(n_days):
        day_dt = start_dt + timedelta(days=d)
        s_i = d * 1440
        e_i = s_i + 1440
        _write_1m_store(
            store_root,
            "BTC",
            day_dt.date().isoformat(),
            day_dt,
            1440,
            prices=px_all[s_i:e_i],
            volumes=vol_all[s_i:e_i],
        )

    # Funding rate: +0.0008 per 8h (+8 bps per 8h) across all 8 days
    fund_ts = [start_dt + timedelta(hours=8 * h) for h in range(n_days * 3)]
    _write_monthly_funding(store_root, "BTC", 2026, 9, fund_ts, [0.0008] * len(fund_ts))

    prices = sd.LazyPrices(oos_grid, store_root=store_root, vike_root=None)

    # 3 wallets: A trades big ($500), B trades medium ($20 -> caught by $30 deadband, passes $15!),
    # C held pre-existing inventory into OOS (tests new vs full difference!)
    def mk_df(rows):
        return pl.DataFrame(rows)

    grouped = {
        "W_A": {
            "BTC": mk_df(
                [
                    {"ts": start_ms + 300_000, "sign": 1, "sz": 5.0, "pre": 0.0, "tid": "1"},
                    {"ts": start_ms + 500 * 300_000, "sign": -1, "sz": 5.0, "pre": 5.0, "tid": "2"},
                    {"ts": start_ms + 600 * 300_000, "sign": 1, "sz": 5.0, "pre": 0.0, "tid": "3"},
                ]
            )
        },
        "W_B": {
            "BTC": mk_df(
                [
                    {"ts": start_ms + 900_000, "sign": 1, "sz": 0.2, "pre": 0.0, "tid": "4"},
                    {"ts": start_ms + 900 * 300_000, "sign": -1, "sz": 0.2, "pre": 0.2, "tid": "5"},
                ]
            )
        },
        "W_C": {
            "BTC": mk_df(
                [
                    # Open inventory prior to start_ms, closed halfway through OOS
                    {"ts": start_ms - 600_000, "sign": 1, "sz": 2.0, "pre": 0.0, "tid": "6"},
                    {"ts": start_ms + 1000 * 300_000, "sign": -1, "sz": 2.0, "pre": 2.0, "tid": "7"},
                ]
            )
        },
    }
    row = {
        "index": 1,
        "aliases": ["W_A", "W_B", "W_C"],
        "train_opt_k": [1.0, 1.0, 1.0],
    }

    vol_cache: dict[str, np.ndarray] = {}
    fund_cache: dict[str, np.ndarray] = {}
    equity_records: list[pl.DataFrame] = []
    block_records: list[pl.DataFrame] = []

    res = v4.evaluate_one(
        row,
        grouped,
        prices,
        oos_grid,
        full_grid=oos_grid,
        store_root=store_root,
        vol_cache=vol_cache,
        fund_cache=fund_cache,
        montecarlo=True,
        pm03=True,
        equity_sink=equity_records,
        block_sink=block_records,
        mc_n_paths=200,
    )

    # Both new and full policies evaluated
    assert set(res) == {"new", "full"}
    # Pre-existing W_C inventory makes full differ from new
    assert (
        res["full"]["deadbands"]["15"]["cost_multipliers"]["1x"]["full_window_net_usd"]
        != res["new"]["deadbands"]["15"]["cost_multipliers"]["1x"]["full_window_net_usd"]
    )

    for pol in ("new", "full"):
        for db in ("15", "30"):
            cm = res[pol]["deadbands"][db]["cost_multipliers"]
            assert set(cm) == {"1x", "2x", "3x"}
            # Cost stress monotonicity: 3x <= 2x <= 1x
            assert cm["3x"]["full_window_net_usd"] <= cm["2x"]["full_window_net_usd"] <= cm["1x"]["full_window_net_usd"]
            assert cm["1x"]["peak_gross_leverage"] > 0.0
            vp = res[pol]["deadbands"][db]["volume_participation"]
            assert vp["trade_count"] >= 1
            assert vp["volume_coverage_frac"] == 1.0
            assert 0.0 < vp["max_participation_pct"] < 100.0

        # Deadband $30 suppresses W_B's ~$20 leg -> fewer trades or lower traded notional than $15
        assert (
            res[pol]["deadbands"]["30"]["volume_participation"]["total_traded_notional_usd"]
            < res[pol]["deadbands"]["15"]["volume_participation"]["total_traded_notional_usd"]
        )

        # Funding stress genuinely models negative long carry
        fs = res[pol]["deadbands"]["15"]["funding_stress"]
        assert fs["historical_usdt_proxy"]["total_funding_usd"] < 0.0
        assert fs["constant_1bp_8h"]["total_funding_usd"] < 0.0

        # MC matrix present
        assert set(res[pol]["deadbands"]["15"]["mc"]) == {
            "1.0x/Blended",
            "1.0x/75%-Chop",
            "1.5x/Blended",
            "1.5x/75%-Chop",
            "2.0x/Blended",
            "2.0x/75%-Chop",
        }

    # PM03 7d reconciliation matches Tier-1 cleanly
    assert abs(res["new"]["pm03"]["drift_usd"]) < 0.05

    # OOS Parquet exports write valid non-empty Parquet files
    cache_dir = tmp_path / "cache"
    oos_paths = v4.export_oos_parquet_artifacts(cache_dir, equity_records, block_records)
    eq_pq = Path(oos_paths["oos_equity_curves_parquet"])
    blk_pq = Path(oos_paths["oos_daily_blocks_parquet"])
    if not eq_pq.is_absolute():
        eq_pq = v4.ROOT / eq_pq
        blk_pq = v4.ROOT / blk_pq
    eq_df = pl.read_parquet(eq_pq)
    blk_df = pl.read_parquet(blk_pq)
    assert eq_df.height > 0
    assert blk_df.height > 0
    assert {"candidate_idx", "policy", "deadband_usd", "cost_mult", "ts_ms", "equity_usd"} <= set(eq_df.columns)
    assert {"candidate_idx", "policy", "deadband_usd", "cost_mult", "day_end_usd", "day_swept"} <= set(blk_df.columns)


def test_end_to_end_run_study_produces_sanitized_reports_and_parquet_caches(tmp_path: Path) -> None:
    v4 = _load_v4()

    # Build minimal synthetic 184-day archive + 1m store + registry with 4 active winning wallets
    # plus TRIAD aliases so end-to-end screening, single-pass replay, clustering, CPCV, OOS & Triad A run.
    db_path = tmp_path / "registry.sqlite"
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE wallets (wallet_id INTEGER PRIMARY KEY, address TEXT);
        CREATE TABLE monitoring (wallet_id INTEGER PRIMARY KEY, configured INTEGER);
        CREATE TABLE exclusions (wallet_id INTEGER PRIMARY KEY, reason TEXT);
        CREATE TABLE wallet_aliases (wallet_id INTEGER, alias TEXT, known_at TEXT);
        """
    )
    wallets = [
        (1, "0xaaaa000000000000000000000000000000000001", "W_A", 1),
        (2, "0xaaaa000000000000000000000000000000000002", "W_B", 1),
        (3, "0xaaaa000000000000000000000000000000000003", "W_C", 1),
        (4, "0xaaaa000000000000000000000000000000000004", "W_D", 1),
        (5, "0xbbbb000000000000000000000000000000000005", "ACT04", 0),
        (6, "0xbbbb000000000000000000000000000000000006", "ACT08", 0),
        (7, "0xbbbb000000000000000000000000000000000007", "ACT09", 0),
    ]
    for wid, addr, alias, cfg in wallets:
        con.execute("INSERT INTO wallets VALUES (?, ?)", (wid, addr))
        con.execute("INSERT INTO monitoring VALUES (?, ?)", (wid, cfg))
        con.execute("INSERT INTO wallet_aliases VALUES (?, ?, '2026-04-01')", (wid, alias))
    con.commit()
    con.close()

    rows_by_day: dict[str, list[dict]] = {}
    cur = datetime(2026, 5, 1, tzinfo=UTC)
    end_dt = datetime(2026, 11, 1, tzinfo=UTC)
    tid = 1
    coins = ["BTC", "ETH", "SOL", "AVAX"]
    store_root = tmp_path / "store"
    px_base = 100.0

    # Generate 184 days of daily archive + 1m store written once per month (6*4 = 24 files, fast!)
    for m in range(5, 11):
        m_start = datetime(2026, m, 1, tzinfo=UTC)
        m_end = datetime(2026, m + 1, 1, tzinfo=UTC)
        n_m = int((m_end - m_start).total_seconds() // 60)
        day_offset = (m_start - datetime(2026, 5, 1, tzinfo=UTC)).days
        p_arr = px_base + day_offset * 0.4 + np.linspace(0.0, (n_m / 1440.0) * 0.4, n_m)
        v_arr = np.full(n_m, 200.0)
        for coin in coins:
            _write_1m_store(store_root, coin, m_start.date().isoformat(), m_start, n_m, prices=p_arr, volumes=v_arr)

    while cur < end_dt:
        day_str = cur.date().isoformat()
        day_rows: list[dict] = []
        day_idx = (cur - datetime(2026, 5, 1, tzinfo=UTC)).days

        if cur == datetime(2026, 5, 1, tzinfo=UTC):
            day_rows.append(
                {
                    "wallet": wallets[0][1],
                    "exchange_ts": cur + timedelta(minutes=5),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": 10.0,
                    "start_position": 0.0,
                    "trade_id": tid,
                }
            )
            day_rows.append(
                {
                    "wallet": wallets[0][1],
                    "exchange_ts": cur + timedelta(hours=1),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "sell",
                    "size": 10.0,
                    "start_position": 10.0,
                    "trade_id": tid + 1,
                }
            )
            tid += 2
        elif day_idx in range(100, 122, 2):  # 11 Train episodes in late Aug + loss spike on day 98
            for w_idx, coin in enumerate(coins):
                addr = wallets[w_idx][1]
                # Small dip & exit on day 100 to guarantee nonzero unit maxdd, winning entries afterwards
                sz = 12.0 + w_idx * 2.0
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(minutes=5 + w_idx * 10),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "buy",
                        "size": sz,
                        "start_position": 0.0,
                        "trade_id": tid,
                    }
                )
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(hours=20, minutes=w_idx * 10),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "sell",
                        "size": sz,
                        "start_position": sz,
                        "trade_id": tid + 1,
                    }
                )
                tid += 2
        elif day_idx == 130:  # OOS trade + ACT08 opposite position for Triad A conflict test
            for w_idx, coin in enumerate(coins):
                addr = wallets[w_idx][1]
                sz = 12.0 + w_idx * 2.0
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(minutes=5),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "buy",
                        "size": sz,
                        "start_position": 0.0,
                        "trade_id": tid,
                    }
                )
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(hours=12),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "sell",
                        "size": sz,
                        "start_position": sz,
                        "trade_id": tid + 1,
                    }
                )
                tid += 2
            day_rows.append(
                {
                    "wallet": wallets[5][1],  # ACT08 short BTC opposite W_A long BTC
                    "exchange_ts": cur + timedelta(minutes=5),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "sell",
                    "size": 50.0,
                    "start_position": 0.0,
                    "trade_id": tid,
                }
            )
            day_rows.append(
                {
                    "wallet": wallets[5][1],
                    "exchange_ts": cur + timedelta(hours=12),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": 50.0,
                    "start_position": -50.0,
                    "trade_id": tid + 1,
                }
            )
            day_rows.append(
                {
                    "wallet": wallets[4][1],  # ACT04
                    "exchange_ts": cur + timedelta(minutes=10),
                    "coin": "ETH",
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": 5.0,
                    "start_position": 0.0,
                    "trade_id": tid + 2,
                }
            )
            day_rows.append(
                {
                    "wallet": wallets[6][1],  # ACT09
                    "exchange_ts": cur + timedelta(minutes=10),
                    "coin": "SOL",
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": 5.0,
                    "start_position": 0.0,
                    "trade_id": tid + 3,
                }
            )
            tid += 4
        elif day_idx == 183:  # Oct 31 final UTC hour coverage
            day_rows.append(
                {
                    "wallet": wallets[0][1],
                    "exchange_ts": cur + timedelta(hours=23, minutes=55),
                    "coin": "BTC",
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": 5.0,
                    "start_position": 0.0,
                    "trade_id": tid,
                }
            )
            tid += 1

        # Add a tiny Train loss bar for W_A..W_D on day 90 (price drops intra-day) so maxdd > 0
        if day_idx == 90:
            for w_idx, coin in enumerate(coins):
                addr = wallets[w_idx][1]
                sz = 12.0 + w_idx * 2.0
                # Short on a rising day -> small Train drawdown!
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(minutes=5),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "sell",
                        "size": sz,
                        "start_position": 0.0,
                        "trade_id": tid,
                    }
                )
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": cur + timedelta(hours=10),
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "buy",
                        "size": sz,
                        "start_position": -sz,
                        "trade_id": tid + 1,
                    }
                )
                tid += 2

        rows_by_day[day_str] = day_rows
        cur += timedelta(days=1)

    archive_root = tmp_path / "archive"
    _write_daily_fills(archive_root, rows_by_day)

    paths = v4.StudyPaths(
        registry=db_path,
        archive=archive_root,
        store=store_root,
        vike=tmp_path / "no_vike",
        cache_dir=tmp_path / "cache",
        report_prefix=tmp_path / "reports" / "cft_v4",
    )
    av_map = {addr: (5000.0 if alias != "W_D" else 200.0, None, None) for _, addr, alias, _ in wallets}

    results = v4.run_study(
        paths=paths,
        through="all",
        bybit_snapshot=(set(coins), {"pages": 1, "base_coins": 4, "lookup_utc": "2026-11-16T00:00Z"}),
        account_value_fetcher=lambda r: av_map[r.address],
        fetch_hyperx=False,
        mc_n_paths=200,
    )

    assert results["historical_eligible_count"] >= 4
    assert results["live_deployable_count"] < results["historical_eligible_count"]
    assert results["frozen_candidates"] >= 1
    assert (paths.cache_dir / "train_unit_streams.parquet").exists()
    assert (paths.cache_dir / "train_unit_equity.parquet").exists()
    assert (paths.cache_dir / "oos_equity_curves.parquet").exists()
    assert (paths.cache_dir / "oos_daily_blocks.parquet").exists()

    md_path = paths.report_prefix.with_name(paths.report_prefix.name + ".md")
    freeze_path = paths.report_prefix.with_name(paths.report_prefix.name + "__freeze.json")
    results_path = paths.report_prefix.with_name(paths.report_prefix.name + "__results.json")
    assert md_path.exists() and freeze_path.exists() and results_path.exists()

    md_text = md_path.read_text()
    for p in (md_path, freeze_path, results_path):
        txt = p.read_text()
        assert "0xaaaa" not in txt and "0xbbbb" not in txt
    assert "$15" in md_text and "$30" in md_text
    assert "1x" in md_text and "2x" in md_text and "3x" in md_text
    assert "peak" in md_text.lower() and "participation" in md_text.lower()
    assert "CPCV" in md_text and "Triad A" in md_text


def test_evaluate_one_wires_lot_specs_5x_leverage_and_guardian_participation(tmp_path: Path) -> None:
    v4 = _load_v4()
    import study_data as sd

    n_days = 2
    n_bars = n_days * 288
    start_dt = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    start_ms = int(start_dt.timestamp() * 1000)
    oos_grid = np.arange(start_ms, start_ms + n_bars * 300_000, 300_000, dtype=np.int64)

    store_root = tmp_path / "store"
    for d in range(n_days):
        day_dt = start_dt + timedelta(days=d)
        _write_1m_store(
            store_root,
            "BTC",
            day_dt.date().isoformat(),
            day_dt,
            1440,
            prices=np.full(1440, 100.0),
            volumes=np.full(1440, 1000.0),  # $100k/min dollar volume
        )

    prices = sd.LazyPrices(oos_grid, store_root=store_root, vike_root=None)
    # Wallet requests 800 BTC ($80,000 = 8.0x on $10,000 eq), then closes.
    # Guardian's 5.0x max_gross_leverage cap + lot step 10.0 BTC caps entry at <= 500 BTC ($50,000).
    # Without Guardian alignment, raw target would trade $80k + $80k = $160k notional.
    grouped = {
        "W_BIG": {
            "BTC": pl.DataFrame(
                [
                    {"ts": start_ms + 300_000, "sign": 1, "sz": 800.0, "pre": 0.0, "tid": "1"},
                    {"ts": start_ms + 3_000_000, "sign": -1, "sz": 800.0, "pre": 800.0, "tid": "2"},
                ]
            )
        }
    }
    row = {"index": 1, "aliases": ["W_BIG"], "train_opt_k": [1.0]}

    res = v4.evaluate_one(
        row,
        grouped,
        prices,
        oos_grid,
        full_grid=oos_grid,
        store_root=store_root,
        montecarlo=False,
        pm03=False,
        deadbands=(15.0,),
        cost_multipliers=(1.0,),
        include_funding_stress=False,
        amount_step_by_coin={"BTC": 10.0},
        min_amount_by_coin={"BTC": 10.0},
    )

    cm = res["new"]["deadbands"]["15"]["cost_multipliers"]["1x"]
    vp = res["new"]["deadbands"]["15"]["volume_participation"]
    assert cm["full_window_peak_gross_leverage"] <= 5.0 + 1e-9
    # Capped round-trip notional (~$50k * 2 = ~$100k), strictly below raw un-capped $160k
    assert 95_000.0 <= vp["total_traded_notional_usd"] <= 100_000.0

