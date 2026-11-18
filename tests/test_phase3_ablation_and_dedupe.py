"""Tests for Phase 3 deduplication, funding loader, Propr filtering, and ablation pipeline."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sqlite3
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = ROOT / "studies"
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import canonical_replay as cr
import pod_constructor as pc
import study_data as sd

V4_SCRIPT = STUDIES_DIR / "2026-11-16__cft__10k__canonical-full-universe-v4.py"
PHASE3_SCRIPT = STUDIES_DIR / "2026-11-18__cft__10k__phase3-pod-shop-ablation.py"
PROP_PRODUCTION_SRC = Path("/home/quant/dev/prop-production/src")
HEX_RE = re.compile(r"0x[0-9a-fA-F]{4,}")


def _load_phase3_module():
    spec = importlib.util.spec_from_file_location("phase3_pod_shop_ablation", PHASE3_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sanitize_public_label_redacts_hex_aliases_and_preserves_plain_labels() -> None:
    redacted = sd.sanitize_public_label("Whale_0x5f94")
    assert redacted.startswith("Whale_H#")
    assert len(redacted) == len("Whale_H#") + 6
    assert not HEX_RE.search(redacted)
    assert sd.sanitize_public_label("Whale_0x5F94") == redacted

    multi = sd.sanitize_public_label("pair:0xDeadBeef/0x1234")
    assert not HEX_RE.search(multi)
    assert multi.count("H#") == 2

    for plain in ("W#107", "W111", "M11", "CFT_Tier2", "short_0x12"):
        assert sd.sanitize_public_label(plain) == plain


def test_v4_deduplicates_volume_and_funding_loaders_by_ast() -> None:
    tree = ast.parse(V4_SCRIPT.read_text(encoding="utf-8"))
    top_defs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "load_1m_volumes_on_grid" not in top_defs
    assert "load_usdt_funding_rates_on_grid" not in top_defs
    assert "_epoch_ms_to_utc_datetime" not in top_defs

    top_assigns = {
        tgt.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for tgt in node.targets
        if isinstance(tgt, ast.Name)
    }
    assert "_VOL_SCAN_SCHEMA" not in top_assigns
    assert "_FUNDING_SCAN_SCHEMA" not in top_assigns

    sd_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sd"
    }
    assert {"load_1m_volumes_on_grid", "load_usdt_funding_rates_on_grid"} <= sd_calls


def test_load_usdt_funding_rates_handles_nan_and_4tuple_cache(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    f_path = (
        store_root
        / "crypto_futures_funding_BTCUSDT_PERP"
        / "year=2026"
        / "month=09"
        / "data.parquet"
    )
    f_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    pl.DataFrame(
        {
            "timestamp": [
                t0,
                t0 + timedelta(hours=8),
                t0 + timedelta(hours=16),
            ],
            "funding_rate": [0.00096, float("nan"), 0.00048],
            "funding_interval_hours": [float("nan"), 8.0, 4.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(f_path)

    start_ms = sd.to_utc_ms(t0)
    grid_a = np.arange(start_ms, start_ms + 24 * 3_600_000, 300_000, dtype=np.int64)
    cache: dict[tuple[str, int, int, int], np.ndarray] = {}

    rates_a, covered_a = sd.load_usdt_funding_rates_on_grid(
        ("BTC", "MISSING"), grid_a, store_root, cache
    )
    assert rates_a.shape == (2, grid_a.size)
    assert covered_a == 1
    assert np.all(np.isfinite(rates_a))
    assert rates_a[0, 0] == pytest.approx(0.00096 / (8.0 * 12.0))
    assert rates_a[0, 8 * 12] == pytest.approx(0.0)
    assert rates_a[0, 16 * 12] == pytest.approx(0.00048 / (4.0 * 12.0))
    assert np.all(rates_a[1] == 0.0)

    key_a = ("BTC", int(grid_a[0]), int(grid_a[-1]), int(grid_a.size))
    assert key_a in cache

    grid_b = grid_a.copy()
    grid_b[-1] += 300_000
    rates_b, covered_b = sd.load_usdt_funding_rates_on_grid(("BTC",), grid_b, store_root, cache)
    assert rates_b.shape == (1, grid_b.size)
    assert covered_b == 1
    assert ("BTC", int(grid_b[0]), int(grid_b[-1]), int(grid_b.size)) in cache

    empty_coins, cov_empty = sd.load_usdt_funding_rates_on_grid((), grid_a, store_root, cache)
    assert empty_coins.shape == (0, grid_a.size)
    assert cov_empty == 0

    empty_grid, cov_empty_g = sd.load_usdt_funding_rates_on_grid(
        ("BTC",), np.array([], dtype=np.int64), store_root, cache
    )
    assert empty_grid.shape == (1, 0)
    assert cov_empty_g == 0


def test_load_prop_contract_spec_splits_propr_programs_and_fails_closed() -> None:
    propr_1 = pc.load_prop_contract_spec("propr", account_usd=10_000.0)
    assert len(propr_1.eval_phases) == 1
    assert propr_1.eval_phases[0].profit_target_usd == pytest.approx(1000.0)
    assert propr_1.eval_phases[0].max_static_dd_usd == pytest.approx(600.0)
    assert propr_1.eval_phases[0].daily_loss_limit_usd == pytest.approx(300.0)
    assert propr_1.funded_phase.max_static_dd_usd == pytest.approx(600.0)
    assert propr_1.funded_phase.daily_loss_limit_usd == pytest.approx(300.0)

    propr_2 = pc.load_prop_contract_spec(
        "propr", account_usd=10_000.0, program_filter="Classic 2-Step"
    )
    assert len(propr_2.eval_phases) == 2
    assert propr_2.eval_phases[0].profit_target_usd == pytest.approx(500.0)
    assert propr_2.eval_phases[1].profit_target_usd == pytest.approx(1000.0)
    assert propr_2.eval_phases[0].max_static_dd_usd == pytest.approx(800.0)
    assert propr_2.eval_phases[0].daily_loss_limit_usd == pytest.approx(500.0)
    assert propr_2.funded_phase.max_static_dd_usd == pytest.approx(800.0)
    assert propr_2.funded_phase.daily_loss_limit_usd == pytest.approx(500.0)

    if PROP_PRODUCTION_SRC.exists():
        if str(PROP_PRODUCTION_SRC) not in sys.path:
            sys.path.insert(0, str(PROP_PRODUCTION_SRC))
        from prop_production.prop_firms import get_challenge_spec

        propr2_rule = get_challenge_spec("propr:classic_2step:phase_1")
    else:
        propr2_rule = SimpleNamespace(
            initial="5000",
            target="250",
            daily_loss="250",
            trailing_loss="400",
            daily_basis="BALANCE",
            trailing_basis="INTRADAY_EQUITY",
            reset_time_hhmm="00:00",
        )
    with pytest.raises(ValueError, match="UNSUPPORTED_TRAILING_BASIS:INTRADAY_EQUITY"):
        cr.GuardianSpec.from_challenge_spec(propr2_rule)


def _write_synthetic_day_fills(
    archive_root: Path,
    day_str: str,
    rows: list[dict[str, object]],
) -> None:
    p = (
        archive_root
        / "reservoir-full-history-synthetic"
        / "hyperliquid"
        / "fills"
        / f"day={day_str}"
        / "data.parquet"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).with_columns(
        pl.col("exchange_ts").cast(pl.Datetime("us", "UTC")),
        pl.col("trade_id").cast(pl.UInt64),
    ).write_parquet(p)


def _write_synthetic_ohlcv_and_funding(
    store_root: Path,
    coins: tuple[str, ...],
    start_dt: datetime,
    n_days: int,
) -> None:
    for coin_idx, coin in enumerate(coins):
        for d in range(n_days):
            day_dt = start_dt + timedelta(days=d)
            day_str = day_dt.date().isoformat()
            y, m, day_n = day_str.split("-")
            pq = (
                store_root
                / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{coin}_PERP"
                / f"year={y}"
                / f"month={m}"
                / f"day={day_n}"
                / "data.parquet"
            )
            pq.parent.mkdir(parents=True, exist_ok=True)
            mins = np.arange(1440, dtype=np.float64)
            phase = 6.0 * np.pi * (mins / 1440.0)
            amp = 4.0 + 0.6 * ((d + coin_idx) % 3)
            base = 100.0 + 10.0 * coin_idx
            pullback = np.where((mins >= 120) & (mins <= 150), -1.0 - 0.2 * (d % 2), 0.0)
            px = base - amp * np.cos(phase) + pullback
            vol = np.full(1440, 200_000.0, dtype=np.float64)
            ts = [day_dt + timedelta(minutes=int(i)) for i in range(1440)]
            pl.DataFrame(
                {
                    "timestamp": ts,
                    "symbol": [coin] * 1440,
                    "open": px.tolist(),
                    "close": px.tolist(),
                    "volume": vol.tolist(),
                    "quote_volume": (vol * px).tolist(),
                }
            ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(pq)

        fq = (
            store_root
            / f"crypto_futures_funding_{coin}USDT_PERP"
            / f"year={start_dt.year:04d}"
            / f"month={start_dt.month:02d}"
            / "data.parquet"
        )
        fq.parent.mkdir(parents=True, exist_ok=True)
        fts = [start_dt + timedelta(hours=8 * h) for h in range(n_days * 3 + 3)]
        pl.DataFrame(
            {
                "timestamp": fts,
                "funding_rate": [0.0001] * len(fts),
                "funding_interval_hours": [8] * len(fts),
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(fq)


def test_phase3_ablation_end_to_end_synthetic_and_no_hex_leakage(tmp_path: Path) -> None:
    p3 = _load_phase3_module()

    start_dt = datetime(2026, 10, 26, 0, 0, tzinfo=UTC)
    n_days = 13
    archive_end_dt = start_dt + timedelta(days=n_days)

    archive_root = tmp_path / "archive"
    store_root = tmp_path / "store"
    db_path = tmp_path / "registry.sqlite"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"

    wallets = [
        (1, "0x5f94aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Whale_0x5f94", "BTC"),
        (2, "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "Alpha_One", "ETH"),
        (3, "0xcccccccccccccccccccccccccccccccccccccccc", "Alpha_Two", "SOL"),
        (4, "0xdddddddddddddddddddddddddddddddddddddddd", "Bench_One", "BTC"),
        (5, "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "Bench_Two", "ETH"),
        (6, "0xffffffffffffffffffffffffffffffffffffffff", "Bench_Three", "SOL"),
    ]

    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE schema_migrations (version INTEGER);
        INSERT INTO schema_migrations VALUES (8);
        CREATE TABLE wallets (wallet_id INTEGER PRIMARY KEY, address TEXT);
        CREATE TABLE cohort_memberships (wallet_id INTEGER, known_at TEXT);
        CREATE TABLE monitoring (wallet_id INTEGER, configured INTEGER, known_at TEXT);
        CREATE TABLE exclusions (wallet_id INTEGER, known_at TEXT);
        CREATE TABLE wallet_aliases (wallet_id INTEGER, alias TEXT, known_at TEXT);
        """
    )
    for wid, addr, alias, _coin in wallets:
        con.execute("INSERT INTO wallets VALUES (?, ?)", (wid, addr))
        con.execute("INSERT INTO cohort_memberships VALUES (?, '2026-11-05T00:00:00Z')", (wid,))
        con.execute("INSERT INTO monitoring VALUES (?, 1, '2026-11-05T00:00:00Z')", (wid,))
        con.execute("INSERT INTO wallet_aliases VALUES (?, ?, '2026-11-05T00:00:00Z')", (wid, alias))
    con.commit()
    con.close()

    coins = ("BTC", "ETH", "SOL")
    _write_synthetic_ohlcv_and_funding(store_root, coins, start_dt, n_days)

    tid = 1
    for d in range(n_days):
        day_dt = start_dt + timedelta(days=d)
        day_rows: list[dict[str, object]] = []
        for w_idx, (_wid, addr, _alias, coin) in enumerate(wallets):
            for open_hour in (0, 16):
                open_dt = day_dt + timedelta(hours=open_hour, minutes=15 + w_idx * 3)
                close_dt = open_dt + timedelta(hours=4)
                qty = 3.0 + 0.5 * w_idx
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": open_dt,
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "buy",
                        "size": qty,
                        "start_position": 0.0,
                        "trade_id": tid,
                    }
                )
                day_rows.append(
                    {
                        "wallet": addr,
                        "exchange_ts": close_dt,
                        "coin": coin,
                        "dex": "hyperliquid",
                        "side": "sell",
                        "size": qty,
                        "start_position": qty,
                        "trade_id": tid + 1,
                    }
                )
                tid += 2
            if d % 2 == 0:
                l_open = day_dt + timedelta(hours=5, minutes=w_idx * 2)
                l_close = l_open + timedelta(hours=1)
                l_qty = round(qty * 0.8, 2)
                day_rows.extend(
                    [
                        {
                            "wallet": addr,
                            "exchange_ts": l_open,
                            "coin": coin,
                            "dex": "hyperliquid",
                            "side": "buy",
                            "size": l_qty,
                            "start_position": 0.0,
                            "trade_id": tid,
                        },
                        {
                            "wallet": addr,
                            "exchange_ts": l_close,
                            "coin": coin,
                            "dex": "hyperliquid",
                            "side": "sell",
                            "size": l_qty,
                            "start_position": l_qty,
                            "trade_id": tid + 1,
                        },
                    ]
                )
                tid += 2
            s_open = day_dt + timedelta(hours=20, minutes=15 + w_idx * 2)
            s_close = day_dt + timedelta(hours=23, minutes=30 + w_idx * 2)
            s_qty = 2.0 + 0.3 * w_idx
            day_rows.append(
                {
                    "wallet": addr,
                    "exchange_ts": s_open,
                    "coin": coin,
                    "dex": "hyperliquid",
                    "side": "sell",
                    "size": s_qty,
                    "start_position": 0.0,
                    "trade_id": tid,
                }
            )
            day_rows.append(
                {
                    "wallet": addr,
                    "exchange_ts": s_close,
                    "coin": coin,
                    "dex": "hyperliquid",
                    "side": "buy",
                    "size": s_qty,
                    "start_position": -s_qty,
                    "trade_id": tid + 1,
                }
            )
            tid += 2
        _write_synthetic_day_fills(archive_root, day_dt.date().isoformat(), day_rows)

    def _mock_challenge_spec(slug: str) -> SimpleNamespace:
        is_eq = "hyrotrader" in slug
        trailing = "INTRADAY_EQUITY" if "classic_2step" in slug else "STATIC"
        return SimpleNamespace(
            key=slug,
            initial="5000",
            target="500" if "hyrotrader" in slug or "propr" in slug else "400",
            daily_loss="250" if "hyrotrader" in slug or "crypto-fund-trader" in slug else "150",
            trailing_loss="500" if "hyrotrader" in slug or "crypto-fund-trader" in slug else "300",
            daily_basis="EQUITY" if is_eq else "BALANCE",
            trailing_basis=trailing,
            reset_time_hhmm="00:00",
        )

    cfg = p3.Phase3AblationConfig(
        start_iso="2026-10-26T00:00:00Z",
        split_iso="2026-11-02T00:00:00Z",
        wf_end_iso="2026-11-05T00:00:00Z",
        oos_end_iso="2026-11-05T00:00:00Z",
        pit_strict_iso="2026-11-06T00:00:00Z",
        archive_end_iso=archive_end_dt.isoformat().replace("+00:00", "Z"),
        registry_db=db_path,
        archive_root=archive_root,
        store_root=store_root,
        vike_root=None,
        rules_json_path=ROOT / "prop_firms_rules.json",
        report_md=report_md,
        report_json=report_json,
        pod_size=3,
        top_per_cluster=2,
        max_candidate_pods=8,
        min_active_bar_frac=0.001,
        min_episodes=3,
        min_overlap_days=1,
        min_lot_usd=15.0,
        min_baseline_order_usd=15.0,
        max_marginal_cdar_share=0.45,
        cpcv_n_blocks=4,
        l1_window_days=1,
        mc_n_paths=50,
        walk_forward_train_days=5,
        walk_forward_eval_days=2,
        pit_strict_lookback_days=7,
        fetch_bybit_specs=False,
        write_reports=True,
    )

    payload = p3.run_phase3_ablation(cfg, challenge_spec_loader=_mock_challenge_spec)

    assert "section_a" in payload and "section_b" in payload
    assert "section_c" in payload and "section_d" in payload
    assert payload["stage_timings_s"]["peak_rss_mb"] > 0.0
    assert payload["section_a"]["counts"]["reservoir_train_active"] == 6
    assert len(payload["section_a"]["oos_outcomes"]) == 18
    cpcv_sec = payload["section_a"]["cpcv"]
    assert cpcv_sec["leakfree_evaluated_splits"] > 0
    assert cpcv_sec["leakfree_total_splits"] >= cpcv_sec["leakfree_evaluated_splits"]
    assert "leakfree_splits" not in cpcv_sec
    assert "full_pipeline_splits" not in cpcv_sec
    assert cpcv_sec["full_pipeline_evaluated_splits"] > 0
    assert cpcv_sec["full_pipeline_total_splits"] >= cpcv_sec["full_pipeline_evaluated_splits"]
    assert sum(cpcv_sec["full_pipeline_status_counts"].values()) == cpcv_sec["full_pipeline_total_splits"]

    for t_key in ("2", "3"):
        arms = payload["section_b"][t_key]["lifecycle_arms"]
        reps = payload["section_b"][t_key]["replacement_modes"]
        assert len(arms) == 4
        assert len(reps) == 3
        for row in list(arms.values()) + list(reps.values()):
            assert row["max_pm03_conservation_err"] < 1e-10
            assert row["static_floor_dd_usd"] <= 0.0

    for firm in payload["section_d"]["firms"]:
        assert 0.0 < firm["payout_split_pct"] <= 1.0
        for reg_name in ("Blended", "75%-Chop"):
            unp = firm["regimes"][reg_name]["fee_unverified"]
            s110 = firm["regimes"][reg_name]["fee_110_sensitivity"]
            assert unp["breakeven_fee_usd"] - 110.0 == pytest.approx(s110["net_ev_usd"], abs=0.02)

    wf_folds = payload["section_c"]["walk_forward_folds"]
    assert len(wf_folds) >= 4
    assert len({row["fold_idx"] for row in wf_folds}) >= 2
    assert {row["label"] for row in wf_folds} == {"Tier2_MarginalEuler", "Tier1_StandaloneProxy"}
    assert all(
        row["verdict"] != "TIMEOUT"
        for row in payload["section_c"]["pit_strict_nov05"]["outcomes"]
    )
    assert (
        payload["section_d"]["blocked_fail_closed"]["status"]
        == "BLOCKED_UNSUPPORTED_TRAILING_BASIS:INTRADAY_EQUITY"
    )

    md_text = report_md.read_text(encoding="utf-8")
    json_text = report_json.read_text(encoding="utf-8")
    p3._check_report_hygiene(md_text, json_text)
    assert not HEX_RE.search(md_text)
    assert not HEX_RE.search(json_text)
    assert "\u2014" not in md_text
    assert "evaluated of" in md_text
    assert "Term net ($)" in md_text
    assert "Full net ($)" in md_text
    assert "Section A CPCV note." in md_text
    assert "Section B continuation note." in md_text
    assert "Section C execution note." in md_text
    assert sd.sanitize_public_label("Whale_0x5f94") in json_text
    assert "### Three-line caveat footer" in md_text
    assert json.loads(json_text)["archive_days"] == n_days


def test_slice_tensor_bars_sparse_3d_bit_identical_and_inactive_zero() -> None:
    n_t, n_s, n_days, bpd = 5, 4, 6, 288
    n_bars = n_days * bpd
    t0 = sd.to_utc_ms(datetime(2026, 9, 1, 0, 0, tzinfo=UTC))
    ts = t0 + np.arange(n_bars, dtype=np.int64) * 300_000

    rng = np.random.default_rng(123)
    notional = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    delta = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    sym_net = np.zeros((n_t, n_s, n_bars), dtype=np.float32)

    active_pairs = [(0, 1), (2, 3), (4, 0)]
    for ti, si in active_pairs:
        notional[ti, si] = rng.normal(0.0, 0.25, size=n_bars).astype(np.float32)
        delta[ti, si] = rng.normal(0.0, 0.05, size=n_bars).astype(np.float32)
        sym_net[ti, si] = rng.normal(0.0, 0.002, size=n_bars).astype(np.float32)

    net = np.sum(sym_net.astype(np.float64), axis=1)
    gross = net + np.sum(np.abs(delta.astype(np.float64)), axis=1) * 6.5e-4
    open_px = 100.0 + np.cumsum(rng.normal(0.0, 0.1, size=(n_s, n_bars)), axis=1)
    close_px = open_px + rng.normal(0.0, 0.05, size=(n_s, n_bars))
    chop = np.arange(n_bars) >= (n_bars // 2)

    tensor = pc.UnitCopy5mTensor(
        aliases=tuple(f"T{i}" for i in range(n_t)),
        symbols=("BTC", "ETH", "SOL", "ARB"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=delta,
        gross_ret_frac=gross,
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
        chop_mask=chop,
        exec_open_px=open_px,
        mark_close_px=close_px,
    )

    mask_2d = pc._compute_active_2d_mask(tensor)
    for ti in range(n_t):
        for si in range(n_s):
            assert bool(mask_2d[ti, si]) == ((ti, si) in active_pairs)

    multi_seg_idx = np.concatenate(
        [np.arange(0, bpd), np.arange(2 * bpd, 4 * bpd), np.arange(5 * bpd, 6 * bpd)]
    ).astype(np.int32)
    many_runs_idx = np.arange(17, n_bars, 3, dtype=np.int32)

    for bar_idx in (multi_seg_idx, many_runs_idx):
        sliced = pc._slice_tensor_bars(tensor, bar_idx, active_mask_2d=mask_2d)
        assert np.array_equal(sliced.timestamps_ms, ts[bar_idx])
        assert np.array_equal(sliced.notional_frac, notional[:, :, bar_idx])
        assert np.array_equal(sliced.delta_notional_frac, delta[:, :, bar_idx])
        assert sliced.sym_net_ret_frac is not None
        assert np.array_equal(sliced.sym_net_ret_frac, sym_net[:, :, bar_idx])
        assert np.array_equal(sliced.gross_ret_frac, gross[:, bar_idx])
        assert np.array_equal(sliced.net_ret_frac, net[:, bar_idx])
        assert sliced.chop_mask is not None
        assert np.array_equal(sliced.chop_mask, chop[bar_idx])
        assert sliced.exec_open_px is not None
        assert np.array_equal(sliced.exec_open_px, open_px[:, bar_idx])
        assert sliced.mark_close_px is not None
        assert np.array_equal(sliced.mark_close_px, close_px[:, bar_idx])
        for ti in range(n_t):
            for si in range(n_s):
                if (ti, si) not in active_pairs:
                    assert not sliced.notional_frac[ti, si].any()
                    assert not sliced.delta_notional_frac[ti, si].any()
                    assert not sliced.sym_net_ret_frac[ti, si].any()

    no_sym = pc._slice_tensor_bars(tensor, multi_seg_idx, include_sym_net_ret=False)
    assert no_sym.sym_net_ret_frac is None
    assert np.array_equal(no_sym.notional_frac, notional[:, :, multi_seg_idx])

    empty_sliced = pc._slice_tensor_bars(tensor, np.array([], dtype=np.int32))
    assert empty_sliced.notional_frac.shape == (n_t, n_s, 0)
    assert empty_sliced.net_ret_frac.shape == (n_t, 0)
    assert empty_sliced.timestamps_ms.shape == (0,)


def test_size_pod_dual_barrier_unrl_parity_and_euler_combo_skip() -> None:
    bpd = 288
    n_days = 14
    n_bars = n_days * bpd
    t0 = sd.to_utc_ms(datetime(2026, 9, 1, 0, 0, tzinfo=UTC))
    ts = t0 + np.arange(n_bars, dtype=np.int64) * 300_000

    bar_i = np.arange(n_bars, dtype=np.float64)
    open_px = np.vstack(
        [
            100.0 + 3.0 * np.sin(bar_i * 0.03) + 0.002 * bar_i,
            200.0 + 5.0 * np.cos(bar_i * 0.025) - 0.001 * bar_i,
            50.0 + 1.5 * np.sin(bar_i * 0.04 + 1.0),
        ]
    )
    close_px = open_px + np.vstack(
        [
            0.15 * np.cos(bar_i * 0.05),
            -0.20 * np.sin(bar_i * 0.04),
            0.08 * np.cos(bar_i * 0.06),
        ]
    )

    n_t, n_s = 4, 3
    notional = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    delta = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    sym_net = np.zeros((n_t, n_s, n_bars), dtype=np.float32)

    for d in range(n_days):
        sb = d * bpd + 10
        eb = d * bpd + 120
        notional[0, 0, sb:eb] = 0.25
        delta[0, 0, sb] = 0.25
        delta[0, 0, eb] = -0.25
        if d == 12:
            sym_net[0, 0, sb + 1 : sb + 21] = -0.0060 / 20.0
            sym_net[0, 0, sb + 21 : sb + 41] = 0.0075 / 20.0
        else:
            sym_net[0, 0, sb + 1 : eb + 1] = (-0.00001 if d in (2, 6) else 0.0020) / 110.0

        notional[1, 0, sb:eb] = 0.20
        delta[1, 0, sb] = 0.20
        delta[1, 0, eb] = -0.20
        sym_net[1, 0, sb + 1 : eb + 1] = (-0.0030 if d in (2, 6) else 0.0018) / 110.0

        sb2 = d * bpd + 130
        eb2 = d * bpd + 200
        notional[2, 1, sb2:eb2] = 0.22
        delta[2, 1, sb2] = 0.22
        delta[2, 1, eb2] = -0.22
        sym_net[2, 1, sb2 + 1 : eb2 + 1] = (
            -0.0050 if d == 2 else (0.0050 if d == 6 else 0.0014)
        ) / 70.0

        sb3 = d * bpd + 210
        eb3 = min(n_bars - 2, (d + 1) * bpd + 15)
        notional[3, 2, sb3:eb3] = 0.22
        delta[3, 2, sb3] = 0.22
        delta[3, 2, eb3] = -0.22
        sym_net[3, 2, sb3 + 1 : eb3 + 1] = (
            0.0050 if d == 2 else (-0.0050 if d == 6 else 0.0014)
        ) / max(1, eb3 - sb3)

    net = np.sum(sym_net.astype(np.float64), axis=1)
    gross = net.copy()
    tensor = pc.UnitCopy5mTensor(
        aliases=("T0", "T1", "T2", "T3"),
        symbols=("BTC", "ETH", "SOL"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=delta,
        gross_ret_frac=gross,
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
        chop_mask=bar_i >= (n_bars // 2),
        exec_open_px=open_px,
        mark_close_px=close_px,
    )

    cfg = pc.PodConstructorConfig(
        pod_size=3,
        top_per_cluster=2,
        max_candidate_pods=8,
        min_episodes=4,
        min_active_bar_frac=0.01,
        reject_unknown_tilt=False,
        risk_share_mode="marginal_euler",
        max_marginal_cdar_share=0.45,
        daily_loss_mode="reset_balance",
    )
    _, metrics, _ = pc.stage1_audit_traders(tensor, cfg)

    pod = (1, 2, 3)
    pod_list = list(pod)
    day_slices = pc._daily_reset_slices(ts, bpd, cfg.reset_bar_offset)
    active_syms = [0, 1, 2]
    pod_notional = tensor.notional_frac[np.ix_(pod_list, active_syms)].astype(np.float64, copy=False)
    for segs in (None, [(0, 6 * bpd), (6 * bpd, n_bars)]):
        seg_slices = pc._daily_reset_slices(ts, bpd, cfg.reset_bar_offset, segments=segs)
        r_targets, sym_cache = pc._build_unrl_reset_cache(
            pod_notional, active_syms, open_px, close_px, seg_slices, segs, tensor.pod_equity_usd
        )
        for w in (
            np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            np.array([0.55, 0.25, 0.20]),
        ):
            u_fast = pc._eval_unrl_reset_cache(w, len(seg_slices), r_targets, sym_cache, tensor.pod_equity_usd)
            u_ref = pc._compute_netted_unrealized_frac_at_resets(
                w, tensor, pod, seg_slices, cfg.cost_bps * 1e-4, active_syms=active_syms, segments=segs
            )
            assert np.allclose(u_fast, u_ref, atol=1e-12, rtol=0.0)
            assert np.any(np.abs(u_ref) > 1e-6)

    sized_ok = pc.size_pod_dual_barrier(pod, tensor, metrics, cfg)
    unrl_ref = pc._compute_netted_unrealized_frac_at_resets(
        sized_ok.multipliers, tensor, pod, day_slices, cfg.cost_bps * 1e-4, active_syms=active_syms
    )
    solo_n, t_ov, d_ov = pc._build_sparse_overlap_cache(
        tensor.delta_notional_frac[pod_list], tensor.net_ret_frac[pod_list]
    )
    _, _, _, _, dcvar_ref, mdaily_ref, _ = pc._evaluate_weights_unit_scale(
        sized_ok.multipliers, solo_n, t_ov, d_ov, cfg.cost_bps * 1e-4, day_slices, cfg, unrl_at_resets=unrl_ref
    )
    assert sized_ok.max_daily_loss == pytest.approx(mdaily_ref, abs=1e-12)
    assert sized_ok.daily_cvar_99 == pytest.approx(dcvar_ref, abs=1e-12)

    with pytest.raises(ValueError, match="INFEASIBLE_MARGINAL_RISK_SHARE_BOUNDS"):
        pc.size_pod_dual_barrier((0, 2, 3), tensor, metrics, cfg)

    pods, _, _, _ = pc.discover_diverse_pods(tensor, cfg)
    assert pods
    assert pods[0].aliases == ("T1", "T2", "T3")
    assert max(pods[0].marginal_cdar_shares) <= 0.45 + 1e-9

    with pytest.raises(ValueError, match="INFEASIBLE_MARGINAL_RISK_SHARE_BOUNDS"):
        pc.discover_diverse_pods(tensor, replace(cfg, max_marginal_cdar_share=0.20))


def test_slice_tensor_bars_negative_idx_validation_and_carried_copy() -> None:
    n_t, n_s, n_bars = 2, 2, 10
    t0 = sd.to_utc_ms(datetime(2026, 9, 1, 0, 0, tzinfo=UTC))
    ts = t0 + np.arange(n_bars, dtype=np.int64) * 300_000
    notional = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    notional[0, 0, 7:] = 0.25
    delta = np.zeros((n_t, n_s, n_bars), dtype=np.float32)
    delta[0, 0, 7] = 0.25
    carried_in = np.zeros((n_t, n_s), dtype=bool)
    carried_in[1, 1] = True

    tensor = pc.UnitCopy5mTensor(
        aliases=("A", "B"),
        symbols=("BTC", "ETH"),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=delta,
        gross_ret_frac=np.zeros((n_t, n_bars), dtype=np.float64),
        net_ret_frac=np.zeros((n_t, n_bars), dtype=np.float64),
        carried_in_mask=carried_in,
    )

    tail = pc._slice_tensor_bars(tensor, np.array([-2, -1], dtype=np.int64))
    assert np.array_equal(tail.timestamps_ms, ts[-2:])
    assert np.array_equal(tail.notional_frac, notional[:, :, -2:])
    assert tail.carried_in_mask is not None
    assert bool(tail.carried_in_mask[0, 0]) is True

    head = pc._slice_tensor_bars(tensor, np.array([0, 1], dtype=np.int64))
    assert head.carried_in_mask is not None
    assert head.carried_in_mask is not tensor.carried_in_mask
    head.carried_in_mask[1, 1] = False
    assert bool(carried_in[1, 1]) is True

    mask2d = pc._compute_active_2d_mask(tensor)
    sparse_neg = pc._slice_3d_sparse_bars(notional, np.array([-2, -1], dtype=np.int64), mask2d)
    assert np.array_equal(sparse_neg, notional[:, :, -2:])

    bool_idx = np.ones(n_bars, dtype=bool)
    idx_2d = np.array([[0, 1]], dtype=np.int64)
    for bad_idx in (bool_idx, idx_2d):
        with pytest.raises(ValueError, match="BAR_IDX_MUST_BE_1D_INT"):
            pc._slice_tensor_bars(tensor, bad_idx)
        with pytest.raises(ValueError, match="BAR_IDX_MUST_BE_1D_INT"):
            pc._slice_3d_sparse_bars(notional, bad_idx, mask2d)

    with pytest.raises(IndexError, match="BAR_IDX_OUT_OF_BOUNDS"):
        pc._slice_tensor_bars(tensor, np.array([n_bars], dtype=np.int64))


def test_unrl_reset_cpcv_segment_boundary_and_nan_open_forward_fill_parity() -> None:
    bpd = 288
    n_days = 4
    n_bars = n_days * bpd
    t0 = sd.to_utc_ms(datetime(2026, 9, 1, 0, 0, tzinfo=UTC))
    ts = t0 + np.arange(n_bars, dtype=np.int64) * 300_000

    open_px = np.full((1, n_bars), 100.0, dtype=np.float64)
    close_px = np.full((1, n_bars), 100.0, dtype=np.float64)

    open_px[0, 50:290] = 100.0
    close_px[0, 288] = 80.0
    open_px[0, 576:] = 200.0
    close_px[0, 576:] = 200.0
    close_px[0, 2 * bpd + 288] = 220.0

    open_px[0, 120] = np.nan
    close_px[0, 120] = 100.0
    open_px[0, 121:124] = np.nan
    close_px[0, 121:124] = np.nan
    open_px[0, 125] = 0.0
    close_px[0, 125] = -5.0

    notional = np.zeros((2, 1, n_bars), dtype=np.float64)
    notional[0, 0, 50:] = 0.40
    notional[1, 0, 50:122] = 0.20
    notional[1, 0, 130:] = 0.20

    delta = np.zeros_like(notional)
    delta[:, 0, 50] = notional[:, 0, 50]
    delta[1, 0, 122] = -0.20
    delta[1, 0, 130] = 0.20

    tensor = pc.UnitCopy5mTensor(
        aliases=("A", "B"),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=delta,
        gross_ret_frac=np.zeros((2, n_bars), dtype=np.float64),
        net_ret_frac=np.zeros((2, n_bars), dtype=np.float64),
        exec_open_px=open_px,
        mark_close_px=close_px,
    )

    segments = [(0, 2 * bpd), (2 * bpd, n_bars)]
    day_slices = pc._daily_reset_slices(ts, bpd, 1, segments=segments)
    active_syms = [0]
    pod_notional = tensor.notional_frac[np.ix_([0, 1], active_syms)].astype(np.float64, copy=False)
    r_targets, sym_cache = pc._build_unrl_reset_cache(
        pod_notional, active_syms, open_px, close_px, day_slices, segments, tensor.pod_equity_usd
    )

    for w in (np.array([1.0, 0.0]), np.array([0.6, 0.4])):
        u_fast = pc._eval_unrl_reset_cache(w, len(day_slices), r_targets, sym_cache, tensor.pod_equity_usd)
        u_ref = pc._compute_netted_unrealized_frac_at_resets(
            w, tensor, (0, 1), day_slices, 0.0, active_syms=active_syms, segments=segments
        )
        assert np.allclose(u_fast, u_ref, atol=1e-12, rtol=0.0)

    w0 = np.array([1.0, 0.0])
    u_seg = pc._compute_netted_unrealized_frac_at_resets(
        w0, tensor, (0, 1), day_slices, 0.0, active_syms=active_syms, segments=segments
    )
    seg1_day3_idx = next(di for di, (sb, _eb) in enumerate(day_slices) if sb == 2 * bpd + 289)
    assert u_seg[seg1_day3_idx] == pytest.approx(0.04, abs=1e-12)


def test_stage1_audit_segmented_enforces_reject_hidden_bagholder() -> None:
    n_bars = 240
    t0 = sd.to_utc_ms(datetime(2026, 9, 1, 0, 0, tzinfo=UTC))
    ts = t0 + np.arange(n_bars, dtype=np.int64) * 300_000
    notional = np.zeros((2, 1, n_bars), dtype=np.float64)
    delta = np.zeros((2, 1, n_bars), dtype=np.float64)
    sym_net = np.zeros((2, 1, n_bars), dtype=np.float64)

    for ti in (0, 1):
        for seg_off in (0, 120):
            for k in range(2):
                sb = seg_off + 10 + k * 25
                eb = sb + 10
                notional[ti, 0, sb:eb] = 0.4
                delta[ti, 0, sb] = 0.4
                delta[ti, 0, eb] = -0.4
                sym_net[ti, 0, sb + 1 : eb + 1] = 0.00125

    notional[1, 0, 190:] = 0.4
    delta[1, 0, 190] = 0.4
    sym_net[1, 0, 200] = -0.035

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("CLEAN", "BAGHOLDER"),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=delta,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    segments = [(0, 120), (120, 240)]
    cfg = pc.PodConstructorConfig(
        min_episodes=2,
        min_active_bar_frac=0.01,
        reject_unknown_tilt=False,
        reject_hidden_bagholder=True,
        min_baseline_order_usd=0.0,
    )
    passed_rej, metrics_rej = pc._stage1_audit_segmented(tensor, segments, cfg)
    assert 0 in passed_rej
    assert 1 not in passed_rej
    assert "right_censored_bagholder" in metrics_rej[1].reject_reason

    passed_allow, metrics_allow = pc._stage1_audit_segmented(
        tensor, segments, replace(cfg, reject_hidden_bagholder=False)
    )
    assert 1 in passed_allow
    assert metrics_allow[1].reject_reason == ""


def test_funding_sanitization_zero_rate_coverage_and_1d_grid_validation(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    t0 = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)

    bad_path = (
        store_root
        / "crypto_futures_funding_BADUSDT_PERP"
        / "year=2026"
        / "month=09"
        / "data.parquet"
    )
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [
                t0,
                t0 + timedelta(hours=4),
                t0 + timedelta(hours=8),
                t0 + timedelta(hours=12),
                t0 + timedelta(hours=16),
            ],
            "funding_rate": [0.001, 0.001, 0.001, float("inf"), float("-inf")],
            "funding_interval_hours": [0.0, -4.0, float("inf"), 8.0, 8.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(bad_path)

    zero_path = (
        store_root
        / "crypto_futures_funding_ZEROUSDT_PERP"
        / "year=2026"
        / "month=09"
        / "data.parquet"
    )
    zero_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [t0, t0 + timedelta(hours=8)],
            "funding_rate": [0.0, 0.0],
            "funding_interval_hours": [8.0, 8.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(zero_path)

    start_ms = sd.to_utc_ms(t0)
    grid = np.arange(start_ms, start_ms + 24 * 3_600_000, 300_000, dtype=np.int64)
    rates, covered = sd.load_usdt_funding_rates_on_grid(("BAD", "ZERO", "MISSING"), grid, store_root)
    assert np.all(np.isfinite(rates))
    assert np.all(rates == 0.0)
    assert covered == 2

    _, cov_zero_only = sd.load_usdt_funding_rates_on_grid(("ZERO",), grid, store_root)
    assert cov_zero_only == 1

    prices = sd.LazyPrices(grid, store_root=store_root, vike_root=None)
    for bad_grid in (np.zeros((2, 2), dtype=np.int64), np.zeros((0, 0), dtype=np.int64)):
        with pytest.raises(ValueError, match="GRID_MS_MUST_BE_1D"):
            sd.load_usdt_funding_rates_on_grid(("ZERO",), bad_grid, store_root)
        with pytest.raises(ValueError, match="GRID_MS_MUST_BE_1D"):
            sd.load_1m_volumes_on_grid(("ZERO",), bad_grid, prices)


def test_prop_contract_vanta_propr_comma_and_hyro_40pct_credit_cap() -> None:
    assert pc._extract_leading_number("$3,500") == pytest.approx(3500.0)
    assert pc._extract_leading_number(True) is None

    vanta = pc.load_prop_contract_spec("vanta-trading", account_usd=10_000.0)
    assert vanta.payout_split_pct == pytest.approx(1.0)
    assert vanta.funded_phase.consistency_max_single_day_pct is None

    propr_c1 = pc.load_prop_contract_spec(
        "propr", account_usd=10_000.0, program_filter="Classic 1-Step"
    )
    assert propr_c1.payout_split_pct == pytest.approx(0.80)
    assert propr_c1.eval_phases[0].profit_target_usd == pytest.approx(1000.0)

    hyro = pc.load_prop_contract_spec("hyrotrader", account_usd=10_000.0, min_eval_trading_days=1)
    assert hyro.eval_phases[0].consistency_max_single_day_pct == pytest.approx(0.40)

    giant_blocks = (
        np.zeros(4, dtype=bool),
        np.array([2500.0, 10.0, 10.0, 10.0]),
        np.array([2500.0, 10.0, 10.0, 10.0]),
        np.zeros(4),
        np.zeros(4, dtype=np.int32),
        np.ones(4, dtype=bool),
    )
    one_day_hyro = replace(
        hyro,
        eval_phases=(replace(hyro.eval_phases[0], horizon_days=1, min_trading_days=1),),
    )
    ev_hyro = pc.evaluate_multiphase_contract_ev(
        giant_blocks, one_day_hyro, n_paths=300, mean_block_days=1.0, rng=np.random.default_rng(5)
    )
    assert ev_hyro["phase_pass_rates_pct"]["phase_1"] == 0.0
