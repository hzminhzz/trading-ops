"""Adversarial planted-trap falsifier suite for deep underwriting gates."""

from __future__ import annotations

import importlib
import math
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = ROOT / "studies"
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

cr = importlib.import_module("canonical_replay")
pc = importlib.import_module("pod_constructor")
sd = importlib.import_module("study_data")


def _grid_ms(n_bars: int, start_dt: datetime = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)) -> np.ndarray:
    start_ms = int(start_dt.timestamp() * 1000)
    return start_ms + np.arange(n_bars, dtype=np.int64) * 300_000


def _write_1m_bars(
    store_root: Path,
    coin: str,
    start_dt: datetime,
    n_minutes: int,
    opens: np.ndarray,
    closes: np.ndarray,
    quote_volumes: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
) -> None:
    day_str = start_dt.date().isoformat()
    y, m, d = day_str.split("-")
    path = (
        store_root
        / f"crypto_futures_ohlcv_1m_HYPERLIQUID_{coin}_PERP"
        / f"year={y}"
        / f"month={m}"
        / f"day={d}"
        / "data.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = [start_dt + timedelta(minutes=i) for i in range(n_minutes)]
    data: dict[str, object] = {
        "timestamp": ts,
        "symbol": [coin] * n_minutes,
        "open": opens.tolist(),
        "close": closes.tolist(),
    }
    if quote_volumes is not None:
        data["quote_volume"] = quote_volumes.tolist()
        data["volume"] = (
            volumes.tolist()
            if volumes is not None
            else (quote_volumes / np.maximum(opens, 1e-8)).tolist()
        )
    elif volumes is not None:
        data["volume"] = volumes.tolist()
    pl.DataFrame(data).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC"))).write_parquet(
        path
    )


def test_falsifier_a_overnight_balance_reset_and_multiphase_n_eff_ci() -> None:
    # Planted trap (a): Q11 overnight counterexample ($9,900 balance vs $9,700 equity at 00:05 UTC).
    n = 310
    grid = _grid_ms(n)
    t_open = np.full((2, n), 100.0)
    m_close = np.full((2, n), 100.0)
    targets = np.zeros((2, n))
    targets[0, 10:20] = 10.0
    m_close[0, 15:] = 90.0
    t_open[0, 16:] = 90.0
    targets[1, 100:] = 10.0
    m_close[1, 200:295] = 80.0
    t_open[1, 201:296] = 80.0
    m_close[1, 295:] = 49.90
    t_open[1, 296:] = 49.90

    res_bal, blocks_bal = cr.simulate_guardian_challenge(
        ["C0", "C1"],
        targets,
        t_open,
        m_close,
        grid,
        cr.GuardianSpec(cost_bps=0.0, guardian_sweep_loss=10_000.0, daily_loss_basis="balance"),
    )
    assert res_bal.verdict == "FAIL_DAILY"
    assert res_bal.worst_daily_usd == pytest.approx(-501.0, abs=1e-9)
    assert len(blocks_bal) == 7

    res_eq, _ = cr.simulate_guardian_challenge(
        ["C0", "C1"],
        targets,
        t_open,
        m_close,
        grid,
        cr.GuardianSpec(cost_bps=0.0, guardian_sweep_loss=400.0, daily_loss_basis="equity"),
    )
    assert res_eq.verdict == "TIMEOUT"

    # Multi-phase Contract EV: Wilson CI uses N_eff = n_days / mean_block_days (never n_paths).
    n_days = 28
    is_chop = np.zeros(n_days, dtype=bool)
    day_end = np.where(np.arange(n_days) % 3 == 2, -60.0, 95.0)
    day_max = np.maximum(day_end, 120.0)
    day_min = np.minimum(day_end, -80.0)
    day_swept = np.zeros(n_days, dtype=np.int32)
    day_mf = np.ones(n_days, dtype=bool)
    blocks = (is_chop, day_end, day_max, day_min, day_swept, day_mf)

    cft_unpriced = pc.load_prop_contract_spec("crypto-fund-trader", account_usd=10_000.0)
    ev_unpriced = pc.evaluate_multiphase_contract_ev(
        blocks, cft_unpriced, n_paths=2_000, mean_block_days=7.0, rng=np.random.default_rng(7)
    )
    assert ev_unpriced["ev_status"] == "BREAK_EVEN_ONLY_FEE_UNVERIFIED"
    assert ev_unpriced["net_ev_usd"] is None
    assert ev_unpriced["breakeven_fee_usd"] > 0.0
    assert ev_unpriced["n_eff_blocks"] == pytest.approx(4.0, abs=1e-6)

    ev_7tuple = pc.evaluate_multiphase_contract_ev(
        blocks_bal, cft_unpriced, n_paths=200, mean_block_days=2.0, rng=np.random.default_rng(7)
    )
    assert ev_7tuple["p_pass_all_eval_pct"] == 0.0
    with pytest.raises(ValueError, match="UNEXPECTED_MC_BLOCKS_LEN"):
        pc.evaluate_multiphase_contract_ev(blocks[:5], cft_unpriced)

    cft_priced = pc.load_prop_contract_spec(
        "crypto-fund-trader", account_usd=10_000.0, challenge_fee_usd=120.0
    )
    ev_5k = pc.evaluate_multiphase_contract_ev(
        blocks, cft_priced, n_paths=5_000, mean_block_days=7.0, rng=np.random.default_rng(11)
    )
    ev_30k = pc.evaluate_multiphase_contract_ev(
        blocks, cft_priced, n_paths=30_000, mean_block_days=7.0, rng=np.random.default_rng(11)
    )
    assert ev_30k["ev_status"] == "VERIFIED_FEE_NET_EV"
    lo_30k, hi_30k = ev_30k["p_funded_payout_ci95_pct"]  # type: ignore[misc]
    assert (hi_30k - lo_30k) > 25.0
    assert abs(ev_5k["n_eff_blocks"] - ev_30k["n_eff_blocks"]) < 1e-9

    # Consistency rule hurdle delays early pass when single-day spike dominates.
    spike_end = np.array([800.0, 50.0, 50.0, 50.0])
    spike_blocks = (
        np.zeros(4, dtype=bool),
        spike_end,
        spike_end + 10.0,
        np.zeros(4),
        np.zeros(4, dtype=np.int32),
        np.ones(4, dtype=bool),
    )
    cons_contract = pc.PropContractSpec(
        firm_slug="hyrotrader",
        program_name="consistency-test",
        account_usd=10_000.0,
        eval_phases=(
            pc.PropPhaseSpec(
                "phase_1",
                800.0,
                1_000.0,
                500.0,
                "equity",
                min_trading_days=2,
                consistency_max_single_day_pct=0.40,
                horizon_days=2,
            ),
        ),
        funded_phase=pc.PropPhaseSpec(
            "funded", None, 1_000.0, 500.0, "equity", min_trading_days=0, horizon_days=2
        ),
        funded_cycle_days=2,
    )
    ev_cons = pc.evaluate_multiphase_contract_ev(
        spike_blocks, cons_contract, n_paths=500, mean_block_days=4.0, rng=np.random.default_rng(3)
    )
    assert ev_cons["p_pass_all_eval_pct"] == 0.0


def test_falsifier_b_high_roi_maker_and_scalper_rejected_by_5m_markouts(tmp_path: Path) -> None:
    # Planted trap (b): passive maker with negative fee & +100 bps pre-CP gain that reverts post-CP,
    # plus a sub-5m churn scalper, contrasted against a clean >=5m drift trader and split_ms boundary.
    start_dt = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    n_minutes = 600
    opens = np.full(n_minutes, 100.0, dtype=np.float64)
    for k in range(10):
        b = 10 + k * 40
        opens[b] = 100.0
        opens[b + 1] = 100.2
        opens[b + 5] = 100.5
        opens[b + 20] = 99.7

    opens_clean = np.full(n_minutes, 100.0, dtype=np.float64)
    for k in range(10):
        b = 10 + k * 40
        opens_clean[b] = 100.0
        opens_clean[b + 1] = 100.1
        opens_clean[b + 5] = 100.2
        opens_clean[b + 20] = 101.2
    # Adversarial OOS crash at minute 460 (post-horizon for a late Train fill at minute 440 -> cp 445 -> +15m 460)
    split_min = 450
    opens_clean[460] = 1.0

    _write_1m_bars(tmp_path, "BTC", start_dt, n_minutes, opens, opens.copy())
    _write_1m_bars(tmp_path, "ETH", start_dt, n_minutes, opens_clean, opens_clean.copy())

    grid = _grid_ms(n_minutes // 5, start_dt=start_dt)
    prices = sd.LazyPrices(grid, store_root=tmp_path, vike_root=None)

    base_ms = int(start_dt.timestamp() * 1000)
    split_ms = base_ms + split_min * 60_000
    maker_rows = []
    scalper_rows = []
    clean_rows = []
    for k in range(10):
        b = 10 + k * 40
        t0 = base_ms + b * 60_000 + 5_000
        maker_rows.append({"ts": t0, "sign": 1, "sz": 1.0, "px": 99.0, "fee": -0.02})
        scalper_rows.append({"ts": t0, "sign": 1, "sz": 2.0, "px": 100.0, "fee": 0.01})
        scalper_rows.append({"ts": t0 + 90_000, "sign": -1, "sz": 2.0, "px": 100.3, "fee": 0.01})
        clean_rows.append({"ts": t0, "sign": 1, "sz": 1.0, "px": 100.0, "fee": 0.02})

    prof_maker = sd.compute_fill_latency_markouts("MAKER_DECAY", {"BTC": pl.DataFrame(maker_rows)}, prices)
    assert prof_maker.leader_pre_cp_edge_bps is not None and prof_maker.leader_pre_cp_edge_bps > 100.0
    assert prof_maker.maker_rebate_frac == pytest.approx(1.0)
    assert (
        prof_maker.follower_post_cp_markout_bps is not None
        and prof_maker.follower_post_cp_markout_bps < -50.0
    )
    assert prof_maker.status == "REJECT_NEGATIVE_5M_MARKOUT"

    prof_scalper = sd.compute_fill_latency_markouts(
        "SCALPER", {"ETH": pl.DataFrame(scalper_rows)}, prices
    )
    assert prof_scalper.sub_5m_churn_frac > 0.90
    assert prof_scalper.status == "REJECT_SUB5M_CHURN"

    prof_clean = sd.compute_fill_latency_markouts(
        "CLEAN_DRIFT", {"ETH": pl.DataFrame(clean_rows)}, prices
    )
    assert prof_clean.status == "COPYABLE_DRIFT"
    assert (
        prof_clean.follower_post_cp_markout_bps is not None
        and prof_clean.follower_post_cp_markout_bps > 50.0
    )

    # Late Train fill whose cp_next + 15m lands at split_ms + 10m must be excluded when split_ms is set.
    late_train_fill = {"ts": base_ms + 440 * 60_000 + 5_000, "sign": 1, "sz": 1.0, "px": 100.0, "fee": 0.02}
    prof_split = sd.compute_fill_latency_markouts(
        "CLEAN_SPLIT",
        {"ETH": pl.DataFrame([*clean_rows, late_train_fill])},
        prices,
        split_ms=split_ms,
    )
    assert prof_split.evaluated_fills == 10
    assert prof_split.follower_post_cp_markout_bps == pytest.approx(
        prof_clean.follower_post_cp_markout_bps, abs=1e-9
    )
    assert prof_split.status == "COPYABLE_DRIFT"


def test_falsifier_c_right_censored_100pct_winrate_bagholder_rejected() -> None:
    # Planted trap (c): 6 closed winning episodes (100% closed WR) + 1 open right-censored -8% bag.
    n_bars = 240
    ts = _grid_ms(n_bars)
    notional = np.zeros((3, 1, n_bars), dtype=np.float64)
    d_notional = np.zeros((3, 1, n_bars), dtype=np.float64)
    sym_net = np.zeros((3, 1, n_bars), dtype=np.float64)

    for ti in range(2):
        for ep in range(6):
            sb = 10 + ep * 30
            eb = sb + 12
            notional[ti, 0, sb:eb] = 0.4
            d_notional[ti, 0, sb] = 0.4
            d_notional[ti, 0, eb] = -0.4
            sym_net[ti, 0, sb + 1 : eb + 1] = 0.001 if ep % 2 == 0 else -0.0003

    for ep in range(6):
        sb = 5 + ep * 25
        eb = sb + 10
        notional[2, 0, sb:eb] = 0.5
        d_notional[2, 0, sb] = 0.5
        d_notional[2, 0, eb] = -0.5
        sym_net[2, 0, sb + 1 : eb + 1] = 0.0005
    notional[2, 0, 170:] = 1.0
    d_notional[2, 0, 170] = 1.0
    sym_net[2, 0, 180] = -0.080

    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("CLEAN_A", "CLEAN_B", "BAGHOLDER"),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    cfg = pc.PodConstructorConfig(min_episodes=4, min_active_bar_frac=0.05, reject_unknown_tilt=False)
    passed_ids, metrics, _ = pc.stage1_audit_traders(tensor, cfg)
    bag = next(m for m in metrics if m.alias == "BAGHOLDER")
    assert bag.num_episodes == 6
    assert bag.right_censored_episodes == 1
    assert bag.right_censored_pnl_frac == pytest.approx(-0.08, abs=1e-9)
    assert not bag.passed_behavioral_gate
    assert "right_censored_bagholder" in bag.reject_reason
    assert 2 not in passed_ids
    assert np.isfinite(bag.eb_shrunk_ropn)


def test_falsifier_d_pit_universe_and_survivorship_leakage(tmp_path: Path) -> None:
    # Planted trap (d): post-cutoff exclusion leakage + pre-split live-AV filtering.
    db_path = tmp_path / "registry.sqlite"
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
        INSERT INTO wallets VALUES (1, '0xaaa'), (2, '0xbbb');
        INSERT INTO cohort_memberships VALUES (1, '2026-11-05T00:00:00Z'), (2, '2026-11-05T00:00:00Z');
        INSERT INTO monitoring VALUES (1, 1, '2026-11-05T00:00:00Z'), (2, 1, '2026-11-05T00:00:00Z');
        INSERT INTO exclusions VALUES (1, '2026-11-16T00:00:00Z'), (2, '2026-11-06T00:00:00Z');
        INSERT INTO wallet_aliases VALUES (1, 'W_EARLY', '2026-11-05T00:00:00Z'), (2, 'W_OLD_EXCL', '2026-11-05T00:00:00Z');
        """
    )
    con.commit()
    con.close()

    with pytest.raises(ValueError, match="EMPTY_PIT_UNIVERSE_BEFORE_RECEIPT_EPOCH"):
        cr.load_universe_asof(db_path, "2026-09-01T00:00:00Z", mode="PIT_STRICT_ASOF")

    res_nov10 = cr.load_universe_asof(db_path, "2026-11-10T00:00:00Z", mode="PIT_STRICT_ASOF")
    assert [r.label for r in res_nov10.records] == ["W_EARLY"]

    res_retro = cr.load_universe_asof(db_path, "2026-09-01T00:00:00Z", mode="RETROSPECTIVE_ASOF_MASKED")
    assert [r.label for r in res_retro.records] == ["W_EARLY", "W_OLD_EXCL"]
    assert res_retro.retrospective_evidence is not None
    assert res_retro.retrospective_evidence["kind"] == "RETROSPECTIVE_ESTIMATE"

    day_file = tmp_path / "reservoir-full-history-test" / "hyperliquid" / "fills" / "day=2026-08-15" / "data.parquet"
    day_file.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "wallet": ["0xaaa", "0xccc"],
            "exchange_ts": [datetime(2026, 8, 15, 12, 0, tzinfo=UTC)] * 2,
            "coin": ["BTC", "BTC"],
            "dex": ["hyperliquid", "hyperliquid"],
            "side": ["buy", "sell"],
            "size": [1.0, 2.0],
            "start_position": [0.0, 0.0],
            "trade_id": [1, 2],
        }
    ).with_columns(
        pl.col("exchange_ts").cast(pl.Datetime("us", "UTC")),
        pl.col("trade_id").cast(pl.UInt64),
    ).write_parquet(day_file)

    recs, _sha, prov = sd.discover_reservoir_pit_cohort(
        [day_file],
        train_start=datetime(2026, 8, 1, tzinfo=UTC),
        train_end=datetime(2026, 9, 1, tzinfo=UTC),
        registry_db=db_path,
    )
    assert prov == "EX_POST_RESERVOIR_ARCHIVE_ZERO_EXCLUSION_LEAKAGE"
    labels = sorted(r.label for r in recs)
    assert "W_EARLY" in labels
    assert any(lbl.startswith("R#") for lbl in labels)

    start_dt = datetime(2026, 8, 15, 0, 0, tzinfo=UTC)
    grid = _grid_ms(12, start_dt=start_dt)
    _write_1m_bars(tmp_path, "BTC", start_dt, 60, np.full(60, 100.0), np.full(60, 100.0))
    prices = sd.LazyPrices(grid, store_root=tmp_path, vike_root=None)
    ing_fills = {
        "W_EARLY": {"BTC": pl.DataFrame({"ts": [int(grid[1])], "sign": [1], "sz": [1.0], "pre": [0.0], "tid": ["1"]})},
        "W_OLD_EXCL": {"BTC": pl.DataFrame({"ts": [int(grid[1])], "sign": [1], "sz": [1.0], "pre": [0.0], "tid": ["2"]})},
    }
    stats_df = pl.DataFrame(
        {
            "wallet": ["0xaaa", "0xbbb"],
            "coin": ["BTC", "BTC"],
            "train_fills": [20, 20],
            "oos_fills": [0, 0],
            "total_fills": [20, 20],
            "first_fill_ms": [int(grid[0]), int(grid[0])],
            "last_train_fill_ms": [int(grid[1]), int(grid[1])],
            "last_fill_ms": [int(grid[1]), int(grid[1])],
        }
    )
    ing_obj = sd.FillIngestionResult(
        grouped=ing_fills,
        stats_df=stats_df,
        min_ts_ms=int(grid[0]),
        max_ts_ms=int(grid[1]),
        explain_plan="SYNTH",
    )
    av_map = {
        "W_EARLY": (5_000.0, None, None),
        "W_OLD_EXCL": (200.0, None, None),
    }
    screen_rows = sd.screen_wallets(
        res_retro.records,
        ing_obj,
        split=datetime(2026, 8, 16, 0, 0, tzinfo=UTC),
        bybit_usdt_perps={"BTC"},
        account_value_fetcher=lambda r: av_map[r.label],
        min_train_fills=10,
    )
    hist_chosen, _ = sd.eligible_fills(
        ing_fills, screen_rows, prices, split=datetime(2026, 8, 16, 0, 0, tzinfo=UTC), scope="historical"
    )
    live_chosen, _ = sd.eligible_fills(
        ing_fills, screen_rows, prices, split=datetime(2026, 8, 16, 0, 0, tzinfo=UTC), scope="live"
    )
    assert set(hist_chosen) == {"W_EARLY", "W_OLD_EXCL"}
    assert set(live_chosen) == {"W_EARLY"}


def test_falsifier_e_user_fills_pagination_truncation_caught() -> None:
    start_ms = 1_700_000_000_000
    end_ms = start_ms + 30 * 86_400_000

    res_cap = sd.audit_user_fills_pagination(
        {"pages": 12, "truncated": True, "fills": [{"t": start_ms + i * 1000} for i in range(24_000)]},
        start_ms=start_ms,
        end_ms=end_ms,
    )
    assert "TRUNCATED_PAGE_CAP" in res_cap["flags"]
    assert not res_cap["screening_complete"]
    assert not res_cap["usable_for_backtest"]

    res_tail = sd.audit_user_fills_pagination(
        {"pages": 1, "truncated": False, "fills": [{"t": end_ms - 86_400_000 + i * 10} for i in range(2_000)]},
        start_ms=start_ms,
        end_ms=end_ms,
    )
    assert "TAIL_WINDOW_TRUNCATED" in res_tail["flags"]
    assert not res_tail["screening_complete"]
    assert not res_tail["usable_for_backtest"]

    res_zero = sd.audit_user_fills_pagination(
        {"pages": 1, "truncated": False, "fills": []}, start_ms=start_ms, end_ms=end_ms
    )
    assert "UNVERIFIED_ZERO_FILLS" in res_zero["flags"]
    assert not res_zero["screening_complete"]
    assert not res_zero["usable_for_backtest"]

    res_ok = sd.audit_user_fills_pagination(
        {"pages": 1, "truncated": False, "fills": [{"t": start_ms + 86_400_000}, {"t": end_ms - 86_400_000}]},
        start_ms=start_ms,
        end_ms=end_ms,
    )
    assert res_ok["flags"] == []
    assert res_ok["screening_complete"]
    assert not res_ok["usable_for_backtest"]


def test_falsifier_f_latched_cusum_euler_marginal_shares_and_capacity(tmp_path: Path) -> None:
    baseline = pc.TraderEpisodeBaseline(
        trader_id=0,
        alias="TRADER_X",
        mean_ep_return=0.004,
        std_ep_return=0.010,
        median_ln_peak_notional=-1.0,
        mad_ln_peak_notional=0.15,
        mean_win_ln_duration=2.0,
        std_win_ln_duration=0.4,
        symbol_notional_dist=np.array([0.8, 0.2]),
        mae_p95=0.02,
    )
    state = pc.RosterCUSUMState(alias="TRADER_X")
    for _ in range(3):
        pc.update_roster_episode_cusum(
            state,
            baseline,
            ep_return=-0.04,
            ep_peak_notional_frac=0.90,
            ep_duration_bars=40.0,
            ep_mae=0.03,
            ep_symbol_notional=np.array([0.1, 0.9]),
        )
    assert state.status == "KILL_CONTAIN_0X"
    assert state.terminal_kill_latched

    restored = pc.RosterCUSUMState.from_dict(state.to_dict())
    for _ in range(5):
        pc.update_roster_episode_cusum(
            restored,
            baseline,
            ep_return=0.05,
            ep_peak_notional_frac=math.exp(-1.0),
            ep_duration_bars=8.0,
            ep_mae=0.001,
            ep_symbol_notional=np.array([0.8, 0.2]),
        )
    assert restored.status == "KILL_CONTAIN_0X"
    assert restored.terminal_kill_latched

    restored.reset_for_new_mandate_generation()
    pc.update_roster_episode_cusum(
        restored,
        baseline,
        ep_return=0.01,
        ep_peak_notional_frac=math.exp(-1.0),
        ep_duration_bars=8.0,
        ep_mae=0.002,
        ep_symbol_notional=np.array([0.8, 0.2]),
    )
    assert restored.status == "OK"
    assert not restored.terminal_kill_latched

    # Euler marginal shares sum to 1.0 (both single- and multi-segment) and reject impossible bounds.
    n_days = 10
    bpd = 288
    n_bars = n_days * bpd
    ts = _grid_ms(n_bars)
    notional = np.zeros((3, 1, n_bars), dtype=np.float64)
    d_notional = np.zeros((3, 1, n_bars), dtype=np.float64)
    sym_net = np.zeros((3, 1, n_bars), dtype=np.float64)
    for ti in range(3):
        for ep in range(6):
            sb = 20 + ep * 400
            eb = sb + 20
            notional[ti, 0, sb:eb] = 0.2
            d_notional[ti, 0, sb] = 0.2
            d_notional[ti, 0, eb] = -0.2
            sym_net[ti, 0, sb + 1 : eb + 1] = 0.001 if ep % 3 != 1 else -0.0015
    net = np.sum(sym_net, axis=1)
    tensor = pc.UnitCopy5mTensor(
        aliases=("A", "B", "C"),
        symbols=("BTC",),
        timestamps_ms=ts,
        notional_frac=notional,
        delta_notional_frac=d_notional,
        gross_ret_frac=net.copy(),
        net_ret_frac=net,
        sym_net_ret_frac=sym_net,
    )
    cfg = pc.PodConstructorConfig(
        pod_size=3, min_episodes=4, min_active_bar_frac=0.01, reject_unknown_tilt=False
    )
    _, metrics, _ = pc.stage1_audit_traders(tensor, cfg)
    sized = pc.size_pod_dual_barrier((0, 1, 2), tensor, metrics, cfg)
    assert sum(sized.marginal_cdar_shares) == pytest.approx(1.0, abs=1e-6)
    assert sum(sized.marginal_daily_cvar_shares) == pytest.approx(1.0, abs=1e-6)

    segs = [(0, 5 * bpd), (5 * bpd, n_bars)]
    sized_seg = pc.size_pod_dual_barrier((0, 1, 2), tensor, metrics, cfg, segments=segs)
    assert sum(sized_seg.marginal_cdar_shares) == pytest.approx(1.0, abs=1e-6)
    assert sum(sized_seg.marginal_daily_cvar_shares) == pytest.approx(1.0, abs=1e-6)

    solo_n, t_ov, d_ov = pc._build_sparse_overlap_cache(
        tensor.delta_notional_frac[[0, 1, 2]], tensor.net_ret_frac[[0, 1, 2]]
    )
    sl_10 = pc._daily_reset_slices(ts, bpd, cfg.reset_bar_offset)
    for u_val in (-0.005, 0.0005):
        c_sh, d_sh = pc.compute_euler_marginal_risk_shares(
            np.array([0.4, 0.35, 0.25]),
            solo_n,
            t_ov,
            d_ov,
            cfg.cost_bps * 1e-4,
            sl_10,
            cfg,
            unrl_at_resets=np.full(len(sl_10), u_val),
        )
        assert sum(c_sh) == pytest.approx(1.0, abs=1e-6)
        assert sum(d_sh) == pytest.approx(1.0, abs=1e-6)

    with pytest.raises(ValueError, match="INFEASIBLE_MARGINAL_RISK_SHARE_BOUNDS"):
        pc.size_pod_dual_barrier(
            (0, 1, 2),
            tensor,
            metrics,
            pc.PodConstructorConfig(
                pod_size=3,
                min_episodes=4,
                min_active_bar_frac=0.01,
                reject_unknown_tilt=False,
                risk_share_mode="marginal_euler",
                max_marginal_cdar_share=0.20,
            ),
        )

    # NaN quote_volume must fall back to volume * close (or volume * open), never return NaN.
    vol_start = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    vol_grid = _grid_ms(4, start_dt=vol_start)
    n_min = 20
    opens = np.full(n_min, 100.0, dtype=np.float64)
    closes = np.full(n_min, 102.0, dtype=np.float64)
    closes[6] = float("nan")
    qvols = np.full(n_min, float("nan"), dtype=np.float64)
    qvols[11] = 1234.5
    vols = np.full(n_min, 50.0, dtype=np.float64)
    vols[6] = 40.0
    _write_1m_bars(
        tmp_path, "SOL", vol_start, n_min, opens, closes, quote_volumes=qvols, volumes=vols
    )
    v_prices = sd.LazyPrices(vol_grid, store_root=tmp_path, vike_root=None, streaming=False)
    sol_qvol = v_prices.get_quote_volume_on_grid("SOL")
    assert sol_qvol[0] == pytest.approx(50.0 * 102.0)
    assert sol_qvol[1] == pytest.approx(40.0 * 100.0)
    assert sol_qvol[2] == pytest.approx(1234.5)

    # 1m bar participation capacity audit
    deltas = np.array([[0.0, 10.0, 0.5]])
    opens_c = np.array([[100.0, 100.0, 100.0]])
    qvols_c = np.array([[10_000.0, 10_000.0, 10_000.0]])
    cap = cr.audit_1m_bar_participation(["BTC"], deltas, opens_c, qvols_c)
    assert cap.status == "CAPACITY_BREACH_GT_5PCT"
    assert cap.trades_gt_5pct == 1
    assert cap.max_participation == pytest.approx(0.10, abs=1e-6)
    assert cap.modeled_sqrt_impact_usd > 0.0

    cap_unp = cr.audit_1m_bar_participation(["BTC"], deltas, opens_c, np.full((1, 3), np.nan))
    assert cap_unp.status == "UNPROVEN_VOLUME"
