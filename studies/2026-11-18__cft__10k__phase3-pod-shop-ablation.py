"""Phase 3 end-to-end pod-shop ablation study across May-Nov 2026 archives."""

from __future__ import annotations

import gc
import json
import math
import os
import re
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = Path(__file__).resolve().parent
PROP_PRODUCTION_SRC = Path("/home/quant/dev/prop-production/src")
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import canonical_replay as cr
import pod_constructor as pc
import portfolio_risk_policy as prp
import study_data as sd

DEFAULT_REGISTRY_DB = Path.home() / ".local/share/copytrade/registry/wallet_registry.sqlite"
COST_MULTIPLIERS: tuple[float, ...] = (1.0, 2.0)


@dataclass(frozen=True)
class Phase3AblationConfig:
    """Configuration for reproducible Phase 3 ablation and synthetic test injection."""

    start_iso: str = "2026-05-01T00:00:00Z"
    split_iso: str = "2026-09-01T00:00:00Z"
    wf_end_iso: str = "2026-10-28T00:00:00Z"
    oos_end_iso: str = "2026-11-01T00:00:00Z"
    pit_strict_iso: str = "2026-11-05T00:00:00Z"
    archive_end_iso: str = "2026-11-14T00:00:00Z"
    registry_db: Path = DEFAULT_REGISTRY_DB
    archive_root: Path = sd.DEFAULT_ARCHIVE_ROOT
    store_root: Path = sd.DEFAULT_STORE_ROOT
    vike_root: Path | None = sd.DEFAULT_VIKE_ROOT
    rules_json_path: Path = ROOT / "prop_firms_rules.json"
    report_md: Path = ROOT / "reports/2026-11-18__cft__10k__phase3-pod-shop-ablation.md"
    report_json: Path = ROOT / "reports/2026-11-18__cft__10k__phase3-pod-shop-ablation__results.json"
    pod_size: int = 3
    top_per_cluster: int = 2
    max_candidate_pods: int = 16
    min_active_bar_frac: float = 0.005
    min_episodes: int = 4
    min_overlap_days: int = 5
    min_lot_usd: float = 15.0
    min_baseline_order_usd: float = 15.0
    max_marginal_cdar_share: float = 0.45
    cpcv_n_blocks: int = 6
    l1_window_days: int = 14
    mc_n_paths: int = 5_000
    walk_forward_train_days: int = 90
    walk_forward_eval_days: int = 30
    pit_strict_lookback_days: int = 90
    fetch_bybit_specs: bool = True
    write_reports: bool = True


@dataclass(frozen=True)
class PodOOSOutcome:
    """Out-of-sample simulation metrics and 1m participation audit for one pod variant."""

    name: str
    tier: int | None
    execution_mode: str
    cost_multiplier: float
    aliases: tuple[str, ...]
    k_weights: tuple[float, ...]
    coins_count: int
    verdict: str
    terminal_net_usd: float
    full_window_net_usd: float
    static_floor_dd_usd: float
    peak_to_trough_dd_usd: float
    worst_daily_usd: float
    sweeps: int
    peak_gross_leverage: float
    total_funding_usd: float
    unpriced_open_steps: int
    capacity_status: str
    max_participation_pct: float
    p95_participation_pct: float
    trades_gt_1pct: int
    trades_gt_5pct: int
    unverified_volume_trades: int
    modeled_sqrt_impact_usd: float
    cwsa_net_usd: float | None = None
    cwsa_clipped_orders: int = 0
    cwsa_missed_orders: int = 0
    cwsa_bars_gt_5pct: int = 0
    cwsa_delta_mismatch_bars: int = 0


def _parse_utc_ms(iso_ts: str) -> int:
    dt = datetime.fromisoformat(iso_ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return sd.to_utc_ms(dt.astimezone(UTC))


def _mem_rss_mb() -> tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb) using /proc/self/statm and resource.getrusage."""
    rss_mb = 0.0
    statm_path = Path("/proc/self/statm")
    if statm_path.is_file():
        try:
            parts = statm_path.read_text(encoding="utf-8").split()
            if len(parts) >= 2:
                rss_mb = int(parts[1]) * float(os.sysconf("SC_PAGE_SIZE")) / 1_048_576.0
        except (OSError, ValueError):
            rss_mb = 0.0
    peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    return rss_mb, peak_rss_mb


def _log_stage(stage: str, elapsed_s: float) -> float:
    rss_mb, peak_mb = _mem_rss_mb()
    print(
        f"[Phase3] {stage} elapsed={elapsed_s:.2f}s rss_mb={rss_mb:.1f} peak_rss_mb={peak_mb:.1f}",
        flush=True,
    )
    return peak_mb


def _unit_frames_to_compact_arrays(
    w_frames: dict[str, pl.DataFrame],
) -> dict[str, dict[str, np.ndarray]]:
    """Extract compact NumPy arrays from Polars replay frames, skipping all-zero coins."""
    compact: dict[str, dict[str, np.ndarray]] = {}
    for coin, df_f in w_frames.items():
        pos = df_f["pos"].to_numpy().astype(np.float64, copy=False)
        delta = df_f["delta"].to_numpy().astype(np.float64, copy=False)
        open_px = df_f["open"].to_numpy().astype(np.float64, copy=False)
        gross = df_f["gross"].to_numpy().astype(np.float64, copy=False)
        net = df_f["net"].to_numpy().astype(np.float64, copy=False)
        if not (np.any(pos) or np.any(delta) or np.any(gross) or np.any(net)):
            continue
        open_safe = np.where(np.isfinite(open_px), open_px, 0.0)
        compact[coin] = {
            "notional": np.where(pos != 0.0, pos * open_safe, 0.0),
            "delta_notional": np.where(delta != 0.0, delta * open_safe, 0.0),
            "gross": gross.copy(),
            "net": net.copy(),
        }
    return compact


def _sanitize_payload(obj: Any) -> Any:
    if isinstance(obj, str):
        return sd.sanitize_public_label(obj)
    if isinstance(obj, dict):
        return {_sanitize_payload(k): _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_sanitize_payload(v) for v in obj]
    if isinstance(obj, list):
        return [_sanitize_payload(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_payload(v) for v in obj.tolist()]
    return obj


def _default_challenge_spec_loader(slug: str) -> Any:
    if str(PROP_PRODUCTION_SRC) not in sys.path:
        sys.path.insert(0, str(PROP_PRODUCTION_SRC))
    from prop_production.prop_firms import get_challenge_spec

    return get_challenge_spec(slug)


def _ensure_tensor_prices(
    tensor: pc.UnitCopy5mTensor,
    prices: sd.LazyPrices,
    start_idx: int,
    stop_idx: int,
) -> pc.UnitCopy5mTensor:
    if tensor.exec_open_px is not None and tensor.mark_close_px is not None:
        return tensor
    n_syms = len(tensor.symbols)
    n_bars = tensor.timestamps_ms.size
    open_buf = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
    close_buf = np.full((n_syms, n_bars), np.nan, dtype=np.float64)
    for si, coin in enumerate(tensor.symbols):
        t_px, m_px = prices.get(coin)
        open_buf[si] = t_px[start_idx:stop_idx]
        close_buf[si] = m_px[start_idx:stop_idx]
    return replace(tensor, exec_open_px=open_buf, mark_close_px=close_buf)


def _get_funding_subgrid(
    coins: Sequence[str],
    start_idx: int,
    stop_idx: int,
    full_grid: np.ndarray,
    store_root: Path,
    fund_cache: dict[tuple[str, int, int, int], np.ndarray],
) -> tuple[np.ndarray, int]:
    if not coins:
        return np.zeros((0, stop_idx - start_idx), dtype=np.float64), 0
    full_mat, _ = sd.load_usdt_funding_rates_on_grid(coins, full_grid, store_root, fund_cache)
    sub = full_mat[:, start_idx:stop_idx]
    covered = int(np.sum(np.any(sub != 0.0, axis=1)))
    return sub, covered


def _reconstruct_member_positions(
    aliases: Sequence[str],
    coins: Sequence[str],
    grouped: dict[str, dict[str, pl.DataFrame]],
    eval_grid: np.ndarray,
    activation_ms: int,
) -> np.ndarray:
    member_pos = np.zeros((len(aliases), len(coins), eval_grid.size), dtype=np.float64)
    for ai, alias in enumerate(aliases):
        by_coin = grouped.get(alias, {})
        for ci, coin in enumerate(coins):
            df_c = by_coin.get(coin)
            if df_c is None:
                continue
            member_pos[ai, ci] = cr.reconstruct_positions_on_grid(
                df_c,
                eval_grid,
                policy="new",
                activation_ms=activation_ms,
            ).pos
    return member_pos


def _load_market_matrices(
    coins: Sequence[str],
    start_idx: int,
    stop_idx: int,
    full_grid: np.ndarray,
    prices: sd.LazyPrices,
    store_root: Path,
    fund_cache: dict[tuple[str, int, int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bars = stop_idx - start_idx
    if not coins:
        empty = np.zeros((0, n_bars), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()
    trade_open = np.vstack([prices.get(c)[0][start_idx:stop_idx] for c in coins])
    mark_close = np.vstack([prices.get(c)[1][start_idx:stop_idx] for c in coins])
    qvol = np.vstack([prices.get_quote_volume_on_grid(c)[start_idx:stop_idx] for c in coins])
    funding, _ = _get_funding_subgrid(coins, start_idx, stop_idx, full_grid, store_root, fund_cache)
    return trade_open, mark_close, qvol, funding


def _build_l1_allowed_slice(
    coins: Sequence[str],
    start_idx: int,
    stop_idx: int,
    full_grid: np.ndarray,
    prices: sd.LazyPrices,
    liq_cfg: cr.CausalLiquidityConfig,
) -> np.ndarray:
    n_bars = stop_idx - start_idx
    if not coins:
        return np.zeros((0, n_bars), dtype=bool)
    l1_start_idx = max(0, start_idx - liq_cfg.l1_window_days * 288)
    grid_l1 = full_grid[l1_start_idx:stop_idx]
    qvol_l1 = np.vstack([prices.get_quote_volume_on_grid(c)[l1_start_idx:stop_idx] for c in coins])
    l1_res = cr.screen_l1_allowed_markets(
        coins,
        qvol_l1,
        grid_l1,
        freeze_ts_ms=int(full_grid[start_idx]),
        cfg=liq_cfg,
    )
    offset = start_idx - l1_start_idx
    return l1_res.rolling_allowed_mask[:, offset:]


def _make_cft_guardian_spec(
    bybit_specs: dict[str, tuple[float, float]],
    coins: Sequence[str],
    *,
    cost_multiplier: float = 1.0,
) -> cr.GuardianSpec:
    step_map = {c: bybit_specs[c][0] for c in coins if c in bybit_specs}
    min_map = {c: bybit_specs[c][1] for c in coins if c in bybit_specs}
    return cr.GuardianSpec(
        initial_equity=10_000.0,
        profit_target=800.0,
        max_static_dd=1_000.0,
        daily_loss_limit=500.0,
        guardian_sweep_loss=400.0,
        cost_bps=cr.DEFAULT_COST_BPS,
        cost_multiplier=float(cost_multiplier),
        daily_loss_basis="balance",
        reenter_after_sweep="new_only",
        min_trade_notional_usd=15.0,
        amount_step_by_coin=step_map or None,
        min_amount_by_coin=min_map or None,
        max_gross_leverage=5.0,
    )


def _eval_pod_on_window(
    name: str,
    tier: int | None,
    aliases: Sequence[str],
    k_weights: Sequence[float],
    grouped: dict[str, dict[str, pl.DataFrame]],
    prices: sd.LazyPrices,
    full_grid: np.ndarray,
    start_idx: int,
    stop_idx: int,
    store_root: Path,
    fund_cache: dict[tuple[str, int, int, int], np.ndarray],
    bybit_specs: dict[str, tuple[float, float]],
    liq_cfg: cr.CausalLiquidityConfig,
    *,
    execution_mode: str = "bar_delta",
    cost_multiplier: float = 1.0,
    censor_timeout: bool = False,
) -> tuple[PodOOSOutcome, tuple[np.ndarray, ...], cr.CWSAReplayResult | None]:
    eval_grid = full_grid[start_idx:stop_idx]
    activation_ms = int(eval_grid[0])
    coins = sorted({c for a in aliases for c in grouped.get(a, {})})
    member_pos = _reconstruct_member_positions(aliases, coins, grouped, eval_grid, activation_ms)
    net_targets = cr.net_member_targets(member_pos, k_weights)
    trade_open, mark_close, qvol, funding = _load_market_matrices(
        coins, start_idx, stop_idx, full_grid, prices, store_root, fund_cache
    )

    cwsa_res: cr.CWSAReplayResult | None = None
    tgt_for_guardian = net_targets
    if execution_mode == "cwsa_one_shot_v3" and coins:
        allowed_oos = _build_l1_allowed_slice(coins, start_idx, stop_idx, full_grid, prices, liq_cfg)
        cwsa_cfg = cr.CWSAReplayConfig(
            fee_bps=cr.DEFAULT_FEE_BPS * cost_multiplier,
            slippage_bps=cr.DEFAULT_SLIPPAGE_BPS * cost_multiplier,
        )
        cwsa_res = cr.simulate_cwsa_one_shot_replay(
            coins,
            net_targets,
            trade_open,
            mark_close,
            eval_grid,
            k_scale=1.0,
            vol_1m_usd=qvol,
            allowed_markets_mask=allowed_oos,
            cwsa_cfg=cwsa_cfg,
            liquidity_cfg=liq_cfg,
        )
        tgt_for_guardian = cwsa_res.executed_positions

    spec = _make_cft_guardian_spec(bybit_specs, coins, cost_multiplier=cost_multiplier)
    res, blocks = cr.simulate_guardian_challenge(
        coins,
        tgt_for_guardian,
        trade_open,
        mark_close,
        eval_grid,
        spec,
        funding_rates=funding,
    )
    if res.executed_deltas is None:
        raise RuntimeError(f"GUARDIAN_MISSING_EXECUTED_DELTAS:{name}")
    cap = cr.audit_1m_bar_participation(coins, res.executed_deltas, trade_open, qvol)

    cwsa_net: float | None = None
    cwsa_clipped = 0
    cwsa_missed = 0
    cwsa_gt5 = 0
    mismatch_bars = 0
    if cwsa_res is not None:
        cwsa_cap = cr.audit_1m_bar_participation(coins, cwsa_res.executed_deltas, trade_open, qvol)
        cwsa_net = round(float(np.sum(cwsa_res.step_pnl)), 2)
        cwsa_clipped = int(cwsa_res.participation_audit.clipped_order_count)
        cwsa_missed = int(cwsa_res.participation_audit.missed_order_count)
        cwsa_gt5 = int(cwsa_cap.trades_gt_5pct)
        if res.executed_deltas.size > 0:
            mismatch_bars = int(
                np.count_nonzero(np.max(np.abs(res.executed_deltas - cwsa_res.executed_deltas), axis=0) > 1e-6)
            )

    verdict = "RIGHT_CENSORED_AT_ARCHIVE_END" if censor_timeout and res.verdict == "TIMEOUT" else res.verdict
    outcome = PodOOSOutcome(
        name=name,
        tier=tier,
        execution_mode=execution_mode,
        cost_multiplier=float(cost_multiplier),
        aliases=tuple(sd.sanitize_public_label(a) for a in aliases),
        k_weights=tuple(round(float(x), 6) for x in k_weights),
        coins_count=len(coins),
        verdict=verdict,
        terminal_net_usd=round(float(res.terminal_net_usd), 2),
        full_window_net_usd=round(float(res.full_window_net_usd), 2),
        static_floor_dd_usd=round(float(res.static_floor_dd_usd), 2),
        peak_to_trough_dd_usd=round(float(res.peak_to_trough_dd_usd), 2),
        worst_daily_usd=round(float(res.worst_daily_usd), 2),
        sweeps=int(res.sweeps),
        peak_gross_leverage=round(float(res.peak_gross_leverage), 3),
        total_funding_usd=round(float(res.total_funding_usd), 2),
        unpriced_open_steps=int(res.unpriced_open_steps),
        capacity_status=cap.status,
        max_participation_pct=round(float(cap.max_participation) * 100.0, 4),
        p95_participation_pct=round(float(cap.p95_participation) * 100.0, 4),
        trades_gt_1pct=int(cap.trades_gt_1pct),
        trades_gt_5pct=int(cap.trades_gt_5pct),
        unverified_volume_trades=int(cap.unverified_volume_trades),
        modeled_sqrt_impact_usd=round(float(cap.modeled_sqrt_impact_usd), 2),
        cwsa_net_usd=cwsa_net,
        cwsa_clipped_orders=cwsa_clipped,
        cwsa_missed_orders=cwsa_missed,
        cwsa_bars_gt_5pct=cwsa_gt5,
        cwsa_delta_mismatch_bars=mismatch_bars,
    )
    return outcome, blocks, cwsa_res


def _project_baseline_to_coins(
    base: pc.TraderEpisodeBaseline,
    train_symbols: Sequence[str],
    target_coins: Sequence[str],
) -> pc.TraderEpisodeBaseline:
    by_sym = {
        sym: float(val) for sym, val in zip(train_symbols, base.symbol_notional_dist, strict=True)
    }
    vec = np.array([by_sym.get(c, 0.0) for c in target_coins], dtype=np.float64)
    return replace(base, symbol_notional_dist=vec)


def _run_section_b_for_tier(
    tier: Literal[1, 2, 3, 4],
    pod: pc.SizedPodCandidate,
    tier_cfg: pc.PodConstructorConfig,
    metrics_list: list[pc.TraderAuditMetrics],
    baselines: dict[int, pc.TraderEpisodeBaseline],
    passed_ids: list[int],
    train_symbols: Sequence[str],
    grouped: dict[str, dict[str, pl.DataFrame]],
    prices: sd.LazyPrices,
    full_grid: np.ndarray,
    cut_split: int,
    cut_oos: int,
    split_ms: int,
    store_root: Path,
    fund_cache: dict[tuple[str, int, int, int], np.ndarray],
    liq_cfg: cr.CausalLiquidityConfig,
    cft_contract: pc.PropContractSpec,
    mc_n_paths: int,
) -> dict[str, Any]:
    d_tier = float(tier_cfg.tier_peak_dd_cap_frac or tier_cfg.baseline_port_dd_frac) * 10_000.0
    selected_set = set(pod.pod_indices)
    bench_ids = sorted(
        [tid for tid in passed_ids if tid not in selected_set],
        key=lambda tid: metrics_list[tid].net_over_max_dd,
        reverse=True,
    )[:3]

    bench_aliases = [metrics_list[tid].alias for tid in bench_ids]
    all_aliases = list(pod.aliases) + bench_aliases
    port_coins = sorted({c for a in all_aliases for c in grouped.get(a, {})})
    eval_grid = full_grid[cut_split:cut_oos]
    trade_open, mark_close, qvol, funding = _load_market_matrices(
        port_coins, cut_split, cut_oos, full_grid, prices, store_root, fund_cache
    )
    allowed_oos = _build_l1_allowed_slice(port_coins, cut_split, cut_oos, full_grid, prices, liq_cfg)

    streams: dict[str, prp.SeatTelemetryStream] = {}
    for alias in all_aliases:
        pos_mat = _reconstruct_member_positions([alias], port_coins, grouped, eval_grid, split_ms)[0]
        streams[alias] = prp.SeatTelemetryStream(seat_id=alias, source_positions=pos_mat)

    seat_specs: list[prp.SeatRiskSpec] = []
    for idx, tid in enumerate(pod.pod_indices):
        m = metrics_list[tid]
        base = _project_baseline_to_coins(baselines[tid], train_symbols, port_coins)
        s_i = float(pod.marginal_cdar_shares[idx]) if pod.marginal_cdar_shares else 1.0 / len(pod.aliases)
        seat_specs.append(
            prp.SeatRiskSpec(
                seat_id=m.alias,
                tier=tier,
                k_scale=float(pod.multipliers[idx]),
                initial_balance_usd=10_000.0,
                tier_seat_count_m=len(pod.aliases),
                s_i=max(1e-6, s_i),
                d_tier=d_tier,
                win_hold_p95_bars=m.win_hold_p95_bars,
                mae_p95_r=float(base.mae_p95),
                episode_baseline=base,
            )
        )

    bench_candidates: list[prp.PITBenchCandidate] = []
    for tid in bench_ids:
        m_b = metrics_list[tid]
        base_b = _project_baseline_to_coins(baselines[tid], train_symbols, port_coins)
        b_dd = max(m_b.standalone_peak_to_trough_dd * 10_000.0, 1.0)
        k_b = min((d_tier / float(max(1, len(pod.aliases)))) / b_dd, float(tier_cfg.max_k_mult))
        bench_spec = prp.SeatRiskSpec(
            seat_id=m_b.alias,
            tier=tier,
            k_scale=max(0.01, float(k_b)),
            initial_balance_usd=10_000.0,
            tier_seat_count_m=len(pod.aliases),
            s_i=1.0 / float(max(1, len(pod.aliases))),
            d_tier=d_tier,
            win_hold_p95_bars=m_b.win_hold_p95_bars,
            mae_p95_r=float(base_b.mae_p95),
            episode_baseline=base_b,
        )
        bench_candidates.append(
            prp.PITBenchCandidate(spec=bench_spec, score=float(m_b.net_over_max_dd))
        )

    bench_snaps = (
        prp.PITBenchSnapshot(
            asof_ts_ms=split_ms,
            candidates=tuple(bench_candidates),
            mode="PIT_STRICT_ASOF",
        ),
    )

    cft_spec = _make_cft_guardian_spec({}, port_coins, cost_multiplier=1.0)
    rng = np.random.default_rng(42)

    def _summarize_policy_res(
        label: str,
        res: prp.PortfolioPolicyResult,
        cfg_used: prp.PortfolioPolicyConfig,
    ) -> dict[str, Any]:
        if res.max_pm03_conservation_err >= 1e-10:
            raise RuntimeError(f"PM03_CONSERVATION_ERR:{label}:{res.max_pm03_conservation_err}")
        if res.guardian_breached:
            cont_res = prp.run_portfolio_policy_replay(
                seat_specs=seat_specs,
                seat_streams=streams,
                coins=port_coins,
                trade_open=trade_open,
                mark_close=mark_close,
                grid_ms=eval_grid,
                vol_1m_usd=qvol,
                allowed_markets_mask=allowed_oos,
                funding_rates=funding,
                bench_snapshots=bench_snaps,
                policy_cfg=replace(cfg_used, stop_on_guardian_breach=False),
            )
            if cont_res.max_pm03_conservation_err >= 1e-10:
                raise RuntimeError(
                    f"PM03_CONSERVATION_ERR:{label}_cont:{cont_res.max_pm03_conservation_err}"
                )
        else:
            cont_res = res

        step_pnl = np.diff(np.concatenate(([cfg_used.initial_balance_usd], cont_res.equity_curve)))
        swept_flags = np.zeros(eval_grid.size, dtype=bool)
        blocks = cr.extract_mc_daily_blocks(
            step_pnl,
            swept_flags,
            eval_grid,
            cft_spec,
            balance_curve=cont_res.balance_curve,
        )
        mc_ev = pc.evaluate_multiphase_contract_ev(
            blocks,
            cft_contract,
            "Blended",
            n_paths=mc_n_paths,
            rng=rng,
        )
        eq_arr = res.equity_curve
        min_eq = float(np.min(eq_arr)) if eq_arr.size else cfg_used.initial_balance_usd
        static_dd = round(max(0.0, cfg_used.initial_balance_usd - min_eq), 2)
        retired = [
            sd.sanitize_public_label(sid)
            for sid, st in res.seat_final_states.items()
            if st.state == prp.SeatState.RETIRED
        ]
        return {
            "label": label,
            "final_equity_usd": round(float(eq_arr[-1]), 2),
            "net_pnl_usd": round(float(eq_arr[-1] - cfg_used.initial_balance_usd), 2),
            "static_floor_dd_usd": -static_dd if static_dd > 0.0 else 0.0,
            "max_pm03_conservation_err": float(res.max_pm03_conservation_err),
            "guardian_breached": bool(res.guardian_breached),
            "guardian_breach_reason": res.guardian_breach_reason,
            "guardian_target_passed": bool(res.guardian_target_passed),
            "guardian_sweeps": int(res.guardian_sweeps),
            "retired_seats": retired,
            "promoted_bench_seats": [sd.sanitize_public_label(s) for s in res.promoted_bench_seats],
            "resized_seat_events": len(res.resized_seat_events),
            "bars_gt_5pct_count": int(res.participation_audit.bars_gt_5pct_count),
            "blended_p_funded_pct": mc_ev["p_funded_payout_pct"],
            "blended_breakeven_fee_usd": mc_ev["breakeven_fee_usd"],
        }

    arm_rows: dict[str, Any] = {}
    for arm in prp.LifecycleArm:
        cfg_arm = prp.PortfolioPolicyConfig(
            initial_balance_usd=10_000.0,
            lifecycle_arm=arm,
            replacement_mode=prp.ReplacementMode.LEAVE_RISK_UNUSED,
            liquidity=liq_cfg,
        )
        res_arm = prp.run_portfolio_policy_replay(
            seat_specs=seat_specs,
            seat_streams=streams,
            coins=port_coins,
            trade_open=trade_open,
            mark_close=mark_close,
            grid_ms=eval_grid,
            vol_1m_usd=qvol,
            allowed_markets_mask=allowed_oos,
            funding_rates=funding,
            bench_snapshots=bench_snaps,
            policy_cfg=cfg_arm,
        )
        arm_rows[arm.value] = _summarize_policy_res(arm.value, res_arm, cfg_arm)

    base_rep_cfg = prp.PortfolioPolicyConfig(
        initial_balance_usd=10_000.0,
        lifecycle_arm=prp.LifecycleArm.RESEARCH_LEADER_REDUCE_FREEZE,
        liquidity=liq_cfg,
    )
    rep_cmp = prp.compare_seat_replacement_modes(
        seat_specs=seat_specs,
        seat_streams=streams,
        bench_snapshots=bench_snaps,
        coins=port_coins,
        trade_open=trade_open,
        mark_close=mark_close,
        grid_ms=eval_grid,
        vol_1m_usd=qvol,
        allowed_markets_mask=allowed_oos,
        funding_rates=funding,
        base_policy_cfg=base_rep_cfg,
    )
    rep_rows: dict[str, Any] = {}
    for mode, m_res in rep_cmp.results_by_mode.items():
        cfg_m = replace(base_rep_cfg, replacement_mode=mode)
        rep_rows[mode.value] = _summarize_policy_res(mode.value, m_res, cfg_m)

    return {
        "tier": tier,
        "aliases": [sd.sanitize_public_label(a) for a in pod.aliases],
        "bench_aliases": [sd.sanitize_public_label(a) for a in bench_aliases],
        "d_tier_usd": d_tier,
        "lifecycle_arms": arm_rows,
        "replacement_modes": rep_rows,
    }


def _evaluate_multifirm_frontier(
    aliases: Sequence[str],
    k_weights: Sequence[float],
    prices: sd.LazyPrices,
    full_grid: np.ndarray,
    cut_split: int,
    cut_oos: int,
    store_root: Path,
    fund_cache: dict[tuple[str, int, int, int], np.ndarray],
    bybit_specs: dict[str, tuple[float, float]],
    cwsa_executed_positions: np.ndarray,
    cwsa_coins: Sequence[str],
    rules_json_path: Path,
    mc_n_paths: int,
    challenge_spec_loader: Callable[[str], Any],
) -> dict[str, Any]:
    eval_grid = full_grid[cut_split:cut_oos]
    trade_open, mark_close, _, funding = _load_market_matrices(
        cwsa_coins, cut_split, cut_oos, full_grid, prices, store_root, fund_cache
    )
    step_map = {c: bybit_specs[c][0] for c in cwsa_coins if c in bybit_specs} or None
    min_map = {c: bybit_specs[c][1] for c in cwsa_coins if c in bybit_specs} or None

    propr_1step = pc.load_prop_contract_spec("propr", account_usd=10_000.0, rules_json_path=rules_json_path)
    propr_2step = pc.load_prop_contract_spec(
        "propr", account_usd=10_000.0, rules_json_path=rules_json_path, program_filter="Classic 2-Step"
    )
    if len(propr_1step.eval_phases) != 1:
        raise ValueError(f"UNEXPECTED_PROPR_1STEP_PHASES:{len(propr_1step.eval_phases)}")
    if len(propr_2step.eval_phases) != 2:
        raise ValueError(f"UNEXPECTED_PROPR_2STEP_PHASES:{len(propr_2step.eval_phases)}")

    propr2_status = "UNEXPECTED_SUCCESS"
    try:
        cr.GuardianSpec.from_challenge_spec(challenge_spec_loader("propr:classic_2step:phase_1"))
    except ValueError as exc:
        propr2_status = f"BLOCKED_{exc}"

    firms_cfg = [
        ("crypto-fund-trader:two_phase:phase_1", "crypto-fund-trader", None),
        ("hyrotrader:two_step:phase_1", "hyrotrader", None),
        ("vanta-trading:classic_1step_3pct:phase_1", "vanta-trading", None),
        ("propr:classic_1step:phase_1", "propr", None),
    ]
    firm_rows: list[dict[str, Any]] = []

    for firm_idx, (slug, firm_slug, prog_filter) in enumerate(firms_cfg):
        cspec = challenge_spec_loader(slug)
        scale = 10_000.0 / float(cspec.initial)
        gspec = cr.GuardianSpec.from_challenge_spec(
            cspec,
            scale=scale,
            min_trade_notional_usd=15.0,
            amount_step_by_coin=step_map,
            min_amount_by_coin=min_map,
            max_gross_leverage=5.0,
        )
        guard_res, blocks = cr.simulate_guardian_challenge(
            cwsa_coins,
            cwsa_executed_positions,
            trade_open,
            mark_close,
            eval_grid,
            gspec,
            funding_rates=funding,
        )
        contract_unpriced = pc.load_prop_contract_spec(
            firm_slug,
            account_usd=10_000.0,
            challenge_fee_usd=None,
            rules_json_path=rules_json_path,
            program_filter=prog_filter,
        )
        contract_110 = replace(contract_unpriced, challenge_fee_usd=110.0)
        regimes: dict[str, Any] = {}
        for reg_idx, regime in enumerate(("Blended", "75%-Chop")):
            seed = 700 + firm_idx * 10 + reg_idx
            ev_unp = pc.evaluate_multiphase_contract_ev(
                blocks,
                contract_unpriced,
                regime,
                n_paths=mc_n_paths,
                rng=np.random.default_rng(seed),
            )
            ev_110 = pc.evaluate_multiphase_contract_ev(
                blocks,
                contract_110,
                regime,
                n_paths=mc_n_paths,
                rng=np.random.default_rng(seed),
            )
            regimes[regime] = {
                "fee_unverified": ev_unp,
                "fee_110_sensitivity": ev_110,
            }
        firm_rows.append(
            {
                "slug": slug,
                "firm_slug": firm_slug,
                "daily_loss_basis": gspec.daily_loss_basis,
                "payout_split_pct": round(float(contract_unpriced.payout_split_pct), 4),
                "consistency_max_single_day_pct": (
                    contract_unpriced.eval_phases[0].consistency_max_single_day_pct
                ),
                "eval_phases_count": len(contract_unpriced.eval_phases),
                "oos_verdict": guard_res.verdict,
                "oos_terminal_net_usd": round(float(guard_res.terminal_net_usd), 2),
                "oos_full_window_net_usd": round(float(guard_res.full_window_net_usd), 2),
                "oos_static_floor_dd_usd": round(float(guard_res.static_floor_dd_usd), 2),
                "oos_worst_daily_usd": round(float(guard_res.worst_daily_usd), 2),
                "regimes": regimes,
            }
        )

    return {
        "pod_aliases": [sd.sanitize_public_label(a) for a in aliases],
        "pod_k_weights": [round(float(k), 6) for k in k_weights],
        "firms": firm_rows,
        "blocked_fail_closed": {
            "slug": "propr:classic_2step:phase_1",
            "status": propr2_status,
            "propr_1step_eval_phases": len(propr_1step.eval_phases),
            "propr_2step_eval_phases": len(propr_2step.eval_phases),
        },
    }


def _check_report_hygiene(md_text: str, json_text: str) -> None:
    if sd.HEX_ALIAS_RE.search(md_text) or sd.HEX_ALIAS_RE.search(json_text):
        raise AssertionError("HEX_ALIAS_LEAK_DETECTED_IN_OUTPUT")
    if "\u2014" in md_text:
        raise AssertionError("EM_DASH_FORBIDDEN_IN_MARKDOWN")
    in_frontmatter = False
    for idx, line in enumerate(md_text.splitlines()):
        stripped = line.strip()
        if idx == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if not stripped or stripped.startswith(("|", "#")):
            continue
        cleaned = re.sub(r"`[^`]+`", "", stripped)
        if ": " in cleaned:
            raise AssertionError(f"MID_SENTENCE_COLON_IN_MARKDOWN:{stripped}")


def _render_markdown(payload: dict[str, Any]) -> str:
    sec_a = payload["section_a"]
    sec_b = payload["section_b"]
    sec_c = payload["section_c"]
    sec_d = payload["section_d"]
    timings = payload["stage_timings_s"]

    lines = [
        "---",
        "firm: crypto-fund-trader",
        "account_usd: 10000",
        "status: RESEARCH",
        "top_candidate: UNPROVEN_RETROSPECTIVE_PHASE3_ABLATION",
        "script: studies/2026-11-18__cft__10k__phase3-pod-shop-ablation.py",
        "---",
        "",
        "# Phase 3 pod-shop ablation study across May to November 2026 archives",
        "",
        "## Single-pass archive coverage and funnel",
        "",
        (
            f"- Archive ingestion covered `{payload['archive_days']}` daily files across "
            f"`{payload['window']['start_iso']}` to `{payload['window']['archive_end_iso']}` "
            f"(`archive_sha256={payload['archive_sha256'][:12]}`, "
            f"`roster_sha256={payload['roster_sha256'][:12]}`)."
        ),
        (
            f"- Stage timings in seconds completed in `{timings['total_s']}` total "
            f"(discover `{timings['discover_archive_s']}`, ingest `{timings['ingest_fills_s']}`, "
            f"markouts and replay `{timings['markouts_and_train_replay_s']}`, "
            f"stage2 and cpcv `{timings['stage2_and_cpcv_s']}`, "
            f"oos ablations `{timings['oos_ablations_s']}`, "
            f"walk-forward and pit strict `{timings['walk_forward_and_pit_strict_s']}`, "
            f"peak RSS `{timings['peak_rss_mb']}` MB)."
        ),
        (
            f"- Reservoir funnel retained `{sec_a['counts']['reservoir_train_active']}` Train active wallets, "
            f"`{sec_a['counts']['historical_eligible']}` historically eligible wallets, "
            f"`{sec_a['counts']['positive_copyable_drift']}` positive COPYABLE_DRIFT wallets, and "
            f"`{sec_a['counts']['passed_stage1_tier2']}` Stage 1 Tier 2 survivors."
        ),
        (
            f"- Tier 2 CPCV PBO measured `{sec_a['cpcv']['leakfree_pbo']}` across "
            f"`{sec_a['cpcv']['leakfree_evaluated_splits']}` evaluated of "
            f"`{sec_a['cpcv']['leakfree_total_splits']}` total leak-free splits and "
            f"`{sec_a['cpcv']['full_pipeline_pbo']}` across "
            f"`{sec_a['cpcv']['full_pipeline_evaluated_splits']}` evaluated of "
            f"`{sec_a['cpcv']['full_pipeline_total_splits']}` total full-pipeline splits."
        ),
        (
            "- Section A CPCV note. Wallet eligibility, 5m markout gating (`COPYABLE_DRIFT`), and positive "
            "unit-replay pre-filtering ran once across the full 123-day pre-split Train window before "
            "CPCV block splitting."
        ),
        "",
        "## Section A primary OOS ablations on September to October 2026",
        "",
        "| Variant | Mode | Cost | Aliases | Verdict | Term net ($) | Full net ($) | Static DD ($) | Worst day ($) | Capacity status | Max part (%) | CWSA net ($) |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]

    for row in sec_a["oos_outcomes"]:
        cwsa_str = f"{row['cwsa_net_usd']:.2f}" if row["cwsa_net_usd"] is not None else "n/a"
        lines.append(
            f"| `{row['name']}` | `{row['execution_mode']}` | {row['cost_multiplier']:.0f}x | "
            f"`{', '.join(row['aliases'])}` | `{row['verdict']}` | {row['terminal_net_usd']:.2f} | "
            f"{row['full_window_net_usd']:.2f} | {row['static_floor_dd_usd']:.2f} | "
            f"{row['worst_daily_usd']:.2f} | `{row['capacity_status']}` | "
            f"{row['max_participation_pct']:.4f} | {cwsa_str} |"
        )

    lines.extend(
        [
            "",
            "## Section B four-arm containment and seat replacement modes",
            "",
            (
                "- Section B continuation note. Policy replay continues trading through the full OOS "
                "window after reaching the Guardian profit target, and `Static DD ($)` uses the same "
                "non-positive floor sign convention as Sections A and C."
            ),
            "",
            "| Tier | Policy | Net PnL ($) | Static DD ($) | Guardian verdict | PM03 max err | Bars gt 5 pct | Retired seats | Promoted bench | Blended P(Funded) pct |",
            "|---:|---|---:|---:|---|---:|---:|---|---|---:|",
        ]
    )
    for t_key, b_data in sorted(sec_b.items()):
        for policy_group in ("lifecycle_arms", "replacement_modes"):
            for p_name, row in b_data[policy_group].items():
                g_state = (
                    f"BREACH ({row['guardian_breach_reason']})"
                    if row["guardian_breached"]
                    else ("PASS" if row["guardian_target_passed"] else "ACTIVE")
                )
                lines.append(
                    f"| T{t_key} | `{p_name}` | {row['net_pnl_usd']:.2f} | "
                    f"{row['static_floor_dd_usd']:.2f} | `{g_state}` | "
                    f"{row['max_pm03_conservation_err']:.2e} | {row['bars_gt_5pct_count']} | "
                    f"`{', '.join(row['retired_seats']) or 'none'}` | "
                    f"`{', '.join(row['promoted_bench_seats']) or 'none'}` | "
                    f"{row['blended_p_funded_pct']:.2f} |"
                )

    lines.extend(
        [
            "",
            "## Section C rolling walk-forward and post-Nov-4 PIT strict evaluation",
            "",
            (
                "- Section C execution note. Rolling walk-forward folds evaluate continuous unit-scale "
                "5m returns without Bybit lot rounding, funding carry, minimum trade notional, or CWSA 1m "
                "caps, whereas `pit_strict_nov05` evaluates full Guardian and CWSA replay."
            ),
            "",
            "| Fold | Config | Train aliases | OOS status | OOS net frac | OOS static DD frac | Provenance kind |",
            "|---:|---|---|---|---:|---:|---|",
        ]
    )
    for row in sec_c["walk_forward_folds"]:
        retro_kind = row["retrospective_evidence"].get("kind", "UNKNOWN")
        lines.append(
            f"| {row['fold_idx']} | `{row['label']}` | `{', '.join(row['aliases'])}` | "
            f"`{row['oos_status']}` | {row['oos_net_ret_frac']:.4f} | "
            f"{row['oos_static_floor_dd']:.4f} | `{retro_kind}` |"
        )

    pit_sec = sec_c["pit_strict_nov05"]
    lines.extend(
        [
            "",
            (
                f"- Post-Nov-4 strict PIT snapshot at `{pit_sec['asof_iso']}` trained over "
                f"`{pit_sec['train_lookback_days']}` days on `{pit_sec['positive_copyable_drift']}` "
                f"positive COPYABLE_DRIFT wallets (`roster_sha256={pit_sec['roster_sha256'][:12]}`)."
            ),
            "",
            "| Variant | Mode | Aliases | Verdict | Full net ($) | Static DD ($) | Capacity status |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in pit_sec["outcomes"]:
        lines.append(
            f"| `{row['name']}` | `{row['execution_mode']}` | `{', '.join(row['aliases'])}` | "
            f"`{row['verdict']}` | {row['full_window_net_usd']:.2f} | "
            f"{row['static_floor_dd_usd']:.2f} | `{row['capacity_status']}` |"
        )

    lines.extend(
        [
            "",
            "## Section D multi-firm conditional EV frontier and fail-closed gates",
            "",
            (
                f"- Blocked gate verification for `propr:classic_2step:phase_1` returned "
                f"`{sec_d['blocked_fail_closed']['status']}` "
                f"(rulebook eval phases `1-Step={sec_d['blocked_fail_closed']['propr_1step_eval_phases']}`, "
                f"`2-Step={sec_d['blocked_fail_closed']['propr_2step_eval_phases']}`)."
            ),
            "",
            "| Firm slug | Daily basis | OOS verdict | Term net ($) | Full net ($) | Regime | P(Funded) pct [95 pct CI] | Breakeven fee ($) | Fee status | Net EV at 110 ($) |",
            "|---|---|---|---:|---:|---|---|---|---|---:|",
        ]
    )
    for firm in sec_d["firms"]:
        for reg_name in ("Blended", "75%-Chop"):
            reg = firm["regimes"][reg_name]
            unp = reg["fee_unverified"]
            s110 = reg["fee_110_sensitivity"]
            ci = unp["p_funded_payout_ci95_pct"]
            net_ev_str = f"{s110['net_ev_usd']:.2f}" if s110["net_ev_usd"] is not None else "n/a"
            lines.append(
                f"| `{firm['slug']}` | `{firm['daily_loss_basis']}` | `{firm['oos_verdict']}` | "
                f"{firm['oos_terminal_net_usd']:.2f} | {firm['oos_full_window_net_usd']:.2f} | `{reg_name}` | "
                f"{unp['p_funded_payout_pct']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
                f"{unp['breakeven_fee_usd']:.2f} | `{unp['ev_status']}` | {net_ev_str} |"
            )

    lines.extend(
        [
            "",
            "## Qualification and caveats",
            "",
            "### Three-line caveat footer",
            "- CFT trading restrictions and account framing require independent compliance confirmation.",
            "- Live broker execution, funding, capacity and PIT source continuity are UNPROVEN.",
            "- No deployments, allocations, orders, service or registry writes occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def run_phase3_ablation(
    config: Phase3AblationConfig | None = None,
    *,
    bybit_fetcher: Callable[[], tuple[dict[str, tuple[float, float]], dict[str, Any]]] | None = None,
    challenge_spec_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Execute the single-pass Phase 3 pod-shop ablation study and write sanitized outputs."""
    cfg = config if config is not None else Phase3AblationConfig()
    t_total_0 = time.monotonic()

    start_ms = _parse_utc_ms(cfg.start_iso)
    split_ms = _parse_utc_ms(cfg.split_iso)
    wf_end_ms = _parse_utc_ms(cfg.wf_end_iso)
    oos_end_ms = _parse_utc_ms(cfg.oos_end_iso)
    pit_strict_ms = _parse_utc_ms(cfg.pit_strict_iso)
    archive_end_ms = _parse_utc_ms(cfg.archive_end_iso)

    full_grid = np.arange(start_ms, archive_end_ms, 300_000, dtype=np.int64)
    cut_split = int(np.searchsorted(full_grid, split_ms))
    cut_wf_end = int(np.searchsorted(full_grid, wf_end_ms))
    cut_oos = int(np.searchsorted(full_grid, oos_end_ms))
    cut_pit = int(np.searchsorted(full_grid, pit_strict_ms))
    cut_archive = int(full_grid.size)
    train_grid = full_grid[:cut_split]

    t0 = time.monotonic()
    archive_paths, inv_manifest = sd.discover_fill_archive(
        start_ms,
        archive_end_ms,
        archive_root=cfg.archive_root,
    )
    if inv_manifest["missing_dates"]:
        raise RuntimeError(f"MISSING_SOURCE_DAYS:{inv_manifest['missing_dates'][:10]}")
    res_records, roster_sha, pit_provenance = sd.discover_reservoir_pit_cohort(
        archive_paths,
        train_start=start_ms,
        train_end=split_ms,
        registry_db=cfg.registry_db,
    )
    wf_union_records, _, _ = sd.discover_reservoir_pit_cohort(
        archive_paths,
        train_start=start_ms,
        train_end=max(split_ms, wf_end_ms),
        registry_db=cfg.registry_db,
    )
    pit_res = cr.load_universe_asof(cfg.registry_db, cfg.pit_strict_iso, mode="PIT_STRICT_ASOF")
    discover_archive_s = round(time.monotonic() - t0, 2)
    _log_stage("discover_complete", discover_archive_s)

    ingested_label_by_addr = {r.address.lower(): r.label for r in res_records}
    used_labels = {r.label for r in res_records}
    all_records = list(res_records)

    def _merge_records(records: Sequence[cr.WalletRecord]) -> dict[str, cr.WalletRecord]:
        by_addr: dict[str, cr.WalletRecord] = {}
        for rec in records:
            addr_l = rec.address.lower()
            if addr_l in ingested_label_by_addr:
                lbl = ingested_label_by_addr[addr_l]
                by_addr[addr_l] = rec if rec.label == lbl else replace(rec, alias=lbl)
                continue
            safe_lbl = rec.label if rec.label not in used_labels else f"{rec.label}__{rec.wallet_id}"
            while safe_lbl in used_labels:
                safe_lbl = f"{safe_lbl}_{addr_l[2:8]}"
            if safe_lbl != rec.label:
                rec = replace(rec, alias=safe_lbl)
            ingested_label_by_addr[addr_l] = safe_lbl
            used_labels.add(safe_lbl)
            all_records.append(rec)
            by_addr[addr_l] = rec
        return by_addr

    wf_record_by_addr = _merge_records(wf_union_records)
    pit_record_by_addr = _merge_records(pit_res.records)

    t0 = time.monotonic()
    ingestion = sd.ingest_fills(all_records, archive_paths, start=start_ms, end=archive_end_ms, split=split_ms)
    if bybit_fetcher is not None:
        bybit_specs, bybit_meta = bybit_fetcher()
    elif cfg.fetch_bybit_specs:
        try:
            bybit_specs, bybit_meta = sd.fetch_bybit_usdt_perp_specs()
        except (OSError, RuntimeError, ValueError, KeyError):
            bybit_specs, bybit_meta = {}, {"status": "BYBIT_UNREACHABLE"}
    else:
        bybit_specs, bybit_meta = {}, {"status": "BYBIT_DISABLED_IN_CONFIG"}
    bybit_coins = set(bybit_specs) if bybit_specs else None
    prices = sd.LazyPrices(
        full_grid,
        store_root=cfg.store_root,
        vike_root=cfg.vike_root,
        split_ms=split_ms,
    )
    fund_cache: dict[tuple[str, int, int, int], np.ndarray] = {}
    ingest_fills_s = round(time.monotonic() - t0, 2)
    _log_stage(f"ingest_complete days={len(inv_manifest['files'])}", ingest_fills_s)

    t0 = time.monotonic()
    res_grouped = {r.label: ingestion.grouped.get(r.label, {}) for r in res_records}
    res_addrs = [r.address.lower() for r in res_records]
    res_stats_df = ingestion.stats_df.filter(pl.col("wallet").is_in(res_addrs))
    res_ingestion = replace(ingestion, grouped=res_grouped, stats_df=res_stats_df)
    screen_rows = sd.screen_wallets(
        res_records,
        res_ingestion,
        split=split_ms,
        bybit_usdt_perps=bybit_coins,
        fetch_live_av=False,
    )
    hist_eligible, _ = sd.eligible_fills(
        res_grouped,
        screen_rows,
        prices,
        split=split_ms,
        scope="historical",
    )

    all_hist_coins = sorted({c for by in hist_eligible.values() for c in by})
    train_prices = {c: tuple(x[:cut_split] for x in prices.get(c)) for c in all_hist_coins}
    markout_counts: dict[str, int] = {}
    retained_drift_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    unit_dd_by_alias: dict[str, float] = {}
    unit_net_by_alias: dict[str, float] = {}
    train_stats: list[dict[str, Any]] = []

    for alias, coins_dict in sorted(hist_eligible.items()):
        prof = sd.compute_fill_latency_markouts(alias, coins_dict, prices, split_ms=split_ms)
        markout_counts[prof.status] = markout_counts.get(prof.status, 0) + 1
        frames, _ = cr.build_unit_replay_by_alias(
            {alias: coins_dict},
            {c: train_prices[c] for c in coins_dict if c in train_prices},
            [alias],
            train_grid,
            policy="new",
            activation_ms=start_ms,
        )
        w_frames = frames[alias]
        if not w_frames:
            continue
        totals = np.zeros(cut_split, dtype=np.float64)
        for df_f in w_frames.values():
            totals += df_f["net"].to_numpy()
        eq = np.cumsum(totals)
        peak = np.maximum.accumulate(np.maximum(eq, 0.0))
        dd = float(np.max(peak - eq)) if eq.size else 0.0
        net = float(totals.sum())
        unit_net_by_alias[alias] = net
        unit_dd_by_alias[alias] = dd
        train_stats.append(
            {
                "alias": alias,
                "net_usd": net,
                "maxdd_usd": dd,
                "net_over_dd": net / max(dd, 0.01),
                "markout_status": prof.status,
            }
        )
        if net > 0.0 and dd > 0.0 and prof.status == "COPYABLE_DRIFT":
            compact = _unit_frames_to_compact_arrays(w_frames)
            if compact:
                retained_drift_arrays[alias] = compact

    del train_prices
    prices.clear_1m_cache()
    gc.collect()
    markouts_and_train_replay_s = round(time.monotonic() - t0, 2)
    _log_stage("markouts_and_train_replay_complete", markouts_and_train_replay_s)

    t0 = time.monotonic()
    chop_idx = min(int(0.75 * max(1, cut_split)), max(0, cut_split - 1))
    train_tensor = pc.build_tensor_from_arrays(
        retained_drift_arrays,
        train_grid,
        pod_equity_usd=10_000.0,
        chop_split_ms=int(train_grid[chop_idx]),
    )
    train_tensor = _ensure_tensor_prices(train_tensor, prices, 0, cut_split)
    positive_drift_aliases = tuple(retained_drift_arrays.keys())
    retained_drift_arrays.clear()

    base_euler = pc.PodConstructorConfig(
        pod_size=cfg.pod_size,
        top_per_cluster=cfg.top_per_cluster,
        max_candidate_pods=cfg.max_candidate_pods,
        min_active_bar_frac=cfg.min_active_bar_frac,
        min_episodes=cfg.min_episodes,
        min_overlap_days=cfg.min_overlap_days,
        min_lot_usd=cfg.min_lot_usd,
        min_baseline_order_usd=cfg.min_baseline_order_usd,
        baseline_port_dd_frac=0.02,
        risk_share_mode="marginal_euler",
        max_marginal_cdar_share=cfg.max_marginal_cdar_share,
        cpcv_n_blocks=cfg.cpcv_n_blocks,
        reject_hidden_bagholder=True,
        reject_unknown_tilt=True,
    )
    legacy_cfg = replace(base_euler, risk_share_mode="standalone_proxy", max_marginal_cdar_share=None)
    legacy_t1_cfg = pc.PodConstructorConfig.for_sizing_tier(1, base=legacy_cfg)
    legacy_pods, _, _, _ = pc.discover_diverse_pods(train_tensor, legacy_t1_cfg)
    legacy_top = legacy_pods[0]
    legacy_k = [float(k) for k in legacy_top.multipliers]

    naive_ranked = sorted(
        [r for r in train_stats if r["net_usd"] > 0.0 and r["maxdd_usd"] > 0.0],
        key=lambda x: (x["net_usd"], x["net_over_dd"]),
        reverse=True,
    )[: cfg.pod_size]
    naive_aliases = [r["alias"] for r in naive_ranked]
    naive_k = [
        min((0.02 * 10_000.0 / float(max(1, len(naive_aliases)))) / max(unit_dd_by_alias[a], 0.01), 5.0)
        for a in naive_aliases
    ]

    tiers_t: tuple[Literal[1, 2, 3, 4], ...] = (1, 2, 3, 4)
    tier_configs: dict[Literal[1, 2, 3, 4], pc.PodConstructorConfig] = {}
    tier_pods: dict[Literal[1, 2, 3, 4], list[pc.SizedPodCandidate]] = {}
    tier_metrics: dict[Literal[1, 2, 3, 4], list[pc.TraderAuditMetrics]] = {}
    tier_baselines: dict[Literal[1, 2, 3, 4], dict[int, pc.TraderEpisodeBaseline]] = {}
    for tier in tiers_t:
        t_cfg = pc.PodConstructorConfig.for_sizing_tier(tier, base=base_euler)
        pods_t, mets_t, bases_t, _ = pc.discover_diverse_pods(train_tensor, t_cfg)
        tier_configs[tier] = t_cfg
        tier_pods[tier] = pods_t
        tier_metrics[tier] = mets_t
        tier_baselines[tier] = bases_t

    t2_passed_ids = [m.trader_id for m in tier_metrics[2] if m.passed_behavioral_gate]
    leakfree_cand_pods = [p.pod_indices for p in tier_pods[2][:6]]
    leakfree_pbo, leakfree_logits = pc.compute_leakfree_cpcv_pbo(
        leakfree_cand_pods,
        train_tensor,
        tier_configs[2],
    )
    n_days_tr = max(2, train_tensor.timestamps_ms.size // tier_configs[2].bars_per_day)
    n_blocks_tr = min(tier_configs[2].cpcv_n_blocks, n_days_tr // 2)
    if n_blocks_tr % 2 != 0:
        n_blocks_tr -= 1
    n_blocks_tr = max(2, n_blocks_tr)
    leakfree_total_splits = math.comb(n_blocks_tr, n_blocks_tr // 2)
    leakfree_evaluated_splits = 0 if len(leakfree_cand_pods) < 2 else int(leakfree_logits.size)
    full_pbo, full_logits, full_split_records = pc.compute_full_pipeline_cpcv_pbo(
        train_tensor,
        tier_configs[2],
    )
    full_status_counts: dict[str, int] = {}
    for rec_fp in full_split_records:
        st_fp = str(rec_fp["status"])
        full_status_counts[st_fp] = full_status_counts.get(st_fp, 0) + 1
    full_evaluated_splits = sum(1 for r in full_split_records if r["status"] == "OK")
    full_total_splits = len(full_split_records)
    stage2_and_cpcv_s = round(time.monotonic() - t0, 2)
    _log_stage("stage2_cpcv_complete", stage2_and_cpcv_s)

    t0 = time.monotonic()
    liq_cfg = cr.CausalLiquidityConfig(l1_window_days=cfg.l1_window_days, max_participation_1m=0.05)
    oos_outcomes: list[dict[str, Any]] = []
    primary_cwsa_res: cr.CWSAReplayResult | None = None

    def _record_eval(
        name: str,
        tier: int | None,
        aliases: Sequence[str],
        k_weights: Sequence[float],
        *,
        execution_mode: str,
        cost_multiplier: float,
    ) -> cr.CWSAReplayResult | None:
        outcome, _, c_res = _eval_pod_on_window(
            name,
            tier,
            aliases,
            k_weights,
            res_grouped,
            prices,
            full_grid,
            cut_split,
            cut_oos,
            cfg.store_root,
            fund_cache,
            bybit_specs,
            liq_cfg,
            execution_mode=execution_mode,
            cost_multiplier=cost_multiplier,
        )
        oos_outcomes.append(asdict(outcome))
        return c_res

    _record_eval(
        "Naive_Leaderboard_1_over_DD",
        None,
        naive_aliases,
        naive_k,
        execution_mode="bar_delta",
        cost_multiplier=1.0,
    )
    _record_eval(
        "Deep_Legacy_BarDelta_StandaloneProxy",
        None,
        legacy_top.aliases,
        legacy_k,
        execution_mode="bar_delta",
        cost_multiplier=1.0,
    )
    for tier in tiers_t:
        top_t = tier_pods[tier][0]
        k_t = [float(x) for x in top_t.multipliers]
        for mult in COST_MULTIPLIERS:
            suffix = f"_{int(mult)}xCost" if mult != 1.0 else ""
            _record_eval(
                f"Deep_Euler_Tier{tier}_BarDelta{suffix}",
                tier,
                top_t.aliases,
                k_t,
                execution_mode="bar_delta",
                cost_multiplier=mult,
            )
            c_res = _record_eval(
                f"Deep_Euler_Tier{tier}_CWSA{suffix}",
                tier,
                top_t.aliases,
                k_t,
                execution_mode="cwsa_one_shot_v3",
                cost_multiplier=mult,
            )
            if tier == 2 and abs(mult - 1.0) < 1e-9:
                primary_cwsa_res = c_res

    cft_contract = pc.load_prop_contract_spec(
        "crypto-fund-trader",
        account_usd=10_000.0,
        challenge_fee_usd=None,
        rules_json_path=cfg.rules_json_path,
    )
    section_b: dict[str, Any] = {}
    section_b_tiers: tuple[Literal[1, 2, 3, 4], ...] = (2, 3)
    for tier in section_b_tiers:
        section_b[str(tier)] = _run_section_b_for_tier(
            tier=tier,
            pod=tier_pods[tier][0],
            tier_cfg=tier_configs[tier],
            metrics_list=tier_metrics[tier],
            baselines=tier_baselines[tier],
            passed_ids=[m.trader_id for m in tier_metrics[tier] if m.passed_behavioral_gate],
            train_symbols=train_tensor.symbols,
            grouped=res_grouped,
            prices=prices,
            full_grid=full_grid,
            cut_split=cut_split,
            cut_oos=cut_oos,
            split_ms=split_ms,
            store_root=cfg.store_root,
            fund_cache=fund_cache,
            liq_cfg=liq_cfg,
            cft_contract=cft_contract,
            mc_n_paths=cfg.mc_n_paths,
        )

    if primary_cwsa_res is None:
        raise RuntimeError("MISSING_PRIMARY_CWSA_RESULT")
    spec_loader = challenge_spec_loader if challenge_spec_loader is not None else _default_challenge_spec_loader
    section_d = _evaluate_multifirm_frontier(
        tier_pods[2][0].aliases,
        [float(x) for x in tier_pods[2][0].multipliers],
        prices,
        full_grid,
        cut_split,
        cut_oos,
        cfg.store_root,
        fund_cache,
        bybit_specs,
        primary_cwsa_res.executed_positions,
        primary_cwsa_res.coins,
        cfg.rules_json_path,
        cfg.mc_n_paths,
        spec_loader,
    )

    train_tier_summaries = {
        f"tier_{t}": {
            "aliases": [sd.sanitize_public_label(a) for a in tier_pods[t][0].aliases],
            "k_weights": [round(float(x), 6) for x in tier_pods[t][0].multipliers],
            "marginal_cdar_shares": [round(float(x), 4) for x in tier_pods[t][0].marginal_cdar_shares],
            "binding_constraint": tier_pods[t][0].binding_constraint,
            "feasible": bool(tier_pods[t][0].feasible),
        }
        for t in tiers_t
    }

    del train_tensor, tier_metrics, tier_baselines
    gc.collect()
    oos_ablations_s = round(time.monotonic() - t0, 2)
    _log_stage("sections_abd_complete", oos_ablations_s)

    t0 = time.monotonic()
    train_bars = cfg.walk_forward_train_days * 288
    eval_bars = cfg.walk_forward_eval_days * 288
    wf_rows: list[dict[str, Any]] = []
    fold_idx = 0
    while True:
        c_tr_s = fold_idx * eval_bars
        c_tr_e = c_tr_s + train_bars
        c_ev_e = c_tr_e + eval_bars
        if c_ev_e > cut_wf_end:
            break
        t_fold0 = time.monotonic()
        f_tr_start_ms = int(full_grid[c_tr_s])
        f_tr_end_ms = int(full_grid[c_tr_e])
        f_ev_end_ms = int(full_grid[c_ev_e]) if c_ev_e < full_grid.size else int(full_grid[-1]) + 300_000
        fold_grid = full_grid[c_tr_s:c_ev_e]

        fold_records: list[cr.WalletRecord] = []
        fold_train_grouped: dict[str, dict[str, pl.DataFrame]] = {}
        for raw_rec in wf_union_records:
            rec = wf_record_by_addr[raw_rec.address.lower()]
            lbl = rec.label
            tr_coins: dict[str, pl.DataFrame] = {}
            for coin, df in ingestion.grouped.get(lbl, {}).items():
                if not cr.is_hl_perp_coin(coin):
                    continue
                sub_df = df.filter((pl.col("ts") >= f_tr_start_ms) & (pl.col("ts") < f_tr_end_ms))
                if sub_df.height > 0:
                    tr_coins[coin] = sub_df
            if tr_coins:
                fold_records.append(rec)
                fold_train_grouped[lbl] = tr_coins

        fold_screen_rows = [
            sd.summarize_wallet(
                rec,
                fold_train_grouped[rec.label],
                split=f_tr_end_ms,
                bybit_usdt_perps=bybit_coins,
            )
            for rec in fold_records
        ]
        fold_eligible, _ = sd.eligible_fills(
            fold_train_grouped,
            fold_screen_rows,
            prices,
            split=f_tr_end_ms,
            scope="historical",
        )

        drift_aliases: list[str] = []
        for alias, fold_train_eligible_coins in sorted(fold_eligible.items()):
            prof = sd.compute_fill_latency_markouts(
                alias,
                fold_train_eligible_coins,
                prices,
                split_ms=f_tr_end_ms,
            )
            if prof.status == "COPYABLE_DRIFT":
                drift_aliases.append(alias)
        gc.collect()

        fold_window_fills: dict[str, dict[str, pl.DataFrame]] = {}
        fold_active_coins_set: set[str] = set()
        for alias in drift_aliases:
            elig_coins = fold_eligible[alias]
            w_src = ingestion.grouped.get(alias, {})
            w_win: dict[str, pl.DataFrame] = {}
            for coin in elig_coins:
                df_c = w_src.get(coin)
                if df_c is None:
                    continue
                sub_win = df_c.filter((pl.col("ts") >= f_tr_start_ms) & (pl.col("ts") < f_ev_end_ms))
                if sub_win.height > 0:
                    w_win[coin] = sub_win
                    fold_active_coins_set.add(coin)
            if w_win:
                fold_window_fills[alias] = w_win

        fold_active_coins = sorted(fold_active_coins_set)
        fold_prices = {c: tuple(x[c_tr_s:c_ev_e] for x in prices.get(c)) for c in fold_active_coins}
        fold_compact: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for alias, w_win in sorted(fold_window_fills.items()):
            frames, _ = cr.build_unit_replay_by_alias(
                {alias: w_win},
                {c: fold_prices[c] for c in w_win if c in fold_prices},
                [alias],
                fold_grid,
                policy="new",
                activation_ms=f_tr_start_ms,
            )
            w_frames = frames[alias]
            if not w_frames:
                continue
            totals = np.zeros(train_bars, dtype=np.float64)
            for df_f in w_frames.values():
                totals += df_f["net"].to_numpy()[:train_bars]
            eq = np.cumsum(totals)
            dd = float(np.max(np.maximum.accumulate(np.maximum(eq, 0.0)) - eq)) if eq.size else 0.0
            net = float(totals.sum())
            if net > 0.0 and dd > 0.0:
                compact = _unit_frames_to_compact_arrays(w_frames)
                if compact:
                    fold_compact[alias] = compact
        del fold_prices, fold_train_grouped, fold_window_fills, fold_eligible

        fold_chop_idx = min(int(0.75 * max(1, train_bars)), max(0, train_bars - 1))
        fold_tensor = pc.build_tensor_from_arrays(
            fold_compact,
            fold_grid,
            pod_equity_usd=10_000.0,
            chop_split_ms=int(fold_grid[fold_chop_idx]),
        )
        fold_compact.clear()
        fold_tensor = _ensure_tensor_prices(fold_tensor, prices, c_tr_s, c_ev_e)

        wf_t2_folds = pc.run_rolling_walk_forward(
            fold_tensor,
            train_days=cfg.walk_forward_train_days,
            eval_days=cfg.walk_forward_eval_days,
            tiers=(2,),
            base_config=base_euler,
        )
        wf_t1_folds = pc.run_rolling_walk_forward(
            fold_tensor,
            train_days=cfg.walk_forward_train_days,
            eval_days=cfg.walk_forward_eval_days,
            tiers=(1,),
            base_config=legacy_cfg,
        )
        del fold_tensor
        gc.collect()

        for label_wf, fold_list in (
            ("Tier2_MarginalEuler", wf_t2_folds),
            ("Tier1_StandaloneProxy", wf_t1_folds),
        ):
            for f_res in fold_list:
                asof_dt = datetime.fromtimestamp(float(f_res.train_end_ms) / 1000.0, tz=UTC)
                retro = cr.load_universe_asof(
                    cfg.registry_db,
                    asof_dt,
                    mode="RETROSPECTIVE_ASOF_MASKED",
                ).retrospective_evidence
                wf_rows.append(
                    {
                        "label": label_wf,
                        "fold_idx": int(fold_idx),
                        "tier": int(f_res.tier),
                        "train_start_iso": datetime.fromtimestamp(f_res.train_start_ms / 1000.0, tz=UTC).isoformat(),
                        "train_end_iso": asof_dt.isoformat(),
                        "eval_end_iso": datetime.fromtimestamp(f_res.eval_end_ms / 1000.0, tz=UTC).isoformat(),
                        "aliases": [sd.sanitize_public_label(a) for a in f_res.aliases],
                        "multipliers": [round(float(x), 6) for x in f_res.multipliers],
                        "oos_status": f_res.oos_status,
                        "oos_net_ret_frac": round(float(f_res.oos_net_ret_frac), 5),
                        "oos_static_floor_dd": round(float(f_res.oos_static_floor_dd), 5),
                        "oos_max_daily_loss": round(float(f_res.oos_max_daily_loss), 5),
                        "binding_constraint": f_res.binding_constraint,
                        "retrospective_evidence": retro or {},
                    }
                )
        _log_stage(f"walk_forward_fold_{fold_idx}_complete", time.monotonic() - t_fold0)
        fold_idx += 1

    pit_train_start_ms = pit_strict_ms - cfg.pit_strict_lookback_days * 86_400_000
    cut_pit_start = int(np.searchsorted(full_grid, pit_train_start_ms))
    pit_train_grid = full_grid[cut_pit_start:cut_pit]
    pit_eval_records = [pit_record_by_addr[r.address.lower()] for r in pit_res.records]
    pit_grouped = {
        rec.label: ingestion.grouped.get(rec.label, {})
        for rec in pit_eval_records
    }
    pit_rows: list[dict[str, Any]] = []
    for rec in pit_eval_records:
        wf_fills = {
            coin: df.filter((df["ts"] >= pit_train_start_ms) & (df["ts"] < pit_strict_ms))
            for coin, df in pit_grouped.get(rec.label, {}).items()
        }
        wf_fills = {coin: df for coin, df in wf_fills.items() if df.height > 0}
        pit_rows.append(
            sd.summarize_wallet(
                rec,
                wf_fills,
                split=pit_strict_ms,
                bybit_usdt_perps=bybit_coins,
            )
        )

    pit_eligible, _ = sd.eligible_fills(
        pit_grouped,
        pit_rows,
        prices,
        split=pit_strict_ms,
        scope="historical",
    )
    pit_coins = sorted({c for by in pit_eligible.values() for c in by})
    pit_prices = {c: tuple(x[cut_pit_start:cut_pit] for x in prices.get(c)) for c in pit_coins}
    pit_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for alias, coins_dict in sorted(pit_eligible.items()):
        sliced = {
            c: df.filter((df["ts"] >= pit_train_start_ms) & (df["ts"] < pit_strict_ms))
            for c, df in coins_dict.items()
        }
        sliced = {c: df for c, df in sliced.items() if df.height > 0}
        if not sliced:
            continue
        prof = sd.compute_fill_latency_markouts(alias, sliced, prices, split_ms=pit_strict_ms)
        if prof.status != "COPYABLE_DRIFT":
            continue
        frames, _ = cr.build_unit_replay_by_alias(
            {alias: sliced},
            {c: pit_prices[c] for c in sliced if c in pit_prices},
            [alias],
            pit_train_grid,
            policy="new",
            activation_ms=pit_train_start_ms,
        )
        w_frames = frames[alias]
        if not w_frames:
            continue
        tot = np.zeros(pit_train_grid.size, dtype=np.float64)
        for df_f in w_frames.values():
            tot += df_f["net"].to_numpy()
        eq = np.cumsum(tot)
        dd = float(np.max(np.maximum.accumulate(np.maximum(eq, 0.0)) - eq)) if eq.size else 0.0
        if float(tot.sum()) > 0.0 and dd > 0.0:
            compact = _unit_frames_to_compact_arrays(w_frames)
            if compact:
                pit_arrays[alias] = compact
    del pit_prices
    prices.clear_1m_cache()
    gc.collect()

    pit_positive_drift_count = len(pit_arrays)
    pit_chop_idx = min(int(0.67 * max(1, pit_train_grid.size)), max(0, pit_train_grid.size - 1))
    pit_tensor = pc.build_tensor_from_arrays(
        pit_arrays,
        pit_train_grid,
        pod_equity_usd=10_000.0,
        chop_split_ms=int(pit_train_grid[pit_chop_idx]),
    )
    pit_arrays.clear()
    pit_tensor = _ensure_tensor_prices(pit_tensor, prices, cut_pit_start, cut_pit)

    pit_t2_cfg = pc.PodConstructorConfig.for_sizing_tier(2, base=base_euler)
    pit_pods, _, _, _ = pc.discover_diverse_pods(pit_tensor, pit_t2_cfg)
    pit_top = pit_pods[0]
    pit_k = [float(x) for x in pit_top.multipliers]
    del pit_tensor
    gc.collect()

    pit_outcomes: list[dict[str, Any]] = []
    for mode_p in ("bar_delta", "cwsa_one_shot_v3"):
        p_out, _, _ = _eval_pod_on_window(
            f"PIT_Strict_Nov05_Tier2_{mode_p}",
            2,
            pit_top.aliases,
            pit_k,
            pit_grouped,
            prices,
            full_grid,
            cut_pit,
            cut_archive,
            cfg.store_root,
            fund_cache,
            bybit_specs,
            liq_cfg,
            execution_mode=mode_p,
            cost_multiplier=1.0,
            censor_timeout=True,
        )
        pit_outcomes.append(asdict(p_out))

    walk_forward_and_pit_strict_s = round(time.monotonic() - t0, 2)
    peak_rss_mb = _log_stage("walk_forward_and_pit_strict_complete", walk_forward_and_pit_strict_s)
    total_s = round(time.monotonic() - t_total_0, 2)

    raw_payload: dict[str, Any] = {
        "status": "RESEARCH_PHASE3_ABLATION",
        "window": {
            "start_iso": cfg.start_iso,
            "split_iso": cfg.split_iso,
            "wf_end_iso": cfg.wf_end_iso,
            "oos_end_iso": cfg.oos_end_iso,
            "pit_strict_iso": cfg.pit_strict_iso,
            "archive_end_iso": cfg.archive_end_iso,
        },
        "archive_days": len(inv_manifest["files"]),
        "archive_sha256": inv_manifest["archive_sha256"],
        "roster_sha256": roster_sha,
        "pit_provenance": pit_provenance,
        "bybit_meta": bybit_meta,
        "stage_timings_s": {
            "discover_archive_s": discover_archive_s,
            "ingest_fills_s": ingest_fills_s,
            "markouts_and_train_replay_s": markouts_and_train_replay_s,
            "stage2_and_cpcv_s": stage2_and_cpcv_s,
            "oos_ablations_s": oos_ablations_s,
            "walk_forward_and_pit_strict_s": walk_forward_and_pit_strict_s,
            "total_s": total_s,
            "peak_rss_mb": round(peak_rss_mb, 2),
        },
        "section_a": {
            "counts": {
                "reservoir_train_active": len(res_records),
                "historical_eligible": len(hist_eligible),
                "markout_breakdown": markout_counts,
                "positive_copyable_drift": len(positive_drift_aliases),
                "passed_stage1_tier2": len(t2_passed_ids),
            },
            "cpcv": {
                "leakfree_pbo": round(float(leakfree_pbo), 4),
                "leakfree_evaluated_splits": leakfree_evaluated_splits,
                "leakfree_total_splits": leakfree_total_splits,
                "leakfree_logits": [round(float(x), 5) for x in leakfree_logits],
                "full_pipeline_pbo": round(float(full_pbo), 4),
                "full_pipeline_evaluated_splits": full_evaluated_splits,
                "full_pipeline_total_splits": full_total_splits,
                "full_pipeline_status_counts": full_status_counts,
                "full_pipeline_logits": [round(float(x), 5) for x in full_logits],
                "full_pipeline_split_records": full_split_records,
            },
            "train_tier_summaries": train_tier_summaries,
            "oos_outcomes": oos_outcomes,
        },
        "section_b": section_b,
        "section_c": {
            "walk_forward_folds": wf_rows,
            "pit_strict_nov05": {
                "asof_iso": cfg.pit_strict_iso,
                "train_lookback_days": cfg.pit_strict_lookback_days,
                "roster_sha256": pit_res.roster_sha256,
                "registry_inventory_sha256": pit_res.registry_inventory_sha256,
                "active_records": len(pit_res.records),
                "historical_eligible": len(pit_eligible),
                "positive_copyable_drift": pit_positive_drift_count,
                "selected_aliases": [sd.sanitize_public_label(a) for a in pit_top.aliases],
                "selected_k_weights": [round(float(x), 6) for x in pit_k],
                "outcomes": pit_outcomes,
            },
        },
        "section_d": section_d,
    }
    payload: dict[str, Any] = _sanitize_payload(raw_payload)
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = _render_markdown(payload)
    _check_report_hygiene(md_text, json_text)

    if cfg.write_reports:
        cfg.report_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_json.write_text(json_text, encoding="utf-8")
        cfg.report_md.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_md.write_text(md_text, encoding="utf-8")

    return payload


def main() -> int:
    """Run Phase 3 ablation on the May-Nov 2026 fill archive and print runtime summary."""
    payload = run_phase3_ablation()
    print(
        f"Phase 3 ablation complete in {payload['stage_timings_s']['total_s']}s -> "
        f"{Phase3AblationConfig().report_md}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
