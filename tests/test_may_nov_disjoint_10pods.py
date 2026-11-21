"""Unit tests for ``studies/2026-11-21__cft__10k__may-nov-disjoint-10pods.py``."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from copytrade_kernel import KnownAtTs
from copytrade_pm.facts import FactStore
from copytrade_pm.roster import RosterStore, SeatSpec
from research_fixtures import (
    DAY_MS,
    START,
    WALLET_A,
    WALLET_B,
    bars_1m_from_fills,
    build_store,
    wallet_round_trips,
)

from trading_ops.research import GRID_MS, BacktestWindow, MarketSpec, ScoreConfig

ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = ROOT / "studies" / "2026-11-21__cft__10k__may-nov-disjoint-10pods.py"


def _load_study_module():
    spec = importlib.util.spec_from_file_location("may_nov_disjoint_10pods", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WALLET_C = "0x" + "c" * 40
ALIAS_MAP = {
    WALLET_A: "R#aaa00001",
    WALLET_B: "R#bbb00002",
    WALLET_C: "R#ccc00003",
}


def test_strict_pit_check_and_stage1_audit_isolated_roster(tmp_path: Path) -> None:
    study = _load_study_module()
    bars_3d = 3 * 288
    fills_a = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=bars_3d, base_price=20000.0, size=0.1
    )
    fills_b = wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=bars_3d, base_price=2000.0, size=1.0
    )
    fills_c = wallet_round_trips(
        WALLET_C, "SOL", start_ms=START, bars=bars_3d, base_price=100.0, size=0.01
    )
    all_fills = fills_a + fills_b + fills_c

    store = build_store(
        tmp_path / "src_ws",
        fills=all_fills,
        equity={
            WALLET_A: (Decimal(12000), 4),
            WALLET_B: (Decimal(9000), 4),
            WALLET_C: (Decimal(5000), 4),
        },
    )
    roster_path = tmp_path / "stage1_roster.sqlite"
    roster = RosterStore(roster_path, create=True)
    as_of = KnownAtTs(START + 3 * DAY_MS)
    try:
        strict = study.run_strict_pit_check(
            store,
            wallets=[WALLET_A, WALLET_B, WALLET_C],
            as_of=as_of,
            lookback_days=3,
            address_to_alias=ALIAS_MAP,
        )
        assert strict["cards_count"] == 0
        assert strict["basis"]["pit"] == "STRICT_KNOWN_AT"

        win = study.audit_stage1_window(
            store,
            name="synthetic_3d",
            as_of=as_of,
            lookback_days=3,
            wallets=[WALLET_A, WALLET_B, WALLET_C],
            address_to_alias=ALIAS_MAP,
            roster=roster,
            config=ScoreConfig(
                min_baseline_order_usd=Decimal(15),
                max_euler_cdar_share=Decimal(1),
                max_p95_participation_1m=Decimal(1),
                max_max_participation_1m=Decimal(1),
            ),
        )
    finally:
        roster.close()
        store.close()

    assert win["total_wallets"] == 3
    assert win["scored_count"] == 2
    assert win["eligible_count"] == 2
    assert win["scorecards_recorded"] == 2
    assert "BASELINE_ORDER_BELOW_FLOOR" in win["excluded_by_reason"]
    assert win["excluded_by_reason"]["BASELINE_ORDER_BELOW_FLOOR"] == ["R#ccc00003"]
    by_alias = {row["alias"]: row for row in win["wallets"]}
    assert by_alias["R#aaa00001"]["eligible"] is True
    assert by_alias["R#aaa00001"]["bench_cdar_share"] is not None
    assert by_alias["R#ccc00003"]["excluded_reason"] == "BASELINE_ORDER_BELOW_FLOOR"
    assert by_alias["R#ccc00003"]["observed"] == bars_3d
    study._assert_sanitized_text("stage1_win", json.dumps(win))


def test_multi_firm_catalog_and_sim_inputs_fees(tmp_path: Path) -> None:
    study = _load_study_module()
    sim_inputs_path = tmp_path / "sim_inputs.json"
    sim_inputs_path.write_text(
        json.dumps(
            {
                "version": 2,
                "challenges": {
                    "hyrotrader:two_step": {
                        "refundableOnPass": True,
                        "payoutFrequency": "on-demand",
                        "verified_at": "2026-11-19",
                        "notes": "Refundable deposit.",
                        "tiers": {"10000": 119},
                    },
                    "vanta-trading:classic_1step_3pct": {
                        "refundableOnPass": False,
                        "payoutFrequency": "on-demand",
                        "verified_at": "2026-11-19",
                        "notes": "3% daily default 100/0.",
                        "tiers": {"10000": 127},
                    },
                    "propr:classic_1step": {
                        "refundableOnPass": False,
                        "payoutFrequency": "on-demand",
                        "verified_at": "2026-11-19",
                        "notes": "Classic 1-Step.",
                        "tiers": {"10000": 110},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    rows = study.load_multi_firm_catalog(sim_inputs_path, account_usd=10_000.0)
    by_firm = {row["firm_slug"]: row for row in rows}

    assert set(by_firm) == {"crypto-fund-trader", "hyrotrader", "vanta-trading", "propr"}
    assert by_firm["crypto-fund-trader"]["challenge_fee_usd"] is None
    assert by_firm["crypto-fund-trader"]["ev_status"] == "BREAK_EVEN_ONLY_FEE_UNVERIFIED"

    assert by_firm["hyrotrader"]["challenge_fee_usd"] == pytest.approx(119.0)
    assert by_firm["hyrotrader"]["refundable_on_pass"] is True
    assert by_firm["hyrotrader"]["ev_status"] == "SOURCED_SIM_INPUTS"

    assert by_firm["vanta-trading"]["program_name"] == "Classic (1-Step Evaluation, 3% daily)"
    assert by_firm["vanta-trading"]["challenge_fee_usd"] == pytest.approx(127.0)
    assert by_firm["vanta-trading"]["eval_phases"][0]["daily_loss_limit_usd"] == pytest.approx(300.0)
    assert by_firm["vanta-trading"]["ev_status"] == "SOURCED_SIM_INPUTS"

    assert by_firm["propr"]["program_name"] == "Classic 1-Step"
    assert by_firm["propr"]["challenge_fee_usd"] == pytest.approx(110.0)
    assert by_firm["propr"]["ev_status"] == "SOURCED_SIM_INPUTS"


def test_build_pod_fact_store_and_stage2_kernel_backtest(tmp_path: Path) -> None:
    study = _load_study_module()
    bars_7d = 7 * 288
    fills_a = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=bars_7d, base_price=20000.0, size=0.1
    )
    fills_b = wallet_round_trips(
        WALLET_B, "kPEPE", start_ms=START, bars=bars_7d, base_price=2000.0, size=1.0
    )
    fills_c = wallet_round_trips(
        WALLET_C, "SOL", start_ms=START, bars=bars_7d, base_price=100.0, size=1.0
    )
    all_fills = fills_a + fills_b + fills_c
    bars_rows = bars_1m_from_fills(all_fills)

    src_root = tmp_path / "src_ws"
    src_store = build_store(
        src_root,
        fills=all_fills,
        equity={
            WALLET_A: (Decimal(12000), 1),
            WALLET_B: (Decimal(9000), 1),
            WALLET_C: (Decimal(8000), 1),
        },
        bars_1m_rows=bars_rows,
    )
    src_store.close()

    pod_store_path = tmp_path / "pod_facts"
    anchor_day_3 = datetime.fromtimestamp((START + 3 * DAY_MS) / 1000, UTC).date().isoformat()
    raw_coins = study.build_pod_fact_store(
        src_root / "facts",
        pod_store_path,
        pod_wallets=[WALLET_A, WALLET_B],
        anchor_days=("2026-10-01", anchor_day_3),
    )
    assert raw_coins == ("BTC", "kPEPE")

    rulebook = study.build_cft_rulebook(evidence_sha="d" * 64)
    markets = tuple(MarketSpec(coin) for coin in raw_coins)
    window = BacktestWindow(START + 3 * DAY_MS, START + 5 * DAY_MS)
    perm_cfg = ScoreConfig(
        min_baseline_order_usd=Decimal("0.01"),
        max_fixed_k=Decimal("0.005"),
        max_euler_cdar_share=Decimal(1),
        max_p95_participation_1m=Decimal(1),
        max_max_participation_1m=Decimal(1),
    )
    seats = (
        SeatSpec("seat_1", Decimal(200), Decimal(1)),
        SeatSpec("seat_2", Decimal(200), Decimal(1)),
        SeatSpec("seat_3", Decimal(200), Decimal(1)),
    )

    with FactStore(pod_store_path, create=False) as pod_facts:
        pod_fills = pod_facts.scan("fills", allow_unknown_known_at=True).collect()
        assert set(pod_fills["wallet"].unique().to_list()) == {WALLET_A, WALLET_B}
        pod_bars = pod_facts.scan("bars_1m", allow_unknown_known_at=True).collect()
        assert set(pod_bars["coin"].unique().to_list()) == {"BTC", "KPEPE"}
        pod_eq = pod_facts.scan("equity", allow_unknown_known_at=True).collect()
        assert pod_eq.height == 4

        run1 = study.run_pod_backtest_case(
            pod_facts,
            pod_id="MPOD-TEST",
            config_name="permissive_constituent",
            window_name="synthetic_2d",
            window=window,
            lookback_days=2,
            seats=seats,
            score_config=perm_cfg,
            rulebook=rulebook,
            markets=markets,
            roster_path=tmp_path / "pod_roster_1.sqlite",
            address_to_alias=ALIAS_MAP,
        )
        run2 = study.run_pod_backtest_case(
            pod_facts,
            pod_id="MPOD-TEST",
            config_name="permissive_constituent",
            window_name="synthetic_2d",
            window=window,
            lookback_days=2,
            seats=seats,
            score_config=perm_cfg,
            rulebook=rulebook,
            markets=markets,
            roster_path=tmp_path / "pod_roster_2.sqlite",
            address_to_alias=ALIAS_MAP,
        )
        empty_run = study.run_pod_backtest_case(
            pod_facts,
            pod_id="MPOD-TEST",
            config_name="strict_gate",
            window_name="empty_nov",
            window=BacktestWindow(START + 20 * DAY_MS, START + 22 * DAY_MS),
            lookback_days=2,
            seats=seats,
            score_config=perm_cfg,
            rulebook=rulebook,
            markets=markets,
            roster_path=tmp_path / "pod_roster_empty.sqlite",
            address_to_alias=ALIAS_MAP,
        )

    assert run1["status"] == "OK"
    assert run1["decision_count"] == 2 * DAY_MS // GRID_MS
    assert run1["reconcile_plans_count"] == 2
    assert run1["scorecards_recorded"] == 4
    assert run1["allocation_mode_share"] == {"PARTIAL": 1.0}
    assert run1["challenge_pass_rate"] == 1.0
    assert len(run1["first_decision_sha"]) == 64
    assert len(run1["last_decision_sha"]) == 64
    assert run1["first_decision_sha"] == run2["first_decision_sha"]
    assert run1["last_decision_sha"] == run2["last_decision_sha"]
    assert empty_run["status"] == "NO_FACT_ROWS_IN_WINDOW"
    assert empty_run["decision_count"] == 0
    study._assert_sanitized_text("stage2_run", json.dumps(run1))


def test_sanitized_outputs_and_hex_leak_guard(tmp_path: Path) -> None:
    study = _load_study_module()
    cleaned = study._sanitize_text(f"NO_DRAWDOWN_EVIDENCE:{WALLET_A}", ALIAS_MAP)
    assert cleaned == "NO_DRAWDOWN_EVIDENCE:R#aaa00001"

    with pytest.raises(RuntimeError, match="HEX_ADDRESS_LEAK"):
        study._sanitize_text(f"unmapped:{'0x' + 'f' * 40}", ALIAS_MAP)

    rulebook = study.build_cft_rulebook(evidence_sha="d" * 64)
    sample_stage2 = [
        {
            "pod_id": "MPOD-06",
            "aliases": ["R#aaa00001", "R#bbb00002", "R#ccc00003"],
            "frozen_k": [0.01, 0.02, 0.03],
            "runs": [
                {
                    "pod_id": "MPOD-06",
                    "config_name": "permissive_constituent",
                    "window_name": "sep_nov_oos",
                    "status": "OK",
                    "allocation_mode_share": {"ACTIVE": 1.0},
                    "ending_equity": 10850.0,
                    "net_pnl_usd": 850.0,
                    "challenge_pass_rate": 1.0,
                    "challenge_reasons": [],
                    "decision_count": 576,
                    "first_decision_sha": "a" * 64,
                    "last_decision_sha": "b" * 64,
                    "reconcile_plans_count": 2,
                    "scorecards_recorded": 6,
                }
            ],
        }
    ]
    top = study.select_top_candidate(sample_stage2)
    assert top["label"] == "MPOD-06_PERMISSIVE_OOS_RETROSPECTIVE"

    dummy_window = {
        "window_name": "full_may_nov",
        "as_of_iso": "2026-11-19T00:00:00+00:00",
        "lookback_days": 202,
        "scored_count": 1,
        "eligible_count": 1,
        "excluded_count": 0,
        "eligible_aliases": ["R#aaa00001"],
        "excluded_by_reason": {},
        "failed_by_gate": {},
        "wallets": [
            {
                "alias": "R#aaa00001",
                "eligible": True,
                "failures": [],
                "excluded_reason": None,
                "observed": 100,
                "q10_order_usd": 50.0,
                "markout_5m_bps": 2.5,
                "p95_participation_1m": 0.001,
                "max_participation_1m": 0.005,
                "cdar_share": 0.25,
                "bench_cdar_share": 0.25,
                "wallet_max_dd_usd": 100.0,
                "suggested_k": 1.0,
                "active_days_14d": 10,
                "cluster_id": "upgma_0",
            }
        ],
    }
    results = {
        "schema": "may_nov_disjoint_10pods_results_v1",
        "provenance": "RETROSPECTIVE_RESEARCH_ONLY",
        "allow_undated": True,
        "stage1": {
            "strict_pit_check": {
                "as_of_iso": "2026-11-19T00:00:00+00:00",
                "lookback_days": 202,
                "cards_count": 0,
                "basis": {"pit": "STRICT_KNOWN_AT"},
            },
            "windows": {
                "train_may_aug": dummy_window,
                "prenov_may_oct": dummy_window,
                "nov_dump": dummy_window,
                "full_may_nov": dummy_window,
            },
        },
        "multi_firm_catalog": [
            {
                "firm_slug": "crypto-fund-trader",
                "program_name": "2-Phase Evaluation",
                "eval_phases": [
                    {
                        "profit_target_usd": 800.0,
                        "max_static_dd_usd": 1000.0,
                        "daily_loss_limit_usd": 500.0,
                        "daily_loss_basis": "balance",
                    }
                ],
                "payout_split_pct": 0.8,
                "challenge_fee_usd": None,
                "refundable_on_pass": False,
                "ev_status": "BREAK_EVEN_ONLY_FEE_UNVERIFIED",
            }
        ],
        "stage2_pods": sample_stage2,
        "top_candidate": top,
    }
    report_md = study.render_markdown_report(results)
    proposal = study.build_proposal(results, rulebook=rulebook)

    assert proposal["approval_required"] is True
    assert proposal["trading_authorized"] is False
    assert proposal["roster_mutated"] is False
    assert proposal["provenance"] == "RETROSPECTIVE_RESEARCH_ONLY"
    assert "allow_undated=True" in report_md
    assert "### Three-line caveat footer" in report_md

    written = study.write_sanitized_outputs(
        tmp_path / "out",
        results=results,
        report_md=report_md,
        proposal=proposal,
    )
    assert written["report_md"].is_file()
    assert written["results_json"].is_file()
    assert written["proposal_json"].is_file()

    leaked_path = tmp_path / "leaked.md"
    leaked_path.write_text(f"wallet={WALLET_A}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="HEX_ADDRESS_LEAK"):
        study._assert_no_hex_leaks(leaked_path)


def test_study_never_touches_live_roster_or_retired_names() -> None:
    text = STUDY_PATH.read_text(encoding="utf-8")
    for forbidden in (
        ".local/share/copytrade/roster.sqlite",
        "wallet_registry.sqlite",
        "prop_firms_rules.json",
        "fast_replay",
        "discover_fill_archive",
        "portfolio_risk_policy",
    ):
        assert forbidden not in text


def test_run_stage2_pods_process_pool_spawn_parity(tmp_path: Path) -> None:
    study = _load_study_module()
    bars_7d = 7 * 288
    fills_a = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=bars_7d, base_price=20000.0, size=0.1
    )
    fills_b = wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=bars_7d, base_price=2000.0, size=1.0
    )
    fills_c = wallet_round_trips(
        WALLET_C, "SOL", start_ms=START, bars=bars_7d, base_price=100.0, size=1.0
    )
    all_fills = fills_a + fills_b + fills_c
    bars_rows = bars_1m_from_fills(all_fills)

    src_root = tmp_path / "src_mp"
    src_store = build_store(
        src_root,
        fills=all_fills,
        equity={
            WALLET_A: (Decimal(12000), 1),
            WALLET_B: (Decimal(9000), 1),
            WALLET_C: (Decimal(8000), 1),
        },
        bars_1m_rows=bars_rows,
    )
    src_store.close()

    anchor_day_3 = datetime.fromtimestamp((START + 3 * DAY_MS) / 1000, UTC).date().isoformat()
    pods = [
        {"pod_id": "MPOD-01", "aliases": ["R#aaa00001", "R#bbb00002"]},
        {"pod_id": "MPOD-02", "aliases": ["R#bbb00002", "R#ccc00003"]},
    ]
    alias_to_address = {alias: addr for addr, alias in ALIAS_MAP.items()}
    rulebook = study.build_cft_rulebook(evidence_sha="d" * 64)

    seq_results = study.run_stage2_pods(
        pods,
        src_store_path=src_root / "facts",
        alias_to_address=alias_to_address,
        address_to_alias=ALIAS_MAP,
        rulebook=rulebook,
        max_workers=1,
        anchor_days=("2026-10-01", anchor_day_3),
    )
    mp_results = study.run_stage2_pods(
        pods,
        src_store_path=src_root / "facts",
        alias_to_address=alias_to_address,
        address_to_alias=ALIAS_MAP,
        rulebook=rulebook,
        max_workers=2,
        anchor_days=("2026-10-01", anchor_day_3),
    )

    assert len(mp_results) == 2
    assert [r["pod_id"] for r in mp_results] == ["MPOD-01", "MPOD-02"]
    for seq_pod, mp_pod in zip(seq_results, mp_results, strict=True):
        assert mp_pod["aliases"] == seq_pod["aliases"]
        assert mp_pod["coins"] == seq_pod["coins"]
        assert len(mp_pod["runs"]) == len(seq_pod["runs"])
        for seq_run, mp_run in zip(seq_pod["runs"], mp_pod["runs"], strict=True):
            assert mp_run == seq_run
    study._assert_sanitized_text("mp_stage2", json.dumps(mp_results))

    direct = study._evaluate_pod_process(
        pods[0],
        src_root / "facts",
        alias_to_address,
        ALIAS_MAP,
        "d" * 64,
        ("2026-10-01", anchor_day_3),
    )
    assert direct["pod_id"] == "MPOD-01"
