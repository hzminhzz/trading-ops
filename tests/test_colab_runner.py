"""Unit tests for ``scripts/colab_research_run.py``."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from copytrade_kernel import KnownAtTs
from copytrade_pm.facts import FactStore
from research_fixtures import (
    DAY_MS,
    START,
    WALLET_A,
    WALLET_B,
    bars_1m_from_fills,
    build_store,
    wallet_round_trips,
)

from trading_ops.research import ScoreConfig, score_universe

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import colab_research_run as runner

PERMISSIVE = ScoreConfig(
    min_baseline_order_usd=Decimal(1),
    max_euler_cdar_share=Decimal(1),
    max_p95_participation_1m=Decimal(1),
    max_max_participation_1m=Decimal(1),
)


def test_package_code_creates_clean_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_tar = tmp_path / "code_bundle.tar.gz"
    rc = runner.main(["package-code", "--worktree", str(ROOT), "--out", str(out_tar)])
    assert rc == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["files"] > 0
    assert summary["bytes"] == out_tar.stat().st_size
    assert len(summary["kernel_rev"]) == 40
    assert len(summary["pm_rev"]) == 40

    with tarfile.open(out_tar, "r:gz") as tar:
        names = tar.getnames()

    assert "pyproject.toml" in names
    assert "src/trading_ops/research/scoring.py" in names
    assert any(n.startswith("_vendor_pkgs/copytrade_kernel/") and n.endswith(".py") for n in names)
    assert any(n.startswith("_vendor_pkgs/copytrade_pm/") and n.endswith(".py") for n in names)

    for forbidden_prefix in (".env", "data/", ".git/", ".venv/"):
        assert not any(n == forbidden_prefix or n.startswith(forbidden_prefix) for n in names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    non_vendor = [n for n in names if not n.startswith("_vendor_pkgs/")]
    assert not any(".sqlite" in n or "private" in n.lower() for n in non_vendor)


def test_package_code_excludes_sensitive_paths_and_checks_vendor_revs(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    (worktree / "src" / "pkg").mkdir(parents=True)
    (worktree / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (worktree / "src" / "pkg" / "__pycache__").mkdir()
    (worktree / "src" / "pkg" / "__pycache__" / "x.cpython-314.pyc").write_bytes(b"pyc")
    (worktree / "src" / "pkg" / "private_notes.txt").write_text("secret", encoding="utf-8")
    (worktree / "studies").mkdir()
    (worktree / "studies" / "study.py").write_text("print(1)\n", encoding="utf-8")
    (worktree / "studies" / "cache.sqlite3").write_bytes(b"sqlite")
    (worktree / "studies" / ".env.prod").write_text("KEY=1\n", encoding="utf-8")
    (worktree / "data").mkdir()
    (worktree / "data" / "dump.parquet").write_bytes(b"pq")
    (worktree / ".git").mkdir()
    (worktree / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (worktree / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (worktree / "pyproject.toml").write_text(
        '[project]\nname = "wt"\nversion = "0.1.0"\n'
        '[tool.uv.sources]\n'
        'copytrade-kernel = { git = "https://example.com/k.git", rev = "aaaa" }\n'
        'copytrade-pm = { git = "https://example.com/p.git", rev = "bbbb" }\n',
        encoding="utf-8",
    )
    sp = worktree / ".venv" / "lib" / "python3.14" / "site-packages"
    for pkg, dist, rev in (
        ("copytrade_kernel", "copytrade_kernel-0.1.0.dist-info", "aaaa"),
        ("copytrade_pm", "copytrade_pm-0.1.0.dist-info", "bbbb"),
    ):
        (sp / pkg).mkdir(parents=True)
        (sp / pkg / "__init__.py").write_text("", encoding="utf-8")
        (sp / pkg / "__pycache__").mkdir()
        (sp / pkg / "__pycache__" / "a.pyc").write_bytes(b"pyc")
        (sp / dist).mkdir(parents=True)
        (sp / dist / "direct_url.json").write_text(
            json.dumps({"vcs_info": {"commit_id": rev}}),
            encoding="utf-8",
        )

    out_tar = tmp_path / "clean.tar.gz"
    stats = runner.package_code(worktree, out_tar)
    assert stats["kernel_rev"] == "aaaa"
    assert stats["pm_rev"] == "bbbb"

    with tarfile.open(out_tar, "r:gz") as tar:
        names = set(tar.getnames())

    assert "pyproject.toml" in names
    assert "src/pkg/__init__.py" in names
    assert "studies/study.py" in names
    assert "_vendor_pkgs/copytrade_kernel/__init__.py" in names
    assert "_vendor_pkgs/copytrade_pm/__init__.py" in names
    for excluded in (
        ".env",
        "studies/.env.prod",
        "data/dump.parquet",
        ".git/HEAD",
        "studies/cache.sqlite3",
        "src/pkg/private_notes.txt",
        "src/pkg/__pycache__/x.cpython-314.pyc",
        "_vendor_pkgs/copytrade_kernel/__pycache__/a.pyc",
    ):
        assert excluded not in names
    assert not any(n.startswith(".venv/") for n in names)

    (sp / "copytrade_kernel-0.1.0.dist-info" / "direct_url.json").write_text(
        json.dumps({"vcs_info": {"commit_id": "deadbeef"}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="VENDOR_REV_MISMATCH:copytrade-kernel"):
        runner.package_code(worktree, tmp_path / "out.tar.gz")


def test_trim_facts_filters_and_preserves_score_universe_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bars_3d = 3 * 288
    fills_a = wallet_round_trips(
        WALLET_A, "BTC", start_ms=START, bars=bars_3d, base_price=20000.0, size=0.1
    )
    fills_b = wallet_round_trips(
        WALLET_B, "ETH", start_ms=START, bars=bars_3d, base_price=2000.0, size=1.0
    )
    all_fills = fills_a + fills_b
    bars_rows = bars_1m_from_fills(all_fills)
    extra_sol_bar = dict(bars_rows[0])
    extra_sol_bar["coin"] = "SOL"
    bars_rows.append(extra_sol_bar)

    src_root = tmp_path / "src_ws"
    src_store = build_store(
        src_root,
        fills=all_fills,
        equity={WALLET_A: (Decimal(12000), 1), WALLET_B: (Decimal(9000), 1)},
        bars_1m_rows=bars_rows,
    )
    src_store.close()

    wallets_json = tmp_path / "wallets.json"
    wallets_json.write_text(json.dumps({"alias_to_address": {"W001": WALLET_A}}), encoding="utf-8")

    out_store_path = tmp_path / "trimmed_store"
    out_tar_path = tmp_path / "trimmed_facts.tar.gz"
    start_day = datetime.fromtimestamp(START / 1000, UTC).date().isoformat()
    anchor_day_2 = datetime.fromtimestamp((START + DAY_MS) / 1000, UTC).date().isoformat()
    end_day = datetime.fromtimestamp((START + 2 * DAY_MS) / 1000, UTC).date().isoformat()

    rc = runner.main(
        [
            "trim-facts",
            "--src-store",
            str(src_root / "facts"),
            "--out-store",
            str(out_store_path),
            "--wallets-json",
            str(wallets_json),
            "--start-day",
            start_day,
            "--end-day",
            end_day,
            "--out-tar",
            str(out_tar_path),
            "--equity-anchor-days",
            f"{start_day},{anchor_day_2},{end_day}",
        ]
    )
    assert rc == 0

    stdout_text = capsys.readouterr().out
    assert not re.search(r"(?i)0x[0-9a-f]{4,}", stdout_text)
    stats = json.loads(stdout_text)
    assert stats["wallets_count"] == 1
    assert stats["coins_count"] == 1
    assert stats["fills_rows"] == len(fills_a)
    assert stats["equity_rows"] == 3
    assert stats["bars_1m_rows"] > 0
    assert stats["tar_bytes"] == out_tar_path.stat().st_size

    with FactStore(out_store_path, create=False) as trimmed, FactStore(
        src_root / "facts", create=False
    ) as untrimmed:
        assert trimmed.verify()["ok"] is True

        trimmed_fills = trimmed.scan("fills", allow_unknown_known_at=True).collect()
        assert trimmed_fills.height == len(fills_a)
        assert trimmed_fills["wallet"].unique().to_list() == [WALLET_A]
        assert trimmed_fills["coin"].unique().to_list() == ["BTC"]

        trimmed_bars = trimmed.scan("bars_1m", allow_unknown_known_at=True).collect()
        assert trimmed_bars["coin"].unique().to_list() == ["BTC"]

        trimmed_eq = trimmed.scan("equity", allow_unknown_known_at=True).collect().sort("event_ts_ms")
        assert trimmed_eq.height == 3
        assert trimmed_eq["first_known_at_ms"].to_list() == [START + 1000, None, None]
        assert trimmed_eq["known_at_basis"].to_list() == [
            "partition_fetch_clock",
            "unrecorded",
            "unrecorded",
        ]
        as_of = KnownAtTs(START + 3 * DAY_MS)
        assert (
            trimmed.scan("equity", as_of_ms=as_of.value, allow_unknown_known_at=False)
            .collect()
            .filter(pl.col("event_ts_ms") > START)
            .height
            == 0
        )
        strict_pit_trimmed = score_universe(
            trimmed, as_of=as_of, lookback_days=3, config=PERMISSIVE, allow_undated=False
        )
        assert strict_pit_trimmed.cards == ()

        expected = score_universe(
            untrimmed,
            as_of=as_of,
            lookback_days=3,
            config=PERMISSIVE,
            wallets=[WALLET_A],
            allow_undated=True,
        )
        actual = score_universe(
            trimmed,
            as_of=as_of,
            lookback_days=3,
            config=PERMISSIVE,
            allow_undated=True,
        )
        assert actual.cards == expected.cards
        assert actual.audit == expected.audit
        assert actual.excluded == expected.excluded
        assert actual.slice_rows == expected.slice_rows

    with tarfile.open(out_tar_path, "r:gz") as tar:
        tar_names = tar.getnames()
    assert "facts.sqlite" in tar_names
    assert any(n.startswith("fills/day=") and n.endswith(".parquet") for n in tar_names)
    assert any(n.startswith("bars_1m/day=") and n.endswith(".parquet") for n in tar_names)


def test_run_dry_run_builds_plan_and_assert_no_hex_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    facts_tar = tmp_path / "facts.tar.gz"
    facts_tar.write_bytes(b"fake-tar-payload")
    inputs_json = tmp_path / "inputs.json"
    inputs_json.write_text('{"aliases": ["W001"]}', encoding="utf-8")
    out_md = tmp_path / "report.md"

    rc = runner.main(
        [
            "run",
            "--worktree",
            str(ROOT),
            "--script",
            "scripts/colab_research_run.py",
            "--facts-tar",
            str(facts_tar),
            "--inputs-json",
            str(inputs_json),
            "--download",
            f"/content/report.md:{out_md}",
            "--session",
            "unit-dry",
            "--high-mem",
            "--dry-run",
            "--script-args",
            "--alpha",
            "0.95",
        ]
    )
    assert rc == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["session"] == "unit-dry"
    assert plan["config_path"] == "/tmp/colab-unit-dry.json"
    assert plan["facts_tar_bytes"] == facts_tar.stat().st_size
    assert plan["code_bundle"]["files"] > 0

    joined_cmds = "\n".join(plan["commands"])
    assert "colab --auth=adc --config /tmp/colab-unit-dry.json whoami" in joined_cmds
    assert "colab --auth=adc --config /tmp/colab-unit-dry.json usage" in joined_cmds
    assert "colab --auth=adc --config /tmp/colab-unit-dry.json new -s unit-dry --high-mem" in joined_cmds
    assert "exec -s unit-dry --timeout 180 -f" in joined_cmds
    assert f"download -s unit-dry /content/report.md {out_md}" in joined_cmds
    assert "stop -s unit-dry" in joined_cmds

    wrapper_code = runner._render_remote_wrapper("studies/example.py", ["--alpha", "0.95"])
    assert 'executable="/bin/bash"' in wrapper_code
    assert "uv python install 3.14" in wrapper_code
    assert "uv venv --python 3.14 /content/venv" in wrapper_code
    assert "[colab-hb] elapsed=" in wrapper_code
    assert "/content/work/src:/content/work/studies:/content/work/_vendor_pkgs" in wrapper_code

    clean_file = tmp_path / "clean.md"
    clean_file.write_text(
        "sha256: d360fd1f8b2a9c4e5f0123456789abcdef0123456789abcdef0123456789abcd\n",
        encoding="utf-8",
    )
    runner._assert_no_hex_leaks(clean_file)

    leaked_file = tmp_path / "leaked.json"
    synthetic_hex = "0x" + "1234abcd9876543210"
    leaked_file.write_text(json.dumps({"wallet": synthetic_hex}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="HEX_ADDRESS_LEAK"):
        runner._assert_no_hex_leaks(leaked_file)


def test_chunked_upload_dry_run_and_remote_reassembly_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    raw_payload = bytes((i * 37 + 13) % 256 for i in range(3500))
    expected_sha256 = hashlib.sha256(raw_payload).hexdigest()
    facts_tar = tmp_path / "facts_slim.tar.gz"
    facts_tar.write_bytes(raw_payload)

    small_inputs = tmp_path / "inputs.json"
    small_inputs.write_text('{"k": 1}', encoding="utf-8")
    out_md = tmp_path / "report.md"

    parts, specs = runner._prepare_upload_parts(
        facts_tar,
        "/content/facts_slim.tar.gz",
        tmp_path / "stage",
        chunk_bytes=1024,
    )
    assert len(parts) == 4
    assert [p[0].stat().st_size for p in parts] == [1024, 1024, 1024, 428]
    assert [p[1] for p in parts] == [
        "/content/facts_slim.tar.gz.part0000",
        "/content/facts_slim.tar.gz.part0001",
        "/content/facts_slim.tar.gz.part0002",
        "/content/facts_slim.tar.gz.part0003",
    ]
    assert specs == [
        (
            "/content/facts_slim.tar.gz",
            [p[1] for p in parts],
            expected_sha256,
            3500,
        )
    ]

    direct_parts, direct_specs = runner._prepare_upload_parts(
        small_inputs,
        "/content/private_inputs.json",
        tmp_path / "stage",
        chunk_bytes=1024,
    )
    assert direct_parts == [(small_inputs, "/content/private_inputs.json")]
    assert direct_specs == []

    plan = runner.build_colab_plan(
        worktree=ROOT,
        rel_script_path="scripts/colab_research_run.py",
        facts_tar=facts_tar,
        inputs_json=small_inputs,
        downloads=[f"/content/report.md:{out_md}"],
        session="chunk-dry",
        chunk_bytes=1024,
    )
    dry_result = runner.run_colab(plan, dry_run=True)
    runner._assert_sanitized_text("dry_result", json.dumps(dry_result))
    by_remote = {item["remote"]: item for item in dry_result["chunked_files"]}
    assert "/content/code_bundle.tar.gz" in by_remote
    assert by_remote["/content/facts_slim.tar.gz"] == {
        "remote": "/content/facts_slim.tar.gz",
        "chunk_count": 4,
        "bytes": 3500,
        "sha256": expected_sha256,
    }
    assert any(
        cmd.endswith("/content/private_inputs.json") for cmd in dry_result["commands"]
    )
    part_upload_cmds = [
        cmd for cmd in dry_result["commands"] if "/content/facts_slim.tar.gz.part" in cmd
    ]
    assert len(part_upload_cmds) == 4
    for idx in range(4):
        assert f"/content/facts_slim.tar.gz.part{idx:04d}" in part_upload_cmds[idx]

    wrapper_code = runner._render_remote_wrapper(
        "scripts/colab_research_run.py",
        [],
        reassembly_specs=specs,
    )
    ast.parse(wrapper_code)
    assert wrapper_code.index("_reassemble_chunks(REASSEMBLY_SPECS)") < wrapper_code.index("BASH_SETUP =")

    vm_root = tmp_path / "vm_root"
    vm_content = vm_root / "content"
    vm_content.mkdir(parents=True)
    for local_part, remote_part in parts:
        shutil.copy2(local_part, vm_root / remote_part.lstrip("/"))

    assert "if REASSEMBLY_SPECS:\n    _reassemble_chunks(REASSEMBLY_SPECS)" in wrapper_code
    preamble = wrapper_code.split("if REASSEMBLY_SPECS:", 1)[0]
    preamble_path = tmp_path / "remote_wrapper_preamble.py"
    preamble_path.write_text(preamble, encoding="utf-8")
    mod_spec = importlib.util.spec_from_file_location("remote_wrapper_preamble", preamble_path)
    assert mod_spec is not None and mod_spec.loader is not None
    preamble_mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(preamble_mod)
    preamble_mod._reassemble_chunks(preamble_mod.REASSEMBLY_SPECS, root_override=vm_root)

    assembled = vm_content / "facts_slim.tar.gz"
    assert assembled.is_file()
    assert assembled.read_bytes() == raw_payload
    assert runner._file_sha256(assembled) == expected_sha256
    for _, remote_part in parts:
        assert not (vm_root / remote_part.lstrip("/")).exists()

    corrupted_root = tmp_path / "vm_corrupt"
    (corrupted_root / "content").mkdir(parents=True)
    for idx, (local_part, remote_part) in enumerate(parts):
        dst = corrupted_root / remote_part.lstrip("/")
        data = bytearray(local_part.read_bytes())
        if idx == 2:
            data[0] ^= 0xFF
        dst.write_bytes(bytes(data))

    with pytest.raises(RuntimeError, match="CHUNK_REASSEMBLY_MISMATCH"):
        runner._reassemble_chunks(specs, root_override=corrupted_root)
    assert not (corrupted_root / "content" / "facts_slim.tar.gz").exists()

    with pytest.raises(SystemExit) as exc_info:
        runner.main(
            [
                "run",
                "--worktree",
                str(ROOT),
                "--script",
                "scripts/colab_research_run.py",
                "--facts-tar",
                str(facts_tar),
                "--download",
                f"/content/report.md:{out_md}",
                "--chunk-mb",
                "0",
                "--dry-run",
            ]
        )
    assert exc_info.value.code == 2
    capsys.readouterr()

    rc = runner.main(
        [
            "run",
            "--worktree",
            str(ROOT),
            "--script",
            "scripts/colab_research_run.py",
            "--facts-tar",
            str(facts_tar),
            "--download",
            f"/content/report.md:{out_md}",
            "--chunk-mb",
            "8",
            "--dry-run",
        ]
    )
    assert rc == 0
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["chunk_bytes"] == 8 * 1024 * 1024
    assert cli_result["chunked_files"] == []
    assert cli_result["rclone_pulls"] == []


def test_rclone_remote_staging_dry_run_and_pull_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    raw_payload = bytes((i * 53 + 7) % 256 for i in range(1_200_000))
    expected_sha256 = hashlib.sha256(raw_payload).hexdigest()
    sha16 = expected_sha256[:16]
    facts_tar = tmp_path / "facts_slim.tar.gz"
    facts_tar.write_bytes(raw_payload)
    out_md = tmp_path / "report.md"
    missing_cfg = tmp_path / "missing_rclone.conf"

    with pytest.raises(FileNotFoundError, match="RCLONE_CONFIG_MISSING"):
        runner.build_colab_plan(
            worktree=ROOT,
            rel_script_path="scripts/colab_research_run.py",
            facts_tar=facts_tar,
            downloads=[f"/content/report.md:{out_md}"],
            rclone_remote="gdrive:quant-data-backup/colab-staging",
            rclone_config=missing_cfg,
        )

    rclone_cfg = tmp_path / "rclone.conf"
    rclone_cfg.write_text("[gdrive]\ntoken = SECRET_RCLONE_TOKEN_DO_NOT_LEAK\n", encoding="utf-8")

    rc = runner.main(
        [
            "run",
            "--worktree",
            str(ROOT),
            "--script",
            "scripts/colab_research_run.py",
            "--facts-tar",
            str(facts_tar),
            "--download",
            f"/content/report.md:{out_md}",
            "--session",
            "rclone-dry",
            "--chunk-mb",
            "1",
            "--rclone-remote",
            "gdrive:quant-data-backup/colab-staging/",
            "--rclone-config",
            str(rclone_cfg),
            "--dry-run",
        ]
    )
    assert rc == 0
    raw_out = capsys.readouterr().out
    assert "SECRET_RCLONE_TOKEN_DO_NOT_LEAK" not in raw_out
    dry_result = json.loads(raw_out)
    runner._assert_sanitized_text("rclone_dry_result", raw_out)

    expected_remote_src = f"gdrive:quant-data-backup/colab-staging/{sha16}_facts_slim.tar.gz"
    assert dry_result["chunked_files"] == []
    pulls_by_remote = {item["remote"]: item for item in dry_result["rclone_pulls"]}
    assert pulls_by_remote["/content/facts_slim.tar.gz"] == {
        "remote": "/content/facts_slim.tar.gz",
        "rclone_src": expected_remote_src,
        "bytes": len(raw_payload),
        "sha256": expected_sha256,
    }

    joined_cmds = "\n".join(dry_result["commands"])
    assert ".part0000" not in joined_cmds
    assert f"rclone copyto --config {rclone_cfg} --checksum {facts_tar} {expected_remote_src}" in joined_cmds
    assert f"upload -s rclone-dry {rclone_cfg} /content/.rclone.conf" in joined_cmds
    assert joined_cmds.index("rclone copyto") < joined_cmds.index("new -s rclone-dry")

    rclone_pulls = [("/content/facts_slim.tar.gz", expected_remote_src, expected_sha256, len(raw_payload))]
    wrapper_code = runner._render_remote_wrapper(
        "scripts/colab_research_run.py",
        [],
        rclone_pulls=rclone_pulls,
    )
    ast.parse(wrapper_code)
    assert "SECRET_RCLONE_TOKEN_DO_NOT_LEAK" not in wrapper_code
    assert "copyto" in wrapper_code
    assert "/content/.rclone.conf" in wrapper_code
    assert "finally:" in wrapper_code
    assert "resolved_config.unlink(missing_ok=True)" in wrapper_code
    assert "_file_sha256" in wrapper_code
    assert wrapper_code.index("_pull_rclone_files(RCLONE_PULLS)") < wrapper_code.index("BASH_SETUP =")

    preamble = wrapper_code.split("if REASSEMBLY_SPECS:", 1)[0]
    preamble_path = tmp_path / "rclone_wrapper_preamble.py"
    preamble_path.write_text(preamble, encoding="utf-8")
    mod_spec = importlib.util.spec_from_file_location("rclone_wrapper_preamble", preamble_path)
    assert mod_spec is not None and mod_spec.loader is not None
    preamble_mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(preamble_mod)

    vm_root = tmp_path / "vm_rclone"
    vm_content = vm_root / "content"
    vm_content.mkdir(parents=True)
    vm_conf = vm_content / ".rclone.conf"
    vm_conf.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rclone" if name == "rclone" else None)

    def _fake_rclone_ok(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd[0] == "/usr/bin/rclone"
        assert cmd[1:4] == ["copyto", "--config", str(vm_conf)]
        assert cmd[4] == expected_remote_src
        Path(cmd[5]).write_bytes(raw_payload)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_rclone_ok)
    preamble_mod._pull_rclone_files(preamble_mod.RCLONE_PULLS, root_override=vm_root)
    assert (vm_content / "facts_slim.tar.gz").read_bytes() == raw_payload
    assert not vm_conf.exists()

    vm_conf.write_text("dummy", encoding="utf-8")

    def _fake_rclone_corrupt(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[5]).write_bytes(b"corrupted-payload")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_rclone_corrupt)
    with pytest.raises(RuntimeError, match="RCLONE_PULL_MISMATCH"):
        preamble_mod._pull_rclone_files(preamble_mod.RCLONE_PULLS, root_override=vm_root)
    assert not (vm_content / "facts_slim.tar.gz").exists()
    assert not vm_conf.exists()


def test_detached_wrapper_and_incremental_poll_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrapper_code = runner._render_remote_wrapper("studies/example.py", ["--max-workers", "4"])
    ast.parse(wrapper_code)
    assert "/content/run_job.py" in wrapper_code
    assert "/content/run.log" in wrapper_code
    assert "/content/run.rc" in wrapper_code
    assert "/content/run.pid" in wrapper_code
    assert "setsid" in wrapper_code
    assert "__COLAB_EXEC_OK__" in wrapper_code

    parsed_running = runner._parse_poll_output("line1\nline2\nCOLAB_JOB_STATUS:RUNNING:12\n")
    assert parsed_running == runner.PollStatus(
        log_text="line1\nline2\n",
        state="RUNNING",
        returncode=None,
        offset=12,
    )
    parsed_done = runner._parse_poll_output("tail\nCOLAB_JOB_STATUS:DONE:0:17\n")
    assert parsed_done == runner.PollStatus(
        log_text="tail\n",
        state="DONE",
        returncode=0,
        offset=17,
    )
    parsed_fail = runner._parse_poll_output("COLAB_JOB_STATUS:DONE:2:17\n")
    assert parsed_fail == runner.PollStatus(
        log_text="",
        state="DONE",
        returncode=2,
        offset=17,
    )
    assert runner._parse_poll_output("Connection was lost.\n") is None

    vm_content = tmp_path / "vm"
    vm_content.mkdir()
    (vm_content / "run.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (vm_content / "run.log").write_bytes(b"first-chunk\n")

    real_subprocess_run = subprocess.run

    def _exec_poll_snippet(offset: int) -> str:
        snippet = runner._build_poll_snippet(offset).replace("/content/", f"{vm_content}/")
        proc = real_subprocess_run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout

    out1 = _exec_poll_snippet(0)
    st1 = runner._parse_poll_output(out1)
    assert st1 is not None
    assert st1.state == "RUNNING"
    assert st1.log_text == "first-chunk\n"
    assert st1.offset == len(b"first-chunk\n")

    with (vm_content / "run.log").open("ab") as fh:
        fh.write(b"second-partial")
    (vm_content / "run.rc").write_text("0\n", encoding="utf-8")
    out2 = _exec_poll_snippet(st1.offset)
    st2 = runner._parse_poll_output(out2)
    assert st2 is not None
    assert st2.state == "DONE"
    assert st2.returncode == 0
    assert st2.log_text == "second-partial\n"
    assert st2.offset == len(b"first-chunk\nsecond-partial")

    dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_proc.wait()
    (vm_content / "run.rc").unlink()
    (vm_content / "run.pid").write_text(f"{dead_proc.pid}\n", encoding="utf-8")
    out_dead = _exec_poll_snippet(st2.offset)
    st_dead = runner._parse_poll_output(out_dead)
    assert st_dead is not None
    assert st_dead.state == "DONE"
    assert st_dead.returncode == 1

    facts_tar = tmp_path / "facts.tar.gz"
    facts_tar.write_bytes(b"tar")
    plan = runner.build_colab_plan(
        worktree=ROOT,
        rel_script_path="scripts/colab_research_run.py",
        facts_tar=facts_tar,
        downloads=[f"/content/out.md:{tmp_path / 'out.md'}"],
        session="poll-unit",
    )

    responses = iter(
        [
            subprocess.CompletedProcess(plan.poll_cmd(), 1, stdout="", stderr="RuntimeError: Connection was lost."),
            subprocess.CompletedProcess(plan.poll_cmd(), 1, stdout="", stderr="RuntimeError: Connection was lost."),
            subprocess.CompletedProcess(plan.poll_cmd(), 0, stdout="pod1\nCOLAB_JOB_STATUS:RUNNING:5\n", stderr=""),
            subprocess.CompletedProcess(plan.poll_cmd(), 1, stdout="", stderr="RuntimeError: Connection was lost."),
            subprocess.CompletedProcess(plan.poll_cmd(), 0, stdout="pod2\nCOLAB_JOB_STATUS:DONE:0:10\n", stderr=""),
        ]
    )
    seen_offsets: list[int] = []

    def _fake_poll_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd == plan.poll_cmd()
        assert kwargs.get("env", {}).get("PYTHONUNBUFFERED") == "1"  # type: ignore[union-attr]
        inp = str(kwargs["input"])
        m = re.search(r"offset = (\d+)", inp)
        assert m is not None
        seen_offsets.append(int(m.group(1)))
        return next(responses)

    monkeypatch.setattr(subprocess, "run", _fake_poll_run)
    capsys.readouterr()
    rc = runner._poll_remote_job(plan, poll_interval_sec=0.0, max_retries=3)
    assert rc == 0
    assert seen_offsets == [0, 0, 0, 5, 5]
    printed = capsys.readouterr().out
    assert "pod1\n" in printed
    assert "pod2\n" in printed

    def _always_drop(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Connection was lost")

    monkeypatch.setattr(subprocess, "run", _always_drop)
    with pytest.raises(RuntimeError, match="REMOTE_POLL_FAILED"):
        runner._poll_remote_job(plan, poll_interval_sec=0.0, max_retries=2)

    def _remote_nonzero(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="boom\nCOLAB_JOB_STATUS:DONE:7:5\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _remote_nonzero)
    with pytest.raises(RuntimeError, match="REMOTE_SCRIPT_FAILED:7"):
        runner._poll_remote_job(plan, poll_interval_sec=0.0, max_retries=1)


def test_sync_facts_and_prune_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    rclone_cfg = tmp_path / "rclone.conf"
    rclone_cfg.write_text("[gdrive]\ntoken = SECRET_TOKEN\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="FACTS_SQLITE_MISSING"):
        runner.sync_facts(
            src_store=store_dir,
            rclone_remote="gdrive:quant-data-backup/facts",
            rclone_config=rclone_cfg,
            dry_run=True,
        )

    (store_dir / "facts.sqlite").write_bytes(b"sqlite")

    with pytest.raises(ValueError, match="TRANSFERS_MUST_BE_POSITIVE"):
        runner.sync_facts(
            src_store=store_dir,
            rclone_remote="gdrive:quant-data-backup/facts",
            rclone_config=rclone_cfg,
            transfers=0,
            dry_run=True,
        )

    def _forbid_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called during sync-facts --dry-run")

    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    rc = runner.main(
        [
            "sync-facts",
            "--src-store",
            str(store_dir),
            "--rclone-remote",
            "gdrive:quant-data-backup/facts/",
            "--rclone-config",
            str(rclone_cfg),
            "--transfers",
            "8",
            "--dry-run",
        ]
    )
    assert rc == 0
    dry_sync = json.loads(capsys.readouterr().out)
    assert dry_sync["dry_run"] is True
    assert dry_sync["rclone_remote"] == "gdrive:quant-data-backup/facts"
    assert len(dry_sync["commands"]) == 2
    assert "rclone sync --config" in dry_sync["commands"][0]
    assert "--transfers 8 --checkers 8 --fast-list --exclude '*.sqlite*'" in dry_sync["commands"][0]
    assert "rclone copyto --config" in dry_sync["commands"][1]
    assert "facts.sqlite gdrive:quant-data-backup/facts/facts.sqlite" in dry_sync["commands"][1]

    executed_cmds: list[list[str]] = []

    def _fake_sync_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        executed_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_sync_run)
    live_sync = runner.sync_facts(
        src_store=store_dir,
        rclone_remote="gdrive:quant-data-backup/facts",
        rclone_config=rclone_cfg,
        transfers=16,
        dry_run=False,
    )
    assert live_sync["dry_run"] is False
    assert len(executed_cmds) == 2
    assert executed_cmds[0][1] == "sync"
    assert executed_cmds[1][1] == "copyto"

    with pytest.raises(ValueError, match="KEEP_LATEST_MUST_BE_NON_NEGATIVE"):
        runner.prune_staging(
            rclone_remote="gdrive:quant-data-backup/colab-staging",
            rclone_config=rclone_cfg,
            keep_latest=-1,
        )

    lsf_output = (
        "2026-11-19 08:00:00;old_bundle.tar.gz\n"
        "2026-11-21 12:30:00;newest_bundle.tar.gz\n"
        "2026-11-20 09:15:00;mid_bundle.tar.gz\n"
    )
    prune_cmds: list[list[str]] = []

    def _fake_prune_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        prune_cmds.append(cmd)
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, 0, stdout=lsf_output, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_prune_run)
    rc = runner.main(
        [
            "prune-staging",
            "--rclone-remote",
            "gdrive:quant-data-backup/colab-staging/",
            "--rclone-config",
            str(rclone_cfg),
            "--keep-latest",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    dry_prune = json.loads(capsys.readouterr().out)
    assert dry_prune["kept"] == [{"name": "newest_bundle.tar.gz", "timestamp": "2026-11-21 12:30:00"}]
    assert [d["name"] for d in dry_prune["deleted"]] == ["mid_bundle.tar.gz", "old_bundle.tar.gz"]
    assert len(prune_cmds) == 1
    assert prune_cmds[0][1] == "lsf"

    prune_cmds.clear()
    live_prune = runner.prune_staging(
        rclone_remote="gdrive:quant-data-backup/colab-staging",
        rclone_config=rclone_cfg,
        keep_latest=1,
        dry_run=False,
    )
    assert [d["name"] for d in live_prune["deleted"]] == ["mid_bundle.tar.gz", "old_bundle.tar.gz"]
    assert [cmd[1] for cmd in prune_cmds] == ["lsf", "deletefile", "deletefile"]
    assert prune_cmds[1][-1] == "gdrive:quant-data-backup/colab-staging/mid_bundle.tar.gz"
    assert prune_cmds[2][-1] == "gdrive:quant-data-backup/colab-staging/old_bundle.tar.gz"


def test_facts_remote_and_venv_cache_plan_and_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    facts_tar = tmp_path / "facts.tar.gz"
    facts_tar.write_bytes(b"tar")
    out_md = tmp_path / "report.md"
    rclone_cfg = tmp_path / "rclone.conf"
    rclone_cfg.write_text("[gdrive]\ntoken = SECRET_RCLONE_TOKEN\n", encoding="utf-8")

    name_a = runner.venv_cache_filename()
    name_b = runner.venv_cache_filename(list(reversed(runner.REMOTE_PIP_PACKAGES)), "3.14")
    name_c = runner.venv_cache_filename([*runner.REMOTE_PIP_PACKAGES, "scipy"], "3.14")
    assert re.fullmatch(r"py314_[0-9a-f]{12}\.tar\.gz", name_a)
    assert name_a == name_b
    assert name_a != name_c

    with pytest.raises(ValueError, match="FACTS_SOURCE_REQUIRED"):
        runner.build_colab_plan(
            worktree=ROOT,
            rel_script_path="scripts/colab_research_run.py",
            downloads=[f"/content/report.md:{out_md}"],
        )
    with pytest.raises(ValueError, match="FACTS_SOURCE_MUTUALLY_EXCLUSIVE"):
        runner.build_colab_plan(
            worktree=ROOT,
            rel_script_path="scripts/colab_research_run.py",
            facts_tar=facts_tar,
            facts_remote="gdrive:quant-data-backup/facts",
            rclone_config=rclone_cfg,
            downloads=[f"/content/report.md:{out_md}"],
        )
    with pytest.raises(ValueError, match="INVALID_DAY_RANGE"):
        runner.build_colab_plan(
            worktree=ROOT,
            rel_script_path="scripts/colab_research_run.py",
            facts_remote="gdrive:quant-data-backup/facts",
            facts_start_day="2026-11-01",
            facts_end_day="2026-05-01",
            rclone_config=rclone_cfg,
            downloads=[f"/content/report.md:{out_md}"],
        )
    with pytest.raises(SystemExit) as exc_neither:
        runner.main(
            [
                "run",
                "--worktree",
                str(ROOT),
                "--script",
                "scripts/colab_research_run.py",
                "--download",
                f"/content/report.md:{out_md}",
                "--dry-run",
            ]
        )
    assert exc_neither.value.code == 2
    with pytest.raises(SystemExit) as exc_both:
        runner.main(
            [
                "run",
                "--worktree",
                str(ROOT),
                "--script",
                "scripts/colab_research_run.py",
                "--facts-tar",
                str(facts_tar),
                "--facts-remote",
                "gdrive:quant-data-backup/facts",
                "--rclone-config",
                str(rclone_cfg),
                "--download",
                f"/content/report.md:{out_md}",
                "--dry-run",
            ]
        )
    assert exc_both.value.code == 2
    capsys.readouterr()

    rc = runner.main(
        [
            "run",
            "--worktree",
            str(ROOT),
            "--script",
            "scripts/colab_research_run.py",
            "--facts-remote",
            "gdrive:quant-data-backup/facts/",
            "--facts-start-day",
            "2026-05-01",
            "--facts-end-day",
            "2026-05-03",
            "--venv-cache-remote",
            "gdrive:quant-data-backup/colab-cache/",
            "--rclone-remote",
            "gdrive:quant-data-backup/colab-staging/",
            "--rclone-config",
            str(rclone_cfg),
            "--chunk-mb",
            "1",
            "--download",
            f"/content/report.md:{out_md}",
            "--session",
            "cache-dry",
            "--dry-run",
        ]
    )
    assert rc == 0
    raw_out = capsys.readouterr().out
    assert "SECRET_RCLONE_TOKEN" not in raw_out
    runner._assert_sanitized_text("cache_dry_out", raw_out)
    plan_out = json.loads(raw_out)
    assert plan_out["facts_tar_bytes"] == 0
    assert plan_out["facts_remote"] == "gdrive:quant-data-backup/facts"
    assert plan_out["facts_start_day"] == "2026-05-01"
    assert plan_out["facts_end_day"] == "2026-05-03"
    assert plan_out["venv_cache_remote"] == "gdrive:quant-data-backup/colab-cache"
    assert plan_out["venv_cache_filename"] == name_a

    conf_uploads = [cmd for cmd in plan_out["commands"] if "/content/.rclone.conf" in cmd]
    assert len(conf_uploads) == 1
    assert not any("/content/facts_slim.tar.gz" in cmd for cmd in plan_out["commands"])

    wrapper_code = runner._render_remote_wrapper(
        "scripts/colab_research_run.py",
        ["--max-workers", "0"],
        rclone_pulls=[("/content/code_bundle.tar.gz", "gdrive:stage/code.tar.gz", "a" * 64, 123)],
        facts_remote="gdrive:quant-data-backup/facts",
        facts_start_day="2026-05-01",
        facts_end_day="2026-05-03",
        venv_cache_remote="gdrive:quant-data-backup/colab-cache",
    )
    ast.parse(wrapper_code)
    assert "SECRET_RCLONE_TOKEN" not in wrapper_code
    assert "_pull_rclone_files(RCLONE_PULLS, unlink_config=False)" in wrapper_code
    assert "_pull_facts_from_remote(" in wrapper_code
    assert "_setup_venv_with_cache(" in wrapper_code
    unlink_idx = wrapper_code.index('Path("/content/.rclone.conf").unlink(missing_ok=True)')
    run_job_idx = wrapper_code.index('Path("/content/run_job.py").write_text')
    setsid_idx = wrapper_code.index("setsid ")
    assert wrapper_code.index("_pull_facts_from_remote(") < unlink_idx
    assert wrapper_code.index("_setup_venv_with_cache(") < unlink_idx
    assert unlink_idx < run_job_idx < setsid_idx

    # Verify batches.path rewrite snippet only updates existing Parquet files
    facts_root = tmp_path / "facts_slim"
    bar_file = facts_root / "bars_1m" / "day=2026-05-01" / "abc123.parquet"
    bar_file.parent.mkdir(parents=True)
    bar_file.write_bytes(b"pq")
    manifest = facts_root / "facts.sqlite"
    with sqlite3.connect(manifest) as con:
        con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        con.execute(
            "CREATE TABLE batches(kind TEXT, day TEXT, batch_sha16 TEXT, path TEXT)"
        )
        con.execute("INSERT INTO meta(key, value) VALUES('store_root', '/old/root')")
        con.execute(
            "INSERT INTO batches(kind, day, batch_sha16, path) VALUES(?, ?, ?, ?)",
            ("bars_1m", "2026-05-01", "abc123", "/old/root/bars_1m/day=2026-05-01/abc123.parquet"),
        )
        con.execute(
            "INSERT INTO batches(kind, day, batch_sha16, path) VALUES(?, ?, ?, ?)",
            ("bars_1m", "2026-04-30", "def456", "/old/root/bars_1m/day=2026-04-30/def456.parquet"),
        )

    assert '_rewrite_facts_manifest(Path("/content/facts_slim"))' in wrapper_code
    runner._rewrite_facts_manifest(facts_root)
    with sqlite3.connect(manifest) as con:
        store_root_val = con.execute("SELECT value FROM meta WHERE key = 'store_root'").fetchone()[0]
        batch_paths = dict(con.execute("SELECT day, path FROM batches").fetchall())
    assert store_root_val == str(facts_root)
    assert batch_paths["2026-05-01"] == str(bar_file)
    assert batch_paths["2026-04-30"] == "/old/root/bars_1m/day=2026-04-30/def456.parquet"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rclone" if name == "rclone" else None)
    captured_filters: list[str] = []

    def _fake_facts_pull(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1] == "lsd":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="          -1 2026-05-01 00:00:00        -1 day=2026-05-01\n          -1 2026-05-02 00:00:00        -1 day=2026-05-02\n", stderr=""
            )
        assert cmd[:4] == ["/usr/bin/rclone", "copy", "--config", str(rclone_cfg)]
        assert "--transfers" in cmd and "16" in cmd
        assert "--fast-list" in cmd
        ff_idx = cmd.index("--filter-from") + 1
        captured_filters.append(Path(cmd[ff_idx]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_facts_pull)
    runner._pull_facts_from_remote(
        "gdrive:quant-data-backup/facts",
        tmp_path / "facts_dst",
        config_path=rclone_cfg,
        start_day="2026-05-01",
        end_day="2026-05-03",
    )
    assert captured_filters[-1].splitlines() == [
        "+ /facts.sqlite",
        "+ /fills/**",
        "+ /equity/**",
        "+ /bars_1m/day=2026-05-01/**",
        "+ /bars_1m/day=2026-05-02/**",
        "+ /bars_1m/day=2026-05-03/**",
        "- **",
    ]
    runner._pull_facts_from_remote(
        "gdrive:quant-data-backup/facts",
        tmp_path / "facts_dst_onesided",
        config_path=rclone_cfg,
        start_day="2026-05-02",
        end_day=None,
    )
    assert captured_filters[-1].splitlines() == [
        "+ /facts.sqlite",
        "+ /fills/**",
        "+ /equity/**",
        "+ /bars_1m/day=2026-05-02/**",
        "- **",
    ]
    runner._pull_facts_from_remote(
        "gdrive:quant-data-backup/facts",
        tmp_path / "facts_dst_all",
        config_path=rclone_cfg,
    )
    assert captured_filters[-1].splitlines() == [
        "+ /facts.sqlite",
        "+ /fills/**",
        "+ /equity/**",
        "+ /bars_1m/**",
        "- **",
    ]

    content_hit = tmp_path / "content_hit"
    content_hit.mkdir()
    hit_cmds: list[object] = []

    def _fake_venv_hit(cmd: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        hit_cmds.append(cmd)
        if isinstance(cmd, list) and cmd[0] == "/usr/bin/rclone" and cmd[1] == "copyto":
            Path(cmd[5]).write_bytes(b"tar")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_venv_hit)
    assert runner._setup_venv_with_cache(
        "gdrive:quant-data-backup/colab-cache",
        name_a,
        runner.REMOTE_PIP_PACKAGES,
        config_path=rclone_cfg,
        content_dir=content_hit,
    )
    assert len(hit_cmds) == 3
    assert not (content_hit / "venv_cache.tar.gz").exists()

    content_miss = tmp_path / "content_miss"
    content_miss.mkdir()
    miss_calls: list[object] = []

    def _fake_venv_miss(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        miss_calls.append(cmd)
        if isinstance(cmd, list) and cmd[0] == "/usr/bin/rclone" and "colab-cache" in cmd[4]:
            raise subprocess.CalledProcessError(1, cmd)
        if isinstance(cmd, list) and cmd[0] == "tar" and cmd[1] == "-czf":
            Path(cmd[2]).write_bytes(b"packed")
            return subprocess.CompletedProcess(cmd, 0)
        if isinstance(cmd, list) and cmd[0] == "/usr/bin/rclone" and cmd[1] == "copyto" and len(cmd) == 6:
            raise subprocess.CalledProcessError(1, cmd)
        if isinstance(cmd, str):
            assert kwargs.get("shell") is True
            assert f"UV_PYTHON_INSTALL_DIR={content_miss / 'py314'}" in cmd
            assert "uv python install 3.14" in cmd
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_venv_miss)
    assert not runner._setup_venv_with_cache(
        "gdrive:quant-data-backup/colab-cache",
        name_a,
        runner.REMOTE_PIP_PACKAGES,
        config_path=rclone_cfg,
        content_dir=content_miss,
    )
    assert not (content_miss / "venv_cache.tar.gz").exists()
    assert any(isinstance(c, str) and "uv pip install" in c for c in miss_calls)
