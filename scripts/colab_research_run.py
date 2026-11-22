#!/usr/bin/env python3
"""Reusable Google Colab portfolio research runner and fact-store trimmer."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.request
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from copytrade_pm.facts import FactIngestor, FactStore
from copytrade_pm.facts import manifest as mf
from copytrade_pm.facts.schemas import KNOWN_AT_BASIS_UNRECORDED, column_names

HEX_ADDRESS_RE = re.compile(r"(?i)0x[0-9a-f]{4,}")
POLL_STATUS_RE = re.compile(r"^COLAB_JOB_STATUS:(?:(RUNNING):(\d+)|(DONE):(-?\d+):(\d+))$")
SENSITIVE_DOWNLOAD_SUFFIXES = frozenset({".md", ".json", ".tsv", ".csv"})
CODE_BUNDLE_ITEMS = ("pyproject.toml", "uv.lock", "src", "studies", "scripts", "tests")
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", "data", ".git", ".venv"})
EXCLUDED_FILE_PATTERNS = ("*.pyc", "*.sqlite*", ".env*", "*private*")
VENDOR_PACKAGES = (
    ("copytrade-kernel", "copytrade_kernel"),
    ("copytrade-pm", "copytrade_pm"),
)
REMOTE_PIP_PACKAGES = (
    "polars",
    "numpy",
    "duckdb",
    "requests",
    "cryptography",
    "pytest",
    "ruff",
)
DEFAULT_UPLOAD_CHUNK_MB = 16
DEFAULT_UPLOAD_CHUNK_BYTES = DEFAULT_UPLOAD_CHUNK_MB * 1024 * 1024


def _assert_no_hex_leaks(path: Path) -> None:
    """Fail closed if any 0x-prefixed hex address appears in a text artifact."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = HEX_ADDRESS_RE.search(text)
    if match is not None:
        raise RuntimeError(f"HEX_ADDRESS_LEAK:{path}:{match.group(0)[:10]}")


def _assert_sanitized_text(label: str, text: str) -> None:
    match = HEX_ADDRESS_RE.search(text)
    if match is not None:
        raise RuntimeError(f"HEX_ADDRESS_LEAK:{label}:{match.group(0)[:10]}")


def _is_excluded_rel_path(rel_parts: tuple[str, ...]) -> bool:
    for part in rel_parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        if any(fnmatch.fnmatch(part, pattern) for pattern in EXCLUDED_FILE_PATTERNS):
            return True
    return False


def _is_excluded_vendor_path(rel_parts: tuple[str, ...]) -> bool:
    for part in rel_parts:
        if part == "__pycache__" or part.endswith(".pyc"):
            return True
    return False


def _find_site_packages(worktree: Path) -> Path:
    candidates = sorted((worktree / ".venv" / "lib").glob("python*/site-packages"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"SITE_PACKAGES_NOT_FOUND:{worktree / '.venv'}")


def _verify_vendor_revisions(worktree: Path, site_packages: Path) -> dict[str, str]:
    pyproject_path = worktree / "pyproject.toml"
    if not pyproject_path.is_file():
        raise FileNotFoundError(f"PYPROJECT_MISSING:{pyproject_path}")
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    uv_sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})

    revisions: dict[str, str] = {}
    for dist_name, import_name in VENDOR_PACKAGES:
        source_entry = uv_sources.get(dist_name)
        if not isinstance(source_entry, dict) or not source_entry.get("rev"):
            raise RuntimeError(f"MISSING_UV_SOURCE_REV:{dist_name}")
        expected_rev = str(source_entry["rev"])
        revisions[dist_name] = expected_rev

        pkg_dir = site_packages / import_name
        if not pkg_dir.is_dir():
            raise RuntimeError(f"VENDOR_PACKAGE_MISSING:{pkg_dir}")

        dist_infos = sorted(site_packages.glob(f"{import_name}-*.dist-info")) + sorted(
            site_packages.glob(f"{dist_name}-*.dist-info")
        )
        for dist_info in dist_infos:
            direct_url = dist_info / "direct_url.json"
            if not direct_url.is_file():
                continue
            payload = json.loads(direct_url.read_text(encoding="utf-8"))
            commit_id = payload.get("vcs_info", {}).get("commit_id")
            if commit_id != expected_rev:
                raise RuntimeError(
                    f"VENDOR_REV_MISMATCH:{dist_name}:installed={commit_id}:expected={expected_rev}"
                )
    return revisions


def package_code(worktree: Path, out_path: Path) -> dict[str, Any]:
    """Pack repo sources and pinned vendor packages into a clean `.tar.gz`."""
    worktree = worktree.resolve()
    out_path = out_path.resolve()
    site_packages = _find_site_packages(worktree)
    revisions = _verify_vendor_revisions(worktree, site_packages)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with tarfile.open(out_path, "w:gz") as tar:
        for item_name in CODE_BUNDLE_ITEMS:
            item_path = worktree / item_name
            if not item_path.exists():
                continue
            if item_path.is_file():
                if not _is_excluded_rel_path((item_name,)):
                    tar.add(item_path, arcname=item_name, recursive=False)
                    file_count += 1
                continue
            for root, dirs, files in os.walk(item_path):
                root_path = Path(root)
                rel_dir = root_path.relative_to(worktree)
                dirs[:] = sorted(
                    d for d in dirs if not _is_excluded_rel_path((*rel_dir.parts, d))
                )
                for filename in sorted(files):
                    rel_file = rel_dir / filename
                    if _is_excluded_rel_path(rel_file.parts):
                        continue
                    tar.add(root_path / filename, arcname=rel_file.as_posix(), recursive=False)
                    file_count += 1

        for _, import_name in VENDOR_PACKAGES:
            pkg_dir = site_packages / import_name
            for root, dirs, files in os.walk(pkg_dir):
                root_path = Path(root)
                rel_pkg = root_path.relative_to(site_packages)
                dirs[:] = sorted(
                    d for d in dirs if not _is_excluded_vendor_path((*rel_pkg.parts, d))
                )
                for filename in sorted(files):
                    rel_file = rel_pkg / filename
                    if _is_excluded_vendor_path(rel_file.parts):
                        continue
                    arcname = f"_vendor_pkgs/{rel_file.as_posix()}"
                    tar.add(root_path / filename, arcname=arcname, recursive=False)
                    file_count += 1

    return {
        "files": file_count,
        "bytes": out_path.stat().st_size,
        "kernel_rev": revisions["copytrade-kernel"],
        "pm_rev": revisions["copytrade-pm"],
    }


def _load_target_wallets(wallets_json: Path) -> tuple[str, ...]:
    raw = json.loads(wallets_json.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("addresses"), list):
            items = raw["addresses"]
        elif isinstance(raw.get("alias_to_address"), dict):
            items = list(raw["alias_to_address"].values())
        else:
            raise TypeError(f"UNSUPPORTED_WALLETS_JSON_SHAPE:{wallets_json}")
    else:
        raise TypeError(f"UNSUPPORTED_WALLETS_JSON_SHAPE:{wallets_json}")

    wallets = sorted({str(item).strip().lower() for item in items if str(item).strip()})
    if not wallets:
        raise ValueError(f"EMPTY_TARGET_WALLETS:{wallets_json}")
    return tuple(wallets)


def _resolve_batch_path(store: FactStore, batch: mf.DayBatch) -> Path:
    recorded = Path(batch.path)
    if recorded.is_file():
        return recorded
    fallback = store.day_dir(batch.kind, batch.day) / f"{batch.batch_sha16}.parquet"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"DAY_BATCH_MISSING:{batch.kind}:{batch.day}:{batch.path}")


def _record_day_batch(
    dst: FactStore,
    *,
    kind: str,
    day: str,
    frame: pl.DataFrame,
    write: Any,
    rel_prefix: str,
) -> None:
    con = dst.connect()
    frame_cols = frame.select(list(column_names(kind)))
    by_source = (
        frame_cols.with_columns(
            pl.when(pl.col("first_known_at_ms").is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("_obs")
        )
        .group_by(["source_id", "_obs"])
        .agg(
            pl.len().alias("rows"),
            pl.col("first_known_at_ms").min().alias("min_known"),
        )
        .sort(["source_id", "_obs"])
    )
    inputs: list[mf.InputRecord] = []
    source_totals: dict[str, int] = {}
    for row in by_source.iter_rows(named=True):
        source_id = str(row["source_id"])
        rows_count = int(row["rows"])
        tier_tag = "observed" if int(row["_obs"]) == 1 else "retrospective"
        min_known = None if row["min_known"] is None else int(row["min_known"])
        input_sha = hashlib.sha256(
            f"{kind}|{day}|{source_id}|{tier_tag}|{write.file_sha256}|{rows_count}".encode()
        ).hexdigest()
        inputs.append(
            mf.InputRecord(
                kind=kind,
                day=day,
                source_id=source_id,
                rel_path=f"{rel_prefix}/{kind}/{day}/{tier_tag}",
                input_sha256=input_sha,
                rows_read=rows_count,
                rows_kept=rows_count,
                rows_duplicate_within=0,
                rows_duplicate_cross=0,
                rows_conflicting_values=0,
                rows_rejected=0,
                rows_null_key=0,
                first_known_at_ms=min_known,
            )
        )
        source_totals[source_id] = source_totals.get(source_id, 0) + rows_count

    fingerprint = hashlib.sha256(
        json.dumps(
            sorted((item.source_id, item.rel_path, item.input_sha256) for item in inputs),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    day_sources = [
        {
            "source_id": source_id,
            "files_read": 1,
            "rows_read": count,
            "rows_kept": count,
            "rows_duplicate_within": 0,
            "rows_duplicate_cross": 0,
            "rows_conflicting_values": 0,
            "rows_rejected": 0,
            "rows_null_key": 0,
            "rows_quantized_to_zero": 0,
        }
        for source_id, count in sorted(source_totals.items())
    ]
    known_values = [
        int(val) for val in frame_cols["first_known_at_ms"].to_list() if val is not None
    ]
    mf.record_day(
        con,
        batch=mf.DayBatch(
            kind=kind,
            day=day,
            batch_sha16=write.batch_sha16,
            path=str(write.path),
            rows=write.rows,
            file_sha256=write.file_sha256,
            inputs_fingerprint=fingerprint,
        ),
        source_ids=sorted(source_totals),
        bytes_count=write.bytes_count,
        inputs=inputs,
        day_sources=day_sources,
        first_known_min_ms=min(known_values) if known_values else None,
        first_known_max_ms=max(known_values) if known_values else None,
        closed_at=datetime.now(tz=UTC).isoformat(),
    )


def _build_equity_anchors(
    equity_df: pl.DataFrame,
    anchor_days: Sequence[str],
) -> list[dict[str, Any]]:
    if equity_df.height == 0 or not anchor_days:
        return []
    ordered = equity_df.sort(["wallet", "event_ts_ms"])
    existing_by_wallet: dict[str, set[str]] = {}
    rows_by_wallet: dict[str, list[dict[str, Any]]] = {}
    for row in ordered.iter_rows(named=True):
        wallet = str(row["wallet"]).lower()
        row_day = datetime.fromtimestamp(int(row["event_ts_ms"]) / 1000, UTC).date().isoformat()
        existing_by_wallet.setdefault(wallet, set()).add(row_day)
        rows_by_wallet.setdefault(wallet, []).append(row)

    anchors: list[dict[str, Any]] = []
    for wallet, wallet_rows in sorted(rows_by_wallet.items()):
        seen_days = existing_by_wallet[wallet]
        for day_str in anchor_days:
            if day_str in seen_days:
                continue
            anchor_dt = datetime.fromisoformat(f"{day_str}T00:00:00+00:00")
            anchor_ms = int(anchor_dt.timestamp() * 1000)
            prior = [r for r in wallet_rows if int(r["event_ts_ms"]) <= anchor_ms]
            base = prior[-1] if prior else wallet_rows[0]
            anchors.append(
                {
                    "wallet": wallet,
                    "account_scope": base["account_scope"],
                    "abstraction_mode": base["abstraction_mode"],
                    "equity_usd": base["equity_usd"],
                    "withdrawable_usd": base["withdrawable_usd"],
                    "margin_used_usd": base["margin_used_usd"],
                    "observed_at": anchor_dt,
                    "event_ts_ms": anchor_ms,
                    "first_known_at_ms": None,
                    "known_at_basis": KNOWN_AT_BASIS_UNRECORDED,
                    "arrival_seq": 0,
                    "source_id": base["source_id"],
                    "source_rank": int(base["source_rank"]),
                    "batch_sha16": "",
                }
            )
    return anchors


def trim_facts(
    *,
    src_store: Path,
    out_store: Path,
    wallets_json: Path,
    start_day: str,
    end_day: str,
    out_tar: Path | None = None,
    equity_anchor_days: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a slim FactStore filtered to target wallets, window, and traded coins."""
    date.fromisoformat(start_day)
    date.fromisoformat(end_day)
    if start_day > end_day:
        raise ValueError(f"INVALID_DAY_RANGE:{start_day}>{end_day}")
    normalized_anchors = tuple(
        dict.fromkeys(date.fromisoformat(d.strip()).isoformat() for d in equity_anchor_days if d.strip())
    )
    target_wallets = _load_target_wallets(wallets_json)

    src_store = src_store.resolve()
    out_store = out_store.resolve()
    if out_store.exists():
        shutil.rmtree(out_store)

    fills_rows = 0
    bars_1m_rows = 0
    equity_rows = 0
    traded_coins: set[str] = set()

    with FactStore(src_store, create=False) as src, FactStore(out_store, create=True) as dst:
        src_con = src.connect()
        dst.connect()

        for batch in mf.day_batches(src_con, "fills", start_day=start_day, end_day=end_day):
            batch_path = _resolve_batch_path(src, batch)
            day_df = (
                pl.scan_parquet(batch_path)
                .filter(pl.col("wallet").is_in(list(target_wallets)))
                .collect()
            )
            if day_df.height == 0:
                continue
            coins_in_day = (
                day_df.select(pl.col("coin").str.strip_chars().str.to_uppercase().unique())
                .to_series()
                .to_list()
            )
            traded_coins.update(str(c) for c in coins_in_day if c is not None and str(c))
            prepared = dst.prepare_day("fills", batch.day, day_df)
            write = dst.write_day(prepared)
            _record_day_batch(
                dst,
                kind="fills",
                day=batch.day,
                frame=prepared.frame,
                write=write,
                rel_prefix="trimmed",
            )
            fills_rows += write.rows

        equity_frames: list[pl.DataFrame] = []
        for batch in mf.day_batches(src_con, "equity"):
            batch_path = _resolve_batch_path(src, batch)
            part = (
                pl.scan_parquet(batch_path)
                .filter(pl.col("wallet").is_in(list(target_wallets)))
                .collect()
            )
            if part.height > 0:
                equity_frames.append(part)

        if equity_frames:
            combined_eq = pl.concat(equity_frames, how="vertical_relaxed")
            ingestor = FactIngestor(dst)
            ingestor.ingest_frame(
                "equity",
                combined_eq,
                source_id="trimmed-equity",
                source_rank=1,
                first_known_at_ms=None,
            )
            anchor_rows = _build_equity_anchors(combined_eq, normalized_anchors)
            if anchor_rows:
                ingestor.ingest_rows(
                    "equity",
                    anchor_rows,
                    source_id="equity-anchor",
                    source_rank=1,
                    first_known_at_ms=None,
                )
            equity_rows = dst.coverage("equity").rows

        sorted_coins = sorted(traded_coins)
        if sorted_coins:
            for batch in mf.day_batches(src_con, "bars_1m", start_day=start_day, end_day=end_day):
                batch_path = _resolve_batch_path(src, batch)
                day_df = (
                    pl.scan_parquet(batch_path)
                    .filter(pl.col("coin").is_in(sorted_coins))
                    .collect()
                )
                if day_df.height == 0:
                    continue
                prepared = dst.prepare_day("bars_1m", batch.day, day_df)
                write = dst.write_day(prepared)
                _record_day_batch(
                    dst,
                    kind="bars_1m",
                    day=batch.day,
                    frame=prepared.frame,
                    write=write,
                    rel_prefix="trimmed",
                )
                bars_1m_rows += write.rows

    tar_bytes = 0
    if out_tar is not None:
        out_tar = out_tar.resolve()
        out_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_tar, "w:gz") as tar:
            for root, dirs, files in os.walk(out_store):
                root_path = Path(root)
                rel_dir = root_path.relative_to(out_store)
                dirs[:] = sorted(dirs)
                for filename in sorted(files):
                    if filename.endswith(("-wal", "-shm")):
                        continue
                    arcname = filename if str(rel_dir) == "." else (rel_dir / filename).as_posix()
                    tar.add(root_path / filename, arcname=arcname, recursive=False)
        tar_bytes = out_tar.stat().st_size

    summary = {
        "wallets_count": len(target_wallets),
        "coins_count": len(sorted_coins),
        "fills_rows": fills_rows,
        "bars_1m_rows": bars_1m_rows,
        "equity_rows": equity_rows,
        "tar_bytes": tar_bytes,
    }
    _assert_sanitized_text("trim_facts_summary", json.dumps(summary))
    return summary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_part_prefix(remote_path: str) -> str:
    rel = remote_path.removeprefix("/content/").lstrip("/")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", rel).strip("_")
    return sanitized or "upload"


def _prepare_upload_parts(
    local_path: Path,
    remote_path: str,
    tmp_root: Path,
    chunk_bytes: int,
) -> tuple[list[tuple[Path, str]], list[tuple[str, list[str], str, int]]]:
    if chunk_bytes <= 0:
        raise ValueError(f"CHUNK_BYTES_MUST_BE_POSITIVE:{chunk_bytes}")
    total_bytes = local_path.stat().st_size
    if total_bytes <= chunk_bytes:
        return [(local_path, remote_path)], []

    expected_sha256 = _file_sha256(local_path)
    chunks_dir = tmp_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    prefix = _remote_part_prefix(remote_path)

    upload_pairs: list[tuple[Path, str]] = []
    part_remote_paths: list[str] = []
    with local_path.open("rb") as src:
        idx = 0
        remaining = total_bytes
        while remaining > 0:
            part_limit = min(chunk_bytes, remaining)
            part_name = f"{prefix}.part{idx:04d}"
            part_local = chunks_dir / part_name
            part_remote = f"/content/{part_name}"
            with part_local.open("wb") as dst:
                to_write = part_limit
                while to_write > 0:
                    buf = src.read(min(to_write, 1024 * 1024))
                    if not buf:
                        raise RuntimeError(f"UNEXPECTED_EOF:{local_path}")
                    dst.write(buf)
                    to_write -= len(buf)
            upload_pairs.append((part_local, part_remote))
            part_remote_paths.append(part_remote)
            remaining -= part_limit
            idx += 1

    return upload_pairs, [(remote_path, part_remote_paths, expected_sha256, total_bytes)]


def _reassemble_chunks(
    specs: Sequence[tuple[str, Sequence[str], str, int]],
    *,
    root_override: Path | None = None,
) -> None:
    def _resolve(raw_path: str) -> Path:
        if root_override is None:
            return Path(raw_path)
        return root_override / raw_path.lstrip("/")

    for remote_path, part_paths, expected_sha256, expected_bytes in specs:
        target = _resolve(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parts = [_resolve(part) for part in part_paths]
        written = 0
        with target.open("wb") as out:
            for part_path in resolved_parts:
                with part_path.open("rb") as src:
                    while buf := src.read(1024 * 1024):
                        out.write(buf)
                        written += len(buf)
        actual_bytes = target.stat().st_size
        actual_sha256 = _file_sha256(target)
        if written != expected_bytes or actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"CHUNK_REASSEMBLY_MISMATCH:{remote_path}:bytes={actual_bytes}/{expected_bytes}:sha256={actual_sha256}/{expected_sha256}"
            )
        for part_path in resolved_parts:
            part_path.unlink(missing_ok=True)


def _ensure_rclone_installed(bin_path: Path = Path("/usr/local/bin/rclone")) -> str:
    existing = shutil.which("rclone")
    if existing:
        return existing
    if bin_path.is_file():
        return str(bin_path)
    url = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
    with tempfile.TemporaryDirectory(prefix="rclone-install-") as tmp_dir:
        zip_path = Path(tmp_dir) / "rclone.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "rclone-installer/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, zip_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        with zipfile.ZipFile(zip_path) as zf:
            member = next(
                (name for name in zf.namelist() if name.endswith("/rclone")),
                None,
            )
            if member is None:
                raise RuntimeError("RCLONE_BINARY_NOT_FOUND_IN_ARCHIVE")
            bin_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, bin_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    bin_path.chmod(0o755)
    return str(bin_path)


def _resolve_rclone_config(rclone_config: Path | None = None) -> Path:
    cfg_candidate = (
        rclone_config if rclone_config is not None else Path.home() / ".config/rclone/rclone.conf"
    )
    resolved = cfg_candidate.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"RCLONE_CONFIG_MISSING:{resolved}")
    return resolved


def _rclone_join(remote: str, name: str) -> str:
    sep = "" if remote.endswith(":") else "/"
    return f"{remote}{sep}{name.lstrip('/')}"


def venv_cache_filename(
    packages: Sequence[str] = REMOTE_PIP_PACKAGES,
    python_version: str = "3.14",
) -> str:
    """Return deterministic cache archive name for a remote uv/Python environment."""
    payload = json.dumps(
        {
            "format": "uv-content-py314-v1",
            "python_version": python_version,
            "packages": sorted(str(pkg).strip() for pkg in packages),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sha12 = hashlib.sha256(payload).hexdigest()[:12]
    py_tag = python_version.replace(".", "")
    return f"py{py_tag}_{sha12}.tar.gz"


def sync_facts(
    *,
    src_store: Path,
    rclone_remote: str,
    rclone_config: Path | None = None,
    transfers: int = 16,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync Parquet partitions first, then facts.sqlite, to an rclone remote."""
    if transfers < 1:
        raise ValueError(f"TRANSFERS_MUST_BE_POSITIVE:{transfers}")
    src_store = src_store.resolve()
    facts_sqlite = src_store / "facts.sqlite"
    if not facts_sqlite.is_file():
        raise FileNotFoundError(f"FACTS_SQLITE_MISSING:{facts_sqlite}")
    normalized_remote = rclone_remote.strip().rstrip("/")
    if not normalized_remote:
        raise ValueError("RCLONE_REMOTE_REQUIRED")
    resolved_cfg = _resolve_rclone_config(rclone_config)

    commands = [
        [
            "rclone",
            "sync",
            "--config",
            str(resolved_cfg),
            "--transfers",
            str(transfers),
            "--checkers",
            str(transfers),
            "--fast-list",
            "--exclude",
            "*.sqlite*",
            str(src_store),
            normalized_remote,
        ],
        [
            "rclone",
            "copyto",
            "--config",
            str(resolved_cfg),
            "--checksum",
            str(facts_sqlite),
            _rclone_join(normalized_remote, "facts.sqlite"),
        ],
    ]
    if not dry_run:
        for cmd in commands:
            _run_cli(cmd, check=True)

    summary = {
        "dry_run": dry_run,
        "src_store": str(src_store),
        "rclone_remote": normalized_remote,
        "commands": [shlex.join(cmd) for cmd in commands],
    }
    _assert_sanitized_text("sync_facts_summary", json.dumps(summary))
    return summary


def prune_staging(
    *,
    rclone_remote: str,
    rclone_config: Path | None = None,
    keep_latest: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Keep the newest ``keep_latest`` files on staging remote and delete older ones."""
    if keep_latest < 0:
        raise ValueError(f"KEEP_LATEST_MUST_BE_NON_NEGATIVE:{keep_latest}")
    normalized_remote = rclone_remote.strip().rstrip("/")
    if not normalized_remote:
        raise ValueError("RCLONE_REMOTE_REQUIRED")
    resolved_cfg = _resolve_rclone_config(rclone_config)

    list_cmd = [
        "rclone",
        "lsf",
        "--config",
        str(resolved_cfg),
        "--files-only",
        "--separator",
        ";",
        "--format",
        "tp",
        normalized_remote,
    ]
    proc = subprocess.run(
        list_cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        env=_unbuffered_env(),
    )

    entries: list[tuple[float, str, str]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        ts_raw, sep, name_raw = line.partition(";")
        name = name_raw.strip()
        if not sep or not name:
            raise ValueError(f"INVALID_RCLONE_LSF_LINE:{line}")
        ts_clean = ts_raw.strip()
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        entries.append((dt.timestamp(), ts_clean, name))

    entries.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    kept = entries[:keep_latest]
    to_delete = entries[keep_latest:]
    delete_cmds = [
        [
            "rclone",
            "deletefile",
            "--config",
            str(resolved_cfg),
            _rclone_join(normalized_remote, name),
        ]
        for _, _, name in to_delete
    ]
    if not dry_run:
        for cmd in delete_cmds:
            _run_cli(cmd, check=True)

    summary = {
        "dry_run": dry_run,
        "rclone_remote": normalized_remote,
        "keep_latest": keep_latest,
        "total_files": len(entries),
        "kept": [{"name": name, "timestamp": ts_clean} for _, ts_clean, name in kept],
        "deleted": [{"name": name, "timestamp": ts_clean} for _, ts_clean, name in to_delete],
        "commands": [shlex.join(cmd) for cmd in delete_cmds],
    }
    _assert_sanitized_text("prune_staging_summary", json.dumps(summary))
    return summary


def _pull_rclone_files(
    specs: Sequence[tuple[str, str, str, int]],
    *,
    config_path: Path = Path("/content/.rclone.conf"),
    root_override: Path | None = None,
    unlink_config: bool = True,
) -> None:
    def _resolve(raw_path: str) -> Path:
        if root_override is None:
            return Path(raw_path)
        return root_override / raw_path.lstrip("/")

    resolved_config = _resolve(str(config_path))
    try:
        rclone_bin = _ensure_rclone_installed()
        for dest_raw, remote_src, expected_sha256, expected_bytes in specs:
            dest = _resolve(dest_raw)
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    rclone_bin,
                    "copyto",
                    "--config",
                    str(resolved_config),
                    remote_src,
                    str(dest),
                ],
                stdin=subprocess.DEVNULL,
                check=True,
            )
            actual_bytes = dest.stat().st_size
            actual_sha256 = _file_sha256(dest)
            if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"RCLONE_PULL_MISMATCH:{dest_raw}:bytes={actual_bytes}/{expected_bytes}:sha256={actual_sha256}/{expected_sha256}"
                )
    finally:
        if unlink_config:
            resolved_config.unlink(missing_ok=True)


def _build_facts_filter_lines(
    start_day: str | None = None,
    end_day: str | None = None,
    *,
    available_bar_days: Sequence[str] | None = None,
) -> list[str]:
    lines = ["+ /facts.sqlite", "+ /fills/**", "+ /equity/**"]
    if start_day is None and end_day is None:
        lines.append("+ /bars_1m/**")
    elif start_day is not None and end_day is not None:
        cur = date.fromisoformat(start_day)
        end = date.fromisoformat(end_day)
        while cur <= end:
            lines.append(f"+ /bars_1m/day={cur.isoformat()}/**")
            cur += timedelta(days=1)
    else:
        for day_str in available_bar_days or ():
            if start_day is not None and day_str < start_day:
                continue
            if end_day is not None and day_str > end_day:
                continue
            lines.append(f"+ /bars_1m/day={day_str}/**")
    lines.append("- **")
    return lines


def _pull_facts_from_remote(
    facts_remote: str,
    dest_dir: Path,
    *,
    config_path: Path = Path("/content/.rclone.conf"),
    start_day: str | None = None,
    end_day: str | None = None,
) -> None:
    rclone_bin = _ensure_rclone_installed()
    dest_dir.mkdir(parents=True, exist_ok=True)
    available_bar_days: list[str] | None = None
    if (start_day is None) != (end_day is None):
        sep = "" if facts_remote.endswith(":") else "/"
        lsd = subprocess.run(
            [rclone_bin, "lsd", "--config", str(config_path), f"{facts_remote}{sep}bars_1m"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        available_bar_days = []
        if lsd.returncode == 0:
            for line in lsd.stdout.splitlines():
                token = line.strip().split()[-1] if line.strip() else ""
                if token.startswith("day="):
                    available_bar_days.append(token.removeprefix("day="))
    filter_lines = _build_facts_filter_lines(
        start_day, end_day, available_bar_days=available_bar_days
    )
    with tempfile.TemporaryDirectory(prefix="facts-filter-") as tmp_dir:
        filter_file = Path(tmp_dir) / "filters.txt"
        filter_file.write_text("\n".join(filter_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                rclone_bin,
                "copy",
                "--config",
                str(config_path),
                "--transfers",
                "16",
                "--checkers",
                "16",
                "--fast-list",
                "--filter-from",
                str(filter_file),
                facts_remote,
                str(dest_dir),
            ],
            stdin=subprocess.DEVNULL,
            check=True,
        )


def _rewrite_facts_manifest(facts_root: Path) -> None:
    manifest = facts_root / "facts.sqlite"
    if not manifest.is_file():
        return
    with sqlite3.connect(manifest) as con:
        con.execute(
            "INSERT INTO meta(key, value) VALUES('store_root', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(facts_root),),
        )
        rows = con.execute("SELECT batch_sha16, kind, day, path FROM batches").fetchall()
        for batch_sha16, kind, day, old_path in rows:
            day_dir = facts_root / kind / f"day={day}"
            cand_sha = day_dir / f"{batch_sha16}.parquet"
            cand_name = day_dir / Path(str(old_path)).name
            new_path: str | None = None
            if cand_sha.is_file():
                new_path = str(cand_sha)
            elif cand_name.is_file():
                new_path = str(cand_name)
            if new_path is not None:
                con.execute(
                    "UPDATE batches SET path = ? WHERE kind = ? AND day = ? AND batch_sha16 = ?",
                    (new_path, kind, day, batch_sha16),
                )


def _setup_venv_with_cache(
    venv_cache_remote: str,
    cache_filename: str,
    packages: Sequence[str],
    *,
    config_path: Path = Path("/content/.rclone.conf"),
    content_dir: Path = Path("/content"),
) -> bool:
    rclone_bin = _ensure_rclone_installed()
    sep = "" if venv_cache_remote.endswith(":") else "/"
    remote_archive = f"{venv_cache_remote}{sep}{cache_filename}"
    local_archive = content_dir / "venv_cache.tar.gz"
    venv_python = content_dir / "venv" / "bin" / "python"

    hit = False
    try:
        local_archive.unlink(missing_ok=True)
        shutil.rmtree(content_dir / "venv", ignore_errors=True)
        shutil.rmtree(content_dir / "py314", ignore_errors=True)
        subprocess.run(
            [rclone_bin, "copyto", "--config", str(config_path), remote_archive, str(local_archive)],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            ["tar", "-xzf", str(local_archive), "-C", str(content_dir)],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            [str(venv_python), "-c", "import polars, numpy, duckdb"],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        hit = True
        print(f"[colab-remote] venv cache hit: {cache_filename}", flush=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[colab-remote] venv cache miss ({cache_filename}): {exc}", flush=True)
    finally:
        local_archive.unlink(missing_ok=True)

    if hit:
        return True

    shutil.rmtree(content_dir / "venv", ignore_errors=True)
    shutil.rmtree(content_dir / "py314", ignore_errors=True)
    pip_pkgs = " ".join(shlex.quote(pkg) for pkg in packages)
    miss_script = "\n".join(
        [
            "set -euo pipefail",
            'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
            "if ! command -v uv >/dev/null 2>&1; then",
            "  curl -LsSf https://astral.sh/uv/install.sh | sh",
            '  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
            "fi",
            f"export UV_PYTHON_INSTALL_DIR={shlex.quote(str(content_dir / 'py314'))}",
            "uv python install 3.14",
            f"uv venv --python 3.14 {shlex.quote(str(content_dir / 'venv'))}",
            f"uv pip install --python {shlex.quote(str(venv_python))} {pip_pkgs}",
        ]
    )
    subprocess.run(miss_script, shell=True, executable="/bin/bash", check=True)
    try:
        subprocess.run(
            ["tar", "-czf", str(local_archive), "-C", str(content_dir), "py314", "venv"],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            [rclone_bin, "copyto", "--config", str(config_path), str(local_archive), remote_archive],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        print(f"[colab-remote] venv cache uploaded: {cache_filename}", flush=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[colab-remote] WARNING: venv cache upload failed ({cache_filename}): {exc}", flush=True)
    finally:
        local_archive.unlink(missing_ok=True)
    return False


@dataclass(frozen=True, slots=True)
class PreparedUploads:
    upload_pairs: list[tuple[Path, str]]
    reassembly_specs: list[tuple[str, list[str], str, int]]
    rclone_stage_cmds: list[list[str]]
    rclone_pulls: list[tuple[str, str, str, int]]


@dataclass(frozen=True, slots=True)
class PollStatus:
    log_text: str
    state: str
    returncode: int | None
    offset: int


@dataclass(frozen=True, slots=True)
class ColabPlan:
    session: str
    config_path: Path
    worktree: Path
    rel_script_path: str
    facts_tar: Path | None
    inputs_json: Path | None
    extra_uploads: tuple[tuple[Path, str], ...]
    downloads: tuple[tuple[str, Path], ...]
    high_mem: bool
    gpu: str | None
    timeout_sec: int
    script_args: tuple[str, ...]
    chunk_bytes: int = DEFAULT_UPLOAD_CHUNK_BYTES
    rclone_remote: str | None = None
    rclone_config: Path | None = None
    facts_remote: str | None = None
    facts_start_day: str | None = None
    facts_end_day: str | None = None
    venv_cache_remote: str | None = None

    def base_cmd(self) -> list[str]:
        return ["colab", "--auth=adc", "--config", str(self.config_path)]

    def whoami_cmd(self) -> list[str]:
        return [*self.base_cmd(), "whoami"]

    def usage_cmd(self) -> list[str]:
        return [*self.base_cmd(), "usage"]

    def new_cmd(self) -> list[str]:
        cmd = [*self.base_cmd(), "new", "-s", self.session]
        if self.high_mem:
            cmd.append("--high-mem")
        if self.gpu:
            cmd.extend(["--gpu", self.gpu])
        return cmd

    def raw_upload_pairs(self, code_tar: Path) -> list[tuple[Path, str]]:
        pairs: list[tuple[Path, str]] = [
            (code_tar, "/content/code_bundle.tar.gz"),
        ]
        if self.facts_tar is not None:
            pairs.append((self.facts_tar, "/content/facts_slim.tar.gz"))
        if self.inputs_json is not None:
            pairs.append((self.inputs_json, "/content/private_inputs.json"))
        pairs.extend(self.extra_uploads)
        return pairs

    def prepare_uploads(self, code_tar: Path, tmp_root: Path) -> PreparedUploads:
        upload_pairs: list[tuple[Path, str]] = []
        reassembly_specs: list[tuple[str, list[str], str, int]] = []
        rclone_stage_cmds: list[list[str]] = []
        rclone_pulls: list[tuple[str, str, str, int]] = []

        for local_path, remote_path in self.raw_upload_pairs(code_tar):
            total_bytes = local_path.stat().st_size
            if total_bytes > self.chunk_bytes and self.rclone_remote is not None:
                if self.rclone_config is None:
                    raise ValueError("RCLONE_CONFIG_REQUIRED_WITH_REMOTE")
                expected_sha256 = _file_sha256(local_path)
                remote_src = _rclone_join(
                    self.rclone_remote, f"{expected_sha256[:16]}_{local_path.name}"
                )
                rclone_stage_cmds.append(
                    [
                        "rclone",
                        "copyto",
                        "--config",
                        str(self.rclone_config),
                        "--checksum",
                        str(local_path),
                        remote_src,
                    ]
                )
                rclone_pulls.append((remote_path, remote_src, expected_sha256, total_bytes))
                continue

            parts, specs = _prepare_upload_parts(
                local_path, remote_path, tmp_root, self.chunk_bytes
            )
            upload_pairs.extend(parts)
            reassembly_specs.extend(specs)

        if rclone_pulls or self.facts_remote is not None or self.venv_cache_remote is not None:
            if self.rclone_config is None:
                raise ValueError("RCLONE_CONFIG_REQUIRED_WITH_REMOTE")
            upload_pairs.append((self.rclone_config, "/content/.rclone.conf"))

        return PreparedUploads(
            upload_pairs=upload_pairs,
            reassembly_specs=reassembly_specs,
            rclone_stage_cmds=rclone_stage_cmds,
            rclone_pulls=rclone_pulls,
        )

    def upload_cmds(
        self,
        code_tar: Path,
        upload_pairs: Sequence[tuple[Path, str]] | None = None,
    ) -> list[list[str]]:
        pairs = list(upload_pairs) if upload_pairs is not None else self.raw_upload_pairs(code_tar)
        return [
            [*self.base_cmd(), "upload", "-s", self.session, str(local), remote]
            for local, remote in pairs
        ]

    def exec_cmd(self, wrapper_path: Path) -> list[str]:
        return [
            *self.base_cmd(),
            "exec",
            "-s",
            self.session,
            "--timeout",
            "180",
            "-f",
            str(wrapper_path),
        ]

    def warmup_cmd(self) -> list[str]:
        return [*self.base_cmd(), "exec", "-s", self.session, "--timeout", "30"]

    def poll_cmd(self) -> list[str]:
        return [*self.base_cmd(), "exec", "-s", self.session, "--timeout", "60"]

    def download_cmds(self) -> list[list[str]]:
        return [
            [*self.base_cmd(), "download", "-s", self.session, remote, str(local)]
            for remote, local in self.downloads
        ]

    def stop_cmd(self) -> list[str]:
        return [*self.base_cmd(), "stop", "-s", self.session]

    def planned_commands(
        self,
        code_tar: Path,
        wrapper_path: Path,
        upload_pairs: Sequence[tuple[Path, str]] | None = None,
    ) -> list[list[str]]:
        return [
            self.whoami_cmd(),
            self.usage_cmd(),
            self.new_cmd(),
            *self.upload_cmds(code_tar, upload_pairs=upload_pairs),
            self.exec_cmd(wrapper_path),
            *self.download_cmds(),
            self.stop_cmd(),
        ]


def _parse_pair(spec: str, *, flag: str) -> tuple[str, str]:
    left, sep, right = spec.partition(":")
    if not sep or not left.strip() or not right.strip():
        raise ValueError(f"INVALID_MAPPING:{flag}:{spec}")
    return left.strip(), right.strip()


def _render_detached_runner(rel_script_path: str, script_args: Sequence[str]) -> str:
    cmd = [
        "/content/venv/bin/python",
        "-u",
        f"/content/work/{rel_script_path}",
        *list(script_args),
    ]
    return f'''"""Detached Colab study runner generated by colab_research_run.py."""
import os
import subprocess
import traceback
from pathlib import Path

log_path = Path("/content/run.log")
rc_path = Path("/content/run.rc")
rc_tmp = Path("/content/run.rc.tmp")
rc = 1
try:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/content/work/src:/content/work/studies:/content/work/_vendor_pkgs"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = {cmd!r}
    with log_path.open("ab", buffering=0) as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd="/content/work",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
except BaseException:
    with log_path.open("a", encoding="utf-8") as log_fh:
        traceback.print_exc(file=log_fh)
    rc = 1
finally:
    rc_tmp.write_text(f"{{rc}}\\n", encoding="utf-8")
    os.replace(rc_tmp, rc_path)
'''


def _render_remote_wrapper(
    rel_script_path: str,
    script_args: Sequence[str],
    *,
    reassembly_specs: Sequence[tuple[str, Sequence[str], str, int]] = (),
    rclone_pulls: Sequence[tuple[str, str, str, int]] = (),
    facts_remote: str | None = None,
    facts_start_day: str | None = None,
    facts_end_day: str | None = None,
    venv_cache_remote: str | None = None,
) -> str:
    pip_pkgs = " ".join(REMOTE_PIP_PACKAGES)
    cache_filename = venv_cache_filename()
    normalized_specs = [
        (remote_path, list(part_paths), expected_sha256, expected_bytes)
        for remote_path, part_paths, expected_sha256, expected_bytes in reassembly_specs
    ]
    normalized_pulls = [
        (dest, remote_src, expected_sha256, expected_bytes)
        for dest, remote_src, expected_sha256, expected_bytes in rclone_pulls
    ]
    helper_srcs = "\n\n".join(
        inspect.getsource(fn).rstrip()
        for fn in (
            _file_sha256,
            _reassemble_chunks,
            _ensure_rclone_installed,
            _pull_rclone_files,
            _build_facts_filter_lines,
            _pull_facts_from_remote,
            _rewrite_facts_manifest,
            _setup_venv_with_cache,
        )
    )
    detached_runner_src = _render_detached_runner(rel_script_path, script_args)
    pull_call = (
        "_pull_rclone_files(RCLONE_PULLS, unlink_config=False)"
        if (facts_remote is not None or venv_cache_remote is not None)
        else "_pull_rclone_files(RCLONE_PULLS)"
    )

    bash_lines = [
        "set -euo pipefail",
        'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
    ]
    if venv_cache_remote is None:
        bash_lines.extend(
            [
                "if ! command -v uv >/dev/null 2>&1; then",
                "  curl -LsSf https://astral.sh/uv/install.sh | sh",
                '  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
                "fi",
                "uv python install 3.14",
                "rm -rf /content/venv",
                "uv venv --python 3.14 /content/venv",
                f"uv pip install --python /content/venv/bin/python {pip_pkgs}",
            ]
        )
    bash_lines.extend(
        [
            "rm -rf /content/work",
            "mkdir -p /content/work",
            "tar -xzf /content/code_bundle.tar.gz -C /content/work",
        ]
    )
    if facts_remote is None:
        bash_lines.extend(
            [
                "rm -rf /content/facts_slim",
                "mkdir -p /content/facts_slim",
                "tar -xzf /content/facts_slim.tar.gz -C /content/facts_slim",
            ]
        )
    bash_setup = "\n".join(bash_lines)
    return f'''"""Remote Colab bootstrap wrapper generated by colab_research_run.py."""
import hashlib
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

REASSEMBLY_SPECS = {normalized_specs!r}
RCLONE_PULLS = {normalized_pulls!r}
FACTS_REMOTE = {facts_remote!r}
FACTS_START_DAY = {facts_start_day!r}
FACTS_END_DAY = {facts_end_day!r}
VENV_CACHE_REMOTE = {venv_cache_remote!r}
VENV_CACHE_FILENAME = {cache_filename!r}
REMOTE_PIP_PACKAGES = {REMOTE_PIP_PACKAGES!r}

{helper_srcs}

print("[colab-remote] bootstrap starting", flush=True)
if REASSEMBLY_SPECS:
    _reassemble_chunks(REASSEMBLY_SPECS)

stop_hb = threading.Event()
start_ts = time.monotonic()

def _heartbeat() -> None:
    while not stop_hb.wait(15.0):
        elapsed = int(time.monotonic() - start_ts)
        print(f"[colab-hb] elapsed={{elapsed}}s", flush=True)

hb_thread = threading.Thread(target=_heartbeat, daemon=True)
hb_thread.start()

try:
    try:
        if RCLONE_PULLS:
            print("[colab-remote] pulling via rclone...", flush=True)
            {pull_call}

        if FACTS_REMOTE is not None:
            print("[colab-remote] pulling FactStore via rclone...", flush=True)
            shutil.rmtree("/content/facts_slim", ignore_errors=True)
            _pull_facts_from_remote(
                FACTS_REMOTE,
                Path("/content/facts_slim"),
                start_day=FACTS_START_DAY,
                end_day=FACTS_END_DAY,
            )

        if VENV_CACHE_REMOTE is not None:
            print("[colab-remote] setting up venv via cache...", flush=True)
            _setup_venv_with_cache(
                VENV_CACHE_REMOTE,
                VENV_CACHE_FILENAME,
                REMOTE_PIP_PACKAGES,
            )
    finally:
        Path("/content/.rclone.conf").unlink(missing_ok=True)
        Path("/content/venv_cache.tar.gz").unlink(missing_ok=True)

    BASH_SETUP = {bash_setup!r}
    print("[colab-remote] running BASH_SETUP...", flush=True)
    subprocess.run(BASH_SETUP, shell=True, executable="/bin/bash", check=True)

    _rewrite_facts_manifest(Path("/content/facts_slim"))

    DETACHED_RUNNER_SRC = {detached_runner_src!r}
    Path("/content/run_job.py").write_text(DETACHED_RUNNER_SRC, encoding="utf-8")
    Path("/content/run.log").write_bytes(b"")
    Path("/content/run.rc").unlink(missing_ok=True)
    Path("/content/run.rc.tmp").unlink(missing_ok=True)
    Path("/content/run.pid").unlink(missing_ok=True)
    launch_cmd = (
        f"setsid {{shlex.quote(sys.executable)}} -u /content/run_job.py "
        "</dev/null >/dev/null 2>&1 & echo $! > /content/run.pid"
    )
    subprocess.run(["/bin/bash", "-c", launch_cmd], check=True)
    pid_txt = Path("/content/run.pid").read_text(encoding="utf-8").strip()
    print(f"[colab-remote] detached job pid={{pid_txt}}", flush=True)
finally:
    stop_hb.set()

print("__COLAB_EXEC_OK__", flush=True)
'''


def _unbuffered_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _build_poll_snippet(offset: int) -> str:
    return (
        "import os\n"
        "from pathlib import Path\n"
        f"offset = {offset}\n"
        "log_path = Path('/content/run.log')\n"
        "rc_path = Path('/content/run.rc')\n"
        "pid_path = Path('/content/run.pid')\n"
        "def _read_rc():\n"
        "    if not rc_path.is_file():\n"
        "        return None\n"
        "    raw = rc_path.read_text(encoding='utf-8').strip()\n"
        "    return int(raw) if raw else None\n"
        "def _pid_alive():\n"
        "    if not pid_path.is_file():\n"
        "        return False\n"
        "    raw = pid_path.read_text(encoding='utf-8').strip()\n"
        "    if not raw:\n"
        "        return False\n"
        "    pid = int(raw)\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except ProcessLookupError:\n"
        "        return False\n"
        "    except PermissionError:\n"
        "        return True\n"
        "    stat_path = Path(f'/proc/{pid}/stat')\n"
        "    if stat_path.is_file():\n"
        "        stat_txt = stat_path.read_text(encoding='utf-8', errors='replace')\n"
        "        rparen = stat_txt.rfind(')')\n"
        "        if rparen != -1 and stat_txt[rparen + 1:].lstrip().startswith('Z'):\n"
        "            return False\n"
        "    return True\n"
        "rc_val = _read_rc()\n"
        "alive = True if rc_val is not None else _pid_alive()\n"
        "if rc_val is None and not alive:\n"
        "    rc_val = _read_rc()\n"
        "text = ''\n"
        "if log_path.is_file():\n"
        "    with log_path.open('rb') as fh:\n"
        "        fh.seek(offset)\n"
        "        data = fh.read()\n"
        "    offset += len(data)\n"
        "    if data:\n"
        "        text = data.decode('utf-8', errors='replace')\n"
        "if text:\n"
        "    if not text.endswith('\\n'):\n"
        "        text += '\\n'\n"
        "    print(text, end='', flush=True)\n"
        "if rc_val is not None:\n"
        "    print(f'COLAB_JOB_STATUS:DONE:{rc_val}:{offset}', flush=True)\n"
        "elif not alive:\n"
        "    print('[colab-remote] detached worker exited without run.rc', flush=True)\n"
        "    print(f'COLAB_JOB_STATUS:DONE:1:{offset}', flush=True)\n"
        "else:\n"
        "    print(f'COLAB_JOB_STATUS:RUNNING:{offset}', flush=True)\n"
    )


def _parse_poll_output(stdout: str) -> PollStatus | None:
    lines = stdout.splitlines(keepends=True)
    for idx in range(len(lines) - 1, -1, -1):
        match = POLL_STATUS_RE.match(lines[idx].rstrip("\r\n"))
        if match is not None:
            log_text = "".join(lines[:idx])
            if match.group(1) == "RUNNING":
                return PollStatus(
                    log_text=log_text,
                    state="RUNNING",
                    returncode=None,
                    offset=int(match.group(2)),
                )
            return PollStatus(
                log_text=log_text,
                state="DONE",
                returncode=int(match.group(4)),
                offset=int(match.group(5)),
            )
    return None


def _run_cli(cmd: list[str], *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        check=check,
        text=True,
        timeout=timeout,
        env=_unbuffered_env(),
    )


def _run_exec_checked(cmd: list[str], *, timeout: float | None = None) -> None:
    """Stream colab exec output and require the completion sentinel (colab exec can exit 0 on tracebacks)."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_unbuffered_env(),
    )
    saw_sentinel = False
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.strip() == "__COLAB_EXEC_OK__":
                saw_sentinel = True
        rc = proc.wait(timeout=timeout)
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if rc != 0 or not saw_sentinel:
        raise RuntimeError(f"REMOTE_EXEC_FAILED:rc={rc}:sentinel={saw_sentinel}")


def _poll_remote_job(
    plan: ColabPlan,
    *,
    deadline: float | None = None,
    poll_interval_sec: float = 15.0,
    max_retries: int = 3,
) -> int:
    """Incrementally tail /content/run.log and check /content/run.rc over short colab exec polls."""
    offset = 0
    consecutive_failures = 0
    while True:
        if poll_interval_sec > 0:
            sleep_for = poll_interval_sec
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"REMOTE_JOB_TIMED_OUT:offset={offset}")
                sleep_for = min(sleep_for, remaining)
            time.sleep(sleep_for)

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"REMOTE_JOB_TIMED_OUT:offset={offset}")

        per_call_timeout = 90.0
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"REMOTE_JOB_TIMED_OUT:offset={offset}")
            per_call_timeout = min(per_call_timeout, max(1.0, remaining))

        try:
            proc = subprocess.run(
                plan.poll_cmd(),
                input=_build_poll_snippet(offset),
                capture_output=True,
                text=True,
                check=False,
                timeout=per_call_timeout,
                env=_unbuffered_env(),
            )
        except subprocess.TimeoutExpired as exc:
            consecutive_failures += 1
            print(
                f"[colab-poll] transient timeout ({consecutive_failures}/{max_retries + 1}): {exc}",
                flush=True,
            )
            if consecutive_failures > max_retries:
                raise RuntimeError(f"REMOTE_POLL_FAILED:timeout:offset={offset}") from exc
            continue

        status = _parse_poll_output(proc.stdout or "")
        if proc.returncode != 0 or status is None:
            consecutive_failures += 1
            combined = "\n".join(part for part in (proc.stderr or "", proc.stdout or "") if part).strip()
            last_line = combined.splitlines()[-1] if combined else f"rc={proc.returncode}"
            print(
                f"[colab-poll] transient error ({consecutive_failures}/{max_retries + 1}): {last_line}",
                flush=True,
            )
            if consecutive_failures > max_retries:
                raise RuntimeError(
                    f"REMOTE_POLL_FAILED:rc={proc.returncode}:offset={offset}:{last_line}"
                )
            continue

        consecutive_failures = 0
        if status.log_text:
            print(status.log_text, end="", flush=True)
        offset = status.offset
        if status.state == "DONE":
            assert status.returncode is not None
            if status.returncode != 0:
                raise RuntimeError(f"REMOTE_SCRIPT_FAILED:{status.returncode}")
            return status.returncode


def build_colab_plan(
    *,
    worktree: Path,
    rel_script_path: str,
    downloads: Sequence[str],
    facts_tar: Path | None = None,
    facts_remote: str | None = None,
    facts_start_day: str | None = None,
    facts_end_day: str | None = None,
    venv_cache_remote: str | None = None,
    inputs_json: Path | None = None,
    extra_uploads: Sequence[str] = (),
    session: str | None = None,
    high_mem: bool = False,
    gpu: str | None = None,
    timeout_sec: int = 0,
    script_args: Sequence[str] = (),
    chunk_bytes: int = DEFAULT_UPLOAD_CHUNK_BYTES,
    rclone_remote: str | None = None,
    rclone_config: Path | None = None,
) -> ColabPlan:
    if chunk_bytes < 1:
        raise ValueError(f"CHUNK_BYTES_MUST_BE_POSITIVE:{chunk_bytes}")
    worktree = worktree.resolve()
    if not worktree.is_dir():
        raise FileNotFoundError(f"WORKTREE_MISSING:{worktree}")

    rel_path = Path(rel_script_path)
    if rel_path.is_absolute():
        raise ValueError(f"SCRIPT_MUST_BE_RELATIVE:{rel_script_path}")
    script_abs = (worktree / rel_path).resolve()
    if not script_abs.is_file() or not script_abs.is_relative_to(worktree):
        raise FileNotFoundError(f"SCRIPT_MISSING:{script_abs}")

    normalized_facts_remote = (
        facts_remote.strip().rstrip("/")
        if facts_remote is not None and facts_remote.strip()
        else None
    )
    if facts_tar is None and normalized_facts_remote is None:
        raise ValueError("FACTS_SOURCE_REQUIRED: provide --facts-tar or --facts-remote")
    if facts_tar is not None and normalized_facts_remote is not None:
        raise ValueError("FACTS_SOURCE_MUTUALLY_EXCLUSIVE: cannot use both --facts-tar and --facts-remote")

    resolved_facts_tar: Path | None = None
    if facts_tar is not None:
        resolved_facts_tar = facts_tar.resolve()
        if not resolved_facts_tar.is_file():
            raise FileNotFoundError(f"FACTS_TAR_MISSING:{resolved_facts_tar}")

    normalized_start_day = (
        date.fromisoformat(facts_start_day.strip()).isoformat()
        if facts_start_day is not None and facts_start_day.strip()
        else None
    )
    normalized_end_day = (
        date.fromisoformat(facts_end_day.strip()).isoformat()
        if facts_end_day is not None and facts_end_day.strip()
        else None
    )
    if (
        normalized_start_day is not None
        and normalized_end_day is not None
        and normalized_start_day > normalized_end_day
    ):
        raise ValueError(f"INVALID_DAY_RANGE:{normalized_start_day}>{normalized_end_day}")

    normalized_venv_cache_remote = (
        venv_cache_remote.strip().rstrip("/")
        if venv_cache_remote is not None and venv_cache_remote.strip()
        else None
    )

    resolved_inputs: Path | None = None
    if inputs_json is not None:
        resolved_inputs = inputs_json.resolve()
        if not resolved_inputs.is_file():
            raise FileNotFoundError(f"INPUTS_JSON_MISSING:{resolved_inputs}")

    normalized_rclone_remote = (
        rclone_remote.strip().rstrip("/")
        if rclone_remote is not None and rclone_remote.strip()
        else None
    )
    resolved_rclone_config: Path | None = None
    if (
        normalized_rclone_remote is not None
        or normalized_facts_remote is not None
        or normalized_venv_cache_remote is not None
    ):
        resolved_rclone_config = _resolve_rclone_config(rclone_config)

    parsed_uploads: list[tuple[Path, str]] = []
    for spec in extra_uploads:
        local_raw, remote = _parse_pair(spec, flag="--extra-upload")
        local_path = Path(local_raw).resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"EXTRA_UPLOAD_MISSING:{local_path}")
        parsed_uploads.append((local_path, remote))

    parsed_downloads: list[tuple[str, Path]] = []
    for spec in downloads:
        remote, local_raw = _parse_pair(spec, flag="--download")
        parsed_downloads.append((remote, Path(local_raw).resolve()))
    if not parsed_downloads:
        raise ValueError("AT_LEAST_ONE_DOWNLOAD_REQUIRED")

    normalized_args = list(script_args)
    if len(normalized_args) == 1 and " " in normalized_args[0] and normalized_args[0].lstrip().startswith("-"):
        normalized_args = shlex.split(normalized_args[0])

    session_name = (session or f"research-{uuid.uuid4().hex[:8]}").strip()
    if not session_name:
        raise ValueError("EMPTY_SESSION_NAME")

    return ColabPlan(
        session=session_name,
        config_path=Path(f"/tmp/colab-{session_name}.json"),
        worktree=worktree,
        rel_script_path=rel_path.as_posix(),
        facts_tar=resolved_facts_tar,
        inputs_json=resolved_inputs,
        extra_uploads=tuple(parsed_uploads),
        downloads=tuple(parsed_downloads),
        high_mem=high_mem,
        gpu=gpu,
        timeout_sec=timeout_sec,
        script_args=tuple(normalized_args),
        chunk_bytes=chunk_bytes,
        rclone_remote=normalized_rclone_remote,
        rclone_config=resolved_rclone_config,
        facts_remote=normalized_facts_remote,
        facts_start_day=normalized_start_day,
        facts_end_day=normalized_end_day,
        venv_cache_remote=normalized_venv_cache_remote,
    )


def run_colab(plan: ColabPlan, *, dry_run: bool = False) -> dict[str, Any]:
    """Package code and either emit dry-run plan or execute end-to-end on Colab."""
    with tempfile.TemporaryDirectory(prefix="colab-run-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        code_tar = tmp_root / "code_bundle.tar.gz"
        wrapper_path = tmp_root / "remote_wrapper.py"
        bundle_stats = package_code(plan.worktree, code_tar)
        prepared = plan.prepare_uploads(code_tar, tmp_root)
        wrapper_path.write_text(
            _render_remote_wrapper(
                plan.rel_script_path,
                plan.script_args,
                reassembly_specs=prepared.reassembly_specs,
                rclone_pulls=prepared.rclone_pulls,
                facts_remote=plan.facts_remote,
                facts_start_day=plan.facts_start_day,
                facts_end_day=plan.facts_end_day,
                venv_cache_remote=plan.venv_cache_remote,
            ),
            encoding="utf-8",
        )
        planned = [
            *prepared.rclone_stage_cmds,
            *plan.planned_commands(code_tar, wrapper_path, upload_pairs=prepared.upload_pairs),
        ]

        result: dict[str, Any] = {
            "dry_run": dry_run,
            "session": plan.session,
            "config_path": str(plan.config_path),
            "code_bundle": bundle_stats,
            "facts_tar_bytes": plan.facts_tar.stat().st_size if plan.facts_tar is not None else 0,
            "facts_remote": plan.facts_remote,
            "facts_start_day": plan.facts_start_day,
            "facts_end_day": plan.facts_end_day,
            "venv_cache_remote": plan.venv_cache_remote,
            "venv_cache_filename": venv_cache_filename() if plan.venv_cache_remote is not None else None,
            "chunk_bytes": plan.chunk_bytes,
            "upload_count": len(prepared.upload_pairs),
            "chunked_files": [
                {
                    "remote": remote_path,
                    "chunk_count": len(part_remotes),
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                }
                for remote_path, part_remotes, expected_sha256, expected_bytes in prepared.reassembly_specs
            ],
            "rclone_pulls": [
                {
                    "remote": dest,
                    "rclone_src": remote_src,
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                }
                for dest, remote_src, expected_sha256, expected_bytes in prepared.rclone_pulls
            ],
            "commands": [shlex.join(cmd) for cmd in planned],
            "downloads": [
                {"remote": remote, "local": str(local)} for remote, local in plan.downloads
            ],
        }
        if dry_run:
            return result

        for cmd in prepared.rclone_stage_cmds:
            _run_cli(cmd, check=True)
        _run_cli(plan.whoami_cmd(), check=True)
        _run_cli(plan.usage_cmd(), check=True)
        _run_cli(plan.new_cmd(), check=True)
        try:
            for attempt in range(3):
                try:
                    warm = subprocess.run(
                        plan.warmup_cmd(),
                        input='print("__COLAB_KERNEL_READY__", flush=True)\n',
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=90,
                        env=_unbuffered_env(),
                    )
                    if warm.returncode == 0 and "__COLAB_KERNEL_READY__" in (warm.stdout or ""):
                        break
                    warm_err = f"rc={warm.returncode}"
                except subprocess.TimeoutExpired:
                    warm_err = "timeout=90s"
                if attempt == 2:
                    raise RuntimeError(f"COLAB_KERNEL_WARMUP_FAILED:{warm_err}")
            for cmd in plan.upload_cmds(code_tar, upload_pairs=prepared.upload_pairs):
                _run_cli(cmd, check=True)
            deadline = time.monotonic() + float(plan.timeout_sec) if plan.timeout_sec > 0 else None
            exec_timeout = float(plan.timeout_sec) if plan.timeout_sec > 0 else None
            _run_exec_checked(plan.exec_cmd(wrapper_path), timeout=exec_timeout)
            _poll_remote_job(plan, deadline=deadline)
            for remote, local in plan.downloads:
                local.parent.mkdir(parents=True, exist_ok=True)
                _run_cli(
                    [*plan.base_cmd(), "download", "-s", plan.session, remote, str(local)],
                    check=True,
                )
                if local.suffix.lower() in SENSITIVE_DOWNLOAD_SUFFIXES and local.is_file():
                    _assert_no_hex_leaks(local)
        finally:
            _run_cli(plan.stop_cmd(), check=False)
            if plan.config_path.exists():
                plan.config_path.unlink(missing_ok=True)

        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable Colab portfolio research runner")
    sub = parser.add_subparsers(dest="command", required=True)

    pkg = sub.add_parser("package-code", help="Bundle worktree code and pinned vendor packages")
    pkg.add_argument("--worktree", type=Path, required=True)
    pkg.add_argument("--out", type=Path, required=True)

    trim = sub.add_parser("trim-facts", help="Trim FactStore to target wallets, window, and coins")
    trim.add_argument("--src-store", type=Path, required=True)
    trim.add_argument("--out-store", type=Path, required=True)
    trim.add_argument("--wallets-json", type=Path, required=True)
    trim.add_argument("--start-day", required=True)
    trim.add_argument("--end-day", required=True)
    trim.add_argument("--out-tar", type=Path, default=None)
    trim.add_argument("--equity-anchor-days", default="")

    sync_cmd = sub.add_parser("sync-facts", help="Sync FactStore Parquet and sqlite manifest to rclone remote")
    sync_cmd.add_argument("--src-store", type=Path, required=True)
    sync_cmd.add_argument("--rclone-remote", required=True)
    sync_cmd.add_argument("--rclone-config", type=Path, default=None)
    sync_cmd.add_argument("--transfers", type=int, default=16)
    sync_cmd.add_argument("--dry-run", action="store_true", default=False)

    prune_cmd = sub.add_parser("prune-staging", help="Delete older staging archives on an rclone remote")
    prune_cmd.add_argument("--rclone-remote", required=True)
    prune_cmd.add_argument("--rclone-config", type=Path, default=None)
    prune_cmd.add_argument("--keep-latest", type=int, default=1)
    prune_cmd.add_argument("--dry-run", action="store_true", default=False)

    run = sub.add_parser("run", help="Run a research script on a provisioned Colab VM")
    run.add_argument("--worktree", type=Path, required=True)
    run.add_argument("--script", required=True)
    run.add_argument("--facts-tar", type=Path, default=None)
    run.add_argument("--facts-remote", default=None)
    run.add_argument("--facts-start-day", default=None)
    run.add_argument("--facts-end-day", default=None)
    run.add_argument("--venv-cache-remote", default=None)
    run.add_argument("--download", action="append", required=True)
    run.add_argument("--inputs-json", type=Path, default=None)
    run.add_argument("--extra-upload", action="append", default=[])
    run.add_argument("--session", default=None)
    run.add_argument("--high-mem", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--gpu", default=None)
    run.add_argument("--timeout-sec", type=int, default=0)
    run.add_argument("--chunk-mb", type=int, default=DEFAULT_UPLOAD_CHUNK_MB)
    run.add_argument("--rclone-remote", default=None)
    run.add_argument("--rclone-config", type=Path, default=None)
    run.add_argument("--dry-run", action="store_true", default=False)
    run.add_argument("--script-args", nargs=argparse.REMAINDER, default=[])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "package-code":
        summary = package_code(args.worktree, args.out)
        print(json.dumps(summary, sort_keys=True))
        return 0

    if args.command == "trim-facts":
        anchors = [item.strip() for item in args.equity_anchor_days.split(",") if item.strip()]
        summary = trim_facts(
            src_store=args.src_store,
            out_store=args.out_store,
            wallets_json=args.wallets_json,
            start_day=args.start_day,
            end_day=args.end_day,
            out_tar=args.out_tar,
            equity_anchor_days=anchors,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0

    if args.command == "sync-facts":
        if args.transfers < 1:
            parser.error("--transfers must be >= 1")
        summary = sync_facts(
            src_store=args.src_store,
            rclone_remote=args.rclone_remote,
            rclone_config=args.rclone_config,
            transfers=args.transfers,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "prune-staging":
        if args.keep_latest < 0:
            parser.error("--keep-latest must be >= 0")
        summary = prune_staging(
            rclone_remote=args.rclone_remote,
            rclone_config=args.rclone_config,
            keep_latest=args.keep_latest,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        if args.chunk_mb < 1:
            parser.error("--chunk-mb must be >= 1")
        has_facts_tar = args.facts_tar is not None
        has_facts_remote = bool(args.facts_remote and args.facts_remote.strip())
        if has_facts_tar == has_facts_remote:
            parser.error("Specify exactly one of --facts-tar or --facts-remote")
        plan = build_colab_plan(
            worktree=args.worktree,
            rel_script_path=args.script,
            facts_tar=args.facts_tar,
            facts_remote=args.facts_remote,
            facts_start_day=args.facts_start_day,
            facts_end_day=args.facts_end_day,
            venv_cache_remote=args.venv_cache_remote,
            downloads=args.download,
            inputs_json=args.inputs_json,
            extra_uploads=args.extra_upload,
            session=args.session,
            high_mem=args.high_mem,
            gpu=args.gpu,
            timeout_sec=args.timeout_sec,
            script_args=args.script_args,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
            rclone_remote=args.rclone_remote,
            rclone_config=args.rclone_config,
        )
        result = run_colab(plan, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
