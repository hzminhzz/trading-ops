"""Publish MCP-retrieved Vike price receipts to private shared read-only cache.

The caller has already used the connected Vike application, which has a
distinct OAuth session unavailable to the VPS CLI. Never write under the
canonical wallet store or Git. Immutable, exclusive-create private files
survive removal of this disposable research worktree.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from market_data_fallback import MCP_LOCAL, MCP_SHARED


def publish(*, source: Path = MCP_LOCAL, destination: Path = MCP_SHARED) -> dict:
    if not source.is_dir():
        raise ValueError("MISSING_RECEIVED_VIKE_MCP_DATA")
    paths = sorted(source.glob("*.json"))
    if not paths:
        raise ValueError("EMPTY_RECEIVED_VIKE_MCP_DATA")
    destination.mkdir(parents=True, mode=0o700, exist_ok=True)
    dest_stat = destination.stat()
    if dest_stat.st_mode & 0o077:
        raise ValueError("PRIVATE_DESTINATION_PERMISSIONS_TOO_BROAD")
    hashes = {}
    rows = {}
    copied = 0
    reused = 0
    for path in paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        obj = json.loads(raw)
        if (
            obj.get("method") != "vike_connected_mcp"
            or obj.get("source") != "1m fills rollup"
            or not obj.get("rows")
            or path.name != (obj.get("symbol", "") + "_" + path.name.split("_", 1)[-1])
        ):
            raise ValueError("INVALID_MCP_PRICE_RECEIPT")
        target = destination / path.name
        # Create exclusively, never overwrite a prior private receipt.
        with tempfile.NamedTemporaryFile(
            dir=destination, prefix=".incoming-", suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(raw)
            temporary.flush()
            os.fsync(temporary.fileno())
            pending = Path(temporary.name)
        try:
            try:
                os.link(pending, target)
                copied += 1
            except FileExistsError:
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError("IMMUTABLE_VIKE_RECEIPT_HASH_CONFLICT:" + path.name)
                reused += 1
        finally:
            pending.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("VIKE_RECEIPT_PERSISTENCE_INTEGRITY_FAILED")
        if target.stat().st_mode & 0o077:
            raise ValueError("VIKE_RECEIPT_PERMISSIONS_TOO_BROAD")
        hashes[path.name] = digest
        rows[path.name] = len(obj["rows"])
    proof = {
        "schema": "private_readonly_vike_ohlcv_recovery_v1",
        "source": "VIKE_CONNECTED_MCP_1M_FILLS_ROLLUP",
        "files": len(paths),
        "copied": copied,
        "already_identical": reused,
        "rows_total": sum(rows.values()),
        "hashes": hashes,
        "rows_by_file": rows,
        "kind": "HISTORICAL_MARKET_SCENARIO_ONLY_NO_FIRST_KNOWN",
        "trading_authorized": False,
    }
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    proof["fingerprint"] = fingerprint
    # The manifest is immutable and non-secret but stays private.
    name = destination / f"receipt-manifest-{fingerprint[:16]}.json"
    payload = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    if not name.exists():
        try:
            fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    return {
        "destination": str(destination),
        "files": len(paths),
        "copied": copied,
        "reused": reused,
        "rows_total": proof["rows_total"],
        "fingerprint": fingerprint,
    }


if __name__ == "__main__":
    print(json.dumps(publish(), sort_keys=True, indent=2))
