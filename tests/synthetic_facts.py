"""Build a synthetic fact store from a synthetic Reservoir-shaped archive tree.

Studies no longer glob an archive directory: they read the ``facts.sqlite``
manifest. Tests that used to lay down ``reservoir-…/hyperliquid/fills/day=…``
files therefore also need a store. This helper runs the real migration engine
over exactly those files, so the fixture exercises the same ingest, dedupe and
ordering path production uses — no test-only store implementation.
"""

from __future__ import annotations

from pathlib import Path

from copytrade_pm.facts.migrate import Migration, new_receipt
from copytrade_pm.facts.sources import build_catalogue
from copytrade_pm.facts.store import FactStore

DEFAULT_ARCHIVE_DIRS = {"full-history": "reservoir-full-history-20261115"}


def build_facts_store(
    archive_root: Path,
    store_root: Path,
    *,
    reservoir: dict[str, Path] | None = None,
) -> Path:
    """Migrate a synthetic archive tree into a fact store and return its root.

    ``reservoir`` maps a reservoir prefix name (``full-history``,
    ``monitored-plus-six``, ``roster-daily``, ``liquid12-discovery``) to its
    directory. The default is the single ``full-history`` tree the tests write.
    """
    if reservoir is None:
        reservoir = {
            name: archive_root / directory for name, directory in DEFAULT_ARCHIVE_DIRS.items()
        }
    sources, absent = build_catalogue(reservoir=reservoir)
    store = FactStore(store_root, create=True)
    try:
        Migration(store=store, sources=sources, kinds=("fills",)).run(
            new_receipt(
                store_root=store_root, sources=sources, not_supplied=absent, kinds=("fills",)
            )
        )
    finally:
        store.close()
    return store_root
