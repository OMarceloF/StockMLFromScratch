"""Parquet cache for the cleaned price frame, with honest invalidation.

Rebuilding from CSV costs ~2.6 s; reading the cache costs ~0.08 s. Across the
dozens of reloads a working session involves, that is the difference between
trying an idea and not bothering.

The hard part of any cache is not writing it -- it is knowing when it is stale.
A cache that silently serves data built by an older version of `clean.py` will
not raise anything; it will just quietly make every downstream number wrong. So
each cache file carries a metadata sidecar recording the source file, the
cleaning parameters and a fingerprint of the cleaning code, and it is discarded
whenever any of those stop matching.

Usage
-----
    from src.data.cache import get_prices
    df = get_prices()                      # build on first call, reuse after
    df = get_prices(rebuild=True)          # force a rebuild
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.data import clean as clean_module
from src.data import loader as loader_module
from src.data.clean import CleaningReport, clean_prices
from src.data.loader import load_prices

#: Bump when the canonical frame's schema changes (columns, dtypes, ordering).
#: Every existing cache is then rejected as stale regardless of anything else.
CACHE_SCHEMA_VERSION = 1

#: zstd, chosen by measurement against the real dataset:
#:
#:     codec    write    read    size
#:     snappy   0.61s    0.29s   47.6 MB
#:     zstd     0.64s    0.08s   36.3 MB   <-
#:     gzip    36.90s    0.07s   33.7 MB
#:     brotli   5.98s    0.11s   32.4 MB
#:
#: zstd reads 3.6x faster than snappy and writes 57x faster than gzip, for a
#: file only 8% larger than the smallest option.
COMPRESSION = "zstd"


@dataclass(frozen=True)
class CacheMetadata:
    """Everything that must still hold for a cache file to be trustworthy."""

    schema_version: int
    source_name: str
    source_bytes: int
    source_mtime_ns: int
    start_date: str | None
    min_history_days: int
    drop_zero_volume: bool
    code_fingerprint: str
    built_at: str
    rows: int
    tickers: int


def _code_fingerprint() -> str:
    """SHA-256 over the loader and cleaning modules.

    Any edit to how data is read or filtered invalidates the cache. This is
    deliberately conservative: reformatting a comment also triggers a rebuild,
    a false positive that costs 2.6 seconds. The opposite error -- a real
    logic change going unnoticed -- costs an entire experiment.
    """
    digest = hashlib.sha256()
    for module in (loader_module, clean_module):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:16]


def cache_paths(source: Path | str, cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Return the (parquet, metadata) paths for a given source file.

    Derived from the source stem so that the full dataset and the sample never
    overwrite one another's cache.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else config.PROCESSED_DIR
    stem = Path(source).stem
    return cache_dir / f"{stem}.clean.parquet", cache_dir / f"{stem}.clean.meta.json"


def _expected_metadata(
    source: Path,
    *,
    start_date: str | None,
    min_history_days: int,
    drop_zero_volume: bool,
) -> dict:
    stat = source.stat()
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_name": source.name,
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "start_date": start_date,
        "min_history_days": min_history_days,
        "drop_zero_volume": drop_zero_volume,
        "code_fingerprint": _code_fingerprint(),
    }


#: Human-readable explanation for each field that can invalidate a cache.
_REASONS = {
    "schema_version": "canonical schema version changed",
    "source_name": "different source file",
    "source_bytes": "source file size changed",
    "source_mtime_ns": "source file was modified",
    "start_date": "start_date changed",
    "min_history_days": "min_history_days changed",
    "drop_zero_volume": "drop_zero_volume changed",
    "code_fingerprint": "loader.py or clean.py changed",
}


def cache_status(
    source: Path | str = config.RAW_CSV,
    *,
    cache_dir: Path | None = None,
    start_date: str | None = config.START_DATE,
    min_history_days: int = config.MIN_HISTORY_DAYS,
    drop_zero_volume: bool = True,
) -> tuple[bool, str]:
    """Whether the cache can be reused, and why not when it cannot."""
    source = Path(source)
    parquet, meta_path = cache_paths(source, cache_dir)

    if not parquet.is_file():
        return False, "no cache file"
    if not meta_path.is_file():
        return False, "cache has no metadata sidecar"
    if not source.is_file():
        return False, "source file is missing"

    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "metadata sidecar is unreadable"

    expected = _expected_metadata(
        source,
        start_date=start_date,
        min_history_days=min_history_days,
        drop_zero_volume=drop_zero_volume,
    )
    for key, value in expected.items():
        if stored.get(key) != value:
            return False, _REASONS[key]

    return True, "up to date"


def build_cache(
    source: Path | str = config.RAW_CSV,
    *,
    cache_dir: Path | None = None,
    start_date: str | None = config.START_DATE,
    min_history_days: int = config.MIN_HISTORY_DAYS,
    drop_zero_volume: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full CSV -> clean pipeline and write the cache.

    Returns the cleaned frame together with the cleaning report, so a caller
    that triggers a rebuild can show what was removed.
    """
    source = Path(source)
    parquet, meta_path = cache_paths(source, cache_dir)
    parquet.parent.mkdir(parents=True, exist_ok=True)

    df = load_prices(source, start_date=start_date)
    df, report = clean_prices(
        df, min_history_days=min_history_days, drop_zero_volume=drop_zero_volume
    )

    df.to_parquet(parquet, engine="pyarrow", compression=COMPRESSION, index=False)

    metadata = CacheMetadata(
        **_expected_metadata(
            source,
            start_date=start_date,
            min_history_days=min_history_days,
            drop_zero_volume=drop_zero_volume,
        ),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rows=len(df),
        tickers=int(df["ticker"].nunique()),
    )
    meta_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    return df, report


def get_prices(
    source: Path | str = config.RAW_CSV,
    *,
    cache_dir: Path | None = None,
    rebuild: bool = False,
    verbose: bool = True,
    start_date: str | None = config.START_DATE,
    min_history_days: int = config.MIN_HISTORY_DAYS,
    drop_zero_volume: bool = True,
) -> pd.DataFrame:
    """Return the cleaned price frame, using the cache when it is still valid.

    This is the entry point every notebook and experiment should call. It is
    indistinguishable from running the pipeline directly, except ~33x faster
    once warm.

    Parameters
    ----------
    source
        CSV to build from. Defaults to the full raw dataset.
    rebuild
        Rebuild even if the cache is valid.
    verbose
        Print one line when a rebuild happens, naming the reason. Silence here
        would make a slow call look like an unexplained hang, and would hide
        the fact that the cached data just changed underneath you.
    """
    kwargs = dict(
        cache_dir=cache_dir,
        start_date=start_date,
        min_history_days=min_history_days,
        drop_zero_volume=drop_zero_volume,
    )

    if rebuild:
        valid, reason = False, "rebuild requested"
    else:
        valid, reason = cache_status(source, **kwargs)

    if valid:
        parquet, _ = cache_paths(source, cache_dir)
        return pd.read_parquet(parquet, engine="pyarrow")

    if verbose:
        print(f"[cache] rebuilding ({reason}) ...")
    df, report = build_cache(source, **kwargs)
    if verbose:
        print(f"[cache] {report.rows_out:,} rows, {report.tickers_out} tickers "
              f"({report.rows_removed:,} removed)")
    return df
