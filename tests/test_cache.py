"""Tests for `src.data.cache`.

Everything here writes into pytest's `tmp_path`, never into `data/processed/`,
so running the suite never disturbs a real cache.

The invalidation tests are the point of this file. A cache that serves stale
data raises nothing and fails nothing -- it just makes every downstream result
quietly wrong. So each trigger gets its own test, and each asserts the *reason*
as well as the outcome, because an invalidation that fires for the wrong reason
is a bug that would otherwise pass unnoticed.
"""

import json
import shutil

import pandas as pd
import pytest

from src import config
from src.data import cache as cache_module
from src.data.cache import (
    build_cache,
    cache_paths,
    cache_status,
    get_prices,
)
from src.data.clean import clean_prices
from src.data.loader import COLUMNS, load_prices


@pytest.fixture
def source(tmp_path):
    """A private copy of the sample CSV, so tests may modify it."""
    dst = tmp_path / "sample_stock_data.csv"
    shutil.copy(config.SAMPLE_CSV, dst)
    return dst


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "processed"


def build(source, cache_dir, **kwargs):
    return get_prices(source, cache_dir=cache_dir, verbose=False, **kwargs)


class TestRoundTrip:
    def test_cached_frame_equals_a_fresh_pipeline_run(self, source, cache_dir):
        """The property everything else depends on: the cache is not an
        approximation of the pipeline, it is the pipeline's exact output."""
        expected, _ = clean_prices(load_prices(source))
        cached = build(source, cache_dir)
        pd.testing.assert_frame_equal(cached, expected)

    def test_second_call_returns_the_identical_frame(self, source, cache_dir):
        first = build(source, cache_dir)
        second = build(source, cache_dir)
        pd.testing.assert_frame_equal(first, second)

    def test_dtypes_survive_parquet(self, source, cache_dir):
        df = build(source, cache_dir)
        assert list(df.columns) == COLUMNS
        assert isinstance(df["ticker"].dtype, pd.CategoricalDtype)
        assert df["volume"].dtype == "int64"
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_category_levels_and_their_order_survive(self, source, cache_dir):
        expected, _ = clean_prices(load_prices(source))
        cached = build(source, cache_dir)
        assert list(cached["ticker"].cat.categories) == list(expected["ticker"].cat.categories)

    def test_writes_both_parquet_and_metadata(self, source, cache_dir):
        build(source, cache_dir)
        parquet, meta = cache_paths(source, cache_dir)
        assert parquet.is_file()
        assert meta.is_file()
        stored = json.loads(meta.read_text(encoding="utf-8"))
        assert stored["rows"] > 0
        assert stored["source_name"] == source.name


class TestCacheIsActuallyUsed:
    def test_second_call_reads_the_file_instead_of_rebuilding(self, source, cache_dir):
        """Replace the cached parquet with a sentinel; if the next call returns
        the sentinel, it genuinely read from disk rather than rebuilding."""
        build(source, cache_dir)
        parquet, _ = cache_paths(source, cache_dir)

        sentinel = pd.DataFrame({"date": [pd.Timestamp("2000-01-01")], "ticker": ["ZZZ"]})
        sentinel.to_parquet(parquet, index=False)

        got = build(source, cache_dir)
        assert list(got["ticker"]) == ["ZZZ"]

    def test_rebuild_flag_overwrites_the_sentinel(self, source, cache_dir):
        build(source, cache_dir)
        parquet, _ = cache_paths(source, cache_dir)
        pd.DataFrame({"date": [pd.Timestamp("2000-01-01")], "ticker": ["ZZZ"]}).to_parquet(
            parquet, index=False
        )
        got = build(source, cache_dir, rebuild=True)
        assert "ZZZ" not in set(got["ticker"].astype(str))
        assert list(got.columns) == COLUMNS


class TestInvalidation:
    def test_reports_valid_after_a_build(self, source, cache_dir):
        build(source, cache_dir)
        valid, reason = cache_status(source, cache_dir=cache_dir)
        assert valid and reason == "up to date"

    def test_no_cache_yet(self, source, cache_dir):
        valid, reason = cache_status(source, cache_dir=cache_dir)
        assert not valid and reason == "no cache file"

    def test_source_modification_invalidates(self, source, cache_dir):
        build(source, cache_dir)
        with source.open("a", encoding="utf-8") as fh:
            fh.write("2026-09-01,AAPL,1,1,0,1,1,1,0,100\n")
        valid, reason = cache_status(source, cache_dir=cache_dir)
        assert not valid
        assert reason in {"source file size changed", "source file was modified"}

    def test_changing_start_date_invalidates(self, source, cache_dir):
        build(source, cache_dir)
        valid, reason = cache_status(source, cache_dir=cache_dir, start_date="2020-01-01")
        assert not valid and reason == "start_date changed"

    def test_changing_min_history_invalidates(self, source, cache_dir):
        build(source, cache_dir)
        valid, reason = cache_status(source, cache_dir=cache_dir, min_history_days=10)
        assert not valid and reason == "min_history_days changed"

    def test_changing_zero_volume_flag_invalidates(self, source, cache_dir):
        build(source, cache_dir)
        valid, reason = cache_status(source, cache_dir=cache_dir, drop_zero_volume=False)
        assert not valid and reason == "drop_zero_volume changed"

    def test_editing_the_cleaning_code_invalidates(self, source, cache_dir, monkeypatch):
        """The trigger that matters most: logic changed, data did not."""
        build(source, cache_dir)
        monkeypatch.setattr(cache_module, "_code_fingerprint", lambda: "deadbeefdeadbeef")
        valid, reason = cache_status(source, cache_dir=cache_dir)
        assert not valid and reason == "loader.py or clean.py changed"

    def test_bumping_the_schema_version_invalidates(self, source, cache_dir, monkeypatch):
        build(source, cache_dir)
        monkeypatch.setattr(cache_module, "CACHE_SCHEMA_VERSION", 99)
        valid, reason = cache_status(source, cache_dir=cache_dir)
        assert not valid and reason == "canonical schema version changed"

    def test_missing_sidecar_invalidates(self, source, cache_dir):
        build(source, cache_dir)
        _, meta = cache_paths(source, cache_dir)
        meta.unlink()
        valid, reason = cache_status(source, cache_dir=cache_dir)
        assert not valid and reason == "cache has no metadata sidecar"

    def test_corrupt_sidecar_invalidates(self, source, cache_dir):
        build(source, cache_dir)
        _, meta = cache_paths(source, cache_dir)
        meta.write_text("{not json", encoding="utf-8")
        valid, reason = cache_status(source, cache_dir=cache_dir)
        assert not valid and reason == "metadata sidecar is unreadable"

    def test_invalid_cache_is_rebuilt_correctly(self, source, cache_dir):
        """Invalidation must lead to correct data, not just to a rebuild."""
        build(source, cache_dir)
        filtered = build(source, cache_dir, start_date="2020-01-01")
        assert filtered["date"].min() >= pd.Timestamp("2020-01-01")


class TestCacheIsolation:
    def test_different_sources_use_different_cache_files(self, tmp_path, cache_dir):
        a = tmp_path / "sample_stock_data.csv"
        b = tmp_path / "other_data.csv"
        shutil.copy(config.SAMPLE_CSV, a)
        shutil.copy(config.SAMPLE_CSV, b)
        pa, ma = cache_paths(a, cache_dir)
        pb, mb = cache_paths(b, cache_dir)
        assert pa != pb and ma != mb

    def test_building_one_does_not_invalidate_the_other(self, tmp_path, cache_dir):
        a = tmp_path / "sample_stock_data.csv"
        b = tmp_path / "other_data.csv"
        shutil.copy(config.SAMPLE_CSV, a)
        shutil.copy(config.SAMPLE_CSV, b)
        build(a, cache_dir)
        build(b, cache_dir)
        assert cache_status(a, cache_dir=cache_dir)[0]
        assert cache_status(b, cache_dir=cache_dir)[0]


class TestBuildCacheReport:
    def test_returns_the_cleaning_report(self, source, cache_dir):
        df, report = build_cache(source, cache_dir=cache_dir)
        assert report.rows_out == len(df)
        assert set(report.removed_tickers) == {"EA", "CICC"}
