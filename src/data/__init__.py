"""Raw data loading and cleaning.

Modules
-------
loader
    Read the raw CSV with explicit dtypes, filter by date, sort by (ticker, date).
clean
    Drop unusable rows and short-history tickers, producing an auditable report
    of everything that was removed.
"""
