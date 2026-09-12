# Data license — `sample_stock_data.csv`

**This file is _not_ covered by the MIT license that applies to the rest of this
repository.** The code is MIT; this data slice is CC BY-SA 4.0. They are
separate works distributed together.

## Attribution

- **Title:** S&P 500 US Stock Market Data
- **Author:** parsalatifi
- **Source:** https://www.kaggle.com/datasets/parsalatifi/s-and-p-500-us-stock-market-data
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Upstream provider:** Yahoo Finance (the dataset's column layout — `Adj Close`,
  `Dividends`, `Stock Splits` — is the standard `yfinance` export)

## Changes made

This file is a **modified** subset of the original dataset, produced by
[`scripts/make_sample.py`](../../scripts/make_sample.py):

- restricted to 12 of the 233 tickers
  (JNJ, KO, CAT, XOM, MSFT, AAPL, NVDA, TSLA, NEE, PLTR, EA, CICC);
- restricted to observations dated 2015-01-01 onward;
- rows sorted by `(Ticker, Date)`.

No values were altered, imputed or recomputed. Column names and units are unchanged.

## Terms

Because the source is licensed under CC BY-SA 4.0, this derived subset is
distributed under the **same** license. Anyone redistributing it — modified or
not — must keep the attribution above and license their version under CC BY-SA 4.0.

The ShareAlike condition applies to the *data*. It does not extend to the source
code in this repository, which processes the data but is not an adaptation of it,
and remains under the MIT license.
