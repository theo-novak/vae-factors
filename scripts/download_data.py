"""
Populate a returns panel from live data (optional).

Pulls adjusted-close daily returns via yfinance and the Fama-French 5 factors from
Kenneth French's library, writing returns to DuckDB. Everything is optional — the
package runs fully on the synthetic generator without any of this.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from vae_factors.data.fetchers import download_returns, fama_french_5
from vae_factors.data.store import init_db, upsert_returns

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
    "JNJ", "UNH", "PFE", "LLY", "XOM", "CVX", "COP", "PG", "KO", "PEP",
    "HD", "LOW", "CAT", "GE", "BA", "DIS", "NFLX", "CRM", "INTC", "AMD",
]


def main() -> None:
    load_dotenv()
    db = os.getenv("DB_PATH", "data/vae.duckdb")
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)

    print(f"Downloading returns for {len(UNIVERSE)} tickers…")
    rets = download_returns(UNIVERSE, start="2015-01-01")
    con = init_db(db)
    n = upsert_returns(con, rets)
    con.close()
    print(f"Wrote {n} return rows to {db}.")

    try:
        ff = fama_french_5(start="2015-01-01")
        print(f"Fama-French 5 factors: {ff.shape[0]} daily rows fetched.")
    except Exception as e:  # pragma: no cover
        print(f"(Fama-French fetch skipped: {e})")


if __name__ == "__main__":
    main()
