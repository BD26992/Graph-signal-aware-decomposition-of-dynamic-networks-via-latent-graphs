"""Build an S&P 100 finance dataset for DGD.

The dynamic graph is built from adjusted-close return correlations in fixed
trading-day windows.  The graph signals are log-volume changes in the same
windows, giving arrays with shapes

    A: N x N x T
    X: N x Q x T

Data are downloaded from Yahoo's public chart endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


SP100_TICKERS = [
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AMAT",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BK",
    "BKNG",
    "BLK",
    "BMY",
    "BRK-B",
    "C",
    "CAT",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "EMR",
    "FDX",
    "GD",
    "GE",
    "GEV",
    "GILD",
    "GM",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "INTU",
    "ISRG",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "LRCX",
    "MA",
    "MCD",
    "MDLZ",
    "MDT",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "MU",
    "NEE",
    "NFLX",
    "NKE",
    "NOW",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PLTR",
    "PM",
    "QCOM",
    "RTX",
    "SBUX",
    "SCHW",
    "SO",
    "SPG",
    "T",
    "TMO",
    "TMUS",
    "TSLA",
    "TXN",
    "UBER",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WFC",
    "WMT",
    "XOM",
]


def unix_time(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch_yahoo_chart(ticker: str, start: str, end: str, pause: float) -> pd.DataFrame:
    period1 = unix_time(start)
    period2 = unix_time(end)
    symbol = quote(ticker)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    time.sleep(pause)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"no chart result for {ticker}: {error}")
    data = result[0]
    timestamps = data.get("timestamp") or []
    quote_data = data["indicators"]["quote"][0]
    adjclose_data = data["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if not timestamps or adjclose_data is None:
        raise RuntimeError(f"missing timestamps/adjusted close for {ticker}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).date,
            "adj_close": adjclose_data,
            "volume": quote_data.get("volume"),
        }
    )
    frame = frame.dropna(subset=["adj_close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[["adj_close", "volume"]].astype(float)


def windowed_dataset(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
    top_k: int,
    binary_graph: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    returns = np.log(prices).diff()
    volume_changes = np.log(volumes.replace(0.0, np.nan)).diff()
    common_dates = returns.dropna().index.intersection(volume_changes.dropna().index)
    returns = returns.loc[common_dates]
    volume_changes = volume_changes.loc[common_dates]

    adjacency_slices: list[np.ndarray] = []
    signal_slices: list[np.ndarray] = []
    window_starts: list[str] = []
    for start in range(0, len(common_dates) - window_size + 1, step_size):
        end = start + window_size
        return_window = returns.iloc[start:end]
        volume_window = volume_changes.iloc[start:end]
        if return_window.isna().any().any() or volume_window.isna().any().any():
            continue

        corr = return_window.corr().to_numpy(dtype=float)
        graph = np.abs(np.nan_to_num(corr, nan=0.0))
        np.fill_diagonal(graph, 0.0)
        if top_k > 0 and top_k < graph.shape[0] - 1:
            sparse = np.zeros_like(graph)
            for i in range(graph.shape[0]):
                idx = np.argpartition(graph[i], -top_k)[-top_k:]
                sparse[i, idx] = graph[i, idx]
            graph = np.maximum(sparse, sparse.T)
        if binary_graph:
            graph = (graph > 0.0).astype(float)
            np.fill_diagonal(graph, 0.0)
        else:
            max_value = float(np.max(graph))
            if max_value > 0.0:
                graph = graph / max_value

        signals = volume_window.to_numpy(dtype=float).T
        mean = signals.mean(axis=1, keepdims=True)
        std = signals.std(axis=1, keepdims=True)
        signals = (signals - mean) / np.maximum(std, 1e-12)

        adjacency_slices.append(graph)
        signal_slices.append(signals)
        window_starts.append(str(common_dates[start].date()))

    adjacency = np.stack(adjacency_slices, axis=2)
    signals = np.stack(signal_slices, axis=2)
    return adjacency, signals, window_starts


def write_ticker_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--binary-graph", action="store_true")
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/finance/sp100_finance_dgd_2018_2025_Q20_topk10.npz"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/finance/sp100_finance_dgd_2018_2025_Q20_topk10_metadata.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for ticker in SP100_TICKERS:
        try:
            frame = fetch_yahoo_chart(ticker, args.start, args.end, args.pause)
            if frame.empty:
                raise RuntimeError("empty frame")
            frames[ticker] = frame
            rows.append(
                {
                    "ticker": ticker,
                    "status": "ok",
                    "n_days": len(frame),
                    "first_date": str(frame.index.min().date()),
                    "last_date": str(frame.index.max().date()),
                    "message": "",
                }
            )
            print(f"{ticker}: {len(frame)} rows", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep download report complete.
            rows.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "n_days": 0,
                    "first_date": "",
                    "last_date": "",
                    "message": str(exc),
                }
            )
            print(f"{ticker}: failed: {exc}", flush=True)

    if not frames:
        raise RuntimeError("no tickers downloaded")

    prices = pd.concat({ticker: frame["adj_close"] for ticker, frame in frames.items()}, axis=1)
    volumes = pd.concat({ticker: frame["volume"] for ticker, frame in frames.items()}, axis=1)
    complete_tickers = [
        ticker
        for ticker in frames
        if prices[ticker].notna().all() and volumes[ticker].notna().all()
    ]
    prices = prices[complete_tickers]
    volumes = volumes[complete_tickers]

    adjacency, signals, window_starts = windowed_dataset(
        prices,
        volumes,
        window_size=args.window_size,
        step_size=args.step_size,
        top_k=args.top_k,
        binary_graph=args.binary_graph,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        adjacency=adjacency,
        signals=signals,
        tickers=np.asarray(complete_tickers, dtype=object),
        window_starts=np.asarray(window_starts, dtype=object),
        start=args.start,
        end=args.end,
        window_size=args.window_size,
        step_size=args.step_size,
        top_k=args.top_k,
        binary_graph=args.binary_graph,
        construction=json.dumps(
            {
                "graph": (
                    "binary top-k absolute adjusted-close return correlation per window"
                    if args.binary_graph
                    else "top-k absolute adjusted-close return correlation per window"
                ),
                "signals": "per-window standardized log-volume changes",
                "source": "Yahoo chart endpoint",
            }
        ),
    )
    write_ticker_csv(args.metadata_output, rows)
    print(f"wrote {args.output}")
    print(f"wrote {args.metadata_output}")
    print(f"A shape: {adjacency.shape}")
    print(f"X shape: {signals.shape}")
    print(f"complete tickers: {len(complete_tickers)} / {len(SP100_TICKERS)}")
    print(f"windows: {len(window_starts)}")


if __name__ == "__main__":
    main()
