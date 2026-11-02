"""Download unadjusted five-minute bars using the existing BaoStock research route."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_dir))
    import baostock as bs

    socket.setdefaulttimeout(25)
    snapshot = ROOT / "reports/research/kronos_watchlist_multitask_v1/watchlist_snapshot.json"
    symbols = sorted(set(json.loads(snapshot.read_text())["symbols"]))
    output = ROOT / "reports/research/overnight_baostock_2024_2026"
    output.mkdir(exist_ok=True)
    protocol = {"watchlist_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "symbols": symbols, "provider": "baostock", "version": "0.9.3",
                "frequency": "5", "adjustflag": "3", "start": "2024-01-01", "end": "2026-09-11",
                "session_contract": "48 bars, 09:35 through 11:30 and 13:05 through 15:00",
                "purpose": "research only; current-watchlist hindsight bias remains"}
    contract = output / "protocol.json"
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError("BaoStock cache protocol differs")
    contract.write_text(json.dumps(protocol, indent=2))
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code}")
    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    audit = []
    try:
        for number, symbol in enumerate(symbols, 1):
            code, exchange = symbol.split(".")
            if exchange not in ("SH", "SZ"):
                audit.append({"symbol": symbol, "status": "unsupported_exchange", "rows": 0})
                continue
            folder = output / symbol
            folder.mkdir(exist_ok=True)
            for year in (2024, 2025, 2026):
                start, end = f"{year}-01-01", "2026-09-11" if year == 2026 else f"{year}-12-31"
                path = folder / f"{year}.parquet"
                if path.exists():
                    frame = pd.read_parquet(path)
                else:
                    result = bs.query_history_k_data_plus(f"{exchange.lower()}.{code}", fields,
                        start_date=start, end_date=end, frequency="5", adjustflag="3")
                    if result.error_code != "0":
                        raise RuntimeError(f"BaoStock query failed: {symbol}/{year}, code {result.error_code}")
                    rows = []
                    while result.next():
                        rows.append(result.get_row_data())
                    if result.error_code != "0":
                        raise RuntimeError(f"BaoStock incomplete stream: {symbol}/{year}, code {result.error_code}")
                    # SDK next() returns False on socket timeout without changing error_code.
                    # A consumed full page is therefore not proof of end-of-stream.
                    if len(result.data) == int(result.per_page_count) and result.cur_row_num >= len(result.data):
                        raise RuntimeError(f"BaoStock stream ended without final page: {symbol}/{year}")
                    frame = pd.DataFrame(rows, columns=fields.split(","))
                    if not frame.empty:
                        frame["trade_time"] = pd.to_datetime(frame.time.str[:14], format="%Y%m%d%H%M%S")
                        frame["ts_code"] = symbol
                        for column in ("open", "high", "low", "close", "volume", "amount"):
                            frame[column] = pd.to_numeric(frame[column], errors="raise")
                        frame["vol"] = frame.volume
                        if (not frame.code.eq(f"{exchange.lower()}.{code}").all()
                                or not frame.adjustflag.eq("3").all()
                                or frame.trade_time.dt.normalize().lt(start).any()
                                or frame.trade_time.dt.normalize().gt(end).any()
                                or frame.trade_time.duplicated().any()):
                            raise ValueError("BaoStock response outside requested data contract")
                        frame = frame.sort_values("trade_time")
                    temporary = path.with_suffix(".pending.parquet")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                    time.sleep(.5)
                audit.append({"symbol": symbol, "year": year, "status": "returned" if len(frame) else "empty",
                              "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                pd.DataFrame(audit).to_csv(output / "coverage.csv", index=False)
                print(f"BaoStock {number}/{len(symbols)} {symbol}/{year}: {len(frame)} bars", flush=True)
    finally:
        bs.logout()
    (output / "complete.json").write_text(json.dumps({"requested_stocks": len(symbols),
        "rows": sum(row["rows"] for row in audit), "chunks": len(audit),
        "unsupported": [row["symbol"] for row in audit if row["status"] == "unsupported_exchange"]}, indent=2))


if __name__ == "__main__":
    main()
