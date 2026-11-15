"""Numerical inner replay, verified against frozen Python cash ledgers."""
import ctypes
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from compare_index_futures_adaptation import POLICIES
from evaluate_index_futures_multistart import VARIANTS


class FastInner:
    def __init__(self, replay):
        self.replay = replay
        self.dates = [g[0].entry_date for g in replay.groups]
        self.ends = [g[0].exit_date for g in replay.groups]
        self.index = {d:i for i,d in enumerate(self.dates)}
        self.products = sorted(replay.panel.symbol.unique())
        self.codes = sorted(replay.panel.ts_code.unique())
        codes = {c:i for i,c in enumerate(self.codes)}
        products = {p:i for i,p in enumerate(self.products)}
        self.data = np.zeros((len(self.dates), 4, 15), dtype=np.float64)
        self.close = np.full((len(self.dates), len(codes)), np.nan)
        self.forecast = np.full((len(self.dates), 21, 4, 2, 5), np.nan)
        self.evidence_names=getattr(replay,"evidence_names",("base","factor","interaction"))
        self.evidence = np.full((len(self.dates),4,2,2,2,len(self.evidence_names),3),np.nan)
        self.soft = np.full((len(self.dates),21,4,2,2,2),np.nan)
        self.calendar = np.array([d in replay.calendar for d in self.dates], dtype=np.int32)
        for i, group in enumerate(replay.groups):
            for r in group:
                self.data[i, products[r.symbol], :14] = [1, r.entry_open, r.exit_open, r.entry_settle,
                    r.multiplier, r.f_volatility_20, r.etf_ret_1, r.long_60, r.short_60,
                    r.long_20, r.short_20, r.win_long, r.win_short, codes[r.ts_code]]
                if hasattr(replay, "market_regime"):
                    self.data[i, products[r.symbol], 14] = replay.market_regime[r.trade_date, r.symbol]
            for code, ci in codes.items():
                self.close[i, ci] = replay.closes.get((group[0].trade_date, code), np.nan)
        for (entry, day, product), r in replay.lookup.items():
            if entry not in self.index or day not in self.index:
                continue
            i, j = self.index[entry], self.index[day]
            age = j-i
            if not 0 <= age < 21:
                continue
            for b, bps in enumerate((3, 6)):
                self.forecast[i, age, products[product], b] = [getattr(r,f"pred_net_5_{bps}"),
                    getattr(r,f"cal_net_{bps}"), getattr(r,f"risk90_{bps}"), r.notional, r.direction]
        for key,values in getattr(replay,"evidence_lookup",{}).items():
            day,product,side,bps,model=key[:5]
            for native in (key[5],) if len(key)==6 else (0,1):
                self.evidence[self.index[day],products[product],int(side>0),int(bps==6),native,
                              self.evidence_names.index(model)] = values
        for (entry,day,product,model),r in getattr(replay,"soft_lookup",{}).items():
            if entry not in self.index or day not in self.index:
                continue
            i,j=self.index[entry],self.index[day];age=j-i
            if 0<=age<21:
                for b,bps in enumerate((3,6)):
                    self.soft[i,age,products[product],b,int(model=="interaction")]=[
                        getattr(r,f"pred_net_5_{bps}"),getattr(r,f"pred_risk_5_{bps}")]
        source = Path(__file__).with_suffix(".cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        library = Path(f"/tmp/index_futures_equity_{digest}.dylib")
        if not library.exists():
            subprocess.run(["clang++", "-std=c++17", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
                            str(source), "-o", str(library)], check=True)
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.equity_replay
        ptr = ctypes.POINTER(ctypes.c_double)
        self.function.argtypes = [ptr, ptr, ptr, ptr, ptr, ctypes.POINTER(ctypes.c_int), *([ctypes.c_int]*13),
                                  *([ctypes.c_double]*4), ptr, ptr]
        self.function.restype = ctypes.c_int
        self.pointers = [a.ctypes.data_as(ptr) for a in (self.data, self.close, self.forecast,self.evidence,self.soft)]
        self.calendar_ptr = self.calendar.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.ptr = ptr

    def run(self, start, count, policy, variant, fee, margin, capital, trace=False,
            risk_overlay="none", principal_floor=0., reentry_rule="none", regime_mode="none",
            evidence_model=0, soft_exit=0):
        i = self.index[start] if isinstance(start, str) else start
        out = np.zeros(9, dtype=np.float64)
        daily = np.zeros((count, 10), dtype=np.float64) if trace else None
        status = self.function(*self.pointers, self.calendar_ptr, len(self.codes), len(self.dates),
            i, count, POLICIES.index(policy), VARIANTS.index(variant), 0 if fee == .0003 else 1,
            ("none", "entry5", "stop5", "fuse20", "combined").index(risk_overlay),
            ("none", "cooldown5", "reset").index(reentry_rule),
            ("none", "entry", "exit", "both").index(regime_mode),
            evidence_model,soft_exit,len(self.evidence_names),
            capital, fee, margin, principal_floor, out.ctypes.data_as(self.ptr),
            daily.ctypes.data_as(self.ptr) if trace else None)
        if status:
            raise AssertionError(f"Inner replay status={status}: {start} {policy} {variant} {capital}")
        record = dict(start=self.dates[i], end=self.ends[i+count-1], horizon=count, policy=policy,
            total_return=out[0], max_drawdown=out[1], trades=int(out[2]), active_days=int(out[3]),
            forced=int(out[4]), signal_opportunities=int(out[5]), blocked_opportunities=int(out[6]),
            blocked_ratio=out[6]/out[5] if out[5] else np.nan, settlement_breaches=int(out[7]), fees=out[8])
        return record, daily
