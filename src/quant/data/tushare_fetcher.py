"""
Tushare数据获取器

基于 Tushare Pro 接口获取股票数据
文档: https://tushare.pro/

使用前需要设置环境变量 TUSHARE_TOKEN 或在配置文件中设置
"""

import os
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import tushare as ts

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig


class TushareDataFetcher:
    """
    Tushare数据获取器
    
    支持获取股票日线、分钟线、财务数据等
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Union[str, Path] = "./data/cache"
    ):
        """
        初始化
        
        Args:
            token: Tushare token，优先使用参数，其次使用环境变量 TUSHARE_TOKEN
            cache_dir: 缓存目录
        """
        # 设置token
        self.token = token or os.environ.get("TUSHARE_TOKEN")
        if not self.token:
            raise ValueError("Tushare token 未设置，请传入参数或设置环境变量 TUSHARE_TOKEN")
        
        # 直接使用token初始化pro接口，避免ts.set_token()写入文件
        self.pro = ts.pro_api(self.token)
        
        # 设置缓存
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.store = MarketDataStore(MarketDataStoreConfig.from_env(root=self.cache_dir))
        self._memory_cache: Dict[str, pd.DataFrame] = {}
    
    def get_stock_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取股票日线数据
        
        Args:
            symbol: 股票代码，支持 sh600000 或 600000 格式
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            adjust: 复权类型，qfq(前复权), hfq(后复权), None(不复权)
        
        Returns:
            DataFrame 包含 date, open, close, high, low, volume, turnover, pct_change
        """
        # 标准化股票代码
        ts_code = self._normalize_symbol(symbol)
        
        cache_key = f"tushare_{ts_code}_{start_date}_{end_date}_{adjust}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df
        
        # 获取基础数据
        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise ValueError(f"未获取到 {symbol} 的数据")
        
        # 获取复权因子
        if adjust == "qfq":
            adj_factor = self.pro.adj_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            df = df.merge(adj_factor[['trade_date', 'adj_factor']], on='trade_date', how='left')
            df['adj_factor'] = df['adj_factor'].fillna(1.0)
            df['open'] = df['open'] * df['adj_factor']
            df['close'] = df['close'] * df['adj_factor']
            df['high'] = df['high'] * df['adj_factor']
            df['low'] = df['low'] * df['adj_factor']
        
        elif adjust == "hfq":
            adj_factor = self.pro.adj_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            df = df.merge(adj_factor[['trade_date', 'adj_factor']], on='trade_date', how='left')
            df['adj_factor'] = df['adj_factor'].fillna(1.0)
            
            # 计算后复权因子（以最后一天为基准）
            last_factor = df['adj_factor'].iloc[-1] if len(df) > 0 else 1.0
            df['hfq_factor'] = last_factor / df['adj_factor']
            df['open'] = df['open'] * df['hfq_factor']
            df['close'] = df['close'] * df['hfq_factor']
            df['high'] = df['high'] * df['hfq_factor']
            df['low'] = df['low'] * df['hfq_factor']
        
        # 标准化列名
        df = df.rename(columns={
            'trade_date': 'date',
            'ts_code': 'symbol',
            'vol': 'volume',
            'amount': 'turnover'
        })
        
        # 计算涨跌幅
        df['pct_change'] = df['close'].pct_change() * 100
        
        # 排序并重置索引
        df = df.sort_values('date').reset_index(drop=True)
        
        # 保存缓存
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        
        return df
    
    def get_stock_minute(
        self,
        symbol: str,
        period: str = "5min",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取股票分钟线数据
        
        Args:
            symbol: 股票代码，支持 sh600000 或 600000 格式
            period: 周期，支持 1min, 5min, 15min, 30min, 60min
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
        
        Returns:
            DataFrame 包含 date, open, close, high, low, volume, turnover
        """
        ts_code = self._normalize_symbol(symbol)
        
        cache_key = f"tushare_{ts_code}_{period}_{start_date}_{end_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df
        
        df = ts.pro_bar(
            ts_code=ts_code,
            freq=period,
            start_date=start_date,
            end_date=end_date,
            adj='qfq'
        )
        
        if df.empty:
            raise ValueError(f"未获取到 {symbol} 的分钟线数据")
        
        df = df.rename(columns={
            'trade_time': 'date',
            'ts_code': 'symbol',
            'vol': 'volume',
            'amount': 'turnover'
        })
        
        df = df.sort_values('date').reset_index(drop=True)
        
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        
        return df

    def get_stock_minutes_history(
        self,
        symbol: str,
        freq: str = "1min",
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取 A 股历史分钟行情。

        Tushare 官方接口为 stk_mins，支持 1min/5min/15min/30min/60min。
        适合补充 B1+1 开盘量比、9:33/9:37、14:55 等执行级回测。

        Args:
            symbol: 股票代码
            freq: 频率，1min/5min/15min/30min/60min
            start_datetime: 开始时间，如 2026-06-03 09:30:00
            end_datetime: 结束时间，如 2026-06-03 15:00:00

        Returns:
            DataFrame 包含 trade_time/open/high/low/close/vol/amount
        """
        ts_code = self._normalize_symbol(symbol)
        cache_key = f"tushare_stk_mins_{ts_code}_{freq}_{start_datetime}_{end_datetime}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        safe_start = str(start_datetime or "").replace(":", "").replace(" ", "_")
        safe_end = str(end_datetime or "").replace(":", "").replace(" ", "_")
        store_key = f"{ts_code}_{freq}_{safe_start}_{safe_end}"
        if self.store.config.backend == "sql":
            df = self.store.read_frame("tushare_stk_mins", store_key)
            if not df.empty:
                self._memory_cache[cache_key] = df
                return df
        file_path = self.cache_dir / f"tushare_stk_mins_{store_key}.parquet"
        if self.store.config.backend != "sql" and file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        df = self.pro.stk_mins(
            ts_code=ts_code,
            freq=freq,
            start_date=start_datetime,
            end_date=end_datetime,
        )
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            if "trade_time" in df.columns:
                df["trade_time"] = pd.to_datetime(df["trade_time"])
            elif "time" in df.columns:
                df = df.rename(columns={"time": "trade_time"})
                df["trade_time"] = pd.to_datetime(df["trade_time"])
            if "vol" in df.columns and "volume" not in df.columns:
                df["volume"] = df["vol"]
            df = df.sort_values("trade_time").reset_index(drop=True)

        if self.store.config.backend == "sql":
            self.store.write_frame(df, "tushare_stk_mins", store_key)
        else:
            df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        return df

    def get_realtime_minutes_daily(self, symbol: str, freq: str = "1MIN") -> pd.DataFrame:
        """
        获取当日开盘以来实时分钟行情，用于例行盘中监控。

        官方接口为 rt_min_daily，freq 使用大写：1MIN/5MIN/15MIN/30MIN/60MIN。
        """
        ts_code = self._normalize_symbol(symbol)
        df = self.pro.rt_min_daily(ts_code=ts_code, freq=freq)
        if df is None:
            return pd.DataFrame()
        if not df.empty:
            if "time" in df.columns and "trade_time" not in df.columns:
                df = df.rename(columns={"time": "trade_time"})
            if "vol" in df.columns and "volume" not in df.columns:
                df["volume"] = df["vol"]
            if "trade_time" in df.columns:
                df["trade_time"] = pd.to_datetime(df["trade_time"])
                df = df.sort_values("trade_time").reset_index(drop=True)
        return df
    
    def get_index_daily(
        self,
        symbol: str = "000001.SH",
        start_date: str = "20200101",
        end_date: str = "20260101"
    ) -> pd.DataFrame:
        """
        获取指数日线数据
        
        Args:
            symbol: 指数代码，如 000001.SH, 399001.SZ
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame 包含 date, open, close, high, low, volume, turnover, pct_change
        """
        ts_code = self._normalize_index_symbol(symbol)
        
        cache_key = f"tushare_index_{ts_code}_{start_date}_{end_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df
        
        df = self.pro.index_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise ValueError(f"未获取到指数 {symbol} 的数据")
        
        df = df.rename(columns={
            'trade_date': 'date',
            'ts_code': 'symbol',
            'vol': 'volume',
            'amount': 'turnover'
        })
        
        df['pct_change'] = df['close'].pct_change() * 100
        
        df = df.sort_values('date').reset_index(drop=True)
        
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        
        return df
    
    def get_stock_basic(self, market: str = "all") -> pd.DataFrame:
        """
        获取股票基本信息
        
        Args:
            market: 市场类型，all/A股/港股/美股
        
        Returns:
            DataFrame 包含 ts_code, symbol, name, industry, list_date 等
        """
        cache_key = f"tushare_stock_basic_{market}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df
        
        df = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date,market')
        
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        
        return df

    def get_daily_basic(self, trade_date: str, fields: Optional[str] = None) -> pd.DataFrame:
        """
        获取 Tushare 每日指标数据。

        Args:
            trade_date: 交易日期，格式 YYYYMMDD
            fields: Tushare fields 参数；为空时使用 B1 建模需要的常用字段

        Returns:
            DataFrame 包含换手率、量比、估值、市值、股本等每日指标
        """
        default_fields = (
            "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
            "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
            "free_share,total_mv,circ_mv"
        )
        fields = fields or default_fields
        cache_key = f"tushare_daily_basic_{trade_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        df = self.pro.daily_basic(trade_date=trade_date, fields=fields)
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        return df

    def get_moneyflow(self, trade_date: str, fields: Optional[str] = None) -> pd.DataFrame:
        """
        获取 Tushare 个股资金流向数据。

        Args:
            trade_date: 交易日期，格式 YYYYMMDD
            fields: Tushare fields 参数；为空时使用策略研究常用字段

        Returns:
            DataFrame 包含大单/超大单净流入等资金流字段
        """
        default_fields = (
            "ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,"
            "buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,"
            "buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,"
            "sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount"
        )
        fields = fields or default_fields
        cache_key = f"tushare_moneyflow_{trade_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        df = self.pro.moneyflow(trade_date=trade_date, fields=fields)
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        return df

    def get_limit_list(self, trade_date: str, fields: Optional[str] = None) -> pd.DataFrame:
        """
        获取涨跌停数据，用于涨停/跌停风险、连板和极端流动性过滤。
        """
        default_fields = (
            "trade_date,ts_code,name,close,pct_chg,amp,fc_ratio,fl_ratio,"
            "fd_amount,first_time,last_time,open_times,strth,limit"
        )
        fields = fields or default_fields
        cache_key = f"tushare_limit_list_{trade_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        df = self.pro.limit_list_d(trade_date=trade_date, fields=fields)
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        return df

    def get_top_list(self, trade_date: str, fields: Optional[str] = None) -> pd.DataFrame:
        """
        获取龙虎榜数据，用于后续游资接力、异常成交和事件型风险研究。
        """
        default_fields = (
            "trade_date,ts_code,name,close,pct_change,turnover_rate,amount,"
            "l_sell,l_buy,l_amount,net_amount,net_rate,amount_rate,float_values,"
            "reason"
        )
        fields = fields or default_fields
        cache_key = f"tushare_top_list_{trade_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        df = self.pro.top_list(trade_date=trade_date, fields=fields)
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        return df
    
    def get_financial_report(self, symbol: str, year: int, quarter: int = 4) -> pd.DataFrame:
        """
        获取财务报表数据
        
        Args:
            symbol: 股票代码
            year: 年份
            quarter: 季度，1-4
        
        Returns:
            DataFrame 财务报表数据
        """
        ts_code = self._normalize_symbol(symbol)
        
        cache_key = f"tushare_finance_{ts_code}_{year}{quarter}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df
        
        df = self.pro.fina_indicator(
            ts_code=ts_code,
            start_date=f"{year}0101",
            end_date=f"{year}1231"
        )
        
        if not df.empty:
            df = df[df['end_date'].str.startswith(str(year))]
        
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        
        return df
    
    def get_dividend(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取分红送股数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame 分红数据
        """
        ts_code = self._normalize_symbol(symbol)
        
        cache_key = f"tushare_dividend_{ts_code}_{start_date}_{end_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df
        
        df = self.pro.dividend(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df
        
        return df
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化股票代码为 Tushare 格式
        
        Args:
            symbol: sh600000 或 600000
        
        Returns:
            600000.SH 格式
        """
        symbol = str(symbol).strip()
        
        if "." in symbol:
            return symbol
        elif symbol.startswith("sh"):
            return f"{symbol[2:]}.SH"
        elif symbol.startswith("sz"):
            return f"{symbol[2:]}.SZ"
        elif symbol.startswith("6"):
            return f"{symbol}.SH"
        elif symbol.startswith("0") or symbol.startswith("3"):
            return f"{symbol}.SZ"
        else:
            return symbol
    
    def _normalize_index_symbol(self, symbol: str) -> str:
        """
        标准化指数代码
        
        Args:
            symbol: 000001 或 000001.SH
        
        Returns:
            000001.SH 格式
        """
        symbol = str(symbol).strip()
        
        if "." in symbol:
            return symbol
        elif symbol.startswith("sh"):
            return f"{symbol[2:]}.SH"
        elif symbol.startswith("sz"):
            return f"{symbol[2:]}.SZ"
        else:
            if len(symbol) == 6:
                if symbol.startswith("0"):
                    return f"{symbol}.SH"
                else:
                    return f"{symbol}.SZ"
        
        return symbol
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        清除缓存
        
        Args:
            symbol: 股票代码，如果为None则清除所有缓存
        """
        if symbol:
            ts_code = self._normalize_symbol(symbol)
            keys_to_remove = [k for k in self._memory_cache.keys() if ts_code in k]
            for k in keys_to_remove:
                del self._memory_cache[k]
        else:
            self._memory_cache.clear()
        
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("tushare_*.parquet"):
                if symbol is None or symbol.lower() in f.name.lower():
                    f.unlink()
