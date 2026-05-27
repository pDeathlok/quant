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
        
        if symbol.startswith("sh"):
            return f"{symbol[2:]}.SH"
        elif symbol.startswith("sz"):
            return f"{symbol[2:]}.SZ"
        elif symbol.startswith("6"):
            return f"{symbol}.SH"
        elif symbol.startswith("0") or symbol.startswith("3"):
            return f"{symbol}.SZ"
        elif "." in symbol:
            return symbol
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
        
        if symbol.startswith("sh"):
            return f"{symbol[2:]}.SH"
        elif symbol.startswith("sz"):
            return f"{symbol[2:]}.SZ"
        elif "." not in symbol:
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