"""
Tushare数据获取测试脚本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

# Tushare Token
TUSHARE_TOKEN = "wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq"


def test_tushare_import():
    """测试Tushare导入"""
    try:
        from quant.data import TushareDataFetcher
        print("✅ TushareDataFetcher 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_tushare_connection():
    """测试Tushare连接"""
    from quant.data import TushareDataFetcher
    
    try:
        fetcher = TushareDataFetcher(token=TUSHARE_TOKEN)
        print("✅ Tushare连接成功")
        return fetcher
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None


def test_get_stock_daily(fetcher):
    """测试获取股票日线数据"""
    try:
        df = fetcher.get_stock_daily(
            symbol="600000",
            start_date="20230101",
            end_date="20230131"
        )
        
        print(f"✅ 获取股票日线数据成功 - {len(df)} 条记录")
        print(f"   列名: {df.columns.tolist()}")
        print(f"   前5条:\n{df.head().to_string()}")
        return True
    except Exception as e:
        print(f"❌ 获取股票日线数据失败: {e}")
        return False


def test_get_index_daily(fetcher):
    """测试获取指数日线数据"""
    try:
        df = fetcher.get_index_daily(
            symbol="000001.SH",
            start_date="20230101",
            end_date="20230131"
        )
        
        print(f"✅ 获取指数日线数据成功 - {len(df)} 条记录")
        print(f"   列名: {df.columns.tolist()}")
        return True
    except Exception as e:
        print(f"❌ 获取指数日线数据失败: {e}")
        return False


def test_get_stock_basic(fetcher):
    """测试获取股票基本信息"""
    try:
        df = fetcher.get_stock_basic()
        
        print(f"✅ 获取股票基本信息成功 - {len(df)} 条记录")
        print(f"   列名: {df.columns.tolist()}")
        print(f"   示例:\n{df[['ts_code', 'name', 'industry']].head(5).to_string()}")
        return True
    except Exception as e:
        print(f"❌ 获取股票基本信息失败: {e}")
        return False


def test_get_financial_report(fetcher):
    """测试获取财务报表"""
    try:
        df = fetcher.get_financial_report("600000", 2023)
        
        print(f"✅ 获取财务报表成功 - {len(df)} 条记录")
        if not df.empty:
            print(f"   列名: {df.columns.tolist()[:10]}...")
        return True
    except Exception as e:
        print(f"❌ 获取财务报表失败: {e}")
        return False


def main():
    print("=" * 60)
    print("Tushare数据获取测试")
    print("=" * 60)
    
    # 测试导入
    if not test_tushare_import():
        return
    
    # 测试连接
    fetcher = test_tushare_connection()
    if not fetcher:
        return
    
    # 测试各接口
    print("\n" + "-" * 60)
    test_get_stock_daily(fetcher)
    
    print("\n" + "-" * 60)
    test_get_index_daily(fetcher)
    
    print("\n" + "-" * 60)
    test_get_stock_basic(fetcher)
    
    print("\n" + "-" * 60)
    test_get_financial_report(fetcher)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()