"""Validated daily amount units shared by live strategy workspaces."""
import numpy as np
import pandas as pd

def resolve_amount_kcny(frame: pd.DataFrame) -> tuple[pd.Series,pd.Series]:
    """Repair the documented Tushare amount->turnover alias, without unit guessing.

    The project fetcher renames amount to turnover without changing its kCNY
    unit. A fallback must also imply a VWAP within the observed price range;
    turnover percentages or yuan-denominated amounts fail that economic check.
    """
    amount=pd.to_numeric(frame['amount'],errors='coerce').copy()
    turnover=pd.to_numeric(frame.get('turnover',pd.Series(np.nan,index=frame.index)),errors='coerce')
    volume=pd.to_numeric(frame.get('vol',pd.Series(np.nan,index=frame.index)),errors='coerce')
    if 'volume' in frame:volume=volume.combine_first(pd.to_numeric(frame.volume,errors='coerce'))
    implied=turnover*10/volume.where(volume.gt(0))
    checked=implied.ge(frame.low*.99)&implied.le(frame.high*1.01)&turnover.gt(0)&np.isfinite(turnover)
    repaired=(amount.isna()|~np.isfinite(amount))&checked
    amount.loc[repaired]=turnover.loc[repaired]
    return amount,repaired
