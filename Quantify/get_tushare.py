import tushare as ts
import pandas as pd

# ts.set_token('22f3885f1f1fb48abf6917324bddf7e97b29b9a772058b3704169fc1')
ts.set_token('1e5aab687e73d645eba8d98be1a0f0c4d83555783adf1c5a7917a3dc')
pro = ts.pro_api()
# 获取均线
# df = ts.pro_bar(ts_code='000725.SZ', start_date='20200101', end_date='20210225', ma=[60])
# df = ts.pro_bar(ts_code='000725.SZ', start_date='2021-03-01', end_date='2021-03-01', freq='5min')
# df = ts.get_today_all()
# print(df.iloc[0])
# print(df)
# df.to_csv('5min_3_1.csv')
