import requests
import json
import tushare as ts
import os
import numpy as np
import datetime
import time

dict = {
    "白银期货": ("AG2106", "SHF")
}
push_users = {
    "self": "kyx6kNQZtoN8XkQXmjxyO1eMD",
    "冯文颂": "g4Elz5ql3N8ZPgItJ0VVWdaah",
}

# step 1
ts.set_token('1e5aab687e73d645eba8d98be1a0f0c4d83555783adf1c5a7917a3dc')
pro = ts.pro_api()


def get_ma60(ts_code, today, saved_ma60_file):
    if os.path.exists(saved_ma60_file):
        file = np.load(saved_ma60_file, allow_pickle=True)
        if ts_code in file.item().keys():
            ma60 = file.item()[ts_code]
            if ma60 is not None:
                return ma60
    df = ts.pro_bar(ts_code=ts_code, start_date='20200101', end_date=today.strftime('%Y%m%d'), ma=[60], asset='FT')
    if len(df) < 1:
        return None
    ma60 = df.iloc[0].loc['ma60']
    ma60_np = {}
    if os.path.exists(saved_ma60_file):
        ma60_np = np.load(saved_ma60_file)
    else:
        ma60_np[ts_code] = ma60
    np.save(saved_ma60_file, ma60_np)
    return ma60


def get_latest_5min(url):
    time.sleep(1)
    r = requests.get(url).content
    try:
        js = json.loads(r)
        if len(js) < 1:
            return None
        return js[0]
    except Exception:
        return None


def calc(ma60, latest_5min):
    print(ma60)
    print(latest_5min)
    low = latest_5min[3]
    close = latest_5min[4]
    # print(latest_5min['day'])
    # print(latest_5min['open'])
    # print(latest_5min['high'])
    # print(latest_5min['low'])
    # print(latest_5min['close'])
    # print(latest_5min['volume'])
    # low = latest_5min['low']
    # close = latest_5min['close']
    if float(low) < float(ma60) < float(close):
        return True, f'ma60:{ma60},latest_5min:{latest_5min}'
    return False, None


def pushWechat(title, content):
    mydata = {
        'text': title,
        'desp': {content}
    }
    for name, user_code in push_users.items():
        url = f'http://wx.xtuis.cn/{user_code}.send'
        requests.post(url, data=mydata)
        print(f'pushed success to {name}')


def main():
    today = datetime.date.today()
    saved_ma60_file = f'{today}-FT.npy'
    title = '期货推送'
    latest_time = None
    for name, v in dict.items():
        ts_code = f'{v[0]}.{v[1]}'
        ma60 = get_ma60(ts_code, today, saved_ma60_file)
        if ma60 is None:
            print('get failed')
            continue
        ts_code = f'{v[0].lower()}'
        url = f'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesMiniKLine5m?symbol={ts_code}'
        latest_5min = get_latest_5min(url)
        if latest_5min is None:
            print('get  latest_5min failed')
            continue
        if latest_time is None:
            latest_time = latest_5min[0]
        isPush, _ = calc(ma60, latest_5min)
        if isPush:
            msg = f'期货:{name}－{ts_code} 五分钟线超过了60日线！,{_}'
            pushWechat(title, msg)
    return latest_time


if __name__ == '__main__':
    latest_time = None

    # 记录当前时间
    print('start')
    while True:
        now = datetime.datetime.now()
        delta = 0
        if latest_time is not None:
            latest_datetime = datetime.datetime.strptime(latest_time, '%Y-%m-%d %H:%M:%S')
            days = (now - latest_datetime).days
            if days >= 0:
                delta = (now - latest_datetime).seconds / 60
        if latest_time is None or delta >= 5:
            print(f'delta:{delta}')
            latest_time = main()
        time.sleep(60)
