import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import pytz

def get_data_from_metatrader5(pair):

    if not mt5.initialize(login=204847, server="ACCapital-Live", password="......."):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # timezone = pytz.timezone("Etc/UTC")
    start_time = pd.to_datetime("2024-01-01 00:00:00")
    end_time = datetime.datetime.now()

    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    rate = mt5.copy_rates_range(
        pair, mt5.TIMEFRAME_M15, start_timestamp, end_timestamp)
    df = pd.DataFrame(rate)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={'time': 'Date', 'open': 'Open',
                   'low': 'Low', 'close': 'Close', 'high': 'High', 'tick_volume': 'Volume'})

    mt5.shutdown()
    return df

# def get_list_symbol():
#     mt5.initialize(login=204847, server="ACCapital-Live", password="........")
#     symbols=mt5.symbols_get()
#     list_symbol =[]
#     for symbol in symbols:
#         i = str(symbol).split("\\")[-1].split("')")[0]
#         list_symbol.append(i)
#     return list_symbol

pairs = ["EURUSD"]  # Liệt kê thêm ...f
# pairs = get_list_symbol()

for pair in pairs:

    try:
        df = get_data_from_metatrader5(pair)
        print(df)

        # filename = f"E:/Ducbui/FX2/{pair}.csv"
        # df.to_csv(filename, index=False)
        print(f'done {pair}\n')
        
    except Exception as e:
        print(e)
        print('fail')
        pass