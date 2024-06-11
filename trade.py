import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import ta
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler
import time

# LOGIN = 204979
# PASSWORD = "HuyManh2002#"
# SERVER = "ACCapital-Live"

# LOGIN = 10002273196
# PASSWORD = "6ZnQkBn"
# SERVER = "MetaQuotes-Demo"

LOGIN = 5026237471
PASSWORD = "FdP+Qw3w"
SERVER = "MetaQuotes-Demo"

BOT_ID = "6991905914:AAEk3_I6DzG_5GcxVxE6-k5qbmtohT0K5uQ"
GROUP_ID = "-4184587836"

VOLUME_PER_TRADE = 0.01
DEVIATION = 1000


def get_data_real_time(pair, days, time_frame):
    """pair là cặp ngoại hối
        days là số ngày lấy tính từ hôm nay
        time_frame là mẫu thời gian sử dụng
    """
    # kết nối mt5 except lỗi
    if not mt5.initialize(login=LOGIN, server=SERVER, password=PASSWORD):
        print("initialize() failed, error code =", mt5.last_error())
        return None

    # thời bắt đầu và kết thúc lấy
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    rate = mt5.copy_rates_range(
        pair, mt5.TIMEFRAME_M1, start_timestamp, end_timestamp)
    df = pd.DataFrame(rate)

    try:
        df["time"] = pd.to_datetime(df["time"], unit="s")
    except Exception as e:
        print(f"Error converting time column: {e}")
        print(df.columns)
    # rename các cột chuẩn OCHL-v
    df = df.rename(columns={'time': 'Date', 'open': 'Open',
                            'low': 'Low', 'close': 'Close', 'high': 'High', 'tick_volume': 'Volume'})
    # resample dữ liệu
    df = df.groupby([pd.Grouper(key='Date', freq=time_frame)]) \
           .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}) \
           .dropna().reset_index()

    mt5.shutdown()
    return df


def Connect_to_MT5():
    if not mt5.initialize(login=LOGIN, server=SERVER, password=PASSWORD):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    account_info = mt5.account_info()
    print("\n\n")
    print("****************************************************")
    print("\t\tLogin:", account_info.login)
    print("\t\tLeverrage:", account_info.leverage)
    print("\t\tBalance:", account_info.balance)
    print("\t\tEquity:", account_info.equity)
    print("\t\tMargin:", account_info.margin)
    print("\t\tProfit:", account_info.profit)
    print("\t\tMargin Free:", account_info.margin_free)
    print("***************************************************")
    print("\n\n")


def send_telegram_message(text):
    token = BOT_ID
    chat_id = GROUP_ID
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text},
    )


def open_buy(lots, PAIR):

    request_buy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PAIR,
        "volume": VOLUME_PER_TRADE * lots,
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(PAIR).ask,
        "slippage": SLIPPAGE,
    }
    order1 = mt5.order_send(request_buy)
    position_id = order1.order
    price_open = order1.price
    print("Open Order Buy ID :", position_id)
    print("Pair :", PAIR)
    print("Buy {} lot at price {}". format(
        VOLUME_PER_TRADE * lots, price_open))

    order_open_buy = {
        'action': 'Buy',
        'volume': VOLUME_PER_TRADE * lots,
        'price': price_open,
        'order_id': position_id,
        'pair': PAIR,
    }
    return position_id


def close_buy(position_id, lots, PAIR):

    request_close_buy = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PAIR,
        "volume": VOLUME_PER_TRADE * lots,
        "type": mt5.ORDER_TYPE_SELL,
        "position": position_id,
        "price": mt5.symbol_info_tick(PAIR).bid,
        "slippage": SLIPPAGE,
    }
    order2 = mt5.order_send(request_close_buy)
    print("Close Order Buy ID :", position_id)
    print("Pair :", PAIR)
    price_close = order2.price
    print("Price :", price_close)
    position_deals = mt5.history_deals_get(position=position_id)
    position_deals = pd.DataFrame(position_deals)
    PNL = position_deals[13].diff(1).iloc[-1]
    print("Profit: {} $". format(PNL))

    order_close = {
        'action': 'Buy',
        'volume': VOLUME_PER_TRADE * lots,
        'price': price_close,
        'order_id': position_id,
        'pair': PAIR,
        'profit': PNL
    }


def open_sell(lots, PAIR):

    request_sell = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PAIR,
        "volume": VOLUME_PER_TRADE * lots,
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(PAIR).bid,
        "slippage": SLIPPAGE,
    }
    order1 = mt5.order_send(request_sell)
    position_id = order1.order
    price_open = order1.price
    print("Open Order SELL ID :", position_id)
    print("Pair :", PAIR)
    print("SELL {} lot at price {}". format(
        VOLUME_PER_TRADE * lots, price_open))

    order_open_sell = {
        'action': 'Sell',
        'volume': VOLUME_PER_TRADE * lots,
        'price': price_open,
        'order_id': position_id,
        'pair': PAIR,
    }
    return position_id


def close_sell(position_id, lots, PAIR):

    request_close_sell = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PAIR,
        "volume": VOLUME_PER_TRADE * lots,
        "type": mt5.ORDER_TYPE_BUY,
        "position": position_id,
        "price": mt5.symbol_info_tick(PAIR).ask,
        "slippage": SLIPPAGE,
    }
    order2 = mt5.order_send(request_close_sell)
    print("Close Order Sell ID :", position_id)
    print("Pair :", PAIR)
    price_close = order2.price
    print("Price :", price_close)
    position_deals = mt5.history_deals_get(position=position_id)
    position_deals = pd.DataFrame(position_deals)
    PNL = position_deals[13].diff(1).iloc[-1]
    print("Profit: {} $". format(PNL))

    order_close = {
        'action': 'Sell',
        'volume': VOLUME_PER_TRADE * lots,
        'price': price_close,
        'order_id': position_id,
        'pair': PAIR,
        'profit': PNL
    }


def portfolio_pnl_future(position_long, position_short, Close):
    ''' tính PNL của một chiến thuật 
    position_long: series position long
    position_short: series position short'''
    intitial_capital_long = (position_long.iloc[0])*(Close.iloc[0])
    cash_long = (position_long.diff(1)*Close)
    cash_long[0] = intitial_capital_long
    cash_cs_long = cash_long.cumsum()
    portfolio_value_long = (position_long*Close)

    intitial_capital_short = (position_short.iloc[0])*(Close.iloc[0])
    cash_short = (position_short.diff(1)*Close)
    cash_short[0] = intitial_capital_short
    cash_cs_short = cash_short.cumsum()
    portfolio_value_short = (position_short*Close)

    backtest = (portfolio_value_long - cash_cs_long) + \
        (cash_cs_short - portfolio_value_short)
    backtest.fillna(0, inplace=True)
    cash_max = (cash_long + cash_short).max()
    pnl = backtest/cash_max

    ''' return PNL, lần vào lệnh lớn nhất, PNL tương đối theo % '''
    return backtest, cash_max, pnl


def DumpCSV_and_MesToTele(name, path_csv_intraday, Position, Close, token, id, position_input=1, fee=0):
    ''' Ghi file csv v� g?i tin nh?n d?n telegram 
        Input: name: t�n c?a chi?n thu?t
               path_csv_intraday: du?ng d?n file csv intraday
               Position: Series v? th? c?a chi?n thu?t 
               Close: Series gi� kh?p l?nh
               token: token c?a bot telegram
               id: id c?a chat group telegram
               position_input: s? h?p d?ng v�o m?i l?nh'''
    # ip_address = output.decode().strip()
    try:
        df = pd.read_csv(path_csv_intraday)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'Position': df.Position.tolist(),
            'Close': df.Close.tolist(),
            'total_gain': df.total_gain.tolist(),
            'gain': df.gain.tolist(),
        }
        try:
            dict_data['Datetime'] = pd.to_datetime(
                dict_data['Datetime']).to_list()
        except:
            for i in range(len(dict_data['Datetime'])):
                dict_data['Datetime'][i] = pd.to_datetime(
                    dict_data['Datetime'][i])
            dict_data['Datetime'] = list(dict_data['Datetime'])
        df = pd.DataFrame(data=dict_data)
    except:
        dict_data = {
            'Datetime': [pd.to_datetime((datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))],
            'Position': [0],
            'Close': [0],
            'total_gain': [0],
            'gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_intraday, index=False)

    Close = Close.iloc[-1]
    new_Pos = int(Position.iloc[-1])
    time_now = datetime.now()
    profit = 0
    profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()
    mes = f'{name}:'

    if new_Pos != dict_data['Position'][-1] :

        inputPos = int(new_Pos - dict_data['Position'][-1])
        dict_data['Datetime'].append(pd.to_datetime(
            time_now.strftime('%Y-%m-%d %H:%M:%S')))
        dict_data['Close'].append(Close)
        dict_data['Position'].append(new_Pos)
        dict_data['total_gain'].append(0)
        dict_data['gain'].append(0)

        df = pd.DataFrame(data=dict_data)
        df['signal_long'] = np.where(df.Position > 0, df.Position, 0)
        df['signal_short'] = np.where(df.Position < 0, np.abs(df.Position), 0)
        df['total_gain'] = portfolio_pnl_future(
            df['signal_long'], df['signal_short'], df.Close)[0]
        df['gain'] = df.total_gain.diff()
        df.fillna(0, inplace=True)
        df['gain'] = np.where(np.abs(df.gain.to_numpy())
                              < 0.00001, 0, df.gain.to_numpy())
        df.loc[df['Position'].diff().fillna(0) != 0, 'gain'] = df.loc[df['Position'].diff(
        ) != 0, 'gain'] - np.abs(fee/2 * inputPos)
        # df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] = df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] - fee
        df['total_gain'] = df.gain.cumsum()
        profit = df.gain.iloc[-1]
        profit_today = df.loc[df.Datetime.dt.date ==
                              time_now.date(), 'gain'].sum()

        if inputPos > 0:
            mes = f'{name}:\nLong {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        elif inputPos < 0:
            mes = f'{name}:\nShort {inputPos*position_input} at {Close}, Current Pos: {new_Pos*position_input}'
        else:
            mes = f'{name}:\nClose at {Close}, Current Pos: {new_Pos*position_input}'

        if np.round(profit*10)/10 != 0:
            mes += f'\nProfit: {np.round(profit*10)/10}'
        mes += f'\nProfit today: {np.round(profit_today*10)/10}'

        df.drop(columns=['signal_long', 'signal_short'], inplace=True)
        send_telegram_message(mes)
        df.to_csv(path_csv_intraday, index=False)

    else:
        inputPos = 0

    profit_today = np.round(profit_today*10)/10
    print(name)
    print(time_now)
    print(Close)
    print('Input Position:', inputPos*position_input)
    print('Current Position:', new_Pos*position_input)
    if np.round(profit*10)/10 != 0:
        print(f'Profit: {np.round(profit*10)/10}')
    print(f'Profit today: {profit_today}')
    print('\n')

    ''' return dataframe intraday, input position, current position'''
    df['profit_today'] = profit_today
    return df, inputPos, new_Pos


def determine_session(data):
    data['Date']=pd.to_datetime(data['Date'])
    data['DayOfWeek']=data['Date'].dt.day_of_week

    data['Tokyo Session']=0
    data["London Session"]=0
    data["New York Session"]=0
    data["Sydney Session"]=0
    
    data.loc[(data['Date'].dt.hour >= 0) & (data['Date'].dt.hour <= 9 )   ,'Tokyo Session']=1
    data.loc[(data['Date'].dt.hour >= 7) & (data['Date'].dt.hour <= 16 ),"London Session"]=1
    data.loc[(data['Date'].dt.hour >= 13) & (data['Date'].dt.hour <= 21),"New York Session"]=1
    data.loc[(data['Date'].dt.hour >= 21) | (data['Date'].dt.hour < 6) ,"Sydney Session"]=1  

    return data


def MACD(df):
    def lowPass_filter(signal, ratio):
        b, a = butter(1, ratio, btype='low', analog=False)
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal
    dt=df.copy()
    dt['Position_MACD']=0
    params_macd={0: {'long_window_slow': 127,  'long_window_fast': 125,  'long_window_sign': 185,  'long_window_roll': 76,  'long_win_ratio': 0.90,  'long_diff': 20,  'short_window_slow': 23,  'short_window_fast': 182,  'short_window_sign': 12,  
            'short_window_roll': 184,  'short_win_ratio': 0.55,  'short_diff': 7}, 
        1: {'long_window_slow': 51,  'long_window_fast': 12,  'long_window_sign': 90,  'long_window_roll': 107,  'long_win_ratio': 0.5,  'long_diff': 36,  'short_window_slow': 114,  'short_window_fast': 162,  'short_window_sign': 135,  
            'short_window_roll': 82,  'short_win_ratio': 0.8,  'short_diff': 40}, 
        2: {'long_window_slow': 197,  'long_window_fast': 61,  'long_window_sign': 56,  'long_window_roll': 200,  'long_win_ratio': 0.99,  'long_diff': 26,  'short_window_slow': 85,  'short_window_fast': 124,  'short_window_sign': 139,  
            'short_window_roll': 166,  'short_win_ratio': 0.72,  'short_diff': 1}, 
        3: {'long_window_slow': 98,  'long_window_fast': 15,  'long_window_sign': 175,  'long_window_roll': 156,  'long_win_ratio': 0.75,  'long_diff': 7,  'short_window_slow': 71,  'short_window_fast': 158,  'short_window_sign': 12,  
            'short_window_roll': 119,  'short_win_ratio': 0.5,  'short_diff': 26}, 
        4: {'long_window_slow': 108,  'long_window_fast': 179,  'long_window_sign': 155,  'long_window_roll': 122,  'long_win_ratio': 0.4,  'long_diff': 6,  'short_window_slow': 174,  'short_window_fast': 155,  'short_window_sign': 88,  
            'short_window_roll': 84,  'short_win_ratio': 0.4,  'short_diff': 35}}
    for i, param in params_macd.items():

        dt['long_lqf_close']=lowPass_filter(dt['Close'],param['long_win_ratio'])

        dt["long_MACD"]=ta.trend.MACD( close=dt['long_lqf_close'], window_slow=param['long_window_slow'],window_fast=param['long_window_fast'], window_sign=param['long_window_sign'],fillna=False).macd_diff()
        dt["long_MACD_ma"]=dt["long_MACD"].rolling(param['long_window_roll']).mean()
        dt["long_MACD_diff"]=dt["long_MACD"].diff(param['long_diff'])


        dt['short_lqf_close']=lowPass_filter(dt['Close'],param['short_win_ratio'])

        dt["short_MACD"]=ta.trend.MACD( close=dt['short_lqf_close'], window_slow=param['short_window_slow'],window_fast=param['short_window_fast'], window_sign=param['short_window_sign'],fillna=False).macd_diff()
        dt["short_MACD_ma"]=dt["short_MACD"].rolling(param['short_window_roll']).mean()
        dt["short_MACD_diff"]=dt["short_MACD"].diff(param['short_diff'])



        entry_long=  (dt['long_MACD_ma'] > dt["long_MACD"] ) & (dt['long_MACD_diff'] > 0) & (dt['DayOfWeek']==i) & (dt['Tokyo Session']==1)
        entry_short= (dt['short_MACD_ma'] < dt["short_MACD"] ) & (dt['short_MACD_diff'] < 0) & (dt['DayOfWeek']==i) & (dt['Tokyo Session']==1)

        dt.loc[entry_long , 'Position_MACD'] =dt['Position_MACD']  +1
        dt.loc[entry_short, 'Position_MACD'] =dt['Position_MACD']  -1
    
    return dt['Position_MACD']

def Stochastic_RSI(df):
    dt= df.copy()
    dt['Position']=0 

    params={0: {'window_short': 56, 'window_long': 74,  'smooth1_long': 36,  'smooth2_long': 24,  'slope_long': 11,  'slope_short': 9,  'smooth1_short': 46,  'smooth2_short': 14},
            1: {'window_short': 90,  'window_long': 76,  'smooth1_long': 90,  'smooth2_long': 78,  'slope_long': 5,  'slope_short': 3,  'smooth1_short': 98,  'smooth2_short': 76},
            2: {'window_short': 36,  'window_long': 28,  'smooth1_long': 36,  'smooth2_long': 48,  'slope_long': 15,  'slope_short': 5, 'smooth1_short': 20,  'smooth2_short': 34},
            3: {'window_short': 30,  'window_long': 10,  'smooth1_long': 22,  'smooth2_long': 58,  'slope_long': 15,  'slope_short': 7,  'smooth1_short': 34,  'smooth2_short': 22},
            4: {'window_short': 40,  'window_long': 28,  'smooth1_long': 18,  'smooth2_long': 42,  'slope_long': 3,  'slope_short': 15,  'smooth1_short': 16,  'smooth2_short': 48}}
    for i, param in params.items():
        dt['RSI_long']=  ta.momentum.stochrsi_k(close=dt['Close'],smooth1=param['smooth1_long'],smooth2=param['smooth2_long'],window=param['window_long'],fillna=False)
        dt['EMA_long']=  ta.momentum.stochrsi_d(close=dt['Close'],smooth1=param['smooth1_long'],smooth2=param['smooth2_long'],window=param['window_long'],fillna=False)
        dt['RSI_long_slope']=  dt['RSI_long'].diff(param['slope_long'])

        dt.loc[ (dt['RSI_long_slope'] > 0) & (dt['RSI_long'] > dt['EMA_long']) & (dt['DayOfWeek']==i) & (dt['Tokyo Session']==1), 'Position'] = dt['Position'] + 1 


        dt['RSI_short']=  ta.momentum.stochrsi_k(close=dt['Close'],smooth1=param['smooth1_short'],smooth2=param['smooth2_short'],window=param['window_short'],fillna=False)
        dt['EMA_short']=  ta.momentum.stochrsi_d(close=dt['Close'],smooth1=param['smooth1_short'],smooth2=param['smooth2_short'],window=param['window_short'],fillna=False)
        dt['RSI_short_slope']=  dt['RSI_short'].diff(param['slope_short'])

        dt.loc[ (dt['RSI_short_slope'] < 0) & (dt['RSI_short'] < dt['EMA_short']) & (dt['DayOfWeek']==i) & (dt['Tokyo Session']==1), 'Position'] = dt['Position'] - 1 

    dt.loc[dt['Position']>0,'Position']=1
    dt.loc[dt['Position']<0,'Position']=-1
    return dt['Position']

def AO(df):
        dt=df.copy()
        dt['Position']=0 
        params={0: {'window1_long': 19,  'window2_long': 11,  'window11_long': 24,  'window12_long': 14,  'window1_short': 17,  'window2_short': 40,  'window11_short': 33,  'window12_short': 7,  'diff_long': 4,  'diff1_long': 14,  'diff_short': 1,  'diff1_short': 1}, 
                1: {'window1_long': 38,  'window2_long': 44,  'window11_long': 22,  'window12_long': 27,  'window1_short': 31,  'window2_short': 8,  'window11_short': 40,  'window12_short': 35,  'diff_long': 2,  'diff1_long': 4,  'diff_short': 10,  'diff1_short': 13}, 
                2: {'window1_long': 39,  'window2_long': 29,  'window11_long': 37,  'window12_long': 39,  'window1_short': 23,  'window2_short': 19,  'window11_short': 32,  'window12_short': 7,  'diff_long': 1,  'diff1_long': 14,  'diff_short': 2,  'diff1_short': 1},
                3: {'window1_long': 35,  'window2_long': 23,  'window11_long': 14,  'window12_long': 30,  'window1_short': 44,  'window2_short': 7,  'window11_short': 20,  'window12_short': 43,  'diff_long': 14,  'diff1_long': 4,  'diff_short': 3,  'diff1_short': 9}, 
                4: {'window1_long': 6,  'window2_long': 34,  'window11_long': 29,  'window12_long': 10,  'window1_short': 41,  'window2_short': 9,  'window11_short': 6,  'window12_short': 33,  'diff_long': 15,  'diff1_long': 12,  'diff_short': 4,  'diff1_short': 15}}
        for i, param in params.items():
                dt['AO_long']=ta.momentum.AwesomeOscillatorIndicator(high=dt['High'], low=dt['Low'], window1=param['window1_long'], window2=param['window2_long'], fillna= False).awesome_oscillator().diff(param['diff_long'])
                dt['AO1_long']=ta.momentum.AwesomeOscillatorIndicator(high=dt['High'], low=dt['Low'], window1=param['window11_long'], window2=param['window12_long'], fillna= False).awesome_oscillator().diff(param['diff1_long'])


                dt['AO_short']=ta.momentum.AwesomeOscillatorIndicator(high=dt['High'], low=dt['Low'], window1=param['window1_short'], window2=param['window2_short'], fillna= False).awesome_oscillator().diff(param['diff_short'])
                dt['AO1_short']=ta.momentum.AwesomeOscillatorIndicator(high=dt['High'], low=dt['Low'], window1=param['window11_short'], window2=param['window12_short'], fillna= False).awesome_oscillator().diff(param['diff1_short'])   

                dt.loc[ (dt['AO_long'] > 0) & (dt['AO1_long'] > 0) & (dt['DayOfWeek']==i) &  (dt['Tokyo Session']==1), 'Position'] = dt['Position'] + 1 
                dt.loc[ (dt['AO_short'] < 0) & (dt['AO1_short'] < 0) & (dt['DayOfWeek']==i) &  (dt['Tokyo Session']==1), 'Position'] = dt['Position'] - 1 

        return dt['Position']


def label(df):
    data=df.copy()
    data['Position']=0


    for i in range(5):
        data['ma_long']=data['Close'].rolling(6).mean()
        data['slope_long']=data['ma_long'].diff(6)
        data['ma_short']=data['Close'].rolling(6).mean()
        data['slope_short']=data['ma_short'].diff(6)


        entry_long=  (data['ma_long'] > data["Close"] ) & (data['slope_long'] > 0) & (data['DayOfWeek']==i) & (data['Tokyo Session']==1)
        entry_short= (data['ma_short'] < data["Close"] ) & (data['slope_short'] < 0) & (data['DayOfWeek']==i) & (data['Tokyo Session']==1)
        data=data[data['Tokyo Session']==1]

        data.loc[entry_long , 'Position'] = +1
        data.loc[entry_short, 'Position'] = -1
    return  data['Position'].shift(-6).fillna(0)

def feature_engineering(df):

    df=determine_session(df)
    df['Position_MACD']=MACD(df)
    df['Position_AO']=AO(df)
    df['Position_RSI']=Stochastic_RSI(df)

    df['trend_1h']=df['Close'].pct_change(4)
    df['trend_4h']=df['Close'].pct_change(4*4)
    df['trend_8h']=df['Close'].pct_change(4*8)
    df['trend_day']=df['Close'].pct_change(4*24)
    df['trend_week']=df['Close'].pct_change(4*24*5)
    df['trend_month']=df['Close'].pct_change(4*24*5*4)
    df['label']=label(df)
    df=df[df['Tokyo Session']==1]
    return df[['Position_MACD'	,'Position_AO'	,'Position_RSI'	,'trend_1h'	,'trend_4h'	,'trend_8h'	,'trend_day',	'trend_week',	'trend_month' ,'Date','Close','label']].dropna()


class PricePredictionDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.features = data.iloc[:, :-3].values
        self.labels = data.iloc[:, -1].values

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.labels[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
def convert_to_classes(predictions, threshold_positive, threshold_negative):
    classes = np.zeros_like(predictions)
    classes[predictions > threshold_positive] = 1
    classes[predictions < threshold_negative] = -1
    return classes

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Connect_to_MT5()

    name = 'EURUSD_vip1'
    PAIR = 'EURUSD'
    path_csv = f'C:/Users/Administrator/Documents/ManhNH/PRV/FX/data/{name}.csv'
    time_frame = '15T'
    list_time15m = [f'{hour:02d}:{minute:02d}:55' for hour in range(
        0, 24) for minute in range(14, 60, int(time_frame[:-1]))]
    # print(list_time15m)
    flag_short = False
    flag_long = False
    order = None

    while True:

        time_now = datetime.now()

        if time_now.strftime('%H:%M:%S') in list_time15m:
        # if True:

            df = get_data_real_time(PAIR, 30, '15min')
            df = feature_engineering(df)
            
            predict_dataset = PricePredictionDataset(df, 60)

            predict_loader = DataLoader(predict_dataset, batch_size=64, shuffle=True)

            input_size = predict_dataset.features.shape[1]
            hidden_size = 50
            output_size = 1  # Dự đoán vị thế
            num_layers = 2

            model = LSTMModel(input_size, hidden_size, output_size, num_layers)
            model_path = 'C:\\Users\\Administrator\\Documents\\ManhNH\\PRV\\FX\\model\\lstm_model.pth'
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            # Tải trọng số mô hình từ model_state_dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            predictions = []
            # actuals = []
            with torch.no_grad():
                for inputs, targets in predict_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    predictions.append(outputs.cpu().numpy())
                    # actuals.append(targets.numpy())

            predictions = np.concatenate(predictions)
            # actuals = np.concatenate(actuals)

            threshold_positive=0.6
            threshold_negative=-0.7

            predicted_classes = convert_to_classes(predictions, threshold_positive, threshold_negative)
            predict_df=df.reset_index(drop=True).loc[60:,:].reset_index(drop=True)
            predict_df['Position'] = pd.Series(predicted_classes.flatten())


            df_csv, inputPos, CP = DumpCSV_and_MesToTele(
                name, path_csv, predict_df['Position'], predict_df.Close, token=BOT_ID, id=GROUP_ID, fee=0)
            print('-------------------------------')

            lots = int(abs(1))

            if (predict_df['Position'].iloc[-1] - predict_df['Position'].iloc[-2]) != 0:
                if flag_long:
                    close_buy(position, lots, PAIR)
                    flag_long = False
                elif flag_short:
                    close_sell(position, lots, PAIR)
                    flag_short = False

            elif predict_df['Position'].iloc[-1] == 1 and not flag_long and not flag_short:
                position = open_buy(lots)
                flag_long = True
            elif predict_df['Position'].iloc[-1] == 1 and flag_long:
                continue

            elif predict_df['Position'].iloc[-1] == -1 and not flag_short and not flag_long:
                position = open_sell(lots)
                flag_short = True
            elif predict_df['Position'].iloc[-1] == -1 and flag_short:
                continue

            else:
                continue

            time.sleep(1)
