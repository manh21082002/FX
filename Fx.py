
import requests
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from datetime import timedelta, time, date
from time import mktime
import matplotlib.pyplot as plt
def send_to_telegram(message, token, id):
    ''' Gửi tin nhắn đến telegram '''
    ''' Input: message: tin nhắn muốn gửi
            token: token của bot
                id: id của chat group '''
    apiToken = token
    chatID = id
    try:
        apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage?chat_id={chatID}&text={message}"
        requests.get(apiURL).json()
    except Exception as e:
        print(e)




def portfolio_pnl_future(position_long, position_short, Close):
    ''' t�nh PNL c?a m?t chi?n thu?t 
    position_long: series position long
    position_short: series position short'''
    intitial_capital_long = (position_long.iloc[0])*(Close.iloc[0])
    cash_long = (position_long.diff(1)*Close) 
    cash_long[0] = intitial_capital_long
    # gain_long = cash_long/Close
    cash_cs_long = cash_long.cumsum()
    # total_gain_long = gain_long.cumsum()
    portfolio_value_long = (position_long*Close)

    intitial_capital_short = (position_short.iloc[0])*(Close.iloc[0])
    cash_short = (position_short.diff(1)*Close)
    cash_short[0] = intitial_capital_short
    # gain_short = cash_short/Close 
    # total_gain_short = gain_short.cumsum()
    cash_cs_short = cash_short.cumsum()
    portfolio_value_short = (position_short*Close)

    backtest = ((portfolio_value_long - cash_cs_long) + (cash_cs_short - portfolio_value_short)) # gain theo di?m c?a chi?n thu?t 
    # total_gain = backtest/Close
    backtest.fillna(0, inplace=True)
    cash_max = (cash_long + cash_short).max()
    gain = backtest.diff()/Close  
    pnl = gain.cumsum()
    # pnl[0] = 0
    # pnl =  backtest/Close # total gain ? d?ng %
    
    ''' return PNL, l?n v�o l?nh l?n nh?t, PNL tuong d?i theo % '''
    return backtest, cash_max, pnl

def Sharp(pnl):
    ''' T�nh Sharp ratio '''
    pnl_daily = pnl.resample("1D").last().dropna()
    r = pnl_daily.diff(1)
    return r.mean()/r.std() * np.sqrt(252)

def maximum_drawdown_future(pnl):
    ''' 
    pnl: c?t gain t�nh ? d?ng %
    T�nh maximum drawdown theo di?m, theo % 
    '''
    pnl_daily = pnl.resample("1D").last().dropna()
    # gain = pnl_daily.diff(1)
    # total_gain = pnl.cumsum()
    total_gain_max = pnl_daily.cummax()
    return (total_gain_max - pnl_daily).max()
    # return (gain.cumsum().cummax() - gain.cumsum()).max(), (gain.cumsum().cummax() - gain.cumsum()).max()/cash_max

def Margin(test): 
    ''' T�nh Margin '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test['inLong'] = test.signal_long.diff()[test.signal_long.diff() > 0].astype(int)
    test['inShort'] = test.signal_short.diff()[test.signal_short.diff() > 0].astype(int)
    test['outLong'] = -test.signal_long.diff()[test.signal_long.diff() < 0].astype(int)
    test['outShort'] = -test.signal_short.diff()[test.signal_short.diff() < 0].astype(int)
    test.loc[test.index[0], 'inLong'] = test.signal_long.iloc[0]
    test.loc[test.index[0], 'inShort'] = test.signal_short.iloc[0]
    test.fillna(0, inplace=True)

    ''' return dataframe chua th�m c�c c?t inLong, inShort, outLong, outShort v� Margin '''
    return test, test.total_gain.iloc[-1]/(test.inLong * test.Close + test.inShort * test.Close + test.outLong * test.Close + test.outShort * test.Close).sum()*10000

def HitRate(test):
    ''' T�nh Hit Rate '''
    test = test.copy()
    try:
        test['signal_long'] = np.where(test['Position'] > 0, 1, 0)
        test['signal_short'] = np.where(test['Position'] < 0, 1, 0)
    except:
        pass
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test = Margin(test)[0]
    test = test[(test.outLong > 0) | (test.outShort > 0) | (test.inLong > 0) | (test.inShort > 0)]
    test['total_gain'] = portfolio_pnl_future(test.signal_long, test.signal_short, test.Close)[0]
    test.fillna(0, inplace=True)
    test['gain'] = test.total_gain.diff()
    test.fillna(0, inplace=True)
    test['gain'] = np.where(np.abs(test.gain) < 0.00001, 0, test.gain)
    try:
        ''' return dataframe thu g?n v� Hit Rate'''
        return test, len(test[test.gain > 0])/(len(test[test.inLong > 0]) + len(test[test.inShort > 0]))
    except:
        return 0









import calendar

def last_thursday_of_month(year, month):
    # Tìm ngày cuối cùng của tháng
    last_day = calendar.monthrange(year, month)[1]

    # Xác định ngày cuối cùng của tháng là thứ mấy
    last_day_of_month_weekday = calendar.weekday(year, month, last_day)

    # Tính toán số ngày cần di chuyển để đến thứ 5 (thứ 3 là 2, thứ 4 là 3, ..., thứ 7 là 6)
    days_to_last_thursday = (last_day_of_month_weekday - calendar.THURSDAY + 7) % 7

    # Lấy ngày thứ 5 cuối cùng của tháng
    last_thursday = last_day - days_to_last_thursday

    return date(year, month, last_thursday)

def last_thursdays_in_range(start_year, end_year):
    last_thursdays = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last_thursday = last_thursday_of_month(year, month)
            last_thursday=last_thursday - timedelta(days=2)
            last_thursdays.append(last_thursday)
    return last_thursdays






def Check_expiry():
    today = date.today()
    third_thursday = today.replace(day=1)

    while third_thursday.weekday() != 3:  # Thursday is 3
        third_thursday += timedelta(days=1)

    for i in range(2):
        third_thursday += datetime.timedelta(days=7)
    last_thursdays=last_thursdays_in_range(2024,2026)

    if today in last_thursdays:
        return True
    else:
        return False

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
            dict_data['Datetime'] = pd.to_datetime(dict_data['Datetime']).to_list()
        except:
            for i in range(len(dict_data['Datetime'])):
                dict_data['Datetime'][i] = pd.to_datetime(dict_data['Datetime'][i])
            dict_data['Datetime'] = list(dict_data['Datetime'])
        df = pd.DataFrame(data=dict_data)
    except:
        dict_data = {
            'Datetime': [pd.to_datetime((datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))],
            'Position': [0],
            'Close': [0],
            'total_gain': [0],
            'gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_intraday, index=False)

    Close = Close.iloc[-1]
    new_Pos = int(Position.iloc[-1])
    time_now = datetime.datetime.now()
    profit = 0
    profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()
    mes = f'{name}:'
    
    if new_Pos != dict_data['Position'][-1] or time_now.time() >= datetime.time(17, 0):

        if time_now.time() >= datetime.time(17, 0):
            new_Pos = dict_data['Position'][-1]
        
        if Check_expiry() and time_now.time() >= datetime.time(17, 0):
            new_Pos = 0

        inputPos = int(new_Pos - dict_data['Position'][-1])
        dict_data['Datetime'].append(pd.to_datetime(time_now.strftime('%Y-%m-%d %H:%M:%S')))
        dict_data['Close'].append(Close)
        dict_data['Position'].append(new_Pos)
        dict_data['total_gain'].append(0)
        dict_data['gain'].append(0)

        df = pd.DataFrame(data=dict_data)
        df['signal_long'] = np.where(df.Position > 0, df.Position, 0)
        df['signal_short'] = np.where(df.Position < 0, np.abs(df.Position), 0)
        df['total_gain'] = portfolio_pnl_future(df['signal_long'], df['signal_short'], df.Close)[0]
        df['gain'] = df.total_gain.diff()
        df.fillna(0, inplace=True)
        df['gain'] = np.where(np.abs(df.gain.to_numpy()) < 0.00001, 0, df.gain.to_numpy())
        df.loc[df['Position'].diff().fillna(0) != 0, 'gain'] = df.loc[df['Position'].diff() != 0, 'gain'] - np.abs(fee/2 * inputPos)
        # df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] = df.loc[np.abs(df['Position'].diff()) == 2, 'gain'] - fee
        df['total_gain'] = df.gain.cumsum()
        profit = df.gain.iloc[-1]
        profit_today = df.loc[df.Datetime.dt.date == time_now.date(), 'gain'].sum()
        
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
        send_to_telegram(mes, token, id)
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

def PNL_per_day(path_csv_daily, profit_today):
    ''' Ghi file csv PNL theo ng�y
        Input: path_csv_daily: du?ng d?n file csv PNL theo ng�y
               profit_today: Series profit_today c?a chi?n thu?t
               (L?y ra t? dataframe df c?a h�m DumpCSV_and_MesToTele)'''
    try:
        df = pd.read_csv(path_csv_daily)
        dict_data = {
            'Datetime': df.Datetime.tolist(),
            'gain': df.gain.tolist(),
        }
    except:
        dict_data = {
            'Datetime': [(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')],
            'gain': [0],
        }
        df = pd.DataFrame(data=dict_data)
        df.to_csv(path_csv_daily, index=False)

    gain = profit_today.iloc[-1]
    time_now = datetime.datetime.now()

    if time_now.strftime('%Y-%m-%d') != pd.to_datetime(dict_data['Datetime'][-1]).strftime('%Y-%m-%d'):
        if gain != dict_data['gain'][-1]:
            dict_data['Datetime'].append(time_now.strftime('%Y-%m-%d'))
            dict_data['gain'].append(gain)
            df = pd.DataFrame(data=dict_data)
    else:
        dict_data['gain'][-1] = gain
        df = pd.DataFrame(data=dict_data)

    df['total_gain'] = df['gain'].cumsum()
    df['total_gain'].apply(lambda x: np.round(x*10)/10)
    df.fillna(0, inplace=True)

    df.to_csv(path_csv_daily, index=False)
    ''' return dataframe PNL theo ng�y '''
    return df

class BacktestInformation:
    ''' Th�ng tin backtest c?a chi?n thu?t 
        Input: Datetime: Series Datetime
                Position: Series Position
                Close: Series Close '''
    ''' CH� �: N�n d�ng class n�y d? l?y du?c c�c th�ng tin c?a chi?n thu?t ch? kh�ng n�n d�ng c�c h�m ri�ng l?
               v� c�c h�m ri�ng l? ph�a tr�n c� th? c� d?nh d?ng position kh�ng d?ng nh?t v?i class n�y '''
    def __init__(self, Datetime, Position, Close, fee=0):
        signal_long = np.where(Position >= 0, Position, 0)
        signal_short = np.where(Position <= 0, np.abs(Position), 0)
        try:
            Datetime = pd.to_datetime(Datetime)
        except:
            Datetime = Datetime.to_list()
            for i in range(len(Datetime)):
                Datetime[i] = pd.to_datetime(Datetime[i])
        self.df = pd.DataFrame(data={'Datetime': Datetime, 'Position': Position, 'signal_long': signal_long, 'signal_short': signal_short, 'Close': Close})
        self.df.set_index('Datetime', inplace=True)
        self.hold_overnight = not (self.df.resample('D').last().dropna()['Position'] == 0).all()
        self.df.index = pd.to_datetime(self.df.index)
        self.df_brief = HitRate(self.df)[0]
        # self.fee = fee + 0.2*self.Trading_per_day()
        self.fee = 0
    
    def PNL(self):
        ''' T�nh PNL c?a chi?n thu?t '''
        total_gain, cash_max, pnl = portfolio_pnl_future(self.df.signal_long, self.df.signal_short, self.df.Close) 

        ''' return Series PNL, cash_max '''
        return total_gain, cash_max, pnl
    
    def Sharp(self):
        ''' T�nh Sharp c?a chi?n thu?t '''
        pnl = self.PNL()[2]
        return Sharp(pnl)
    
    # def Sharp_after_fee(self):
    #     ''' T�nh Sharp sau khi tr? ph� c?a chi?n thu?t '''
    #     return Sharp(self.Plot_PNL(plot=False)['total_gain_after_fee'].resample("1D").last().dropna())
    
    def Margin(self):
        ''' T�nh Margin c?a chi?n thu?t '''
        return Margin(self.df_brief)[1]
    
    def MDD(self):
        ''' T�nh MDD c?a chi?n thu?t '''
        pnl = self.PNL()[2]
        return maximum_drawdown_future(pnl)
    
    def Hitrate(self):
        ''' T�nh Hitrate c?a chi?n thu?t '''
        return HitRate(self.df_brief)[1]
    
    def Number_of_trade(self):
        ''' T�nh s? l?n giao d?ch c?a chi?n thu?t '''
        return len(self.df_brief[self.df_brief.inLong != 0]) + len(self.df_brief[self.df_brief.inShort != 0])
    
    def Profit_per_trade(self):
        ''' T�nh Profit trung b�nh c?a 1 giao d?ch '''
        return self.Plot_PNL(plot=False)['total_gain'].iloc[-1]/self.Number_of_trade()
    
    def Profit_after_fee(self):
        ''' T�nh Profit sau khi tr? ph� '''
        return np.round(self.Plot_PNL(plot=False)['total_gain_after_fee'].iloc[-1]*10)/10
    
    def Profit_per_day(self):
        ''' T�nh Profit trung b�nh theo ng�y '''
        return self.Profit_after_fee()/len(self.PNL()[0].resample("1D").last().dropna())
    
    def Trading_per_day(self):
        ''' T�nh s? l?n giao d?ch trung b�nh theo ng�y '''
        return self.Number_of_trade()/len(self.PNL()[0].resample("1D").last().dropna())
    
    def Hitrate_per_day(self):
        ''' T�nh Hitrate theo ng�y '''
        if self.PNL()[0].resample("1D").last().dropna().iloc[0] != 0:
            Profit = self.PNL()[0].resample("1D").last().dropna().diff()
            Profit.loc[Profit.index[0]] = self.PNL()[0].resample("1D").last().dropna().iloc[0]
        else:
            Profit = self.PNL()[0].resample("1D").last().dropna().iloc[1:].diff()
            Profit.loc[Profit.index[0]] = self.PNL()[0].resample("1D").last().dropna().iloc[1:].iloc[0]
        return Profit, len(Profit[Profit > 0])/len(Profit)

    def Return(self):
        ''' T�nh Return trung b�nh m?i nam (theo %) c?a chi?n thu?t '''
        # cash_max = self.PNL()[1]
        # return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D") / 365)/cash_max  
        
        pnl = self.PNL()[2]
        pnl_daily = pnl.resample("1D").last().dropna()
        r = pnl_daily.diff().dropna()
        # r = pnl.diff(1)
        # (len(self.PNL()[0]) 
        return r.mean()*252
    
    def Profit_per_year(self):
        ''' T�nh Profit trung b�nh theo nam '''
        return self.Profit_after_fee()/(len(self.PNL()[0].resample("1D").last()) / 365)

    def Plot_PNL(self, window_MA=None, plot=True):
        ''' Print th�ng tin v� V? bi?u d? PNL c?a chi?n thu?t 
            Input: after_fee: bool, True: plot c� tr? ph�, False: plot kh�ng tr? ph�'''

        total_gain, cash_max, pnl = self.PNL()
        total_gain = pd.DataFrame(total_gain.to_numpy(), index=total_gain.index, columns=['total_gain'])
        total_gain.loc[self.df['Position'].diff().fillna(0) != 0, 'fee'] = np.abs(self.fee/2 * self.df['Position'].diff().fillna(0))
        total_gain['fee'] = total_gain['fee'].fillna(0).cumsum()
        total_gain['total_gain_after_fee'] = total_gain['total_gain'] - total_gain['fee']

        if total_gain['total_gain'].resample('1D').last().dropna().iloc[0] != 0:
            total_gain.reset_index(inplace=True)
            previous_day = pd.DataFrame(total_gain.iloc[0].to_numpy(), index=total_gain.columns).T
            previous_day.loc[previous_day.index[0], 'Datetime'] = pd.to_datetime(previous_day['Datetime'].iloc[0]) - timedelta(days = 1) 
            previous_day.loc[previous_day.index[0], 'total_gain'] = 0
            total_gain = pd.concat([previous_day, total_gain]).set_index('Datetime')

        df_buy_hold = self.df_brief.copy() 
        df_buy_hold['gain'] = df_buy_hold.Close.diff()/df_buy_hold.Close
        df_buy_hold['pnl'] = df_buy_hold.gain.cumsum() + 1
        df_buy_hold = df_buy_hold.dropna()
        if plot:

            print('Margin:',Margin(self.df_brief)[1])
            print(f'MDD: {self.MDD()}\n')

            data = [('Total trading quantity', self.Number_of_trade()),
                    # ('Profit per trade',self.Profit_per_trade()),
                    # ('Total Profit', np.round(total_gain.total_gain.iloc[-1]*10)/10),
                    # ('Profit after fee', self.Profit_after_fee()),
                    ('Trading quantity per day', self.Number_of_trade()/len(total_gain.total_gain.resample("1D").last().dropna())),
                    # ('Profit per day after fee', self.Profit_per_day()),
                    ('Return per year', self.Return()),
                    # ('Profit per year', self.Profit_per_year()),
                    ('HitRate', self.Hitrate()),
                    ('HitRate per day', self.Hitrate_per_day()[1]),
                    ]
            for row in data:
                print('{:>25}: {:>1}'.format(*row))

            # total_gain[f'MA{window_MA}'] = total_gain['total_gain'].rolling(window_MA).mean().fillna(0)
            # (total_gain.total_gain.resample("1D").last().dropna()).plot(figsize=(15, 4), label= str(self.Sharp()))
            
            # if window_MA != None:
            #     (total_gain.total_gain.resample("1D").last().dropna().rolling(window_MA).mean()).plot(figsize=(15, 4), label=f'MA{window_MA}')
            # plt.grid()
            # plt.legend()
            # plt.xlabel('Time')
            # plt.ylabel('PNL')
            # plt.show()

            pnl = self.PNL()[2]
            plt.figure()
            (1 + pnl).plot(figsize=(15, 4), label= 'Alpha: ' +  str(self.Sharp()))
            (df_buy_hold.pnl.resample("1D").last().dropna()).plot(figsize=(15, 4), label= 'Buy and Hold: ' + str(Sharp(df_buy_hold.pnl.resample("1D").last().dropna())))

            plt.legend()
            plt.grid()
            plt.xlabel('Time')
            plt.ylabel('Return')
            plt.show()

        # self.df.set_index('Datetime', inplace=True)
        total_gain['Position'] = self.df['Position']
        total_gain['Close'] = self.df['Close']
        total_gain['Return'] = 1 + pnl
        total_gain.reset_index(inplace=True)

        return total_gain.set_index('Datetime')