import datetime
import math
import Calculated_Implied_Vol as CIV
import pandas as pd
from warnings import simplefilter

"""
Reference: Some of codes from Extract_Next_Monthly_Maturity is referred from:
        https://github.com/yhilpisch/eurexas/blob/master/vstoxx/scripts/index_date_functions.py
"""
"""
Stage: Stage 2 file
Document type: Data collection + Preprocessing only.

Need to run? No.

Main purpose: to extract the option data from the API

Dependency:
    use -> Calculated_Implied_Vol.py
    be used -> toy_model.py

Methods:
    Extract_Next_Monthly_Maturity(date: str) -> datetime.datetime: to extract the next third Friday in a month
    Extract_Most_Close_Price(price : float, interval : int) -> float: to find the closest strike price
    Request_Option_Info(price : float, maturity : datetime.datetime, symbol : str, callorput: str) -> list: This method is to get the history price of the option
    
    Extract_Option_Info(symbol: str, price_open : float, price_close : float, dates:str): 
    Extract_Option_Close_Info(symbol: str, price_open : float, price_close : float, dates:str):
        These two methods are the similar, to produce the implied volatility, the option price, and the Greeks 
        The difference is, the first one only produce the Greeks of open price, another is only producing the Greeks of close price.
"""


def Extract_Next_Monthly_Maturity(date: str) -> datetime.datetime:
    """
    This method is to extract the next third Friday of the month
    Reference from: https://github.com/yhilpisch/eurexas/blob/master/vstoxx/scripts/index_date_functions.py
    para:
        @date: the input date
    return:
        @output: the next third friday of the month.
    """
    import datetime as dt
    year = int(date[:date.find("-")])
    month = int(date[date.find("-")+1 : date.rfind("-")])
    day = int(date[date.rfind("-")+1:])
    new_date = dt.datetime(year=year,month=month,day=day)



    def third_friday(date):
        ''' Returns the third friday of the month given by the datetime object date
        This is the day options expiry on.
        :param date: datetime object
        :return: the next third friday in a month
            date of month for which third Friday is to be found
        '''

        number_days = date.day
        first_day = date - dt.timedelta(number_days - 1)
          # Reduce the given date to the first of the month.
          # Year and month stay the same.
        week_day = first_day.weekday()
          # What weekday is the first of the month (Mon=0, Tue=1, ...)
        day_delta = 4 - week_day  # distance to the next Friday
        if day_delta < 0:
            day_delta += 7
        third_friday = first_day + dt.timedelta(day_delta + 14)
          # add that distance plus two weeks to the first of month
        return third_friday


    def first_settlement_day(date):
        ''' Returns the next settlement date (third Friday of a month) following
        the date date.
        date: datetime object
            date for which following third Friday is to be found
        '''

        settlement_day_in_month = third_friday(date)
          # settlement date in the given month

        delta = (settlement_day_in_month - date).days
          # where are we relative to the settlement date in that month?

        if delta > 1:  # more than 1 day before ?
            return settlement_day_in_month
             # yes: take the settlement dates of this and the next month
        else:
            next_month = settlement_day_in_month + dt.timedelta(20)
              # no: shift the date of next month into the next month but one and ...
            settlement_day_next_month = third_friday(next_month)
              # ... compute that settlement day
            return settlement_day_next_month

    return first_settlement_day(new_date)

def Extract_Most_Close_Price(price : float, interval : int) -> float:
    """
    This method is to find the closet strike price
    @para:
        @price : the recent price
        @interval : to distinguish the interval for different strike price, to ensure we could find the most closet one.
    @return:
        @output: the most cloest strike price as try
    """
    if interval == 0.5:
        if price < 50:
            a = int(price)
            b = int(price)+1
            c = int(price) + 0.5
            if math.fabs(a-price) < math.fabs(b-price):
                if math.fabs(a-price) > math.fabs(c-price):
                    return c
                else:
                    return a
            else:
                if math.fabs(b-price) > math.fabs(c-price):
                    return c
                else:
                    return b
        else:
            return Extract_Most_Close_Price(price, 1)
    if interval == 1:
        if price < 100:
            return float(round(price))
        else:
            return Extract_Most_Close_Price(price, 2.5)
    else:
        round_price = round(price, -1)
        if interval == 2.5:
            if price < 125:
                if price >= round_price:
                    if math.fabs(price-round_price) <= 1.25:
                        return round_price
                    if math.fabs(price-round_price-2.5) <= 1.25:
                        return round_price + 2.5
                    else:
                        return round_price + 5
                else:
                    if math.fabs(price-round_price) <= 1.25:
                        return round_price
                    if math.fabs(price-round_price-2.5) <= 1.25:
                        return round_price - 2.5
                    else:
                        return round_price - 5
            else:
                return Extract_Most_Close_Price(price, 5)
        if interval == 5:
            if price < 250:
                if price >= round_price:
                    if math.fabs(price-round_price) <= 2.5:
                        return round_price
                    else:
                        return round_price + 5
                else:
                    if math.fabs(price-round_price) <= 2.5:
                        return round_price
                    else:
                        return round_price - 5
            else:
                return Extract_Most_Close_Price(price, 10)
        if interval == 10:
            return round(price, -1)
        if interval == 20:
            if round(price, -1) > price:
                return round(price, -1) - 10
            else:
                return round(price, -1) + 10

def Request_Option_Info(price : float, maturity : datetime.datetime, symbol : str, callorput: str) -> list:
    """
    This method is to get the history price of the option
    @para:
        @price: Strike Price
        @maturity : the maturity date
        @symbol: the stock name
    @return:
        @output: list of option history data
    """
    import requests

    url = "https://quotient.p.rapidapi.com/options/historical"

    querystring = {"symbol": symbol,"type":callorput,"expiration":str(maturity.date()),"strike":str(price)}

    headers = {
        "X-RapidAPI-Key": "5c6e6d2a33msh9a7d4240d33299ep12de8fjsne14b52e19c30",
        "X-RapidAPI-Host": "quotient.p.rapidapi.com"
    }
    disconnect = False
    while disconnect == False:
        try:
            response = requests.request("GET", url, headers=headers, params=querystring, timeout = 10)
            disconnect = True
        except requests.exceptions.RequestException as e:
            print(e)

    a = response.json()

    return a

def Extract_Option_Info(symbol: str, price_open : float, price_close : float, dates:str):
    """
    This is the main method to extract the option price data
    @para:
        @symbol: the company symbol
        @price: The stock price recently
        @data: the closest trading date
    @output: call: [Implied volatilty (open), Implied volatility (close), Option price (open), Option price (close), Greeks...]
            put: [Implied volatilty (open), Implied volatility (close), Option price (open), Option price (close), Greeks...]

    """
    from datetime import date

    def time_difference(Expiration, Current_day):
        year = int(Expiration[:Expiration.find("-")])
        month = int(Expiration[Expiration.find("-")+1 : Expiration.rfind("-")])
        day = int(Expiration[Expiration.rfind("-")+1:])
        #print(year,month,day)
        adate = datetime.date(year=year,month=month,day=day)

        year1 = int(Current_day[:Current_day.find("-")])
        month1 = int(Current_day[Current_day.find("-")+1 : Current_day.rfind("-")])
        day1 = int(Current_day[Current_day.rfind("-")+1:])
        adates = datetime.date(year=year1,month=month1,day=day1)
        return (adate - adates).days

    def string_to_date(dates : str):
        year = int(dates[:dates.find("-")])
        month = int(dates[dates.find("-")+1 : dates.rfind("-")])
        day = int(dates[dates.rfind("-")+1:])
        #print(year,month,day)
        return datetime.date(year=year,month=month,day=day)

    def corresponding_line (Date: date) -> int:
        start = 0
        if Date.year == 2022:
            return 8 - Date.month
        else:
            return 8 + (2021 - Date.year) * 12 + (12 - Date.month)

    maturity = Extract_Next_Monthly_Maturity(dates)
    success_extraction = False
    interval_choice = [0.5,1, 2.5, 5, 10,20]
    n = 0
    while success_extraction is False:
        closet_price = Extract_Most_Close_Price(price_open, interval_choice[n])
        output_call = Request_Option_Info(closet_price, maturity, symbol, 'Call')
        output_put = Request_Option_Info(closet_price, maturity, symbol, 'Put')

        if len(output_call) == 0 or len(output_put) == 0:
            if n == 5:
                return None
            n += 1

        else:
            success_extraction = True

    df = pd.read_csv("AvgInterestRate_20170831_20220831.csv")

    call_extraction = False
    put_extraction = False
    for each in range(0, len(output_call)):
        if output_call[each]['Date'].__eq__(dates) :
            Option_call_open = output_call[each]['Open']
            Option_call_close = output_call[each]['Close']
            sigma0 = 0.1
            t = time_difference(output_call[each]['Expiration'],output_call[each]['Date']) / 365
            r = df.loc[corresponding_line(string_to_date(dates))]['Average Interest Rate Amount'] * 0.01
            try:
                simplefilter('error')
                open_call_IV = CIV.GetImpVol(sigma0,Option_call_open,price_open,output_call[each]['Strike'],t,r,0,CIV.BlackScholesCall,"fsolve")
                cLose_call_IV = CIV.GetImpVol(sigma0,Option_call_close,price_close,output_call[each]['Strike'],t - (1/365),r,0,CIV.BlackScholesCall,"fsolve")
            except RuntimeWarning as e:
                simplefilter('default')
                print(e)
                return None
            delta_call =CIV.Delta(open_call_IV, price_open, output_call[each]['Strike'], t,r, 0, CIV.BlackScholesCall)[0]
            gamma_call = CIV.Gamma(open_call_IV, price_open, output_call[each]['Strike'], t,r, 0, CIV.BlackScholesCall)[0]
            vega_call = CIV.Vega(open_call_IV, price_open, output_call[each]['Strike'], t,r, 0, CIV.BlackScholesCall)[0]
            theta_call = CIV.Theta(open_call_IV, price_open, output_call[each]['Strike'], t,r, 0, CIV.BlackScholesCall)[0]
            call_extraction = True
            print(closet_price,price_open,price_close,open_call_IV,cLose_call_IV)
        #except:
         #   continue
    for each in range(0, len(output_put)):
        if output_put[each]['Date'].__eq__(dates):
            Option_put_open = output_put[each]['Open']
            Option_put_close = output_put[each]['Close']
            sigma0 = 0.1
            t = time_difference(output_call[each]['Expiration'],output_call[each]['Date']) / 365
            r = df.loc[corresponding_line(string_to_date(dates))]['Average Interest Rate Amount'] * 0.01
            try:
                simplefilter('error')
                open_put_IV = CIV.GetImpVol(sigma0,Option_put_open,price_open,output_call[each]['Strike'],t,r,0,CIV.BlackScholesPut,"fsolve")
                cLose_put_IV = CIV.GetImpVol(sigma0,Option_put_close,price_close,output_call[each]['Strike'],t - (1/365),r,0,CIV.BlackScholesPut,"fsolve")
            except RuntimeWarning as e:
                simplefilter('default')
                print(e)
                return None
            delta_put =CIV.Delta(open_put_IV, price_open, output_put[each]['Strike'], t,r, 0, CIV.BlackScholesPut)[0]
            gamma_put = CIV.Gamma(open_put_IV, price_open, output_put[each]['Strike'], t,r, 0, CIV.BlackScholesPut)[0]
            vega_put = CIV.Vega(open_put_IV, price_open, output_put[each]['Strike'], t,r, 0, CIV.BlackScholesPut)[0]
            theta_put = CIV.Theta(open_put_IV, price_open, output_put[each]['Strike'], t,r, 0, CIV.BlackScholesPut)[0]

            put_extraction = True
            print(open_put_IV,cLose_put_IV)

    #Sturcture of the output price
    if call_extraction and put_extraction:
        output = [open_call_IV[0],cLose_call_IV[0],Option_call_open,Option_call_close,delta_call,gamma_call,vega_call,theta_call,open_put_IV[0],cLose_put_IV[0],Option_put_open,Option_put_close,delta_put,gamma_put,vega_put,theta_put]
        return output
    else:
        return None


def Extract_Option_Close_Info(symbol: str, price_open : float, price_close : float, dates:str):
    """
    This is the main method to extract the option price data
    @para:
        @symbol: the company symbol
        @price: The stock price recently
        @data: the closest trading date
    @output: call: [Implied volatilty (open), Implied volatility (close), Option price (open), Option price (close), Greeks...]
            put: [Implied volatilty (open), Implied volatility (close), Option price (open), Option price (close), Greeks...]
    """
    from datetime import date

    def time_difference(Expiration, Current_day):
        year = int(Expiration[:Expiration.find("-")])
        month = int(Expiration[Expiration.find("-")+1 : Expiration.rfind("-")])
        day = int(Expiration[Expiration.rfind("-")+1:])
        #print(year,month,day)
        adate = datetime.date(year=year,month=month,day=day)

        year1 = int(Current_day[:Current_day.find("-")])
        month1 = int(Current_day[Current_day.find("-")+1 : Current_day.rfind("-")])
        day1 = int(Current_day[Current_day.rfind("-")+1:])
        adates = datetime.date(year=year1,month=month1,day=day1)
        return (adate - adates).days

    def string_to_date(dates : str):
        year = int(dates[:dates.find("-")])
        month = int(dates[dates.find("-")+1 : dates.rfind("-")])
        day = int(dates[dates.rfind("-")+1:])
        #print(year,month,day)
        return datetime.date(year=year,month=month,day=day)

    def corresponding_line (Date: date) -> int:
        start = 0
        if Date.year == 2022:
            return 8 - Date.month
        else:
            return 8 + (2021 - Date.year) * 12 + (12 - Date.month)

    maturity = Extract_Next_Monthly_Maturity(dates)
    success_extraction = False
    interval_choice = [0.5,1, 2.5, 5, 10,20]
    n = 0
    while success_extraction is False:
        closet_price = Extract_Most_Close_Price(price_open, interval_choice[n])
        output_call = Request_Option_Info(closet_price, maturity, symbol, 'Call')
        output_put = Request_Option_Info(closet_price, maturity, symbol, 'Put')
        if len(output_call) == 0 or len(output_put) == 0:
            if n == 5:
                return None
            n += 1

        else:
            success_extraction = True


    df = pd.read_csv("AvgInterestRate_20170831_20220831.csv")

    call_extraction = False
    put_extraction = False
    for each in range(0, len(output_call)):
        if output_call[each]['Date'].__eq__(dates) :
            Option_call_open = output_call[each]['Open']
            Option_call_close = output_call[each]['Close']
            sigma0 = 0.1
            t = time_difference(output_call[each]['Expiration'],output_call[each]['Date']) / 365
            r = df.loc[corresponding_line(string_to_date(dates))]['Average Interest Rate Amount'] * 0.01
            try:
                simplefilter('error')
                open_call_IV = CIV.GetImpVol(sigma0,Option_call_open,price_open,output_call[each]['Strike'],t,r,0,CIV.BlackScholesCall,"fsolve")
                cLose_call_IV = CIV.GetImpVol(sigma0,Option_call_close,price_close,output_call[each]['Strike'],t - (1/365),r,0,CIV.BlackScholesCall,"fsolve")
            except RuntimeWarning as e:
                simplefilter('default')
                print(e)
                return None
            delta_call =CIV.Delta(cLose_call_IV, price_close, output_call[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesCall)[0]
            gamma_call = CIV.Gamma(cLose_call_IV, price_close, output_call[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesCall)[0]
            vega_call = CIV.Vega(cLose_call_IV, price_close, output_call[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesCall)[0]
            theta_call = CIV.Theta(cLose_call_IV, price_close, output_call[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesCall)[0]
            call_extraction = True
            print(closet_price,price_open,price_close,open_call_IV,cLose_call_IV)
        #except:
         #   continue
    for each in range(0, len(output_put)):
        if output_put[each]['Date'].__eq__(dates):
            Option_put_open = output_put[each]['Open']
            Option_put_close = output_put[each]['Close']
            sigma0 = 0.1
            t = time_difference(output_call[each]['Expiration'],output_call[each]['Date']) / 365
            r = df.loc[corresponding_line(string_to_date(dates))]['Average Interest Rate Amount'] * 0.01
            try:
                simplefilter('error')
                open_put_IV = CIV.GetImpVol(sigma0,Option_put_open,price_open,output_call[each]['Strike'],t,r,0,CIV.BlackScholesPut,"fsolve")
                cLose_put_IV = CIV.GetImpVol(sigma0,Option_put_close,price_close,output_call[each]['Strike'],t - (1/365),r,0,CIV.BlackScholesPut,"fsolve")
            except RuntimeWarning as e:
                simplefilter('default')
                print(e)
                return None
            delta_put =CIV.Delta(cLose_put_IV, price_close, output_put[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesPut)[0]
            gamma_put = CIV.Gamma(cLose_put_IV, price_close, output_put[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesPut)[0]
            vega_put = CIV.Vega(cLose_put_IV, price_close, output_put[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesPut)[0]
            theta_put = CIV.Theta(cLose_put_IV, price_close, output_put[each]['Strike'], t - (1/365),r, 0, CIV.BlackScholesPut)[0]

            put_extraction = True
            print(open_put_IV,cLose_put_IV)

    #Sturcture of the output price
    if call_extraction and put_extraction:
        output = [open_call_IV[0],cLose_call_IV[0],Option_call_open,Option_call_close,delta_call,gamma_call,vega_call,theta_call,open_put_IV[0],cLose_put_IV[0],Option_put_open,Option_put_close,delta_put,gamma_put,vega_put,theta_put]
        return output
    else:
        return None


