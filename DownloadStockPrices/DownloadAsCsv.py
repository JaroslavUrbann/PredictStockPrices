import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import csv
import requests
import os.path
import time

ts = TimeSeries(key='DWCHWE05UICT02QO', output_format='csv')
# ti = TechIndicators(key='DWCHWE05UICT02QO', output_format='csv')

# with requests.Session() as s:
with open("1500 companies.csv", "r") as companies:
    companieslist = csv.reader(companies)
    for row in companieslist:
        if os.path.isfile("Companies/" + row[0] + ".csv"):
            continue

        data1, meta_data = ts.get_daily_adjusted(symbol=row[0], outputsize='full')
        print(row[0])
        print(data1)
        time.sleep(15)

        with open("Companies/" + row[0] + ".csv", "w", newline="") as newfile:
            csv_writer = csv.writer(newfile)
            for row1 in data1:
                # if row1 and row2 and row3 and row4 and row5 and row6 and row7 and row8 and row9 and row10:
                csv_writer.writerow(row1)

        # download = s.get("https://www.alphavantage.co/query?function=MACD&symbol=" + row[0] + "&interval=daily&series_type=close&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data2 = csv.reader(decoded_content.splitlines(), delimiter=',')
        # # jaký další indikátory???
        # download = s.get("https://www.alphavantage.co/query?function=STOCH&symbol=" + row[0] + "&interval=daily&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data3 = csv.reader(decoded_content.splitlines(), delimiter=',')
        #
        # download = s.get("https://www.alphavantage.co/query?function=RSI&symbol=" + row[0] + "&interval=daily&time_period=10&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data4 = csv.reader(decoded_content.splitlines(), delimiter=',')
        #
        # download = s.get("https://www.alphavantage.co/query?function=ADX&symbol=" + row[0] + "&interval=daily&time_period=14&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data5 = csv.reader(decoded_content.splitlines(), delimiter=',')
        #
        # download = s.get("https://www.alphavantage.co/query?function=AROON&symbol=" + row[0] + "&interval=daily&time_period=25&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data6 = csv.reader(decoded_content.splitlines(), delimiter=',')
        #
        # download = s.get("https://www.alphavantage.co/query?function=BBANDS&symbol=" + row[0] + "&interval=daily&series_type=close&time_period=20&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data7 = csv.reader(decoded_content.splitlines(), delimiter=',')
        #
        # download = s.get("https://www.alphavantage.co/query?function=ATR&symbol=" + row[0] + "&interval=daily&time_period=14&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data8 = csv.reader(decoded_content.splitlines(), delimiter=',')
        #
        # download = s.get("https://www.alphavantage.co/query?function=AD&symbol=" + row[0] + "&interval=daily&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data9 = csv.reader(decoded_content.splitlines(), delimiter=',')
        #
        # download = s.get("https://www.alphavantage.co/query?function=RSI&symbol=" + row[0] + "&interval=daily&datatype=csv&apikey=DWCHWE05UICT02QO")
        # decoded_content = download.content.decode('utf-8')
        # data10 = csv.reader(decoded_content.splitlines(), delimiter=',')

        # with open(row[0] + ".csv", "w", newline="") as newfile:
        #     csv_writer = csv.writer(newfile)
        #     for row1, row2, row3, row4, row5, row6, row7, row8, row9, row10 in zip(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10):
        #         # if row1 and row2 and row3 and row4 and row5 and row6 and row7 and row8 and row9 and row10:
        #         csv_writer.writerow(row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8 + row9 + row10)
        # break