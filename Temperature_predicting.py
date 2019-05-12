import time
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
def predictTemperature(startDate, endDate, temperature, n):
    # Write your code here
    y = temperature
    # first, I converted date/time to values
    start_time = time.mktime(datetime.datetime.strptime(startDate+' 00:00', "%Y-%m-%d %H:%M").timetuple())
    end_time = time.mktime(datetime.datetime.strptime(endDate+' 23:00', "%Y-%m-%d %H:%M").timetuple())
    interval = (end_time - start_time)/(len(temperature)-1) # representative interval for each hour
    x = []
    temp = start_time
    while temp<end_time:
        x.append([temp])
        temp += interval
    x.append([temp])
    #print(x)
    reg = LinearRegression().fit(x, y)  #use a linear regression to predict temperature

    end_time_predict = n*24*interval + end_time
    temp = end_time
    x_predict = []
    while temp<end_time_predict:
        #x_predict.append([temp])
        temp += interval
        x_predict.append([temp])
    #x_predict.append([temp])
    #print(x_predict)
    y_predict = reg.predict(x_predict)
    return y_predict