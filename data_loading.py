import numpy as np
import math
import pandas as pd
from pandas import datetime
from option import opt

def logret(X):
    log_ret = np.zeros_like(X)
    log_ret[0] = 0
    for i in range(1, X.shape[0]):
        log_ret[i] = math.log(X[i] / X[i-1])
    return log_ret


class DataLoader:
    def __init__(self):
        dates = pd.date_range(opt.date_start, opt.date_end)
        pf = pd.DataFrame(index=dates)
        df = pd.read_csv(opt.data_path+opt.dataset+'.csv',
                      index_col='Date',
                      parse_dates=True,
                      usecols=['Date', 'Close'],
                      na_values=['nan'])
        df = df.rename(columns={'Close': opt.dataset})
        data = np.ravel(df.to_numpy(), order='C')

        if opt.data_type == 'log':
            data_return = logret(data) 
        elif opt.data_type == 'price':
            data_return = data
        elif opt.data_type == 'volat':
            print('ERROR::Volatality is not ready...')
            exit()

        self.train_size = data_return.shape[0] // opt.window_size

        self.data_return = data_return[:self.train_size * opt.window_size]
        self.X = self.data_return.reshape(self.train_size, opt.window_size)

