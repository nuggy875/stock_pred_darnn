import numpy as np
import math

def logret(X):
    log_ret = np.zeros_like(X)
    log_ret[0] = 0
    for i in range(1, X.shape[0]):
        log_ret[i] = math.log(X[i] / X[i-1])
    return log_ret


class Loader:
    def __init__(self, data_path, filename, window_size, data_type):
        if filename.split('.')[0]=='KOSPI':
            print('Using KOSPI200 Data')
            adjusted_close = np.genfromtxt(data_path+filename, delimiter = ',', skip_header = 1, usecols = (4))
        else:
            adjusted_close = np.genfromtxt(data_path+filename, delimiter = ',', skip_header = 1, usecols = (5))
        if data_type == 'log':
            log_return = logret(adjusted_close) 
        elif data_type == 'price':
            log_return = adjusted_close
        elif data_type == 'volat':
            print('volatality is not ready...')
            exit()

        self.train_size = log_return.shape[0] // window_size

        self.log_return = log_return[:self.train_size * window_size]
        self.X = self.log_return.reshape(self.train_size, window_size)
