import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch


def binary_result(t):
    val = t.numpy()[0][0]
    if val >= 0:
        return 1
    else:
        return 0


def data_process_price(X, train_size, num_steps):
    X_result = X[:num_steps, :, :]  # [30, 1, 3]
    Y_result = X[num_steps, :, :]   #     [1, 3]
    for s in range(1, train_size - num_steps):
        X_result = torch.cat((X_result, X[s : s + num_steps, :, :]), dim = 1)
        Y_result = torch.cat((Y_result, X[s + num_steps, :, :]), dim = 0)
    return X_result, Y_result


def data_process_log(X, train_size, num_steps):
    y_target=[]
    X_result = X[:num_steps, :, :]  # [30, 1, 1]
    X_last = X[num_steps-1, :, :]
    Y_result = X[num_steps, :, :]   #     [1, 1]
    y_target.append(binary_result(Y_result))
    for s in range(1, train_size - num_steps):
        X_result = torch.cat((X_result, X[s : s + num_steps, :, :]), dim = 1)
        y_result = X[s + num_steps, :, :]
        X_last = torch.cat((X_last, X[s + num_steps - 1, :, :]), dim = 0)
        y_target.append(binary_result(y_result))
    Y_target = torch.FloatTensor(y_target)
    return X_result, Y_target