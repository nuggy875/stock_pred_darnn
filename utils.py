import torch
from torch.autograd import Variable
from option import opt, DaRnnNet, TrainData, TrainConfig
import os
import numpy as np
import matplotlib.pyplot as plt

def get_result_bin(trend_dat, y_train_pred, y_test_pred, T, train_size, e_i):
    trend_gt_train = trend_dat[T-1:train_size]
    train_pos_score = 0
    train_neg_score = 0
    train_pos_total = 0
    train_neg_total = 0
    for i, val in enumerate(trend_gt_train):
        if val==1:
            train_pos_total += 1
            if y_train_pred[i][0]>=0.5:
                train_pos_score += 1
        else:
            train_neg_total += 1
            if y_train_pred[i][0]<0.5:
                train_neg_score += 1
    train_result_pos = train_pos_score / train_pos_total
    train_result_neg = train_neg_score / train_neg_total
    train_result_total = (train_pos_score + train_neg_score) / (train_pos_total + train_neg_total)

    # Testing Result
    trend_gt_test = trend_dat[train_size:]
    test_pos_score = 0
    test_neg_score = 0
    test_pos_total = 0
    test_neg_total = 0

    for i, val in enumerate(trend_gt_test):
        if val==1:
            test_pos_total += 1
            if y_test_pred[i][0]>=0.5:
                test_pos_score += 1
        else:
            test_neg_total += 1
            if y_test_pred[i][0]<0.5:
                test_neg_score += 1

    test_result_pos = test_pos_score / test_pos_total
    test_result_neg = test_neg_score / test_neg_total
    test_result_total = (test_pos_score + test_neg_score) / (test_pos_total + test_neg_total)

    f = open(os.path.join("saves", "score.txt"), "a")
    f.write("====== EPOCH:{} ======\n".format(e_i))
    f.write(">> Train\n")
    f.write("Positive Score: {}\n".format(train_result_pos))
    f.write("Negative Score: {}\n".format(train_result_neg))
    f.write("Total Score: {}\n".format(train_result_total))
    
    f.write(">> Test\n")
    f.write("Positive Score: {}\n".format(test_result_pos))
    f.write("Negative Score: {}\n".format(test_result_neg))
    f.write("Total Score: {}\n\n".format(test_result_total))
    
    f.close()


def get_result(trend_dat, y_train_pred, y_test_pred, T, train_size, e_i):

    # Training Result
    trend_gt_train = trend_dat[T-1:train_size]
    train_pos_score = 0
    train_neg_score = 0
    train_pos_total = 0
    train_neg_total = 0
    for i, val in enumerate(trend_gt_train):
        if val>=0:
            train_pos_total += 1
            if y_train_pred[i][0]>=0:
                train_pos_score += 1
        else:
            train_neg_total += 1
            if y_train_pred[i][0]<0:
                train_neg_score += 1

    train_result_pos = train_pos_score / train_pos_total
    train_result_neg = train_neg_score / train_neg_total
    train_result_total = (train_pos_score + train_neg_score) / (train_pos_total + train_neg_total)

    # Testing Result
    trend_gt_test = trend_dat[train_size:]
    test_pos_score = 0
    test_neg_score = 0
    test_pos_total = 0
    test_neg_total = 0

    for i, val in enumerate(trend_gt_test):
        if val>=0:
            test_pos_total += 1
            if y_test_pred[i][0]>=0:
                test_pos_score += 1
        else:
            test_neg_total += 1
            if y_test_pred[i][0]<0:
                test_neg_score += 1

    test_result_pos = test_pos_score / test_pos_total
    test_result_neg = test_neg_score / test_neg_total
    test_result_total = (test_pos_score + test_neg_score) / (test_pos_total + test_neg_total)

    f = open(os.path.join("saves", "score.txt"), "a")
    f.write("====== EPOCH:{} ======\n".format(e_i))
    f.write(">> Train\n")
    f.write("Positive Score: {}\n".format(train_result_pos))
    f.write("Negative Score: {}\n".format(train_result_neg))
    f.write("Total Score: {}\n".format(train_result_total))
    
    f.write(">> Test\n")
    f.write("Positive Score: {}\n".format(test_result_pos))
    f.write("Negative Score: {}\n".format(test_result_neg))
    f.write("Total Score: {}\n\n".format(test_result_total))
    
    f.close()


# 수익률 ( p[t] - p[t-1] ) / p[t] = r[t]
def get_return_data(data:np.ndarray):
    return_data = np.ones(data.shape, dtype=float)
    # from 0 to 43
    for i in range(data.shape[1]):
        # from 0 to 2866
        for j in range(data.shape[0]):
            if j==0:
                return_data[0, i] = 0
            else:
                return_data[j, i] = (data[j, i] - data[j-1, i]) / data[j, 1]

    return return_data


def get_return_data_bin(data:np.ndarray):
    return_data = np.ones(data.shape, dtype=float)
    # from 0 to 43
    for i in range(data.shape[1]):
        # from 0 to 2866
        for j in range(data.shape[0]):
            if j==0:
                return_data[j, i] = 0
            else:
                return_data[j, i] = (data[j, i] - data[j-1, i]) / data[j, 1]

                trend = (data[j, i] - data[j-1, i]) / data[j, 1]
                if trend >=0:
                    return_data[j,i] = 1
                else:
                    return_data[j,i] = 0

    return return_data


def get_trend_data(data:np.ndarray, Q:int):
    trend_data = np.ones(int(data.shape[0] - Q), dtype=float)
    trend_data_bin = np.ones(int(data.shape[0] - Q), dtype=float)
    # Train data label
    for i in range(trend_data.shape[0]):
        trend_data[i] = (data[i+Q][0] - data[i][0]) / data[i][0]
        trend_data_bin[i] = binary_result(trend_data[i])
    return trend_data, trend_data_bin




def binary_result(val):
    if val >= 0:
        return 1
    else:
        return 0


# Q = 19 / T = 10
def prep_train_data(batch_idx: np.ndarray, config: TrainConfig, train_data: TrainData, y_trend):
    feats = np.zeros((len(batch_idx), config.T, train_data.feats.shape[1]))             # 0~9 KOSPI 외의 데이터 (43)     (b, 10, 43)
    y_history = np.zeros((len(batch_idx), config.T, train_data.targs.shape[1]))         # 0~9 data (학습)               (b, 10, 1)                 
    y_target = y_trend[batch_idx+config.T]                                             # 28(9+19)번째 data (결과값)     (b, 1)

    
    if opt.bin:
        y_target_bin = np.ones(y_target.shape,dtype=float)
        for i, value in enumerate(y_target):
            y_target_bin[i] = binary_result(value)
        y_target = y_target_bin
    
    
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + config.T)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet):
    for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
        enc_params['lr'] = enc_params['lr'] * 0.9
        dec_params['lr'] = dec_params['lr'] * 0.9


def numpy_to_tvar(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(opt.device))
