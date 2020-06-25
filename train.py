import collections
import typing
from typing import Tuple
import json
import os

import torch
from torch import nn
from torch import optim
from torch.nn import functional as tf
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model import Encoder, Decoder
from utils import *
from option import opt, DaRnnNet, TrainData, TrainConfig


def save_json(kwargs: str, filename:str):
    with open(os.path.join("saves", filename+".json"), "w") as fi:
        json.dump(kwargs, fi, indent=4)


def test(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = t_dat.targs.shape[1]     # 1
    if on_train:
        # for train data
        y_pred = np.zeros((train_size - T + 1, out_size))                       # 1997, 1
    else:
        # for test data
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))    # 861, 1

    for y_i in range(0, len(y_pred), batch_size):
        # 128 개의 batch에 대해
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)

        X = np.zeros((b_len, T, t_dat.feats.shape[1]))                      # (128, 10, 43)
        y_history = np.zeros((b_len, T, t_dat.targs.shape[1]))              # (128, 10, 1)

        for b_i, b_idx in enumerate(batch_idx):
            # 각 input에 대해
            if on_train:
                idx = range(b_idx, b_idx + T)                               # T-1 to T
            else:
                idx = range(b_idx + train_size - T + 1, b_idx + train_size + 1)     # train_size-1 to train_Size

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()
    
    return y_pred


def train():
    encoder_hidden_size = opt.ehs
    decoder_hidden_size = opt.dhs
    T = opt.t
    Q = opt.q
    learning_rate = opt.lr
    batch_size=opt.bs


    # ====== Read Data ======
    use_feats=["Date", "KOSPI200",
                "KOSDAQ", "S&P500", "SSEC", "ShanghaiA", "N225",
                "CMX Gold", "CMX Siver", "Dollar Index", "Dollar", "Hongkong"]

    data_path = os.path.join(opt.data_path, opt.dataset+'.csv')
    raw_data = pd.read_csv(data_path,
                            index_col='Date',
                            usecols=use_feats)
    targ_cols = ("KOSPI200",)                               # target Column

    if opt.data_mode == 'price':                            # Price
        input_dat = raw_data.to_numpy()
    elif opt.data_mode == 'standardized':                   # Data Scaling
        scaler = StandardScaler().fit(raw_data)
        input_dat = scaler.transform(raw_data)
    elif opt.data_mode == 'return':                         # Data 수익률
        price_dat = raw_data.to_numpy()         # Data Price
        input_dat = get_return_data(price_dat)  # Data Return
        trend_dat, trend_dat_bin = get_trend_data(price_dat, Q)
    
    mask = np.ones(input_dat.shape[1], dtype=bool)
    dat_cols = list(raw_data.columns)
    for col_name in targ_cols:
        mask[dat_cols.index(col_name)] = False
    feats = input_dat[:, mask]
    targs = input_dat[:, ~mask]
    train_data = TrainData(feats, targs)
    train_size = int(train_data.feats.shape[0] * 0.7)
    print('Input Data Length :{}'.format(len(input_dat)))
    print('Feature : {}'.format(feats.shape[1]))
    print("Training Size (70%) : {}".format(train_size))

    # ====== LOSS ======
    if opt.bin:
        criterion = nn.BCELoss().to(opt.device)
    else:
        criterion = nn.MSELoss().to(opt.device)

    # ====== Config ======
    config = TrainConfig(T, Q, train_size, batch_size, criterion)

    # ====== define Network ======
    net_kwargs = {"batch_size": batch_size, "T": T}
    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size, "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": len(targ_cols)}
    save_json(net_kwargs, "kwargs_net")
    save_json(enc_kwargs, "kwargs_enc")
    save_json(dec_kwargs, "kwargs_dec")
    encoder = Encoder(**enc_kwargs).to(opt.device)
    decoder = Decoder(**dec_kwargs).to(opt.device)
    encoder_optimizer = optim.Adam(params=[p for p in encoder.parameters() if p.requires_grad], lr=learning_rate)
    decoder_optimizer = optim.Adam(params=[p for p in decoder.parameters() if p.requires_grad], lr=learning_rate)

    net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)                  # DaRNN Network

    # ====== TRAIN ======
    iter_per_epoch = int(np.ceil(config.train_size * 1. / config.batch_size))
    print('iter_per_epoch:{}'.format(iter_per_epoch))

    iter_losses = np.zeros(opt.epoch * iter_per_epoch)
    epoch_losses = np.zeros(opt.epoch)
    n_iter = 0
    for e_i in range(opt.epoch):
        print("Epoch:{})".format(e_i))
        perm_idx = np.random.permutation(config.train_size - config.T)
        for t_i in range(0, config.train_size, config.batch_size):                          # train_size = 2006
            batch_idx = perm_idx[t_i:(t_i + config.batch_size)]
            X, y_history, y_trend = prep_train_data(batch_idx, config, train_data, trend_dat)         # get Data
            y_true = numpy_to_tvar(y_trend).unsqueeze(1)

            net.enc_opt.zero_grad()
            net.dec_opt.zero_grad()
            
            # Prediction using Attention (DaRNN)
            input_weighted, input_encoded = net.encoder(numpy_to_tvar(X))                   # Encoder
            y_pred = net.decoder(input_encoded, numpy_to_tvar(y_history))                   # Decoder
            
            loss = config.loss_func(y_pred, y_true)                                        # 현 시점에서 Q(=19)일 후를 예측
            loss.backward()

            net.enc_opt.step()
            net.dec_opt.step()

            iter_losses[e_i * iter_per_epoch + t_i // config.batch_size] = loss
            n_iter += 1
            # Learning Rate 조절 (n_iter 10000 이후): # https://www.jeremyjordan.me/nn-learning-rate/
            if n_iter % 10000 == 0 and n_iter > 0:
                adjust_learning_rate(net)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])
        
        # 100 epoch 마다 Testing
        if e_i % opt.testing_epoch == 0:
            
            # ===== 테스팅 Testing Data =====
            y_test_pred = test(net, train_data, config.train_size, config.batch_size, config.T, on_train=False)
            test_loss = y_test_pred[:-Q] - train_data.targs[config.train_size+Q:]
            
            print("Epoch {}, train Loss:{}, test Loss:{}".format(e_i, epoch_losses[e_i], np.mean(np.abs(test_loss))))

            # ===== 테스팅 Trainind Data =====
            y_train_pred = test(net, train_data, config.train_size, config.batch_size, config.T, on_train=True)
            
            # get Result
            if opt.bin:
                get_result_bin(trend_dat_bin, y_train_pred, y_test_pred, T, train_size, e_i)
                plt.figure()
                plt.plot(range(1, 1 + len(trend_dat_bin)), trend_dat_bin.reshape(trend_dat_bin.size, 1), label="Ground Truth Trend")                     # 1 ~ 2848
                plt.plot(range(config.T - 1, len(y_train_pred) + config.T - 1), y_train_pred, label='Predicted - Train')        # 9 ~ 2005
                plt.plot(range(len(y_train_pred) + config.T - 1, len(train_data.targs)), y_test_pred, label='Predicted - Test') # 2006 ~ 2867 + 19
                plt.legend(loc='upper left')
                plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "pred_{}.png".format(e_i)))
            else:
                get_result(trend_dat, y_train_pred, y_test_pred, T, train_size, e_i)

                plt.figure()
                plt.plot(range(1, 1 + len(trend_dat)), trend_dat.reshape(trend_dat.size, 1), label="Ground Truth Trend")                     # 1 ~ 2848
                plt.plot(range(config.T - 1, len(y_train_pred) + config.T - 1), y_train_pred, label='Predicted - Train')        # 9 ~ 2005
                plt.plot(range(len(y_train_pred) + config.T - 1, len(train_data.targs)), y_test_pred, label='Predicted - Test') # 2006 ~ 2867 + 19
                plt.legend(loc='upper left')
                plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "pred_{}.png".format(e_i)))

            '''
            # For Price & Standardized
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs, label="Ground Truth Trend")                     # 1 ~ 2867
            plt.plot(range(config.T + 1, len(y_train_pred) + config.T + 1), y_train_pred, label='Predicted - Train')        # 11 ~ 2007
            plt.plot(range(config.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred, label='Predicted - Test') # 2008 ~ 2868
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "pred_{}.png".format(e_i)))
            '''
        if e_i % (opt.testing_epoch*10) == 0:

            final_y_pred = test(net, train_data, config.train_size, config.batch_size, config.T)
            
            plt.figure()
            plt.semilogy(range(len(iter_losses)), iter_losses)
            plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "loss_iter_{}.png".format(e_i)))

            plt.figure()
            plt.semilogy(range(len(epoch_losses)), epoch_losses)
            plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "loss_epoch_{}.png".format(e_i)))

            plt.figure()
            plt.plot(final_y_pred, label='Predicted')
            plt.plot(trend_dat[config.train_size-Q:].reshape(trend_dat[config.train_size-Q:].size , 1), label="True")
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "pred_final_{}.png".format(e_i)))
            
            # Save Model
            torch.save(net.encoder.state_dict(), os.path.join("saves", "encoder_{}.torch".format(e_i)))
            torch.save(net.decoder.state_dict(), os.path.join("saves", "decoder_{}.torch".format(e_i)))

    if opt.data_mode=='standardized':
        joblib.dump(scaler, os.path.join("saves", "scaler.pkl"))
    

if __name__ == "__main__":
    train()
