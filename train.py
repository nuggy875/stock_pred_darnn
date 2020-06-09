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
    out_size = t_dat.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred


def train():
    # ====== Read Data ======
    raw_data = pd.read_csv(os.path.join(opt.data_path, opt.dataset+'.csv'), index_col='Date')
    targ_cols = ("KOSPI200",)                           # target Column
    scale = StandardScaler().fit(raw_data)              # Data Scaling
    proc_dat = scale.transform(raw_data)
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(raw_data.columns)
    for col_name in targ_cols:
        mask[dat_cols.index(col_name)] = False
    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    train_data, scaler = [TrainData(feats, targs), scale]

    # ====== hyperparameter ======
    encoder_hidden_size = opt.ehs
    decoder_hidden_size = opt.dhs
    T = opt.t
    learning_rate = opt.lr
    batch_size=opt.bs

    criterion = nn.MSELoss()
    train_size = int(train_data.feats.shape[0] * 0.7)
    config = TrainConfig(T, train_size, batch_size, criterion)
    print("Training Size : {}".format(config.train_size))
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

    net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    iter_per_epoch = int(np.ceil(config.train_size * 1. / config.batch_size))
    print('iter_per_epoch:{}'.format(iter_per_epoch))
    iter_losses = np.zeros(opt.epoch * iter_per_epoch)
    epoch_losses = np.zeros(opt.epoch)
    n_iter = 0
    for e_i in range(opt.epoch):
        print("Epoch:{})".format(e_i))
        perm_idx = np.random.permutation(config.train_size - config.T)

        for t_i in range(0, config.train_size, config.batch_size):
            batch_idx = perm_idx[t_i:(t_i + config.batch_size)]
            X, y_history, y_target = prep_train_data(batch_idx, config, train_data)

            net.enc_opt.zero_grad()
            net.dec_opt.zero_grad()

            input_weighted, input_encoded = net.encoder(numpy_to_tvar(X))
            y_pred = net.decoder(input_encoded, numpy_to_tvar(y_history))
            y_true = numpy_to_tvar(y_target)
            loss = config.loss_func(y_pred, y_true)
            loss.backward()
            net.enc_opt.step()
            net.dec_opt.step()

            iter_losses[e_i * iter_per_epoch + t_i // config.batch_size] = loss
            n_iter += 1
            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])
        
        if e_i % 10 == 0:
            y_test_pred = test(net, train_data, config.train_size, config.batch_size, config.T, on_train=False)
            # TODO: make this MSE and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[config.train_size:]
            print("Epoch {}, train Loss:{}, val Loss:{}".format(e_i, epoch_losses[e_i], np.mean(np.abs(val_loss))))
            y_train_pred = test(net, train_data,
                                    config.train_size, config.batch_size, config.T,
                                    on_train=True)
            
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                    label="True")
            plt.plot(range(config.T, len(y_train_pred) + config.T), y_train_pred,
                    label='Predicted - Train')
            plt.plot(range(config.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                    label='Predicted - Test')
            plt.legend(loc='upper left')
            save_or_show_plot("pred_{}.png".format(e_i), opt.show)
            

    final_y_pred = test(net, train_data, config.train_size, config.batch_size, config.T)
    
    plt.figure()
    plt.semilogy(range(len(iter_losses)), iter_losses)
    save_or_show_plot("iter_loss.png", opt.show)

    plt.figure()
    plt.semilogy(range(len(epoch_losses)), epoch_losses)
    save_or_show_plot("epoch_loss.png", opt.show)

    plt.figure()
    plt.plot(final_y_pred, label='Predicted')
    plt.plot(train_data.targs[config.train_size:], label="True")
    plt.legend(loc='upper left')
    save_or_show_plot("final_predicted.png", opt.show)
    
    joblib.dump(scaler, os.path.join("saves", "scaler.pkl"))
    torch.save(net.encoder.state_dict(), os.path.join("saves", "encoder.torch"))
    torch.save(net.decoder.state_dict(), os.path.join("saves", "decoder.torch"))
    

if __name__ == "__main__":
    train()
