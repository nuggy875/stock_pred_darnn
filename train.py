import sys
import time
import visdom

import torch
from torch import nn
from torch.backends import cudnn

import utils
from option import opt
from data_loading import DataLoader
from model import BinaryLSTM


class TrainModel:
    def __init__(self):
        self.prices = DataLoader()

    def __call__(self, split_rate=.9, seq_length=30, num_layers=2, hidden_size=128):

        if opt.visdom:
            vis = visdom.Visdom()
        else:
            vis = None

        batch_size = opt.batch_size
        train_size = int(self.prices.train_size * split_rate)
        X = self.prices.X[:train_size, :]
        X = torch.unsqueeze(torch.from_numpy(X).float(), 1)

        if opt.data_type=='price':
            X_train, Y_target = utils.data_process_price(X, train_size = X.shape[0], num_steps = seq_length)
        elif opt.data_type=='log':
            X_train, Y_target = utils.data_process_log(X, train_size = X.shape[0], num_steps = seq_length)

        X_train = X_train.to(opt.device)
        Y_target = Y_target.to(opt.device)

        model = BinaryLSTM(opt.window_size, hidden_size, num_layers=num_layers)
        model = model.to(opt.device)

        criterion = nn.BCELoss().to(opt.device)
        optimizer = torch.optim.Adam(model.parameters())

        timeStart = time.time()

        for epoch in range(opt.epoch):
            loss_sum=0
            # First Value
            Y_pred = model(X_train[:, :batch_size, :])
            if batch_size==1:
                Y_pred = torch.unsqueeze(Y_pred, 1)
            if opt.window_size == 1:
                Y_pred = torch.unsqueeze(Y_pred, 2)
            Y_pred = torch.squeeze(Y_pred[num_layers-1, :, :])

            loss = criterion(Y_pred, Y_target[:batch_size])
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(batch_size, X_train.shape[1], batch_size):
                y = model(X_train[:, i : i + batch_size, :])
                if batch_size==1:
                    y = torch.squeeze(y, 1)
                if opt.window_size==1:
                    y = torch.unsqueeze(y, 2)
                y = torch.squeeze(y[num_layers - 1, :, :])
                Y_pred = torch.cat((Y_pred, y))

                loss = criterion(y, Y_target[i : i + batch_size])
                loss_sum += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Visdom    
            if epoch % 10==0 and opt.visdom:
                vis.line(X=torch.ones((1, 1)).cpu() * i + epoch * train_size,
                        Y=torch.Tensor([loss_sum]).unsqueeze(0).cpu(),
                        win='loss',
                        update='append',
                        opts=dict(xlabel='step',
                            ylabel='Loss',
                            title='Training Loss for {}, type:{} (bs={})'.format(opt.dataset, opt.data_type, batch_size),
                            legend=['Loss'])
                    )

            print('epoch [%d] finished, Loss Sum: %f' % (epoch, loss_sum))


        timeSpent = time.time() - timeStart
        print('Time Spend : {}'.format(timeSpent))
        torch.save(model, 'trained_model/'+opt.model + '_'+ opt.dataset + '_bin_' + opt.type + '.model')


if __name__ == "__main__":
    print('--- Train Mode ---')
    Trainer = TrainModel()
    Trainer()