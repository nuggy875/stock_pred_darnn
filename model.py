import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from option import opt


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (43)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1)

    def forward(self, input_data):
        # input_data: (batch_size, T, input_size)
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T, self.input_size).to(opt.device))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T, self.hidden_size).to(opt.device))
        # hidden, cell: initial states with dimension hidden_size
        hidden = Variable(torch.zeros(1, input_data.size(0), self.hidden_size).to(opt.device))
        cell = Variable(torch.zeros(1, input_data.size(0), self.hidden_size).to(opt.device))

        for t in range(self.T):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T)
            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T))  # (batch_size * input_size) * 1
            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T, encoder_hidden_size)
        # y_history: (batch_size, (T))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        # hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        # cell = init_hidden(input_encoded, self.decoder_hidden_size)

        hidden = Variable(torch.zeros(1, input_encoded.size(0), self.decoder_hidden_size).to(opt.device))
        cell = Variable(torch.zeros(1, input_encoded.size(0), self.decoder_hidden_size).to(opt.device))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size).to(opt.device))

        for t in range(self.T):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T),
                    dim=1)  # (batch_size, T)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        if opt.bin:
            return self.sigmoid(self.fc_final(torch.cat((hidden[0], context), dim=1)))
        else:
            return self.fc_final(torch.cat((hidden[0], context), dim=1))