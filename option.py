import argparse
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='LSTM')

parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='KOSPI200')
parser.add_argument('--data_type', type=str, help='price, log, volat', default='log')

parser.add_argument('--bs', dest='batch_size', help='batch size', default=8, type=int)
parser.add_argument('--epoch', default=10000, type=int)
parser.add_argument('--window_size', type=int, default=1)

parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)

parser.add_argument('--visdom', dest='visdom', action='store_true')
parser.set_defaults(visdom=False)


opt = parser.parse_args()

opt.device='cuda' if torch.cuda.is_available() else 'cpu'

print(opt)