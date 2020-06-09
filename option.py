import argparse
import torch

parser=argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='data_NAX')

parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--ehs', type=int, default=64, help='dimension of Encoder hidden state')
parser.add_argument('--dhs', type=int, default=64, help='dimension of Decoder hidden state')
parser.add_argument('--t', type=int, default=10, help='number of time steps')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--epoch', type=int, default=10, type=int)

parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)

parser.add_argument('--visdom', dest='visdom', action='store_true')
parser.set_defaults(visdom=False)

opt = parser.parse_args()

opt.device='cuda' if torch.cuda.is_available() else 'cpu'

print(opt)
