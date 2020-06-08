from train import TrainModel
from test import TestModel
from option import opt

if __name__ == "__main__":
    if not opt.test:
        print('--- Train Mode ---')
        Trainer = TrainModel()
        Trainer()
    else:
        print('--- Test Mode ---')
        Tester = TestModel()
        Tester()