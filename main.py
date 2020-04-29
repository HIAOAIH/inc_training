import torch

import argparse
from iCaRL import ICaRL
from dataset_with_class import dataset_with_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='icarl', choices=['icarl', 'e2e', 'ls'])
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--init_class_num', type=int, default='0')
    parser.add_argument('--k', type=int, default='2000')
    parser.add_argument('--lr', type=float, default='2')
    parser.add_argument('--weight_decay', type=float, default='0.00001')
    parser.add_argument('--use_gpu', type=bool, default=True)

    return check_args(parser.parse_args())


def check_args(args):
    if args.use_gpu:
        args.use_gpu = False if not torch.cuda.is_available() else True
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.method == 'icarl':
        train_data = dataset_with_class()
        eval_data = dataset_with_class(False)
        icarl = ICaRL(args)
        icarl.train(train_data[:20], 20)
        icarl.test(eval_data[:20])

        trained_class_num = 20
        for i in range(8):
            icarl.train(train_data[trained_class_num:trained_class_num + 10], 10)
            icarl.test(eval_data[trained_class_num:trained_class_num + 10])
            trained_class_num += 10




