import torch
import os
import argparse
from iCaRL import ICaRL
from E2E import EndToEnd
from LS import LargeScale, BicLayer
from dataset_with_class import dataset_with_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='icarl', choices=['icarl', 'e2e', 'ls'])
    parser.add_argument('--train_epoch', type=int, default='70')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--init_class_num', type=int, default='0')
    parser.add_argument('--k', type=int, default='2000')
    parser.add_argument('--lr', type=float, default='2')
    parser.add_argument('--weight_decay', type=float, default='0.00001')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--network_dir', type=str, default='networks')
    parser.add_argument('--confusion_mat_dir', type=str, default='confusion_matrix')
    parser.add_argument('--device_num', type=int, choices=list(range(torch.cuda.device_count())), default=0)

    return check_args(parser.parse_args())


def check_args(args):
    if args.use_gpu:
        args.use_gpu = False if not torch.cuda.is_available() else True
    if not os.path.exists(args.network_dir):
        os.makedirs(args.network_dir)
    if not os.path.exists(args.network_dir + '/' + args.method):
        os.makedirs(args.network_dir + '/' + args.method)
    if not os.path.exists(args.confusion_mat_dir):
        os.makedirs(args.confusion_mat_dir)
    if not os.path.exists(args.confusion_mat_dir + '/' + args.method):
        os.makedirs(args.confusion_mat_dir + '/' + args.method)

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    train_data = dataset_with_class()
    eval_data = dataset_with_class(False)
    trained_class_num = 0
    if args.method == 'icarl':
        icarl = ICaRL(args)
        for i in range(10):
            if i == 0:
                momentum = 0
            else:
                momentum = 0.9
            icarl.train(train_data[trained_class_num:trained_class_num + 10], momentum)
            icarl.test(eval_data[:trained_class_num + 10], True)
            icarl.test(train_data[:trained_class_num + 10], False)
            trained_class_num += 10
            # torch.save(icarl.discriminator.state_dict(), args.network_dir + '/iCaRL_' + str(icarl.class_num) + '.pt')

    elif args.method == 'e2e':
        e2e = EndToEnd(args)
        for i in range(10):
            e2e.train(train_data[trained_class_num:trained_class_num + 10])
            e2e.test(eval_data[:trained_class_num + 10])
            trained_class_num += 10

    elif args.method == 'ls':
        ls = LargeScale(args)
        trained_class_num = args.init_class_num
        if trained_class_num != 0:
            ls.net.cuda(args.device_num)
            ls.bias_layers.append(BicLayer())
            ls.bias_layers[-1].cuda(args.device_num)
            ls.construct_exemplar_set(train_data[:trained_class_num], 2000 // trained_class_num)
            ls.test(eval_data[:trained_class_num])

        for i in range((100 - args.init_class_num) // 20):
            ls.train(train_data[trained_class_num:trained_class_num + 20], eval_data[:trained_class_num + 20])
            torch.save(ls.net.state_dict(), './networks/ls/large_scale_to_' + str(trained_class_num + 20) + '.pt')
            ls.test(eval_data[:trained_class_num + 20], save_conf_mat=True)
            trained_class_num += 20
