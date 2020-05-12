import torch
import os
import argparse
from iCaRL import ICaRL
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
    parser.add_argument('--device_num', type=int, choices=list(range(torch.cuda.device_count())), default=0)

    return check_args(parser.parse_args())


def check_args(args):
    if args.use_gpu:
        args.use_gpu = False if not torch.cuda.is_available() else True
    if not os.path.exists(args.network_dir):
        os.makedirs(args.network_dir)
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.method == 'icarl':
        train_data = dataset_with_class()
        eval_data = dataset_with_class(False)
        icarl = ICaRL(args)
        # trained_class_num = 0
        # for i in range(9):
        #     icarl.train(train_data[trained_class_num:trained_class_num + 10], 10)
        #     icarl.test(eval_data[:trained_class_num + 10])
        #     trained_class_num += 10
        #     torch.save(icarl.discriminator.state_dict(), args.network_dir + '/iCaRL_' + str(icarl.class_num) + '.pt')

        # icarl.train(train_data[:50], 50)
        # icarl.test(eval_data[:50], True)
        # icarl.test(train_data[:50], False)

# '''
        # train 20 classes first
        icarl.train(train_data[:20], 20)
        icarl.test(eval_data[:20], True)
        icarl.test(train_data[:20], False)
        torch.save(icarl.discriminator.state_dict(), args.network_dir + '/iCaRL_' + str(icarl.class_num) + '.pt')

        trained_class_num = 20
        for i in range(8):
            icarl.train(train_data[trained_class_num:trained_class_num + 10], 10)
            icarl.test(eval_data[:trained_class_num + 10], True)
            icarl.test(train_data[:trained_class_num + 10], False)
            trained_class_num += 10
            
# '''
        # torch.save(icarl.discriminator.state_dict(), args.network_dir + '/iCaRL_' + str(icarl.class_num) + '.pt')
