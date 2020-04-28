import argparse
import os
from iCaRL import ICaRL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='icarl', choices=['icarl', 'e2e', 'ls'])


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.method == 'icarl':
        icarl = ICaRL(args)

