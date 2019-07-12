#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
import pickle
import pandas as pd
import csv
import multiprocessing

from cgp import CGP
from cgp_config import CgpInfoConvSet
from cnn_train import CNN_train
from utils import create_folder

# For debugging in vscode
multiprocessing.set_start_method('spawn', True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evolving CAE structures')
    parser.add_argument('--gpu_num', '-g', type=int,
                        default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=2,
                        help='Num. of offsprings')
    parser.add_argument('--net_info_file', default='network_info.pickle',
                        help='Network information file name')
    parser.add_argument(
        '--log_file', default='./log_cgp.txt', help='Log file name')
    parser.add_argument('--mode', '-m', default='evolution',
                        help='Mode (evolution / retrain / reevolution)')
    parser.add_argument('--init', '-i', action='store_true')
    parser.add_argument('--gpuID', '-p', type=int, default=0, help='GPU ID')
    parser.add_argument('--save_dir', default='./logs/', help='Log file name')
    parser.add_argument('--archs_per_task', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--num_train', type=int, default=500)
    # parser.add_argument('--genotype', type=str, default='resnet')
    parser.add_argument('--num_depth', type=int, default=900)
    parser.add_argument('--num_min_depth', type=int, default=300)
    parser.add_argument('--num_max_depth', type=int, default=1000)
    parser.add_argument('--num_breadth', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--arch_type', type=str, default='vgg')
    # parser.add_argument('--data_dir', type=str, default='./')
    args = parser.parse_args()
    config = vars(args)

    # --- Optimization of the CNN architecture ---
    if args.mode == 'evolution':
        # Create CGP configuration and save network information
        network_info = CgpInfoConvSet(
            arch_type=config['arch_type'], rows=args.num_breadth,
            cols=args.num_depth, level_back=1,
            min_active_num=args.num_min_depth,
            max_active_num=args.num_max_depth)
        with open(args.net_info_file, mode='wb') as f:
            pickle.dump(network_info, f)

        img_size = args.img_size
        num_epoch = args.num_epoch
        batchsize = args.batch_size
        accs = {}
        create_folder(config['save_dir'])
        for i in range(config['archs_per_task']):
            with open(f"{config['save_dir']}accuracy{args.gpuID}.txt", 'at') as f:
                with open(f"{config['save_dir']}config.json", 'w+') as cfg_f:
                    json.dump(config, cfg_f, indent=2)

                cgp = CGP(network_info, None, arch_type=config['arch_type'],
                          lam=1, img_size=img_size, init=args.init)

                print(cgp.pop[0].active_net_list())
                full = CNN_train('cifar10', validation=True, verbose=True,
                                 batchsize=batchsize, data_num=args.num_train,
                                 mode="full", config=config)
                acc_full, acc_curr = full(cgp.pop[0].active_net_list(),
                                          args.gpuID, num_epoch=num_epoch,
                                          out_model='retrained_net.model')
                accs[i] = acc_curr
                f.write(str(acc_full)+"\n")
                with open(f"{config['save_dir']}accuracies.json", 'w+') as fw:
                    json.dump(accs, fw, indent=2)
                with open(f"{config['save_dir']}log.txt", 'a') as fw:
                    writer = csv.writer(fw, lineterminator='\n')
                    writer.writerow(cgp._log_data(net_info_type='full'))
                with open(f"{config['save_dir']}log-active.txt", 'a') as fw:
                    writer = csv.writer(fw, lineterminator='\n')
                    writer.writerow(cgp._log_data(net_info_type='active_only'))
                print(acc_full)

        # TinyImageNet
        # for _ in range(10):
        #     cgp = CGP(network_info, None, lam=1, img_size=img_size, init=args.init)
        #     print(cgp.pop[0].active_net_list())
        #     d_list = [10000, 20000, 50000]
        #     with open("accuracy%s.txt" % str(args.gpuID), 'at') as f:
        #         f.write(str(d_list[0])+"\t"+str(d_list[1])+"\t"+str(d_list[2])+"\n")
        #         for d in d_list:
        #             # temp = CNN_train('cifar10', validation=False, verbose=True, batchsize=128, data_num=d, mode="part")
        #             temp = CNN_train('tinyimagenet', validation=False, verbose=True, batchsize=batchsize, data_num=d, mode="part")
        #             acc_part = temp(cgp.pop[0].active_net_list(), args.gpuID, num_epoch=num_epoch, out_model='retrained_net.model')
        #             f.write(str(acc_part)+"\t")
        #         full = CNN_train('tinyimagenet', validation=False, verbose=True, batchsize=batchsize, data_num=args.data_num, mode="full")
        #         acc_full = full(cgp.pop[0].active_net_list(), args.gpuID, num_epoch=num_epoch, out_model='retrained_net.model')
        #         f.write(str(acc_full)+"\n")

    # --- Retraining evolved architecture ---
    elif args.mode == 'retrain':
        print('Retrain')
        # In the case of existing log_cgp.txt
        # Load CGP configuration
        with open(args.net_info_file, mode='rb') as f:
            network_info = pickle.load(f)
        # Load network architecture
        cgp = CGP(network_info, None)
        data = pd.read_csv(args.log_file, header=None)  # Load log file
        # Read the log at final generation
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))
        print(cgp._log_data(net_info_type='active_only', start_time=0))
        # Retraining the network
        test_dir = '/ceph/blume/datasets/CIFAR10-C/test/'
        test_dists = [
            'brightness.npy',
            'contrast.npy',
            'defocus_blur.npy',
            'elastic_transform.npy',
            'fog.npy',
            'frost.npy',
            'gaussian_blur.npy',
            'gaussian_noise.npy',
            'glass_blur.npy',
            'impulse_noise.npy',
            'jpeg_compression.npy',
            # 'labels.npy',
            'motion_blur.npy',
            'pixelate.npy',
            'saturate.npy',
            'shot_noise.npy',
            'snow.npy',
            'spatter.npy',
            'speckle_noise.npy',
            'zoom_blur.npy'
        ]
        temp = CNN_train('cifar10', validation=False,
                         verbose=True, batchsize=128, test_dists=test_dists)
        acc = temp(cgp.pop[0].active_net_list(), 0,
                   num_epoch=200, out_model='retrained_net.model')
        print(acc)

        # # otherwise (in the case where we do not have a log file.)
        # temp = CNN_train('haze1', validation=False, verbose=True, img_size=128, batchsize=16)
        # cgp = [['input', 0], ['S_SumConvBlock_64_3', 0], ['S_ConvBlock_64_5', 1], ['S_SumConvBlock_128_1', 2], ['S_SumConvBlock_64_1', 3], ['S_SumConvBlock_64_5', 4], ['S_DeConvBlock_3_3', 5]]
        # acc = temp(cgp, 0, num_epoch=500, out_model='retrained_net.model')
