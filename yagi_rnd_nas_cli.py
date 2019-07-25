import sh
import argparse
import datetime
import sys
import os
import time
import subprocess
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, choices=['local', 'qsub'],
                        required=True)
    parser.add_argument(
        '--server_names', nargs='+', metavar='yagiXX', type=str)
    parser.add_argument(
        '--file_to_run', metavar='XXXX.sh', type=str)
    parser.add_argument('--num_train', type=int)
    parser.add_argument('--archs_per_task', type=int, default=5)
    parser.add_argument('--archs_per_num_train', type=int, default=0)
    parser.add_argument('--arch_type', type=str, default='resnet')

    parser.add_argument('--init_channels', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_nodes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu_start', type=int, default=0)
    parser.add_argument('--gpu_end', type=int, default=None)
    parser.add_argument('--gpus_per_task', type=int, default=1)
    parser.add_argument('--num_train_determ',
                        dest='num_train_determ',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--genotype', type=str,
                        default='PRIMITIVES', nargs='+')
    # parser.add_argument('--num_train', type=int, default=5000, nargs='+')
    args = parser.parse_args()

    return args


def ssh(command, yagi):
    """Runs an ssh command on a yagi server

    Arugments:
        command: string, command to execute
        yagi: string, yagi to run the command on"""
    return sh.ssh('-i', '/home/blume/.ssh/id_ed25519', command, yagi)


def run_nas_shell(num_train, gpu_num, id,
            batch_size, archs_per_task, arch_type):
    """Runs the influence function calculation on a specified yagi

    Arguments:
        start: int, per class test sample index at which to start
        per_class: int, how many images to process per class
        gpu_num: str, gpu id to run the influence function on. can be a single
            number or a comma seperated string of multiple ids
        file_to_run: str, filename of the script to run on yagi
        batch_size: int, reduce for small GPU mem machines
        recursion_depth: int, pass
        r_avg: int, pass"""
    print(f'running random NAS: num_train: {num_train}')
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpu_num
    with open(f"./log/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}.log", 'w+') as logf:
        # sh.python3(
        subprocess.Popen(f'python3 exp_main.py --arch_type {arch_type} --archs_per_task {archs_per_task} --save_dir save_dir/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}/ --num_train {num_train}',
            shell=True,
            env=env_vars,
            stdout=logf,
            stderr=logf)
        # _out=f"./log/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}.log",
        # _err=f"./log/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}-err.log",
        # _bg=True,
        # _env=env_vars)

        # with open(f"./log/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}.log", 'w+') as logf:
        # # sh.python3(
        # subprocess.Popen([
        #         # 'python3',
        #         'exp_main.py',
        #         '--arch_type',
        #         f'{arch_type}',
        #         '--archs_per_task',
        #         f'{archs_per_task}',
        #         '--save_dir',
        #         f'save_dir/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}/',
        #         '--num_train',
        #         f'{num_train}'
        #     ],
        #     shell=True,
        #     env=env_vars,
        #     stdout=logf,
        #     stderr=logf)
        # # _out=f"./log/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}.log",
        # # _err=f"./log/{arch_type}-{time}-{num_train}-{archs_per_task}-id{id}-err.log",
        # # _bg=True,
        # # _env=env_vars)


def run_nas(yagi, num_train, gpu_num, file_to_run, id,
            batch_size, archs_per_task=5):
    """Runs the influence function calculation on a specified yagi

    Arguments:
        start: int, per class test sample index at which to start
        per_class: int, how many images to process per class
        gpu_num: str, gpu id to run the influence function on. can be a single
            number or a comma seperated string of multiple ids
        file_to_run: str, filename of the script to run on yagi
        batch_size: int, reduce for small GPU mem machines
        recursion_depth: int, pass
        r_avg: int, pass"""
    print(f'running random NAS: {yagi}, num_train: {num_train}')
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpu_num
    sh.qsub(
        '-q',
        f'main.q@{yagi}.vision.is.tohoku',
        '-v', f'time={time}',
        '-v', f'num_train={num_train}',
        '-v', f'id={id}',
        '-v', f'batch_size={batch_size}',
        '-v', f'archs_per_task={archs_per_task}',
        file_to_run,
        _env=env_vars)


def get_available_gpus(yagi=None):
    """Returns the number of available gpus on a yagi

    Arguments:
        yagi: string, name of yagi

    Returns:
        num_gpus: int, number of gpus available on the yagi"""
    if not yagi:
        num_gpus = int(sh.wc(sh.grep(sh.grep(sh.lspci(), 'VGA'), 'NVIDIA'), '-l'))
    else:
        num_gpus = int(ssh(yagi, 'lspci | grep VGA | grep NVIDIA | wc -l'))
    return num_gpus


def train_per_dset_size(args, training_samples, file_to_run, availabe_yagis):
    for yagi in availabe_yagis:
        gpus = get_available_gpus(yagi)
        print(f'Available gpus on {yagi}: {gpus}')

        for gpu, num_train in zip(range(args.gpu_start, gpus,
                                        args.gpus_per_task), training_samples):
            gpulst = [gpu+i for i in range(0, args.gpus_per_task)]
            gpus_to_use = ','.join(str(x) for x in gpulst)
            print(f'Running on GPU ids: {gpus_to_use}')

            # run_nas(
            # yagi, num_train, args.init_channels, gpus_to_use,
            # file_to_run, args.batch_size, args.num_layers,
            # args.num_nodes, args.num_train_determ)


def train_once_per_gpu_local(args):
    total_gpus = get_available_gpus()
    if 0 != args.archs_per_num_train:
        archs_per_task = round(args.archs_per_num_train /
                               (total_gpus - args.gpu_start))
    else:
        archs_per_task = args.archs_per_task

    print(f'Running with: {archs_per_task} archs per task')

    gpus = get_available_gpus()
    print(f'Available gpus: {gpus}')

    for gpu in range(args.gpu_start, gpus):
        gpulst = [gpu+i for i in range(args.gpu_start, args.gpus_per_task)]
        gpus_to_use = ','.join(str(x) for x in gpulst)
        print(f'Running on GPU ids: {gpus_to_use}')

        run_nas_shell(
            args.num_train, gpus_to_use,
            gpu, args.batch_size, archs_per_task, args.arch_type)


def train_once_per_gpu(args, file_to_run, availabe_yagis):
    total_gpus = 0
    for yagi in availabe_yagis:
        total_gpus += get_available_gpus(yagi)
    if 0 != args.archs_per_num_train:
        archs_per_task = round(args.archs_per_num_train /
                               (total_gpus - args.gpu_start))
    else:
        archs_per_task = args.archs_per_task

    print(f'Running with: {archs_per_task} archs per task')

    for yagi in availabe_yagis:
        gpus = get_available_gpus(yagi)
        print(f'Available gpus on {yagi}: {gpus}')

        for gpu in range(args.gpu_start, gpus):
            gpulst = [gpu+i for i in range(args.gpu_start, args.gpus_per_task)]
            gpus_to_use = ','.join(str(x) for x in gpulst)
            print(f'Running on GPU ids: {gpus_to_use}')

            run_nas(
                yagi, args.num_train, gpus_to_use,
                file_to_run, gpu, args.batch_size, archs_per_task)


def train_per_genotype(args, genotypes, file_to_run, availabe_yagis):
    if isinstance(args.num_train, list):
        num_train = args.num_train[0]
    else:
        num_train = args.num_train

    for yagi in availabe_yagis:
        gpus = get_available_gpus(yagi)
        print(f'Available gpus on {yagi}: {gpus}')

        for gpu, genotype in zip(range(args.gpu_start, gpus,
                                       args.gpus_per_task), genotypes):
            gpulst = [gpu+i for i in range(0, args.gpus_per_task)]
            gpus_to_use = ','.join(str(x) for x in gpulst)
            print(f'Running on GPU ids: {gpus_to_use}')

            # run_nas(
            #     yagi, num_train, args.init_channels, gpus_to_use,
            #     file_to_run, args.batch_size, args.num_layers,
            #     args.num_nodes, args.num_train_determ, genotype)


def check_if_low_mem_yagi_involved(availabe_yagis):
    return_value = False
    low_gpu_mem_yagis = ['yagi10', 'yagi13']
    for yagi in availabe_yagis:
        if yagi in low_gpu_mem_yagis:
            return_value = True
            print(f"ATTENTION, {yagi} has only a low amount of memory"
                  f" available")
            print("Cancel now if you want")
            time.sleep(5)
    return return_value


if __name__ == "__main__":
    args = parse_args()
    availabe_yagis = args.server_names
    file_to_run = args.file_to_run

    training_samples = [500, 1000, 5000, 10000, 25000]

    if 'local' == args.run_type:
        train_once_per_gpu_local(args)
    elif 'qsub' == args.run_type:
        check_if_low_mem_yagi_involved(availabe_yagis)
        # train_per_dset_size(args, training_samples, file_to_run, availabe_yagis)
        # train_per_genotype(args, args.genotype, file_to_run, availabe_yagis)
        train_once_per_gpu(args, file_to_run, availabe_yagis)

    # for yagi in availabe_yagis:
    #     gpus = get_available_gpus(yagi)
    #     print(f'Available gpus on {yagi}: {gpus}')

    #     for gpu, num_train in zip(range(args.gpu_start, gpus,
    #                                     args.gpus_per_task), training_samples):
    #         gpulst = [gpu+i for i in range(0, args.gpus_per_task)]
    #         gpus_to_use = ','.join(str(x) for x in gpulst)
    #         print(f'Running on GPU ids: {gpus_to_use}')

    #         if yagi in low_gpu_mem_yagis and 8 < args.batch_size:
    #             run_nas(
    #                 yagi, num_train, args.init_channels, gpus_to_use,
    #                 file_to_run, args.batch_size/2, args.num_layers,
    #                 args.num_nodes, args.num_train_determ)
    #         else:
    #             run_nas(
    #                 yagi, num_train, args.init_channels, gpus_to_use,
    #                 file_to_run, args.batch_size, args.num_layers,
    #                 args.num_nodes, args.num_train_determ)
