#%%
import torch
import torch.nn as nn
import numpy as np
import json
import re

from pathlib import Path

from cnn_model import CGP2CNN
from cgp_config import CgpInfoConvSet
from cgp import CGP
from my_data_loader import get_train_valid_loader
from datastuff import get_test_loader

MODEL_SAVE_DIR = 'save_dir'

#%%


def get_model_files_complete(model_folder_name, save_dir=MODEL_SAVE_DIR):
    folder_path = Path(f'{save_dir}/{model_folder_name}')

    model_state = torch.load(folder_path / "model_0.pth")
    config_pth = folder_path / "config.json"
    with open(config_pth, 'r') as config_f:
        config_dict = json.load(config_f)
    
    layer_cfg_pth = folder_path / "log.txt"
    with open(layer_cfg_pth, 'r') as layer_cfg_f:
        lines = layer_cfg_f.readlines()
        line = lines[-1]
        idx = line.index(',,')
        idx_depth_end = line.index(',', idx+2)
        arch_str = line[idx_depth_end+2:]
        num_depth = line[idx+2:idx_depth_end]
        config_dict['num_depth'] = int(num_depth)
        flat_gene = arch_str

    gene_list = flat_gene.split(',')
    even_cnt = 0
    pop = {}
    pop[0] = {}
    gene = []

    for idx, gene_i in enumerate(gene_list):
        if even_cnt % 2 == 0:
            gene.append([gene_i, gene_list[idx+1]])
        even_cnt += 1
        if len(gene_list) == idx+1:
            break

    pop[0]['gene'] = gene
    pop[0]['is_active'] = [True for _ in range(len(gene))]
    pop[0]['is_pool'] = [True for _ in range(len(gene))]

    return model_state, config_dict, pop

def get_model_files(model_folder_name, save_dir=MODEL_SAVE_DIR):
    folder_path = Path(f'{save_dir}/{model_folder_name}')

    model_state = torch.load(folder_path / "model_0.pth")
    config_pth = folder_path / "config.json"
    with open(config_pth, 'r') as config_f:
        config_dict = json.load(config_f)
    
    layer_cfg_pth = folder_path / "log-active.txt"
    with open(layer_cfg_pth, 'r') as layer_cfg_f:
        lines = layer_cfg_f.readlines()
        line = lines[-1]
        idx = line.index(',,')
        idx_depth_end = line.index(',', idx+2)
        arch_str = line[idx_depth_end+2:]
        num_depth = line[idx+2:idx_depth_end]
        config_dict['num_depth'] = int(num_depth)

    gene_list = arch_str.split('],')
    cgp_genes = []

    for idx, item in enumerate(gene_list):
        layer = item.split(',')[0]
        layer_start_idx = layer.find("'")
        layer_end_idx = layer.find("'", layer_start_idx+1)
        layer = layer[layer_start_idx+1:layer_end_idx]
        number = int(re.sub(r'[^0-9]', '', item.split(',')[1]))
        if 'resnet' in model_folder_name:
            number2 = int(re.sub(r'[^0-9]', '', item.split(',')[2]))
            cgp_genes.append([layer, number, number2])
        else:
            cgp_genes.append([layer, number])

    return model_state, config_dict, cgp_genes


def load_model(model_folder_pth, num_layer_eig, layer_eig_spacing):
    # gpuID = 0
    model_state, config, gene = get_model_files(model_folder_pth)

    # recreate the model
    # network_info = CgpInfoConvSet(
    #     arch_type=config['arch_type'], rows=1,
    #     cols=config['num_depth'], level_back=2,
    #     min_active_num=config['num_min_depth'],
    #     max_active_num=config['num_max_depth'])
    # cgp = CGP(network_info, None, arch_type=config['arch_type'],
    #         lam=1, img_size=32, init=config['init'])
    # cgp.pop[0].gene = np.array(gene[0]['gene'])
    # cgp.pop[0].is_active = np.array(gene[0]['is_active'])
    # cgp.pop[0].is_pool = np.array(gene[0]['is_pool'])
    # fix cgp with the loaded configuration

    torch.backends.cudnn.benchmark = True
    model = CGP2CNN(gene, in_channel=3, n_class=10, img_size=config['img_size'],
                    arch_type=config['arch_type'], register_hook=True,
                    num_layer_eig=num_layer_eig,
                    layer_eig_spacing=layer_eig_spacing)
    # criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda(gpuID)
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=0.001, betas=(0.5, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(config['num_epoch']))
    model.load_state_dict(model_state)
    model.eval()
    return model


def get_dataloaders(num_train=5000):
    return get_train_valid_loader(
        data_dir='./', batch_size=2, augment=True,
        random_seed=2018, num_workers=0, pin_memory=True,
        data_num=num_train)


def get_folder_names(net_type='vgg', num_train=500, save_dir=MODEL_SAVE_DIR):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    base_str = f'{net_type}-*-{num_train}-*'
    folder_names = sorted(save_dir.glob(base_str))

    return [(i.stem + i.suffix) for i in folder_names], folder_names


def split_eig_vectors(eig, num_eig_vec):
    eig_vec_dict = {}
    ctr = 0
    for idx, vec in enumerate(eig):
        if ctr % num_eig_vec == 0:
            ctr = 0
        if idx < num_eig_vec:
            eig_vec_dict[ctr] = []
        eig_vec_dict[ctr].append(vec)
        ctr += 1

    return eig_vec_dict



#%%
ex_path = "densenet-2019-08-08-20-16-57.800772-1000-38-id0"
# ex_path = 'resnet-2019-07-26-10-54-57.660333-25000-20-id2'
# ex_path = "vgg-2019-07-24-18-24-07.952185-10000-20-id2"
num_train = 500
num_test = num_train
net_type = 'resnet'
num_layer_eig = 3
layer_eig_spacing = 2
# f_names, _ = get_folder_names(net_type, num_train)

#%%

avg_eigenvalues = []
# for ex_path in f_names:
model = load_model(ex_path, num_layer_eig, layer_eig_spacing)
train_dl, valid_dl = get_dataloaders(num_train=num_train)
test_dl = get_test_loader('test-distortions/impulse_noise.npy', batch_size=16, num_test=num_test)
# test_dl = torch.utils.data.DataLoader(
    # test_dataset, batch_size=128, shuffle=True, num_workers=int(4), drop_last=True)

for _, (data, target) in enumerate(valid_dl):
    __ = model(data)
    print(f'len: {len(model.eigenvalues)}')

eig_vecs = model.eigenvalues
eig_vec_dict = split_eig_vectors(eig_vecs, num_layer_eig)

mean_real_dict = {}
for idx, eigenvalues in eig_vec_dict.items():
    # covariance_matrices = model.covariance_matrices
    real_stack = torch.stack([i[0] for i in eigenvalues], dim=0)
    # complex_stack = torch.stack([i[1] for i in eigenvalues], dim=1)
    mean_real_dict[idx] = torch.mean(real_stack, dim=0)
    # mean_complex = torch.mean(complex_stack, dim=0)
    print(f'mean: {mean_real_dict[idx]}')
    # print(f'mean_complex: {mean_complex}')
    # avg_eigenvalues.append(mean_real)

#%%
