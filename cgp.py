#!/usr/bin/env python3

import csv
import time
import numpy as np
import math

# gene[f][c] f:function type, c:connection (nodeID)


class Individual(object):
    """Contains the gene which randomises the layers chosen for the DNN"""

    def __init__(self, net_info, init, arch_type):
        self.arch_type = arch_type
        self.net_info = net_info
        self.gene = np.zeros(
            (self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(
            self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.is_pool = np.empty(
            self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = None
        self.init_gene()           # generate initial individual randomly

    def init_gene(self):
        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            # type gene
            if n < self.net_info.node_num:
                type_num = self.net_info.func_type_num
            else:
                type_num = self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            if n == 0:
                for i in range(self.net_info.max_in_num):
                    self.gene[n][i + 1] = 0
            else:
                for i in range(self.net_info.max_in_num):
                    # self.gene[n][i + 1] = np.random.randint(0, n)
                    # self.gene[n][i + 1] = n-1  # prev. used for ResNet/VGG calc
                    self.gene[n][i + 1] = n

        self.check_active()

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]  # Gets layer_type value
            if n >= self.net_info.node_num:    # output node
                in_num = self.net_info.out_in_num[t]
            else:    # intermediate node
                in_num = self.net_info.func_in_num[t]

            # looping because there might be multiple inputs to the NN
            for i in range(in_num):
                # >Stopping condition<
                # Gets currents elements `rnd val 0..index`
                # `input_num` is 1 --> checking if not at beginning of layers
                if self.gene[n][i+1] >= self.net_info.input_num:
                    # 
                    self.__check_course_to_out(
                        # self.gene[n][i+1])  # prev used for ResNet/VGG calc
                        self.gene[n][i+1] - self.net_info.input_num)

    def check_active(self):
        # clear
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)

    def check_pool(self):
        is_pool = True
        pool_num = 0
        for n in range(self.net_info.node_num + self.net_info.out_num):
            if self.is_active[n]:
                if self.gene[n][0] > 12:
                    is_pool = False
                    pool_num += 1
        return is_pool, pool_num

    def __mutate(self, current, min_int, max_int):
        mutated_gene = current
        # Will result in an endless loop, because `np.random.randint(1) = 0`
        # Means when `current = 0` and `mutated_gene = 0` the loop is endless
        # TODO: is the error that `mutated_gene` can become `0` or is the while
        # condition incorrect?
        while current == mutated_gene:
            # NOTE: inserting +1 into the `randint()` leads to a lot of index
            # out of range errors and thus is not correct.
            mutated_gene = min_int + np.random.randint(max_int - min_int)
        return mutated_gene

    def mutation(self, mutation_rate=0.01):
        active_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            if n < self.net_info.level_back:
                for i in range(self.net_info.max_in_num):
                    if np.random.rand() < mutation_rate:
                        self.gene[n][i + 1] = 0
                        if self.is_active[n] and i < in_num:
                            active_check = True
            else:
                for i in range(self.net_info.max_in_num):
                    if np.random.rand() < mutation_rate:
                        # self.gene[n][i+1] = self.__mutate(self.gene[n][i+1], min_connect_id, max_connect_id)
                        self.gene[n][i+1] = self.__mutate(
                            self.gene[n][i+1], n-self.net_info.level_back, n)
                        if self.is_active[n] and i < in_num:
                            active_check = True

        self.check_active()
        return active_check

    def count_active_node(self):
        return self.is_active.sum()

    def active_net_list(self):
        if self.arch_type == 'resnet':
            net_list = [["input", 0, 0]]
        elif self.arch_type == 'vgg' or self.arch_type == 'densenet':
            net_list = [["input", 0]]
        active_cnt = np.arange(self.net_info.input_num +
                               self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:    # intermediate node
                    type_str = self.net_info.func_type[t]
                else:    # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i+1]]
                               for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list


class CGP(object):
    def __init__(self, net_info, eval_func, arch_type, lam=4, img_size=32,
                 init=False):
        self.lam = lam
        self.max_pool_num = int(math.log2(img_size) - 2)
        self.pop = [Individual(net_info, init, arch_type)
                    for _ in range(1 + self.lam)]
        active_num = self.pop[0].count_active_node()
        _, pool_num = self.pop[0].check_pool()
        while active_num < self.pop[0].net_info.min_active_num or pool_num > self.max_pool_num:
            self.pop[0].mutation(1.0)
            active_num = self.pop[0].count_active_node()
            _, pool_num = self.pop[0].check_pool()
        self.eval_func = eval_func
        self.num_gen = 0
        self.num_eval = 0
        self.init = init

    def _log_data(self, net_info_type='active_only', start_time=0):
        log_list = [self.num_gen, self.num_eval, time.time(
        )-start_time, self.pop[0].eval, self.pop[0].count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(self.pop[0].active_net_list())
        elif net_info_type == 'full':
            log_list += self.pop[0].gene.flatten().tolist()
        else:
            pass
        return log_list

    def _log_data_children(self, net_info_type='active_only', start_time=0, pop=None):
        log_list = [self.num_gen, self.num_eval,
                    time.time()-start_time, pop.eval, pop.count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        else:
            pass
        return log_list

    def load_log(self, log_data):
        self.num_gen = log_data[0]
        self.num_eval = log_data[1]
        net_info = self.pop[0].net_info
        self.pop[0].eval = log_data[3]
        self.pop[0].gene = np.array(log_data[5:]).reshape(
            (net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()
