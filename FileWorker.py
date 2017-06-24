# -*- coding: utf-8 -*-
# !/usr/bin/env python


"""
四件事
遍历训练文件目录，根据annotations.csv中的结节坐标，切出训练用的npy
训练unet网络用来预测
遍历test文件目录，先切出整体肺部，之后用滑窗遍历肺部的文件，保存为npy
根据model预测npy为结节的概率
"""

import os
import pandas as pd
from TrainNodCrop import TrainNodCrop
from LungCrop import LungCrop
from NodeCrop import NodeCrop

test_path_list = ['../test_subset00/', '../test_subset01/', '../test_subset02/', '../test_subset03/',
                  '../test_subset04/']
train_path_list = ['../train_subset00/']
train_file_name = '../csv/train/annotations.csv'
train_npy_path = './train_npy'
test_npy_path = './test_npy'


class FileWorker(object):
    def train_start(self):
        for train_path in train_path_list:
            for file_name in os.listdir(train_path):
                file_path = os.path.join(train_path, file_name)
                if file_path.endswith(".mhd"):
                    df_nodes = self.get_node_position(file_name)
                    nod_crop = TrainNodCrop(file_path, df_nodes)
                    nod_crop.crop()

    def train_new_start(self):
        for train_path in train_path_list:
            for file_name in os.listdir(train_path):
                file_path = os.path.join(train_path, file_name)
                if file_path.endswith(".mhd"):
                    df_nodes = self.get_node_position(file_name)
                    node_crop = NodeCrop(file_path, train_npy_path, df_nodes)
                    node_crop.create_lung_nodule_mask()

    def get_node_position(self, file_name):
        name = file_name.split('.')[0]
        df = pd.read_csv(train_file_name)
        # df.set_index('seriesuid', inplace=True)
        return df.loc[df['seriesuid'] == name]

    def test_start(self):
        for test_path in test_path_list:
            for file_name in os.listdir(test_path):
                file_path = os.path.join(test_path, file_name)
                if file_path.endswith(".mhd"):
                    lung_crop = LungCrop(file_path, test_npy_path)
                    lung_crop.create_lung_mask()


if __name__ == '__main__':
    worker = FileWorker()
    worker.train_new_start()
    # worker.test_start()
    # worker.get_node_position("LKDS-00036.mhd")
