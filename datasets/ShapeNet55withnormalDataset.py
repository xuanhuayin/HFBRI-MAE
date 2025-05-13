import os
import torch
import numpy as np
import torch.utils.data as data
# from .io import IO
from .build import DATASETS
from utils.logger import *
import h5py

@DATASETS.register_module()
class ShapeNet_withnormal(data.Dataset):
    def __init__(self, config):

        in_filepath = config.PC_PATH + "/shapenetcore55_2048pts_withnormal_" + config.subset + ".h5"
        h5 = h5py.File( in_filepath )
        self.pc = h5["data"][:]
        self.label = h5["label"][:]
        self.name = h5["name"][:]

        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55_withnormal')
        print_log(f'[DATASET] Open file {in_filepath}', logger = 'ShapeNet-55_withnormal')
        print_log(f'[DATASET] {self.pc.shape[0]} instances were loaded', logger = 'ShapeNet-55_withnormal')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        pos = pc[:, 0:3 ]
        ori = pc[:, 3:6 ]
        centroid = np.mean( pos, axis=0 )
        pos = pos - centroid
        m = np.max(np.sqrt(np.sum(pos**2, axis=1)))
        pos = pos / m
        pc = np.hstack( [ pos, ori ] )
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):

        pointset = self.pc[ idx ]
        pointset = self.random_sample( pointset, self.sample_points_num )
        pointset = self.pc_norm( pointset )
        pointset = torch.from_numpy( pointset ).float()

        taxonomy_id = self.label[ idx ]
        model_id = self.name[ idx ].decode()
        instance_id = idx
        return taxonomy_id, model_id, instance_id, pointset

    def __len__(self):
        return self.pc.shape[0]
