import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *
from pytorch3d.io import load_ply

@DATASETS.register_module()
class OmniObject3D(data.Dataset):
    def __init__(self, config):
        self.samplelist = []
        self.data_path = config.DATA_PATH
        path = self.data_path + "/omniobject_" + config.subset + ".txt"
        with open( path ) as f:
            for line in f:
                line = line.rstrip().split(" ")
                self.samplelist.append( [ line[0], int(line[1]) ] )

        self.npoints = config.N_POINTS

        self.cache = {}  # from index to (point_set, cls) tuple
        self.cache_size = 10000

        print_log('The number of data: %d' % len( self.samplelist ), logger = 'OmniObject3D')

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

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            path = self.data_path + "/" + self.samplelist[index][0] + "/pcd_1024.ply"
            point_set, _ = load_ply( path )
            point_set = point_set.numpy().copy()
            cls = self.samplelist[index][1]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        point_set = self.pc_norm( point_set )

        # resample points
        if( self.npoints < point_set.shape[0] ):
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
            point_set = point_set[choice, :]

        # shuffle points
        permutation = np.arange(self.npoints)
        np.random.shuffle(permutation)
        point_set = point_set[permutation]

        return 'OmniObject3D', 'sample', (point_set, cls)

    def __len__(self):
        return len( self.samplelist )
