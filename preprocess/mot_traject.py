import os.path as osp
from re import I

import cv2
import numpy as np
import torch
from PIL import Image
import pickle
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)

from preprocess.mot15_sequence import MOT15Sequence


class MOTraject(MOT15Sequence):
    """Multiple Object Tracking Dataset.

    This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

    Values for P are normally 18 and K 4.
    """

    def __init__(self, seq_name):
        super().__init__(seq_name)

        self.K = 20
        self.seq_name = seq_name

        self.data = self.build_samples()

    def build_samples(self):
        """Builds the samples out of the sequence."""

        tracks = {}
        num_frames = len(self.data)
        for i in range(num_frames):
            sample = self.data[i]
            for k, v in sample['gt'].items():
                track = {'person_id': k, 'im_path': sample['im_path'], 'gt': v}
                if k not in tracks:
                    tracks[k] = []
                tracks[k].append(track)

        # sample maximal self.max_per_person images per person and
        # filter out tracks smaller than self.K samples
        res = {}
        for k,v in tracks.items():
            l = len(v)
            if l >= self.K:
                # print(l)
                res[k] = v


        return res

if __name__ == '__main__':
    # data = MOTraject(seq_name='Venice-2').data
    # print(len(data))
    path = "./2DMOT2015/train/Venice-2.pkl"
    pkl_file = open(path, 'rb')
    res = pickle.load(pkl_file)
    pkl_file.close()
    print(res[1])