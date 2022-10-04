from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import torch
import cv2
import json
import torchvision.transforms as T
import torch.nn.functional as F
import math
import os
import sys
sys.path.insert(0,"./")
from utils.parser import args
from utils.data_utils import get_affine_transform,exec_affine_transform,generate_root_heatmaps,generate_root_distance_maps
from utils.data_utils import get_bezier_parameters,bezier_curve
# 26 9 19 27 24 79 24 98 19 7 10


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class ETH(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, opt,seq_name, split=0,obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(ETH, self).__init__()
        
        self.data_dir = "./data/eth_ucy/datasets"
        if split == 0:
            self.data_dir = os.path.join(self.data_dir, seq_name,'train')
        elif split == 1:
            self.data_dir = os.path.join(self.data_dir,seq_name, 'val')
        else:
            self.data_dir = os.path.join(self.data_dir,seq_name, 'test')

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.image_size = np.array(opt.image_size)
        self.heatmap_size = np.array(opt.heatmap_size)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ])

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        img_path_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()  # frames_id :0,10,20....
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):

                path_split = path.split('/')
                img_path = "/".join(path_split[:5])
                img_name = path_split[-1].split('_')[0]
                img_path = os.path.join(img_path,f"img/{img_name}/{int(frames[idx+self.obs_len])}.jpg")

                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)  # (T*P) *4, P is the average num person in the scene
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # num Person in  the sequence
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # (2,T)
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    
                    # Linear vs Non-Linear Trajectory
                    # ******
                    # points = get_bezier_parameters(curr_ped_seq[0,:],curr_ped_seq[1,:],degree=2)
                    # xvals, yvals = bezier_curve(points, nTimes=1000)
                    # curve_idx = torch.linspace(0,999,20)
                    # curve_idx = curve_idx.int()
                    # root_x = torch.from_numpy(xvals[::-1][curve_idx])
                    # root_y = torch.from_numpy(yvals[::-1][curve_idx])

                    # root_curve = torch.stack([root_x,root_y],dim=1)
                    # root_curve = root_curve.transpose(0,1)
                    # curr_ped_seq = root_curve.numpy()
                    # ******

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    img_path_list.append(img_path)

        self.num_seq = len(seq_list)
        self.img_path_list = img_path_list
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        img_path = self.img_path_list[index]
        data_numpy = cv2.imread(img_path)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        data_numpy = cv2.resize(data_numpy,(self.image_size[0],self.image_size[1]),interpolation=cv2.INTER_AREA)
        raw_img = self.transform(data_numpy)
        raw_img = raw_img.unsqueeze(0).expand(end-start,-1,-1,-1).float()
        
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            raw_img
        ]
        return out