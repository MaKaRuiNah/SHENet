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
import pickle
import os
import sys
sys.path.insert(0,"./")
from utils.parser import args
from utils.data_utils import get_affine_transform,exec_affine_transform,generate_root_heatmaps,generate_root_distance_maps
from utils.data_utils import get_bezier_parameters,bezier_curve
# 26 9 19 27 24 79 24 98 19 7 10

class MOT(Dataset):

    def __init__(self, opt, sequences=None,split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        if split == 0:
            self.path_to_data = "./data/2DMOT2015/train/"
        else:
            self.path_to_data = "./data/2DMOT2015/test/"
        self.split = split
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.input_dim = opt.input_dim

        self.image_size = np.array(opt.image_size)
        self.heatmap_size = np.array(opt.heatmap_size)

        self.sample_rate = 1

        self.pred_kpt = {}
        self.data_idx = []
        self.gt_kpt = {}
        self.skip_rate = opt.skip_rate

        
        
        self.sigma = 10
        self.vis = False

        self.use_scene = True
        self.use_rough_data = True

        #subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        # acts = data_utils.define_actions(actions)
        
        if sequences is None:
            self.train_sequences = ['Venice-2','ADL-Rundle-6', 'PETS09-S2L1']
        else:
            self.train_sequences = sequences
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ])

        self.data = self.load_data()

    def load_data(self):
        idx = 0
        seq_len = self.in_n + self.out_n
        data = []
        #img_path_all = []
        for seq_name in self.train_sequences:
            print("Reading sequences {}".format(seq_name))
            file_path = "{}/{}.pkl".format(self.path_to_data,seq_name)

            pkl_file = open(file_path, 'rb')
            sample = pickle.load(pkl_file)
            pkl_file.close()

            key_list = sample.keys()

            person_num = len(key_list)
            if person_num == 0:
                continue

            # person_list = [key_list[i] for i in sample_list]
            
            for i in key_list:
                person = sample[i]

                num_frames = len(person)
                if num_frames < seq_len:
                    continue

                root = np.zeros((num_frames,2))
                img_path_list = []

                for j in range(num_frames):
                    point = np.array(person[j]['gt'])
                    img_path = person[j]['im_path']
                    img_path = "/home"+img_path

                    root[j,:] = point
                    img_path_list.append(img_path)


                valid_frames = np.arange(0, num_frames - seq_len + 1, 1)

                # tmp_data_idx_1 = [(seq, person)] * len(valid_frames)
                tmp_data_idx_1 = [idx] * len(valid_frames)
                tmp_data_idx_2 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                
                data.append({
                    "seq_name": seq_name,
                    "data_id":zip(tmp_data_idx_1, tmp_data_idx_2),
                    "img_path": img_path_list,
                    "root": root,
                    "num_frames":num_frames,
                })
                
                idx += 1
        return data

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        
        num_frames = self.data[key]['num_frames']
        fs = np.arange(start_frame, num_frames)
        
        img_path = self.data[key]['img_path'][fs[self.in_n-1]]
        # img_path_list = [img_path_list[i] for i in fs]
        root = self.data[key]['root'][fs]
        meta = torch.tensor([key,start_frame])
        return self.get_raw_data(img_path,root,meta)
    def get_raw_data(self,img_path,root,meta):
        data_numpy = cv2.imread(img_path)

        height,width,_ = data_numpy.shape
        if self.use_scene:
            
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            data_numpy = cv2.resize(data_numpy,(self.image_size[0],self.image_size[1]),interpolation=cv2.INTER_AREA)
            raw_img = self.transform(data_numpy)
        if self.use_rough_data:
            # input root data from the timestep input-1, input-1 as the origin point
            root = root[:self.in_n+self.out_n]
            points = get_bezier_parameters(root[:,0],root[:,1],degree=2)
            xvals, yvals = bezier_curve(points, nTimes=1000)
            idx = torch.linspace(0,999,root.shape[0])
            idx = idx.int()
            root_x = torch.from_numpy(xvals[::-1][idx])
            root_y = torch.from_numpy(yvals[::-1][idx])

            root_curve = torch.stack([root_x,root_y],dim=1)
            root_curve = root_curve - root_curve[0,:]
            target_points = torch.from_numpy(points)

            return root_curve,target_points,torch.tensor([width,height]),meta,raw_img
            
        else:
            root = root[:self.in_n+self.out_n]
            root_curve = root-root[0,:]
            return root_curve,root[0,:],torch.tensor([width,height]),meta,raw_img
    
    def evaluate(self,opt,all_preds,all_gts,mem_his=None):
        sample_num = len(self.data_idx)
        data = self.data
        pred_save = []
        
        out_dir = os.path.join(args.model_path,'SHENet/Venice/outputs/')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        output_path = out_dir+"venice.json"

        output_n = self.out_n
        frames_seq = [[] for _ in range(len(self.train_sequences))]
        assert sample_num == all_preds.shape[0]
        all_frames = 0
        for n in range(sample_num):
            key,start_frame = self.data_idx[n]
            num_frames = self.data[key]['num_frames']
            
            fs = np.arange(start_frame, num_frames)

            root = data[key]['root'][fs]
            gt_root = root

            img_path = data[key]['img_path'][fs[self.in_n-1]]
            # mem_his = memory_his[n]

            pred_root = all_preds[n,:,:2][self.in_n:]
            gt_root = all_gts[n,:,:2]
            
            seq_idx = self.train_sequences.index(img_path.split("/")[6])

            frames_seq[seq_idx].append(output_n)

            all_frames += output_n

            pred_save.append({'image_id': n, 'img_path':img_path,'root':gt_root.tolist(),
            'pred_root':pred_root.tolist(),
            })


        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)
