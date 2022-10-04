import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
import sys
sys.path.insert(0,"./")
from utils.parser import args
from model.position_embedding import add_seq_pos_emb
from model.transformers import generate_square_subsequent_mask,TransformerEncoder,TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer
from utils.loss_funcs import ClassificationLoss
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class SHENet(nn.Module):
    def __init__(self, opt):
        """
        Construct a MulT model.
        """
        super(SHENet, self).__init__()
        # self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.orig_d_tra, self.orig_d_scene = opt.origin_d_tra,opt.origin_d_scene   #56*56
        # self.d_tra, self.d_pose, self.d_v = 768, 0, 768
        self.embed_dim = opt.embed_dim
        self.nhead = opt.nhead
        self.num_encoder_layers = opt.num_encoder_layers
        self.num_decoder_layers = opt.num_decoder_layers
        self.max_seq_len = opt.max_seq_len
        self.out_dropout = opt.out_dropout

        self.input_n = opt.input_n
        self.output_n = opt.output_n

        self.output_dim = opt.output_dim

        # 1. Temporal convolutional layers
        self.proj_tra = nn.Conv1d(self.orig_d_tra, self.embed_dim, kernel_size=1, padding=0, bias=False)
        self.proj_img = nn.Linear(self.orig_d_scene, self.embed_dim)
        # self.conv_img = conv3x3(self.orig_d_v, self.embed_dim)

        self.seq_pos_emb = add_seq_pos_emb(self.max_seq_len,self.embed_dim)
        # 2. trajectory encoder: choose transformer encoder or lstm encoder
        # self.encoder_tra = self.get_encoder()
        self.encoder_tra = nn.LSTM(input_size=self.embed_dim, 
                            hidden_size=self.embed_dim,
                            num_layers=1,batch_first = True)

        self.backbone = self.load_vis_backbone(opt)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.dec_tra = self.get_decoder()
        self.dec_scene = self.get_decoder()

        # Projection layers
        self.norm_img = nn.LayerNorm(self.embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.combined_dim = 2*self.embed_dim

        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, self.output_dim)

    def load_static_memory(self,path):
        file_path = "./output/mot15/distances/bestTrajectoryAfterCluster.pickle"
        file = open(file_path, 'rb')
        memory_fut = np.array(pickle.load(file))
        return memory_fut
    
    def load_vis_backbone(self,opt):
        config = opt.vit_config_file
        checkpoint = opt.vit_checkpoint_file

        config = mmcv.Config.fromfile(config)
        config.model.pretrained = None
        config.model.train_cfg = None

        model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
        if checkpoint is not None:
            checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
            model.CLASSES = checkpoint['meta']['CLASSES']
            model.PALETTE = checkpoint['meta']['PALETTE']
 
        return model

    def extract_features(self,x):
        # build the model from a config file and a checkpoint file
        bkb = self.backbone.backbone
        decode_head = self.backbone.decode_head
        out = bkb(x)
        out = decode_head(out)  # (bz,150,64,64)
        out = out.reshape(out.shape[0],out.shape[1],-1)
        out = self.norm_img(self.proj_img(out))
        return out
    
    def extract_interact_features(self,x,cur_tra,scale):
        # build the model from a config file and a checkpoint file
        bkb = self.backbone.backbone
        decode_head = self.backbone.decode_head
        out = bkb(x)
        out = decode_head(out)  # (bz,150,64,64)
        bz,c,h,w = out.shape
        cur_tra = cur_tra/scale.unsqueeze(1).expand(-1,self.input_n,-1) * h  # (N,T,2)
        cur_tra = cur_tra.long()
        tra_index = w*cur_tra[:,:,1] + cur_tra[:,:,0]

        # out = out.reshape(out.shape[0],out.shape[1],-1).transpose(1,2)
        # out = self.norm_img(self.conv_img(out))  # (N,h*w,512)
        out = self.conv_img(out)
        out = out.reshape(bz,self.embed_dim,-1).transpose(1,2)
        person_fea = torch.index_select(out,1,tra_index.flatten())         # (N,T,512)
        enc_scene,_ = self.encoder_scene(person_fea)
        # torch.cat([person_fea,out],dim=1)
        return enc_scene
    
    def get_encoder(self):
        encoder_layer = TransformerEncoderLayer(self.embed_dim, self.nhead, self.embed_dim*4)
        encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers)
        return encoder

    def get_decoder(self):
        decoder_layer = TransformerDecoderLayer(self.embed_dim, self.nhead, self.embed_dim*4)
        decoder_norm = nn.LayerNorm(self.embed_dim)
        decoder = TransformerDecoder(decoder_layer, self.num_encoder_layers,decoder_norm)
        
        return decoder

    def forward(self, x_tra, x_scene,target_points=None,scale =None):
        """
        x_tra: [N,T,2]
        x_scene: [N,3,224,224]
        """
        batch_size,in_seq_len,_ = x_tra.shape

        # Project the trajectory features
        proj_tra =  self.proj_tra(x_tra.transpose(1,2))
        proj_tra = proj_tra.transpose(1,2)  # (N,T,D)

        # position encoding and mask
        seq_pos_emb = self.seq_pos_emb[None,:in_seq_len]  # (1,T,D)

        #x_emb = proj_tra + seq_pos_emb    # (N,T,D)
        x_emb = proj_tra
        self_mask = generate_square_subsequent_mask(in_seq_len).cuda()
        # encode the trajectory features
        enc_tra,_ = self.encoder_tra(x_emb)

        if scale is not None:
            cur_tra = x_tra + target_points[:,0].unsqueeze(1).expand(-1,in_seq_len,-1)
            enc_scene = self.extract_interact_features(x_scene,cur_tra,scale)
        else:
            enc_scene = self.extract_features(x_scene)

        dec_tra = self.dec_tra(enc_tra,enc_scene,None,None)
        dec_scene = self.dec_scene(enc_scene,enc_tra,None,None)


        last_dec_tra = dec_tra[:,-1]  # N,512
        last_dec_scene = self.avgpool(dec_scene.transpose(1,2)).squeeze(-1) # N,512

        last_dec_fea = torch.cat([last_dec_tra,last_dec_scene],dim=1)

        pred_proj = self.proj2(F.dropout(F.relu(self.proj1(last_dec_fea)), p=self.out_dropout, training=self.training))
        pred_proj += last_dec_fea    #(N,768)
        decoder_output = self.out_layer(pred_proj)
        outputs = decoder_output.reshape(batch_size,-1,2)

        outputs = torch.cat([x_tra,outputs],dim=1)
        
        return outputs