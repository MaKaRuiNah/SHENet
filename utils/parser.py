import argparse


parser = argparse.ArgumentParser(description='Arguments for running the scripts')

#ARGS FOR LOADING THE DATASET
parser.add_argument('--image_size',type=list,default=[224,224],help='transformed image size')
parser.add_argument('--heatmap_size',type=list,default=[224,224],help='transformed heatmap size')

parser.add_argument('--data_dir',type=str,default='../datasets/',help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
parser.add_argument('--input_n',type=int,default=10,help="number of model's input frames")
parser.add_argument('--output_n',type=int,default=50,help="number of model's output frames")
parser.add_argument('--skip_rate',type=int,default=1,choices=[1,20],help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')


#ARGS FOR THE MODEL

parser.add_argument('--origin_d_tra',type=int,default=2,help= 'dimensions of the origin trajecotry dim')
parser.add_argument('--origin_d_scene',type=int,default=3136,help= 'dimensions of the scene feature dim')

parser.add_argument('--input_dim',type=int,default=2,help= 'dimensions of the input coordinates')
parser.add_argument('--output_dim',type=int,default=100,help= 'dimensions of the output coordinates')
parser.add_argument('--embed_dim',type=int,default=512,help= 'dimensions of the embed dimensions')

parser.add_argument('--num_goal',type=int,default=1,help= 'order of cheb')
parser.add_argument('--memory_size',type=int,default=100,help= 'memory size')

parser.add_argument('--vit_config_file',type=str,default='./pretrained_models/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py',help= 'config_file')
parser.add_argument('--vit_checkpoint_file',type=str,default='./pretrained_models/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth',help= 'config_file')

parser.add_argument('--nhead',type=int,default=8,help= 'nhead in transformers')
parser.add_argument('--num_encoder_layers',type=int,default=6,help= 'num encoder layers in transformers')
parser.add_argument('--num_decoder_layers',type=int,default=6,help= 'num decoder layers in transformers')
parser.add_argument('--max_seq_len',type=int,default=512,help= 'max_seq_len for transformers postion embedding')
parser.add_argument('--out_dropout',type=float,default=0.,help= 'output dropout')

#ARGS FOR THE TRAINING
parser.add_argument('--gpus', type=str, default='3,4,5', help='gpu ids')
parser.add_argument('--mode',type=str,default='train',choices=['train','test','viz'],help= 'Choose to train,test or visualize from the model.Either train,test or viz')
parser.add_argument('--n_epochs',type=int,default=50,help= 'number of epochs to train')
parser.add_argument('--batch_size',type=int,default=32,help= 'batch size')
parser.add_argument('--batch_size_test',type=int,default=64,help= 'batch size for the test set')
parser.add_argument('--lr',type=int,default=0.005,help= 'Learning rate of the optimizer')
parser.add_argument('--use_scheduler',type=bool,default=True,help= 'use MultiStepLR scheduler')
parser.add_argument('--milestones',type=list,default=[15,30],help= 'the epochs after which the learning rate is adjusted by gamma')
parser.add_argument('--gamma',type=float,default=0.1,help= 'gamma correction to the learning rate, after reaching the milestone epochs')
parser.add_argument('--clip_grad',type=float,default=10000,help= 'select max norm to clip gradients')

# FFNet/MOT15/checkpoints/
parser.add_argument('--model_path',type=str,default='./output/',help= 'directory with the models checkpoints ')




args = parser.parse_args()




