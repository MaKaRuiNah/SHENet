import pickle
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
import random
import torch.cuda as cuda
import os
import sys
sys.path.insert(0,"./")

from datasets import mot15_curve as datasets
from torch.utils.data import DataLoader

from model.SHENet import SHENet
from utils.loss_funcs import CurveLoss
from utils.data_utils import define_actions
from utils.parser import args


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed: int):
    if seed == -1:
        seed = random.randint(0, 99999)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    return seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

model = SHENet(args)
model = torch.nn.DataParallel(model, device_ids=[0,1,2]).to(device)

print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# PETS
model_name = "SHENet/PETS/checkpoints/"
ckpt = os.path.join(args.model_path,model_name)

print(ckpt)
if not os.path.isdir(ckpt):
  os.makedirs(ckpt)

def train():
    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loss = []
    val_loss = []

    train_sequences = ["PETS09-S2L1"]
    val_sequences = ["PETS09-S2L1"]

    dataset = datasets.MOT(args,sequences=train_sequences, split=0)
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,pin_memory=True,drop_last=True)

    vald_dataset = datasets.MOT(args,sequences=val_sequences, split=1)
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True)

    best_loss = 1000
    cur_loss = -1
    static_memory = model.module.load_static_memory(ckpt)

    print("static memory size (before): ",len(static_memory))
    criterion = CurveLoss(static_memory)

    for epoch in range(args.n_epochs):
        running_loss=0
        n=0
        model.train()

        for cnt,(input_root,target,scale,meta,raw_img) in enumerate(data_loader):
            batch_dim=input_root.shape[0]
            n+=batch_dim
            input_root = input_root.float().cuda()
            target = target.float().cuda()
            meta = meta.cuda()
            scale = scale.cuda()
            raw_img = raw_img.cuda()

            preds = model(input_root[:,:args.input_n],raw_img)

            loss,_ = criterion(preds,input_root,target,True)

            if cnt % 500 == 0:
                print('[%d, %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
            
            optimizer.zero_grad() 
            
            loss.backward()

            optimizer.step()

            running_loss += loss*batch_dim

        train_loss.append(running_loss.detach().cpu()/n)  
        model.eval()

        with torch.no_grad():
            running_loss=0 
            n=0
            for cnt,(input_root,target,scale,meta,raw_img) in enumerate(vald_loader):
                batch_dim=input_root.shape[0]
                n+=batch_dim
                input_root = input_root.float().cuda()
                target = target.float().cuda()
                scale = scale.cuda()

                raw_img = raw_img.cuda()

                preds = model(input_root[:,:args.input_n],raw_img)

                loss,_ = criterion(preds,input_root,target,False)

                if cnt % 500 == 0:
                    print('[%d, %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
                running_loss+=loss*batch_dim
            cur_loss = running_loss.detach().cpu()/n
            val_loss.append(cur_loss)
        dynamic_memory = criterion.memory_curves
        print("dynamic memory size: ",len(dynamic_memory))

        if args.use_scheduler:
            scheduler.step()
        if best_loss > cur_loss:
            best_loss = cur_loss
            print('Epoch %d, best loss: %.3f' %(epoch + 1, best_loss)) 
            torch.save(dynamic_memory,ckpt+"mem_curves_bank.pt")
            torch.save(model.state_dict(),ckpt+'model.pth.tar')
    
    pickle_out = open(ckpt+"val_loss.pkl", "wb")
    pickle.dump(val_loss, pickle_out)
    pickle_out.close()
  
if __name__ == '__main__':
    train()
   