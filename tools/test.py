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
   
def test():
    model.load_state_dict(torch.load(ckpt+'model.pth.tar'))
    model.eval()

    running_loss=0
    n=0
    
    test_sequences = ["PETS09-S2L1"]                  

    dataset_test = datasets.MOT(args,sequences=test_sequences, split=1)
    print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)

    num_samples = len(dataset_test)
    all_preds = np.zeros((num_samples,args.output_n+args.input_n,2),dtype=np.float)
    all_gts = np.zeros((num_samples,args.output_n+args.input_n,2),dtype=np.float)

    memory_his = np.zeros(num_samples,dtype=np.int16)
    all_gt_index = np.zeros(num_samples,dtype=np.int16)
    static_memory = model.module.load_static_memory(ckpt)
    curve_loss = CurveLoss(static_memory)
    dynamic_memory = curve_loss.write_memory(ckpt+"mem_curves_full.pt")
    print("static memory size: ", dynamic_memory.shape)
    seq_len = args.input_n + args.output_n

    for cnt,(input_root,target,scale,meta,raw_img) in enumerate(test_loader):
        with torch.no_grad():
            batch_dim=input_root.shape[0]
            input_root = input_root.float().cuda()
            target = target.float().cuda()

            raw_img = raw_img.cuda()

            preds = model(input_root[:,:args.input_n],raw_img)
            # for concta and cross modal
            # loss,_= curve_loss(preds,input_root,target,False)
            # for search
            loss,preds= curve_loss(preds,input_root,target,False)


            running_loss+=loss*batch_dim
            gts = input_root.cpu().data.numpy() + target.unsqueeze(1).expand(-1,seq_len,-1).cpu().data.numpy()
            preds = preds.cpu().data.numpy() + target.unsqueeze(1).expand(-1,seq_len,-1).cpu().data.numpy()

            all_preds[n:n+batch_dim,:,:] =  preds
            all_gts[n:n+batch_dim,:,:] =  gts

            n+=batch_dim
            
    cur_loss = running_loss.detach().cpu()/n
    print('test loss: %.3f' %(cur_loss))
    dataset_test.evaluate(args,all_preds,all_gts)
  
if __name__ == '__main__':
    test()
   
