#!/usr/bin/env python
# coding: utf-8

import torch
from utils import data_utils
import torch.nn as nn
from torch.nn import functional as F
from utils.data_utils import get_bezier_parameters,bezier_curve

class CurveLoss(nn.Module):
    """
    Computes the loss for the bezier curve loss
    """

    def __init__(self, memory):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """
        super(CurveLoss, self).__init__()
        # self.memory = torch.Tensor(memory)
        self.memory_curves,self.memory_points = self.get_memory(memory)
        self.curve_length = self.memory_curves.shape[1]
        self.criterion = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def write_memory(self,path):
        self.memory_curves = torch.load(path)
        return self.memory_curves

    def get_memory(self,memory):
        memory_curves = []
        memory_points = []
        for i in range(memory.shape[0]):

            points = get_bezier_parameters(memory[i,:,0],memory[i,:,1],degree=2)
            xvals, yvals = bezier_curve(points, nTimes=1000)
            idx = torch.linspace(0,999,memory.shape[1])
            idx = idx.int()
            root_x = torch.from_numpy(yvals[::-1][idx])
            root_y = torch.from_numpy(xvals[::-1][idx])

            root_curve = torch.stack([root_x,root_y],dim=1)   # (T,2)
            root_points = torch.from_numpy(points)    # (3,2)
            memory_curves.append(root_curve)
            memory_points.append(root_points)
        return torch.stack(memory_curves,dim=0).float(),torch.stack(memory_points,dim=0).float()
    
    def mean_pointwise_l2_distance(self,memory, ground_truth):

        stacked_ground_truth = ground_truth.repeat(memory.shape[0], 1, 1)
        return torch.pow(memory - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()
    
    def points2curve(self,points):
        xvals, yvals = bezier_curve(points, nTimes=1000)
        idx = torch.linspace(0,999,self.curve_length)
        idx = idx.int()
        root_x = torch.from_numpy(xvals[::-1][idx])
        root_y = torch.from_numpy(yvals[::-1][idx])

        root_curve = torch.stack([root_x,root_y],dim=1)   # (T,2)
        return root_curve
    
    def mse_loss(self,pred,target_curve,target_points):
        loss = self.criterion(pred,target_curve)

        return torch.sqrt(loss)
    
    def curve_loss(self,pred,target_curve,target_points):
        pred = pred+target_points[0,:]
        pred_points = get_bezier_parameters(pred[:,0].detach().cpu().numpy(),pred[:,1].detach().numpy(),degree=2)
        pred_points = torch.from_numpy(pred_points)

        pred_curve = self.points2curve(pred_points.numpy())  # (T,2)
        

        relative_pred_curve = pred_curve-pred_curve[0,:]
        relative_target_curve = target_curve-target_curve[0,:]
        loss_curve = self.criterion(relative_pred_curve,relative_target_curve)

        loss_points = self.criterion(pred_points,target_points)

        return torch.sqrt(loss_curve)
    
    def search_curve(self,preds):
        past_curve = preds[:10].reshape(1,-1)   # (T,2)->(1,2T)
        mem_len = len(self.memory_curves)
        past_curve = past_curve.expand(mem_len,-1)   # (M,2T)
        cmp_curves = self.memory_curves[:,:10,:].reshape(mem_len,-1)  # (M,T*2)
        # print(past_curve.shape,cmp_curves.shape)
        idx = torch.argmax(self.cos_sim(past_curve,cmp_curves))
        # idx = torch.argmin(torch.sum((past_curve-cmp_curves)**2,dim=1))
        return self.memory_curves[idx]
    
    def update_memory(self,loss,target,thresh=38.):
        if loss >= thresh:
            self.memory_curves = torch.cat([self.memory_curves,target.unsqueeze(0)],dim=0)

    def forward(self, preds, target,target_points,set_update=False):
        """
        Computes the loss on a batch.
        :param preds: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param target: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.memory_curves.device != preds.device:
            self.memory_curves = self.memory_curves.to(preds.device)
            self.memory_points = self.memory_points.to(preds.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(preds.device)
        batch_size = preds.shape[0]

        # if set_criterion:
        #     criterion = self.mse_loss
        # else:
        #     criterion = self.curve_loss

        true_index = torch.zeros(batch_size)
        true_preds = target.clone()
        # for logit, ground_truth in zip(clusters, target):
        for idx in range(batch_size):
            searched_curve = self.search_curve(preds[idx])
            pred = searched_curve[10:] + preds[idx,10:]
            true_preds[idx,10:] = pred
            loss = torch.sqrt(self.criterion(pred,target[idx,10:]))
            # loss use for others without search 
            # loss = torch.sqrt(self.criterion(preds[idx][10:],target[idx][10:]))
            if set_update:
                self.update_memory(loss,target[idx])
            loss = loss.to(preds.device)
            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        return batch_losses.mean(),true_preds

class ClassificationLoss(nn.Module):
    """
    Computes the loss for a constant lattice CoverNet model.
    """

    def __init__(self, memory):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """
        super(ClassificationLoss, self).__init__()
        self.memory = torch.Tensor(memory)
        # self.memory = torch.stack(memory,dim=0)
    
    def mean_pointwise_l2_distance(self,memory, ground_truth):
        """
        Computes the index of the closest trajectory in the lattice as measured by l1 distance.
        :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
        :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
        :return: Index of closest mode in the lattice.
        """

        stacked_ground_truth = ground_truth.repeat(memory.shape[0], 1, 1)
        return torch.pow(memory - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()

    def forward(self, preds, target):
        """
        Computes the loss on a batch.
        :param preds: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param target: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.memory.device != preds.device:
            self.memory = self.memory.to(preds.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(preds.device)
        batch_size = preds.shape[0]
        # pred_tra = preds[:,:100].reshape(batch_size,-1,2)
        # pred_clusters = preds[:,100:] 
        pred_clusters = preds

        true_index = torch.zeros(batch_size)
        # for logit, ground_truth in zip(clusters, target):
        for idx in range(batch_size):

            closest_memory_trajectory = self.mean_pointwise_l2_distance(self.memory, target[idx].unsqueeze(0))
            label = torch.LongTensor([closest_memory_trajectory]).to(preds.device)
            true_index[idx] = closest_memory_trajectory
            index = F.softmax(pred_clusters[idx],dim=0)
            index = torch.argmax(index,dim=0)
            # pred_tra_single = pred_tra[idx]+self.memory[index]
            # regression_loss = torch.mean(torch.norm(pred_tra_single - target[idx], dim=1))
        
            classification_loss = F.cross_entropy(pred_clusters[idx].unsqueeze(0), label)
            # regression_loss = torch.mean(torch.norm(pred_tra[idx] - target[idx], dim=1))
            # loss = classification_loss+regression_loss
            loss = classification_loss

            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        return batch_losses.mean() ,true_index

class AllClassificationLoss(nn.Module):
    """
    Computes the loss for a constant lattice CoverNet model.
    """

    def __init__(self, memory):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """
        super(AllClassificationLoss, self).__init__()
        self.memory = torch.Tensor(memory)
        # self.memory = torch.stack(memory,dim=0)
    
    def mean_pointwise_l2_distance(self,memory, ground_truth):
        """
        Computes the index of the closest trajectory in the lattice as measured by l1 distance.
        :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
        :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
        :return: Index of closest mode in the lattice.
        """

        stacked_ground_truth = ground_truth.repeat(memory.shape[0], 1, 1)
        return torch.pow(memory - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()

    def forward(self, logits, target):
        # 
        pred_trac = F.softmax(logits,dim=-1)
        pred_trac = torch.argmax(pred_trac,dim=-1)
        label_hot = F.one_hot(target,logits.shape[2]).float()
        loss = nn.CrossEntropyLoss()(logits,label_hot)
        return loss

def _neg_loss(preds, gts):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
    pred (batch x c x h x w)
    gt_regr (batch x c x h x w)
    '''
    pos_inds = gts.eq(1).float()
    neg_inds = gts.lt(1).float()
    
    neg_weights = torch.pow(1 - gts, 4)
    loss = 0
    for pred in preds:
        pred = torch.clamp(pred,min=1e-4,max=1-1e-4)
    
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        
        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss/len(preds)

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss
    def forward(self, output, target):
        
        return self.neg_loss(output, target)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target,target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

def mpjpe_error(batch_pred,batch_gt): 

    # batch_pred=batch_pred.contiguous().view(-1,2)
    # batch_gt=batch_gt.contiguous().view(-1,2)
    # error = torch.mean(torch.norm(batch_gt-batch_pred,2,1))
    # error =  torch.mean(torch.norm(batch_pred - batch_gt, dim=3))
    joints_weight = torch.tensor([3., 1.5, 1.2, 1., 1.5, 1.2, 1., 1.5, 1.2, 1., 1., 1.2, 1., 1., 1.2, 1., 1.]).cuda()
    #joints_weight = torch.tensor([5., 3., 2., 1., 3., 2., 1., 3., 2., 1., 1., 2., 1., 1., 2., 1., 1.]).cuda()
    error = torch.mean(torch.multiply(torch.norm(batch_pred- batch_gt, dim=3),joints_weight))

    return error
    
    
def euler_error(ang_pred, ang_gt):

    # only for 32 joints
    
    dim_full_len=ang_gt.shape[2]

    # pred_expmap[:, 0:6] = 0
    # targ_expmap[:, 0:6] = 0
    pred_expmap = ang_pred.contiguous().view(-1,dim_full_len).view(-1, 3)
    targ_expmap = ang_gt.contiguous().view(-1,dim_full_len).view(-1, 3)

    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors




