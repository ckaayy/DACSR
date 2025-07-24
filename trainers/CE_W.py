from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def Coral(source,target):
    d1 = source.data.shape[0]
    d2 = target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(np.sqrt(d1*d2))
    return loss
# def Coral(source,target):
#     d1 = source.data.shape[0]
#     #print(d1)
#     d2 = target.data.shape[0]
#     #print(d2)
#     xm = torch.mean(source, 0, keepdim=True) 
#     xmt = torch.mean(target, 0, keepdim=True) 
#     # frobenius norm between source and target
#     loss = torch.mean(torch.mul((xm - xmt), (xm - xmt)))
#     loss = loss#/(np.sqrt(d1*d2))
#     return loss

def Similarity(source,target):
    #print(source.size())
    #print(target.size())
    #z_s = torch.mean(source,1)
    #z_t = torch.mean(target,1)
    z_s = F.normalize(source, dim=1)#
    z_t = F.normalize(target, dim=1)
    loss = 1 - F.cosine_similarity(z_s, z_t).mean()

    return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, verbose,device):
        super().__init__()
        self.temperature= temperature
        self.verbose = verbose
        self.device = device    
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        #print(z_i.size())
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
        batch_size = z_i.size()[0]   
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * batch_size, )).scatter_(0, torch.tensor([i]), 0.0).to(self.device)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss

def KL_loss(p, q, pad_mask=None):
        #print('source logit:', p)
        #print('source sample logit:',q)
        p_loss = F.kl_div(torch.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        #print(p_loss)
        q_loss = F.kl_div(torch.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        #print('KL1:', p_loss)
        q_loss = q_loss.sum()
        #print('KL2:', q_loss)

        loss = (p_loss + q_loss) / 2
        #print('KL:', loss)
        return loss
    
def MMD(x, y, kernel,device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    #print(xx.size())
    #print(yy.size())
    #print(zz.size())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    #print(rx.size())
    #print(ry.size())
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
          
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)
    
class CEWTrainer(AbstractTrainer):
 
    # def __init__(self, args, model,  train_loader_source,train_loader_source_sample, train_loader_target, val_loader, test_loader, export_root):
    #     super().__init__(args,model, train_loader_source, train_loader_source_sample,train_loader_target, val_loader, test_loader, export_root)    
    def __init__(self, args, model, train_loader_source, train_loader_target, train_loader_combine, val_loader, test_loader, export_root):
        super().__init__(args,model, train_loader_source, train_loader_target, train_loader_combine, val_loader, test_loader, export_root) 
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.device = args.device
        self.ContrastiveLoss = ContrastiveLoss(args.temprature,False,self.device)
        self.context_is_mean = args.context_is_mean
        self.regularization = args.regularization
        self.alpha = args.alpha
        self.alpha_sst = args.alpha_sst
        self.beta_st = args.beta_st
        self.beta_t = args.beta_t
        self.beta_s = args.beta_s
    @classmethod
    def code(cls):
        return 'CEW_trainer'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    #def calculate_loss(self, batch_source, batch_source_sample, batch_target):
    def calculate_loss(self, batch_combine):    
            
            seqs_t, pos_t = batch_combine
           
            context_t, logits_t = self.model(seqs_t) 
           
            assert not torch.isnan(logits_t).any()
           
            logits_t = logits_t.view(-1, logits_t.shape[-1])  # (B*T) x V
            labels_t = pos_t.view(-1)  # (B*T) x 1
            pos_labelt= torch.where(labels_t != 0)
            context_t = context_t.contiguous().view(-1, context_t.shape[-1])
            context_t = context_t[pos_labelt]
            
            loss = self.ce(logits_t, labels_t)
        
            loss_t, loss_s, loss_s_s,loss_c,loss_sst = 0, 0, 0, 0, 0
            return loss,loss_t,loss_s,loss_s_s,loss_c,loss_sst

    def calculate_metrics(self, batch, which_trainer):
        seqs, candidates, labels = batch
        #print(candidates.shape)
        _, scores = self.model(seqs)  # B x T x V
        #print(scores)
        scores = scores[:, -1, :]  # B x V
        #print(scores.shape)
        scores = scores.gather(1, candidates)  # B x C
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics



