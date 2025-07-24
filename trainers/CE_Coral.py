import sys
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
    def __init__(self,
                 temperature: float,
                 static_weight: float,
                 verbose: bool,
                 device: torch.device):
        super().__init__()
        self.temperature = temperature
        self.static_weight = static_weight
        self.verbose = verbose
        self.device = device
        # self.projector = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64,64)
        # ).to(self.device)

   
    # def forward(self, emb_i, emb_j):
    #     """
    #     emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
    #     z_i, z_j as per SimCLR paper
    #     """
    #     z_i = F.normalize(emb_i, dim=1)
    #     #print(z_i.size())
    #     z_j = F.normalize(emb_j, dim=1)

    #     representations = torch.cat([z_i, z_j], dim=0)
    #     similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    #     if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
    #     batch_size = z_i.size()[0]   
    #     def l_ij(i, j):
    #         z_i_, z_j_ = representations[i], representations[j]
    #         sim_i_j = similarity_matrix[i, j]
    #         if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
    #         numerator = torch.exp(sim_i_j / self.temperature)
    #         one_for_not_i = torch.ones((2 * batch_size, )).scatter_(0, torch.tensor([i]), 0.0).to(self.device)
    #         if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
    #         denominator = torch.sum(
    #             one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
    #         )    
    #         if self.verbose: print("Denominator", denominator)
                
    #         loss_ij = -torch.log(numerator / denominator)
    #         #loss_ij = -torch.log(numerator)# / denominator
    #         if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
    #         return loss_ij.squeeze(0)

    #     N = batch_size
    #     loss = 0.0
    #     for k in range(0, N):
    #         loss += l_ij(k, k + N) + l_ij(k + N, k)
    #     return 1.0 / (2*N) * loss
    # def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         emb_i (Tensor): [batch_size, D] subsequence embeddings.
    #         emb_j (Tensor): [batch_size, D] full-sequence embeddings.
    #     Returns:
    #         Tensor: scalar contrastive loss.
    #     """
    #     # 1) Normalize and move to device
    #     z_i = F.normalize(emb_i, dim=1).to(self.device)  # [N, D]
    #     z_j = F.normalize(emb_j, dim=1).to(self.device)  # [N, D]
    #     N = z_i.size(0)

    #     # 2) Cross-similarity between subsequence and full-sequence
    #     sim = torch.matmul(z_i, z_j.t())  # [N, N]
    #     if self.verbose:
    #         print("Cross-view similarity matrix:\n", sim)

    #     # logits and log-probabilities via log-softmax for stability
    #     logits = sim / self.temperature    # [N, N]
    #     log_prob = F.log_softmax(logits, dim=1)  # stable log probabilities

    #     # weights: use softmax over raw similarities (or logits) with separate temp
    #     weights = torch.softmax(logits, dim=1)  # [N, N]

    #     # weighted loss
    #     loss_i = - (weights * log_prob).sum(dim=1)  # [N]
    #     return loss_i.mean()
    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_i (Tensor): [batch_size, D] full sequence embeddings.
            emb_j (Tensor): [batch_size, D] subsequence embeddings.
        Returns:
            Tensor: scalar combined contrastive loss.
        """
        # normalise
        # h_i = self.projector(emb_i.to(self.device))  # [N, D]
        # h_j = self.projector(emb_j.to(self.device))  # [N, D]
        z_full = F.normalize(emb_i, dim=1)  # [N, D]
        z_sub  = F.normalize(emb_j, dim=1)  # [N, D]
        N = z_full.size(0)

        # cos similarity matrix
        sim = torch.matmul(z_sub, z_full.t())  # [N, N]
        sim_long = torch.matmul(z_full, z_full.t())  # [N, N]

        N = sim_long.size(0)
        mask_self = torch.eye(N, device=self.device, dtype=torch.bool)
        sim_long = sim_long.masked_fill(mask_self, float('-inf'))
        if self.verbose:
            print("similarity matrix:\n", sim)

        logits = sim / self.temperature
        log_prob = F.log_softmax(logits, dim=1)     # [N, N] original contrastive loss

        # soft similarity loss
        weights = torch.softmax(sim_long / self.temperature, dim=1)      # [N, N]
        dynamic_loss_i = - (weights * log_prob).sum(dim=1)    # [N]
        dynamic_loss = dynamic_loss_i.mean()

        # statically define pos pairs
        # pos are (i,j) and (j,i)
        reps = torch.cat([z_full, z_sub], dim=0)        # [2N, D]
        sim_static = torch.matmul(reps, reps.t())       # [2N, 2N]
        # mask out self-sim
        mask_self = torch.eye(2*N, device=self.device, dtype=torch.bool)
        sim_static = sim_static.masked_fill(mask_self, float('-inf'))
        # compute logits with stability
        logits_static = sim_static / self.temperature
        log_prob_static = F.log_softmax(logits_static, dim=1)
        # static positives: (k, k+N) and (k+N, k)
        idx = torch.arange(N, device=self.device)
        pos_full = idx
        pos_sub  = idx + N
        loss_list = []
        for i in range(N):
            lp1 = log_prob_static[pos_full[i], pos_sub[i]]
            lp2 = log_prob_static[pos_sub[i], pos_full[i]]
            loss_list.append(- (lp1 + lp2) / 2)
        static_loss = torch.stack(loss_list).mean()

        # weighted sum loss
        loss = self.static_weight * static_loss + (1.0 - self.static_weight) * dynamic_loss
        return loss


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
    
class CETrainer(AbstractTrainer):
 
    # def __init__(self, args, model,  train_loader_source,train_loader_source_sample, train_loader_target, val_loader, test_loader, export_root):
    #     super().__init__(args,model, train_loader_source, train_loader_source_sample,train_loader_target, val_loader, test_loader, export_root)    
    def __init__(self, args, model,  train_loader_source, train_loader_target, train_loader_combine,val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader_source,train_loader_target, train_loader_combine, val_loader, test_loader, export_root) 
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.device = args.device
        self.temprature = args.temprature
        self.static_weight = args.static_weight
        self.ContrastiveLoss = ContrastiveLoss(self.temprature, self.static_weight,False,self.device)
        # self.ContrastiveLoss = ContrastiveLoss(self.temprature,False,self.device)
        self.context_is_mean = args.context_is_mean
        self.regularization = args.regularization
        self.alpha = args.alpha
        self.alpha_sst = args.alpha_sst
        self.beta_st = args.beta_st
        self.beta_t = args.beta_t
        self.beta_s = args.beta_s
        self.args = args
        self.train_loader_source  = train_loader_source
        self.train_loader_target  = train_loader_target
        self.train_loader_combine = train_loader_combine
    @classmethod
    def code(cls):
        return 'CE_trainer'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    #def calculate_loss(self, batch_source, batch_source_sample, batch_target):
    def calculate_loss(self, batch_source, batch_target):    
            seqs_s, pos_s, popular_s, seqs_s_s, pos_s_s, popular_s_s, mask = batch_source
            #print('source:',seqs_s)
            #print('source sample:',seqs_s_s)
            #print('source popularity:',popular)
            #print('source sample popularity:',popular_s_s)
            #print('mask:',mask)
            # seqs_s_s = seqs_s_s[:,-10:]
            # pos_s_s = pos_s_s[:,-10:]
            seqs_t, pos_t = batch_target
           
            context_s, logits_s,context_s_1,logits_s_1, context_s_s, logits_s_s,context_t, logits_t = self.model(seqs_s, seqs_s_s, seqs_t)  # B x T x V
            
            assert not torch.isnan(logits_s).any()
            assert not torch.isnan(logits_t).any()
            assert not torch.isnan(logits_s_s).any()
            logits_s_0 = logits_s[:,-1,:]
            logits_s_0 = logits_s_0.view(-1, logits_s_0.shape[-1])
            logits_s_1_0 = logits_s_1[:,-1,:]
            logits_s_1_0 = logits_s_1_0.view(-1, logits_s_1_0.shape[-1])
            logits_s = logits_s.view(-1, logits_s.shape[-1])  # (B*T) x V 
            logits_s_1 = logits_s_1.view(-1, logits_s_1.shape[-1])
            labels_s = pos_s.view(-1)  # (B*T) x 1
            pos_labels= torch.where(labels_s != 0)
            mask = mask.view(-1)
            mask_s= torch.where(mask != 0)
            mask_labels = labels_s[mask_s].unsqueeze(-1)
            #print(mask_labels.size())
            mask_logits_s = logits_s[mask_s]
            #print(mask_logits_s.size())
            mask_logits_s_pos = mask_logits_s.gather(-1, mask_labels).squeeze(-1)
            #print(mask_logits_s_pos.size())
            #print(mask_logits_s.size()
            context_s_1 = context_s.contiguous().view(-1, context_s.shape[-1])
            unmask_context_s = context_s_1[pos_labels]
            mask_context_s = context_s_1[mask_s]
            zero_labels = torch.where(pos_s == 0)
            context_s[zero_labels] = 0
            # Expand weights to match the dimensions of the input matrix
            if self.args.weighted_mean:
                weights = popular_s.unsqueeze(-1)  # shape (batch_size, sequence, 1)
                
                weighted_values = context_s * weights 
                weighted_sum = torch.sum(weighted_values, dim=1)
                # Sum the weights along the sequence dimension
                weights_sum = torch.sum(popular_s, dim=1, keepdim=True)  # shape (batch_size, 1)
                
                weights_sum = weights_sum + 1e-8

                # Calculate the weighted mean
                mean_context_s = weighted_sum / weights_sum
            else:
                mean_context_s = context_s.mean(1)
            
            #print(context_s.shape)
            loss_s = (self.ce(logits_s, labels_s)+self.ce(logits_s_1, labels_s))/2
            
           
            logits_s_s = logits_s_s.view(-1, logits_s_s.shape[-1])  # (B*T) x V  
            labels_s_s = pos_s_s.view(-1)  # (B*T) x 1
            pos_labels_s= torch.where(labels_s_s != 0)
            mask_labels_s = labels_s_s[pos_labels_s].unsqueeze(-1)
            #print(mask_labels_s.size())
            mask_logits_s_s = logits_s_s[pos_labels_s]
            #print(mask_logits_s_s.size())
            mask_logits_s_s_pos = mask_logits_s_s.gather(-1, mask_labels_s).squeeze(-1)
            #print(mask_logits_s_s_pos.size())
            context_s_s_1 = context_s_s.contiguous().view(-1, context_s_s.shape[-1])
            context_s_s_2 = context_s_s_1[pos_labels_s]
            #zero_labels = torch.where(pos_s_s == 0)
            context_s_s[zero_labels] = 0
            if self.args.weighted_mean:
                #print('weighted mean')
                weights = popular_s_s.unsqueeze(-1)  # shape (batch_size, sequence, 1)
                
                weighted_values = context_s_s * weights 
                weighted_sum = torch.sum(weighted_values, dim=1)
                # Sum the weights along the sequence dimension
                weights_sum = torch.sum(popular_s_s, dim=1, keepdim=True)  # shape (batch_size, 1)
                
                weights_sum = weights_sum + 1e-8

                # Calculate the weighted mean
                mean_context_s_s = weighted_sum / weights_sum
            else:
                #print('mean not weighted')
                mean_context_s_s = torch.mean(context_s_s,1)
            #print(context_s_s.shape)
            loss_s_s = self.ce(logits_s_s, labels_s_s)
           

            logits_t = logits_t.view(-1, logits_t.shape[-1])  # (B*T) x V
            labels_t = pos_t.view(-1)  # (B*T) x 1
            pos_labelt= torch.where(labels_t != 0)
            context_t = context_t.contiguous().view(-1, context_t.shape[-1])
            context_t = context_t[pos_labelt]
            
            loss_t = self.ce(logits_t, labels_t)

            #print(loss_t)
            # print(mask_logits_s.size()[0])
            # print(mask_logits_s_s.size()[0]) 
            
            #assert mask_logits_s.size()[0] == mask_logits_s_s.size()[0]
            
            if self.regularization in ['KL']:
                #####aligned KL divergence of the prediction on the masked source samples
                loss_c = KL_loss(mask_logits_s,mask_logits_s_s)
                #####KL divergence on the maksed source sample on prediction of the positve items only
                #loss_c = KL_loss(mask_logits_s_pos,mask_logits_s_s_pos)
                #print(loss_c)
            elif self.regularization in ['MMD']:
                loss_c = MMD(mask_context_s,context_s_s_2, 'rbf',self.device)
            elif self.regularization in ['Cosine']:
                # if self.context_is_mean == True:
                #     loss_c = Similarity(mean_context_s,mean_context_s_s)
                # else:
                loss_c = Similarity(context_s[:,-1,:],context_s_s[:,-1,:])
            elif self.regularization in ['Coral']:
                loss_c = Coral(context_s,context_s_s)
            elif self.regularization in ['Contrastive']:
                #print(mask_context_s.size())
                #print(context_s_s_1.size())
                if self.context_is_mean == 'true':
                    # Using mean of the context of source and sampled source of one user as positive pair 
                    #print('mean:', self.context_is_mean)
                    #print('Mean constrastive')
                    loss_c = self.ContrastiveLoss(mean_context_s,mean_context_s_s)
                else:
                    #Using same time step of the contexts of source and sampled source as positive pair
                    #print('mean:',self.context_is_mean)
                    #loss_c = self.ContrastiveLoss(mask_context_s,context_s_s_2)
                    #loss_c = self.ContrastiveLoss(context_s,context_s_s_1)
                    #print('last contrastive')
                    loss_c = self.ContrastiveLoss(context_s[:,-1,:],context_s_s[:,-1,:])
                
                # for ii in range(self.args.max_len):
                

            #loss_sst = Coral(context_s_s_1,context_t) # Coral loss between the contexts of sampled source and target
            #print(loss_sst)
            if self.args.KL == 1:
                loss_kl = KL_loss(logits_s_0,logits_s_1_0)
            else:
                loss_kl = 0 
            loss = self.beta_t*loss_t + self.beta_s*loss_s + self.beta_st*loss_s_s + self.alpha*(loss_c)+self.alpha_sst*loss_kl
            
            return loss,loss_t,loss_s,loss_s_s,loss_c,loss_kl

    def calculate_metrics(self, batch, which_model):
        seqs, candidates, labels = batch
        #print(candidates.shape)
        _,_,_,_,_,_,_, scores_t = self.model(seqs,seqs,seqs)  # B x T x V
        #print(scores)
        
        if which_model in ['Source-only','Target-only','DASR']:
            scores = scores_t
        else:
            print('Error:Unregonized trainer type')
            sys.exit(1)
        
        scores = scores[:, -1, :]  # B x V
        #print(scores.shape)
        scores = scores.gather(1, candidates)  # B x C
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def train_one_epoch(self, epoch, accum_iter):
        # —— anneal static_weight —— 
        warmup_epochs = 15
        if epoch < warmup_epochs:
            alpha = 1.0
        else:
            warmup = getattr(self.args, 'warmup_epochs', 40)

            frac   = min(epoch, warmup) / float(warmup)
            alpha  = 1.0 - frac
            # update the ContrastiveLoss module
        if hasattr(self, 'ContrastiveLoss'):
            self.ContrastiveLoss.static_weight = alpha

        # now delegate back to the base class to run the usual training loop
        return super().train_one_epoch(epoch, accum_iter)




