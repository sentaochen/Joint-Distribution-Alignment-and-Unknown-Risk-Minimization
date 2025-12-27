import torch

def pairwise_distances(x, y, power=2, sum_dim=2):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)
    dist = torch.pow(x-y, power).sum(sum_dim)
    return dist

def StandardScaler(x,with_std=False):
    mean = x.mean(0, keepdim=True)
    std = x.std(0, unbiased=False, keepdim=True)
    x -= mean
    if with_std:
        x /= (std + 1e-10)
    return x



def RCS_loss(Xs,ys,Xt,yt0, DEVICE, lamda=1e-2):
    X_batch = torch.cat((Xs,Xt), dim=0)
    pairwise_dist = torch.cdist(X_batch, X_batch, p=2)**2 
    sigma = torch.median(pairwise_dist[pairwise_dist!=0]) 
    FX_norm = torch.sum(X_batch ** 2, axis = -1)
    K = torch.exp(-(FX_norm[:,None] + FX_norm[None,:] - 2 * torch.matmul(X_batch, X_batch.t())) / sigma) # feature kernel matrix    
    y = torch.cat((ys, yt0), dim = 0)
    ns, nt = len(ys), len(yt0)
    Deltay = torch.as_tensor(y[:,None]==y, dtype=torch.float32, device=DEVICE) # label kernel matrix  
    P = torch.as_tensor(K * Deltay, dtype=torch.float32) # product kernel matrix
    Ps, Pt = P[:ns], P[ns:]
    H = 1.0 / nt * torch.matmul(Pt.t(), Pt) 
    invM = torch.inverse(H + lamda * torch.eye(ns+nt, device=DEVICE))
    b = (torch.mean(Ps,axis=0)[:,None])
    theta = torch.mm(invM, b)
    RCS = 2.0 * b.T @ theta - theta.T @ H @ theta - 1
    
    return torch.mean(RCS)


