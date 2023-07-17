import torch

def normalize_min_max(A):
    AA = A.clone()
    AA = AA.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(A.size())
    return AA

def psnr(img1,img2) :
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10( 1. / torch.sqrt(mse))


def burg_bregman_divergence(x,y):
    x = x.reshape((x.shape[0], -1)) 
    y = y.reshape((y.shape[0], -1)) 
    return ((x/(y + 1e-10)) - torch.log(x + 1e-10) + torch.log(y + 1e-10) - 1).sum(axis=-1)

def KL_divergence(x,y):
    x = x.reshape((x.shape[0], -1)) 
    y = y.reshape((y.shape[0], -1)) 
    x = x.clamp(0.001,100)
    y = y.clamp(0.001,100)
    return (x*torch.log(x) -x*torch.log(y)).sum(axis=-1)
