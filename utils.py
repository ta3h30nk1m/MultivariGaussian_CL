import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.utils import _standard_normal
import os,sys
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)

class own_MN(MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        super(own_MN, self).__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        return
    
    #@torch.no_grad()
    def rsample(self, eps):
        with torch.no_grad():
            #shape = self._extend_shape(sample_shape)
            #eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)
    
    def _extend_shape(self, size):
        shape = super()._extend_shape(size)
        return shape
    


########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def create_normal_sample(means, variances, n_samples=1, seed=1993):
    rng = np.random.RandomState(seed)
    size = means.shape[0]
    X = rng.normal(means, variances, (n_samples, size))
    X[X<0] = 0
    return X

def create_multivariate_sample(means, covariances, n_samples=1, seed=1993):
    #rng = np.random.RandomState(seed)
    X = np.random.multivariate_normal(means, covariances, n_samples)
    X[X<0] = 0
    return X

def regenerate_multivari_feature(task, Means, Covs):
    reg_x = create_multivariate_sample(Means[0], Covs[0], 1000)
    reg_y = np.full(1000, 0)
    for i in range(1, 2*task):
        reg_x = np.concatenate((reg_x, create_multivariate_sample(Means[i], Covs[i], 1000)), axis=0)
        reg_y = np.concatenate((reg_y, np.full(1000, i)), axis=0)
        
    reg_x, _, reg_y, _, = train_test_split(reg_x, reg_y, test_size = 0.0001)    
    return torch.tensor(reg_x, dtype=torch.float32), torch.LongTensor(reg_y)

def generate(task, buffer):
    reg_x = torch.empty(0)
    reg_y = torch.empty(0, dtype=torch.long)
    d_k = own_MN(torch.Tensor(buffer[0].means_[0]).cuda(), torch.Tensor(buffer[0].covariances_[0]).cuda())
    for i in range(0, 2*task):
        for _ in range(3):
            shape = d_k._extended_shape(torch.Size((111,)))
            eps = _standard_normal(shape, dtype=d_k.loc.dtype, device=d_k.loc.device)
            for j in range(3):
                mean, covariance = buffer[i].means_[j], buffer[i].covariances_[j]
                mean, covariance = torch.Tensor(mean).cuda(), torch.Tensor(covariance).cuda()
                d_k = own_MN(mean, covariance)
                x_k = d_k.rsample(eps).cpu()
                x_k[x_k<0]=0
                reg_x = torch.cat((reg_x, x_k))
            reg_y = torch.cat((reg_y, torch.full((333,), i, dtype=torch.long)))
    
    reg_x, _, reg_y, _, = train_test_split(reg_x, reg_y, test_size = 0.0001)    
    return reg_x, reg_y