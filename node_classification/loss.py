import torch.nn.functional as F
import torch


def sparse_identity(size):
    i = torch.arange(size)
    diag = torch.stack([i, i])
    return torch.sparse_coo_tensor(
        diag, torch.ones(size, dtype=torch.float32), (size, size))


class RegularizedLoss:
    
    def __init__(self, inv_w=1, cov_w=0.01, use_improved_loss=True):
        self.inv_w = inv_w
        self.cov_w = cov_w
        self.use_improved_loss = use_improved_loss
        
    def compute_inv(self, z1, z2):
        inv = 2 - 2 * (z1 * z2).sum(dim=-1).mean()
        return self.inv_w * inv
    
    def compute_cov_reg(self, z1, z2):
        if self.use_improved_loss:
            I = sparse_identity(size=z1.shape[1]).to(z1.device)
            cov_reg = self.cov_w * (
                (z1.t().matmul(z1) - I).norm() + 
                (z2.t().matmul(z2) - I).norm()
            )
        else:
            I = sparse_identity(size=z1.shape[0]).to(z1.device)
            cov_reg =  (
                (z1.matmul(z1.t()) - I).norm() + 
                (z2.matmul(z2.t()) - I).norm()
            )
        return self.cov_w * cov_reg
        
    def __call__(self, z1, z2):
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        inv = self.compute_inv(z1, z2)
        cov_reg = self.compute_cov_reg(z1, z2)
        return inv + cov_reg
    
    
class ModelRegularizer:
    
    def __init__(self, mod_w=1.):
        self.mod_w = mod_w
        
    def __call__(self, augmentor):
        aug1 = augmentor.augmentor1
        aug2 = augmentor.augmentor2
        regularize = ((isinstance(aug1, torch.nn.Linear) and
                       isinstance(aug2, torch.nn.Linear)) or
                      (isinstance(aug1, torch.nn.Sequential) and
                       isinstance(aug2, torch.nn.Sequential)))
        mod_reg = 0
        if regularize:
            params1 = aug1.parameters()
            params2 = aug2.parameters()
            for p1, p2 in zip(params1, params2):
                if p1.shape[0] == p2.shape[0]:
                    if len(p1.shape) == 1:
                        p = torch.stack([p1, p2])
                    else:
                        p = torch.cat([p1, p2])
                    p = F.normalize(p)
                    i = torch.eye(p.shape[0]).to(p.device)
                    mod_reg += (p @ p.T - i).norm()
            if mod_reg != 0:
                mod_reg = mod_reg * self.mod_w
        return mod_reg