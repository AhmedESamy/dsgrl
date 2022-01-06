import torch
import torch.nn.functional as F


class RegularizedLoss:
    
    def __init__(self, α=1.0, β=1., γ=1.0):
        self.α = α
        self.β = β
        self.γ = γ
        self.log = []
        self.reset_losses()
        
    def compute_inv_loss(self, z1, z2):
        inv_loss = self.α * F.mse_loss(z1, z2)
        self.losses[0] = float(inv_loss.data)
        return inv_loss
    
    def compute_var_loss(self, z1, z2):
        var_loss = self.β * variance_loss(z1, z2)
        self.losses[1] = float(var_loss.data)
        return var_loss
    
    def compute_cov_loss(self, z1, z2):
        cov_loss = self.γ * covariance_loss(z1, z2)
        self.losses[2] = float(cov_loss.data)
        return cov_loss
    
    def __call__(self, z1, z2, **kwargs):
        inv_loss = self.compute_inv_loss(z1, z2, **kwargs)
        var_loss = self.compute_var_loss(z1, z2, **kwargs)
        cov_loss = self.compute_cov_loss(z1, z2, **kwargs)
        self.update_log()
        self.reset_losses()
        return inv_loss + var_loss + cov_loss
    
    def update_log(self):
        self.log.append(self.losses)
        
    def reset_losses(self):
        self.losses = [None, None, None]
    
    
class ModelRegularizer:
    
    """
    A model regularizer for the learnable feature augmentaion
    """
    
    def __init__(self, λ=1.):
        self.λ = λ
        self.log = []
        
    def __call__(self, augmentor):
        """
        Returns the augmentor model regularization loss
        
        Args:
            augmentor: The augmentor model
        
        """
        mod_reg = 0
        if augmentor.name == "f":
            params1 = augmentor.augmentor1.parameters()
            params2 = augmentor.augmentor2.parameters()
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
                mod_reg = mod_reg * self.λ
                self.log.append(float(mod_reg.data))
        return -mod_reg

    
"""
The following code is borrowed from https://github.com/vturrisi/solo-learn/blob/main/solo/losses/vicreg.py
"""

def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()
    
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss