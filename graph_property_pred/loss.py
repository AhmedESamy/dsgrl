import torch
import torch.nn.functional as F

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class RegularizedLoss:
    
    def __init__(self, α=1.0, β=1., γ=1.0):
        self.α = α
        self.β = β
        self.γ = γ
        self.log = []
        
    def __update_log(self):
        self.log.append(self.terms)
        
    def __reset_terms(self):
        self.terms = [None, None, None]
        
    def __call__(self, z1, z2, **kwargs):
        self.__reset_terms()
        inv_loss = self.compute_inv_loss(z1, z2, **kwargs)
        var_loss = self.compute_var_reg(z1, z2, **kwargs)
        cov_loss = self.compute_cov_reg(z1, z2, **kwargs)
        self.__update_log()
        self.__reset_terms()
        return inv_loss + var_loss + cov_loss
    
    def __repr__(self):
        args = ",\n".join(f"\t{field}={value}" for field, value in self.__dict__.items())
        return f"{type(self).__name__}(\n{args}\n)"
        
    def compute_inv_loss(self, z1, z2):
        inv_loss = self.α * F.mse_loss(z1, z2)
        self.terms[0] = float(inv_loss.data)
        return inv_loss
    
    # def compute_var_reg(self, z1, z2):
    #     eps = 1e-4
    #     one = torch.Tensor([1.]).to(z1.device)
    #     std_1 = z1.var(dim=0).sqrt() + eps
    #     std_2 = z2.var(dim=0).sqrt() + eps
    #     v_z1 = torch.maximum(one, std_1).mean()
    #     v_z2 = torch.maximum(one, std_2).mean()
    #     var_reg = self.β * (v_z1 + v_z2)
    #     self.terms[1] = float(var_reg.data)
    #     return var_reg
    
    def compute_var_reg(self, z1, z2):
        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        std_loss = self.β * std_loss
        self.terms[1] = float(std_loss.data)
        return std_loss
    
    def compute_cov_reg(self, z1, z2):
        """
        The following code is borrowed from :
                https://github.com/vturrisi/solo-learn/blob/main/solo/losses/vicreg.py#L58
        """
        N, D = z1.size()
    
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        diag = torch.eye(D, device=z1.device)
        cov_reg = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D

        cov_reg = self.γ * cov_reg
        self.terms[2] = float(cov_reg.data)
        return cov_reg
    
    
    
    
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
        return mod_reg
