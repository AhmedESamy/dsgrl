import torch.nn.functional as F
import torch


def sparse_identity(size):
    i = torch.arange(size)
    diag = torch.stack([i, i])
    return torch.sparse_coo_tensor(
        diag, torch.ones(size, dtype=torch.float32), (size, size))


class RegularizedLoss:
    
    def __init__(self, α=1.0, β=1., γ=1.0, logger=None):
        self.α = α
        self.β = β
        self.γ = γ
        self.log = []
        self.logger = logger
        
    def __update_log(self):
        self.log.append(self.terms)
        
    def __reset_terms(self):
        self.terms = [None, None, None]
        
    def __compute_loss(self, z1, z2, **kwargs):
        inv_loss = self.compute_inv_loss(z1, z2, **kwargs)
        var_loss = self.compute_var_reg(z1, z2, **kwargs)
        cov_loss = self.compute_cov_reg(z1, z2, **kwargs)
        return inv_loss, var_loss, cov_loss
    
    def __compute_hetro_loss(self, z1, z2, **kwargs):
        inv_loss = var_loss = cov_loss = 0
        for key in z1:
            losses = self.__compute_loss(
                z1[key], z2[key]
            )
            inv_loss += losses[0]
            var_loss += losses[1]
            cov_loss += losses[2]
        return inv_loss, var_loss, cov_loss
        
    def __call__(self, z1, z2, **kwargs):
        self.__reset_terms()
        if isinstance(z1, dict):
            loss_fn = self.__compute_hetro_loss
        else:
            loss_fn = self.__compute_loss
        inv_loss, var_loss, cov_loss = loss_fn(
            z1, z2, **kwargs
        )
        if self.logger is not None:
            self.logger.log({"inv_loss": inv_loss, "var_loss": var_loss, "cov_loss": cov_loss})
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


class SimplifiedRegularizedLoss:
    
    def __init__(self, inv_w=1, cov_w=0.01, use_improved_loss=True, logger=None):
        self.inv_w = inv_w
        self.cov_w = cov_w
        self.use_improved_loss = use_improved_loss
        self.log = []
        self.logger = logger
        
    def __call__(self, z1, z2):
        self.__reset_terms()
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        inv = self.compute_inv(z1, z2)
        cov_reg = self.compute_cov_reg(z1, z2)
        if self.logger is not None:
            self.logger.log({"inv_loss": inv, "cov_reg": cov_reg})
        self.__update_log()
        self.__reset_terms()
        return inv + cov_reg
    
    def __repr__(self):
        args = ",\n".join(f"\t{field}={value}" for field, value in self.__dict__.items())
        return f"{type(self).__name__}(\n{args}\n)"
    
    def __reset_terms(self):
        self.terms = [None, None]
        
    def __update_log(self):
        self.log.append(self.terms)
        
    def compute_inv(self, z1, z2):
        inv = 2 - 2 * (z1 * z2).sum(dim=-1).mean()
        inv = self.inv_w * inv
        self.terms[0] = inv
        return inv
    
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
        cov_reg = self.cov_w * cov_reg
        self.terms[1] = cov_reg
        return cov_reg
    
    
class ModelRegularizer:
    
    def __init__(self, mod_w=0., logger=None):
        self.mod_w = mod_w
        self.log = []
        self.logger = logger
        
    def __call__(self, augmentor):
        aug1 = augmentor.augmentor1
        aug2 = augmentor.augmentor2
        mod_reg = 0
        if self.mod_w > 0:
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
                self.log.append(mod_reg)
        if self.logger is not None:
            self.logger.log({"mod_reg": mod_reg})
        return mod_reg