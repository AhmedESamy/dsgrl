import torch.nn.functional as F
import torch

import numpy as np

from tqdm import tqdm
import time

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class Trainer:
    
    """
    Carries out the self-supervised training and inference phase of an experiment.
    
    Args:
        loader (list or PyG DataLoader): A data loader
        model (nn.Module) : The model to be trained
        optimizer: The optimizer
        loss_fn (callable): The loss function
        agg_fn (callable): A callable function used for aggregating the latent 
                        representations from multiple views.
        mod_reg_fn (callable): A callable function for model regularization
        epochs (int): The number of training epochs
    """
    
    def __init__(self, loader, model, optimizer, loss_fn, mod_reg_fn, agg_fn, epochs=10):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.mod_reg_fn = mod_reg_fn
        self.agg_fn = agg_fn
        self.epochs = epochs
        
    def _train_step(self, pbar=None):
        """
        A single training step for a batch of PyG Data objects in 
        the dataloader
        """
        self.model.train()
        for batch_counter, data in enumerate(self.loader):
            z1, z2 = self.model(data)
            loss = self.loss_fn(z1, z2)
            mod_reg = self.mod_reg_fn(self.model.augmentor)
            loss = loss + mod_reg
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if pbar is not None:
                pbar.update(batch_counter)
            
    def infer(self, loader, desc="", as_numpy=True):
        """
        Infers the lattent representation and collects their corresponding
        labels.
        Returns an embedding matrix (Tensor or numpy array) and the labels
        (1d Tensor or numpy array)
        
        Args:
            loader (List or PyG DataLoader): The dataloader
            desc (str): A description message
            as_numpy (bool): Whether to return the outputs as numpy array
        """
        self.model.eval()
        z = y = None
        
        for data in tqdm(loader, desc=desc):
            z1, z2 = self.model.infer(data)
            z_ = self.agg_fn(z1, z2).detach().cpu()
            y_ = data.y
            if z is None:
                z, y = z_, y_
            else:
                z = torch.cat([z, z_])
                y = torch.cat([y, y_])
        if as_numpy:
            return z.numpy(), y.numpy()
        return z, y
    
    def fit(self):
        """
        Fits a self-supervised model
        """
        # TODO: Dynamic progress bar
        # show_batch_progress = len(self.loader) > 500
        # iters = len(self.loader) if show_batch_progress else self.epochs
        delta = []
        for i in tqdm(range(self.epochs), desc="Self-supervised training"):
            start = time.time()
            self._train_step()
            delta.append(time.time() - start)
            
        delta = delta[1:] if len(delta) > 1 else delta
        print("Average runtime per epoch "
              f"{np.mean(delta):.6f} +/-"
              f"{np.std(delta):.6f}")
            
    def log_loss(self, path):
        """
        Logs the trace of the loss to a specified path 
        """
        if len(self.loss_fn.log):
            print(f"Logging training loss to {path}")
            with open(path, "w") as f:
                f.write("Invariance,Variance,Covariance\n")
                for inv_l, var_l, cov_l in self.loss_fn.log:
                    f.write(f"{inv_l},{var_l},{cov_l}\n")
            
    def log_reg(self, path):
        """
        Logs the trace of the regularization loss to a specified path 
        """
        if len(self.mod_reg_fn.log) > 0:
            print(f"Logging the regulrarization values for the augmenter to {path}")
            with open(path, "w") as f:
                f.write("AugmentationRegularizer\n")
                for v in self.mod_reg_fn.log:
                    f.write(f"{v}\n")