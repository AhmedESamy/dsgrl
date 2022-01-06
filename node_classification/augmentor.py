from gnn import Encoder
from utils import compute_sim, topk_edges

from torch_geometric.utils import softmax
import torch.nn.functional as F
import torch.nn as nn
import torch

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class FeatureAugmentor(nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.augmentor1 = nn.Linear(in_dim, out_dim)
        self.augmentor2 = nn.Linear(in_dim, out_dim)
        
    @property
    def name(self):
        return "f"
        
    def forward(self, x, **kwargs):
        device = next(self.parameters()).device
        x1 = self.augmentor1(x.to(device))
        x1 = F.dropout(x1, self.dropout, self.training)
        x2 = self.augmentor2(x.to(device))
        x2 = F.dropout(x2, self.dropout, self.training)
        return x1, x2
    
    @torch.no_grad()
    def inference(self, x_all, batch_size=1000):
        """
        Subgraph inference code adapted from PyTorch Geometric:
        https://github.com/rusty1s/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py#L36
        
        Compute representations of nodes layer by layer, using *all*
        available edges. This leads to faster computation in contrast to
        immediately computing the final representations of each batch.
        
        """
        pbar = tqdm(total=x_all.size(0))
        pbar.set_description('Inferring views')
        
        x1s, x2s = [], []
        for i in range(0, x_all.shape[0], batch_size):
            end = i + batch_size if x_all.shape[0] - batch_size > i else x_all.shape[0]
            x = x_all[i:end]
            x1, x2 = self(x)
            x1s.append(x1.cpu())
            x2s.append(x2.cpu())
            
            pbar.update(end)
            
        pbar.close()
        
        x1, x2 = torch.cat(x1s), torch.cat(x2s)
        return x1, x2
    
    
class IdentityAugmentor(nn.Module):
    
    """
    Identity augmenter
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, **kwargs):
        return {k: v for k, v in kwargs.items()}


class TopologyAugmentor(nn.Module):
    
    """
    Topology Augmenter
    
    Args:
        encoder (GNN): A graph neural network encoder to learn a higher-order topology
        keep_edges (bool): Whether to keep the original edges when augmenting the topology
                        with the learned higher-order connections
        edge_mul (int): The factor to determine the number of newly added edges
    """
    
    def __init__(self, in_dim, out_dim, dropout, num_layers=2, keep_edges=True, edge_mul=4):
        super().__init__()
        self.augmentor1 = IdentityAugmentor()
        self.augmentor2 = Encoder(
            in_dim=in_dim, out_dim=out_dim, dropout=dropout, 
            num_layers=num_layers, skip=True)
        self.keep_edges = keep_edges
        self.edge_mul = edge_mul
        
    @property
    def name(self):
        return "t"
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        view1 = self.augmentor1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x_prime = self.augmentor2(x=x, edge_index=edge_index)
        with torch.no_grad():
            k = self.edge_mul * edge_index.shape[1]
            self.adj = compute_sim(x_prime, batch=None)
            if self.keep_edges:
                self.adj[edge_index[0], edge_index[1]] += self.adj.max()
                
            edge_index_,  edge_weight = topk_edges(self.adj, k)
            edge_weight = softmax(edge_weight, edge_index_[0])
        
        view2 = {"x": x, "edge_index": edge_index_, "edge_attr": edge_weight}
        return view1, view2
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        view1 = {"x"}
    
    
    
    
def get_augmentor(in_dim, out_dim, conf):
    if conf.aug_type == "feature":
        return FeatureAugmentor(
            in_dim=in_dim, out_dim=out_dim, dropout=conf.dropout)
    elif conf.aug_type == "topology":
        return TopologyAugmentor(
            in_dim=in_dim, out_dim=out_dim, dropout=conf.dropout, num_layers=conf.num_aug_layers, 
            keep_edges=conf.keep_edges, edge_mul=conf.edge_mul)