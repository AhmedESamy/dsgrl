from utils import compute_sim, topk_edges
from gnn import GINEncoder

from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.utils import softmax
import torch.nn.functional as F
import torch.nn as nn
import torch

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class FeatureAugmentor(nn.Module):
    
    """
    Learnable augmenter for node features under the graph classification setting
    
    Args:
        in_dim (int): The number of incoming features
        out_dim (int): The number of outgoing features
        encode_atom (bool) : Whether to encode atoms (for molecular graphs)
    """
    
    def __init__(self, in_dim, out_dim, encode_atom=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encode_atom = encode_atom
        self._build_atom_encoder()
        self._build_ffn()
    
    def _build_ffn(self):
        self.ffn = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim), 
            nn.BatchNorm1d(self.out_dim), 
            nn.ReLU(), 
            nn.Linear(self.out_dim, self.out_dim), 
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU()
        )
    
    def _build_atom_encoder(self):
        if self.encode_atom:
            self.atom_encoder = AtomEncoder(self.out_dim)
            self.in_dim = self.out_dim
    
    def forward(self, x, **kwargs):
        device = next(self.parameters()).device
        if self.encode_atom:
            x = self.atom_encoder(x.to(device))
        return self.ffn(x.to(device))
    
class IdentityAugmentor(nn.Module):
    
    """
    Identity augmenter
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, **kwargs):
        return {k: v for k, v in kwargs.items()}


class HONAugmentor(nn.Module):
    
    """
    A high-order network (HON) Augmenter
    
    Args:
        encoder (GNN): A graph neural network encoder to learn a higher-order topology
        keep_edges (bool): Whether to keep the original edges when augmenting the topology
                        with the learned higher-order connections
        edge_mul (int): The factor to determine the number of newly added edges
    """
    
    def __init__(self, in_dim, edge_dim, conf):
        super().__init__()
        self.encoder = GINEncoder(
                in_dim=in_dim, hidden_dim=conf.aug_dim, 
                edge_dim=edge_dim, num_layers=conf.num_gnn_layers, 
                encode_atom=conf.encode_atom, encode_bond=conf.encode_bond)
        self.keep_edges = conf.keep_edges
        self.edge_mul = conf.edge_mul
        # if conf.use_linear:
        #     self.linear = nn.Linear(encoder.in_dim, encoder.out_dim)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x_prime = self.encoder(x, edge_index, batch=None)
        with torch.no_grad():
            k = self.edge_mul * edge_index.shape[1]
            self.adj = compute_sim(x_prime, batch)
            if self.keep_edges:
                self.adj[edge_index[0], edge_index[1]] += self.adj.max()
                
            edge_index_,  edge_weight = topk_edges(self.adj, k)
            edge_weight = softmax(edge_weight, edge_index_[0])
            
        if hasattr(self, "linear"):
            x = self.linear(x.to(self.device))
        return {"x": x, "edge_index": edge_index_, "edge_attr": edge_weight, "batch": batch}


class FeatureAugmentorWrapper(nn.Module):
    
    """
    A wrapper for a dual view feature augmentor
    
    Args:
        in_dim (int): The number of incoming features
        out_dim (int): The number of outgoing features
        dropout (float): The dropout rate / probability
        encode_atom (bool) : Whether to encode atoms (for molecular graphs)
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.5, encode_atom=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.augmentor1 = FeatureAugmentor(in_dim, out_dim, encode_atom=encode_atom)
        self.augmentor2 = FeatureAugmentor(in_dim, out_dim, encode_atom=encode_atom)
        
    @property
    def name(self):
        return "f"
        
    def forward(self, x, **kwargs):
        x1 = self.augmentor1(x)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.augmentor2(x)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        return x1, x2
    
    
class TopologyAugmentorWrapper(nn.Module):
    
    def __init__(self, in_dim, edge_dim, conf):
        super().__init__()
        self.augmentor1 = IdentityAugmentor()
        self.augmentor2 = HONAugmentor(in_dim, edge_dim, conf)
    
    @property
    def name(self):
        return "t"
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        view1 = self.augmentor1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        view2 = self.augmentor2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return view1, view2

def get_augmentor(conf, num_features, num_edge_features=None):
    if conf.aug_type.lower() == "feature":
        return FeatureAugmentorWrapper(
            in_dim=num_features, out_dim=conf.aug_dim,
            encode_atom=conf.encode_atom)
    elif conf.aug_type.lower() == "topology":
        return TopologyAugmentorWrapper(in_dim=num_features, edge_dim=num_edge_features, conf=conf)