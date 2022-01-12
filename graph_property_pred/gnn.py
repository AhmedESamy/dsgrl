from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch_geometric.nn import GINConv, GINEConv, global_add_pool

import torch.nn.functional as F
import torch.nn as nn
import torch


def _build_nn(in_dim, out_dim):
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim), 
        nn.BatchNorm1d(out_dim),
        nn.ReLU()
    )


class GINEncoder(torch.nn.Module):
    
    """
    GIN based graph neural network encoder
    """
    
    
    def __init__(self, in_dim, hidden_dim, edge_dim=None, num_layers=4, dropout=0.5, encode_atom=False, encode_bond=False):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_atom = encode_atom
        self.encode_bond = encode_bond
        
        self.has_edge_attr = edge_dim is not None
        edge_dim = in_dim if edge_dim is None else edge_dim
        
        self.convs = nn.ModuleList(
            [self.get_layer(in_dim, hidden_dim, edge_dim=hidden_dim)])
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim)])
        for _ in range(1, num_layers):
            self.convs.append(self.get_layer(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.pool = global_add_pool
        
        if encode_atom:
            self.atom_encoder = AtomEncoder(hidden_dim)
        
        if encode_bond:
            self.bond_encoder = BondEncoder(emb_dim=in_dim)
            
    def get_layer(self, in_dim, out_dim, edge_dim=None):
        nn_ = _build_nn(in_dim, out_dim)
        if self.encode_bond:
            return GINEConv(nn_, edge_dim=edge_dim)
        else:
            return GINConv(nn_)
        

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        device = next(self.parameters()).device
        edge_attr = None if edge_attr is None else edge_attr.to(device)
        if self.encode_atom:
            x = self.atom_encoder(x.to(device))
        if self.encode_bond:
            edge_attr = self.bond_encoder(edge_attr)
            
        x_skip = x
        for idx in range(self.num_layers):
            conv = self.convs[idx]
            linear = self.linears[idx]
            if edge_attr is None:
                x = conv(x.to(device), edge_index.to(device))
            else:
                x = conv(x.to(device), edge_index.to(device), edge_attr)
                
            if batch is not None:
                x_skip = linear(self.pool(x, batch=batch.to(device)))
                x_skip = F.dropout(x_skip, self.dropout, training=self.training)
            else:
                x_skip = x
        
        return x_skip
    
    
def get_encoder(conf, num_features=None, edge_dim=None):
    if conf.aug_type == "feature":
        return GINEncoder(
            in_dim=conf.aug_dim, hidden_dim=conf.model_dim, edge_dim=edge_dim,
            num_layers=conf.num_gnn_layers, encode_bond=conf.encode_bond)
    elif conf.aug_type == "topology":
        edge_dim = 1 if edge_dim is None else edge_dim + 1
        return GINEncoder(
            in_dim=num_features, hidden_dim=conf.model_dim,
            edge_dim=edge_dim, num_layers=conf.num_gnn_layers,
            encode_atom=conf.encode_atom, encode_bond=conf.encode_bond)