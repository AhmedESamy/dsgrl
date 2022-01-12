from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm

import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm

class Encoder(nn.Module):

    def __init__(self, 
                 in_dim, out_dim, dropout, encoder='gcn', 
                 use_norm=False, num_layers=2, skip=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.encoder = encoder
        self.use_norm = use_norm
        self.num_layers = num_layers
        self.skip = skip
        self._init_modules()
        
    def _init_modules(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        if self.encoder == "gcn":
            GNNLayer = GCNConv
        elif self.encoder == "sage":
            GNNLayer = SAGEConv
            
        self.stacked_gnn = nn.ModuleList()
        
        self.stacked_gnn.append(GNNLayer(self.in_dim, self.out_dim))
        for _ in range(1, self.num_layers):
            self.stacked_gnn.append(GNNLayer(self.out_dim, self.out_dim))
            
        if self.skip:
            self.skips = nn.ModuleList()
            self.skips.append(nn.Linear(self.in_dim, self.out_dim))
            for _ in range(self.num_layers - 1):
                self.skips.append(nn.Linear(self.out_dim, self.out_dim))
            
        self.norm = BatchNorm(self.out_dim) if self.use_norm else lambda x: x

    def __full_batch_forward(self, x, edge_index, edge_attr):
        device = next(self.parameters()).device
        edge_attr = edge_attr if edge_attr is None else edge_attr.to(device)
        outputs = []
        for conv in self.stacked_gnn[:-1]:
            x = conv(x.to(device), edge_index=edge_index.to(device))
            x = F.relu(self.norm(x))
            x = F.dropout(input=x, p=self.dropout, training=self.training)
            outputs.append(x)
        
        x = self.stacked_gnn[-1](x.to(device), edge_index=edge_index.to(device))
        outputs.append(x)
        if self.skip:
            x = F.dropout(input=x, p=self.dropout, training=self.training)
            outputs[-1] = x
            return torch.stack(outputs, dim=0).sum(dim=0)
        return x
    
    def __mini_batch_forward(self, x, adj):
        """
        Subgraph inference code adapted from PyTorch Geometric:
        https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_gat.py#L58
        
        `train_loader` computes the k-hop neighborhood of a batch of nodes,
        and returns, for each layer, a bipartite graph object, holding the
        bipartite edges `edge_index`, the index `e_id` of the original edges,
        and the size/shape `size` of the bipartite graph.
        Target nodes are also included in the source nodes so that one can
        easily apply skip-connections or add self-loops.
        
        """
        device = next(self.parameters()).device
        outputs = []
        for i, (edge_index, _, size) in enumerate(adj):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.stacked_gnn[i]((x.to(device), x_target.to(device)),
                                    edge_index.to(device))
            if self.skip:
                x = x + self.skips[i](x_target.to(device))
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(edge_index, torch.Tensor):
            return self.__full_batch_forward(x, edge_index, edge_attr)
        else:
            return self.__mini_batch_forward(x, edge_index)
    
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, desc='Inferring embeddings of a view'):
        """
        Subgraph inference code adapted from PyTorch Geometric:
        https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py#L47
        
        Compute representations of nodes layer by layer, using *all*
        available edges. This leads to faster computation in contrast to
        immediately computing the final representations of each batch.
        
        """
        pbar = tqdm(total=x_all.size(0) * len(self.stacked_gnn))
        pbar.set_description(desc)
        device = next(self.parameters()).device
        for i, conv in enumerate(self.stacked_gnn):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj
                x = x_all[n_id]
                x_target = x[:size[1]]
                x = conv((x.to(device), x_target.to(device)), 
                         edge_index.to(device)).cpu()
                if i != len(self.stacked_gnn) - 1:
                    x = F.relu(self.norm(x.to(device)))
                xs.append(x.cpu())

                pbar.update(batch_size)
                torch.cuda.empty_cache()

            x_all = torch.cat(xs, dim=0)
        torch.cuda.empty_cache()
        pbar.close()

        return x_all