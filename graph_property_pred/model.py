import torch.nn.functional as F
import torch.nn as nn
import torch

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class DSGRL(nn.Module):
    
    """
    A self supervised model wrapper for learning latent representation of graphs
    based on learnable feature augmentations
    
    Args:
        augmenter (Augmenter): the multi-view augmenter object
        encoder (GNN): A graph neural network encoder
    """
    
    def __init__(self, augmentor, encoder):
        super().__init__()
        self.augmentor = augmentor
        self.encoder = encoder
        
    def forward(self, data):
        view1, view2 = self.augmentor(
            x=data.x, edge_index=data.edge_index, 
            edge_attr=data.edge_attr, batch=data.batch)
        if self.augmentor.name == "f":
            x1, x2 = view1, view2
            edge_index1 = edge_index2 = data.edge_index
            edge_attr1 = edge_attr2 = data.edge_attr
        else:
            x1 = x2 = data.x
            edge_index1 = view1["edge_index"]
            edge_index2 = view2["edge_index"]
            edge_attr1 = view1["edge_attr"]
            edge_attr2 = view2["edge_attr"]
        if edge_attr1 is None or edge_attr2 is None:
            edge_attr1 = edge_attr2 = None
        z1 = self.encoder(x1, edge_index1, edge_attr1, data.batch)
        z2 = self.encoder(x2, edge_index2, edge_attr2, data.batch)
        return z1, z2
    
    def infer(self, data):
        return self(data)