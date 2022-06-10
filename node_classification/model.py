import torch.nn.functional as F
import torch.nn as nn
import torch

import copy


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class SurgeonModule(nn.Module):
    
    @property
    def device(self):
        return next(self.parameters()).device


class LogisticRegression(SurgeonModule):
    """
    A simple logistic regression classifier for evaluating SelfGNN 
    """
    def __init__(self, num_dim, num_class, task='mcc'):
        super().__init__()
        assert task in {"bc", "mcc", "mlc", "hetro"}
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        if task in {'bc', 'mcc', 'hetro'}:
            self.loss_fn = nn.CrossEntropyLoss()
        elif task == "mlc":
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        prd = self.linear(x.to(self.device))
        loss = self.loss_fn(prd, y.to(self.device))
        return prd, loss


class Surgeon(SurgeonModule):

    def __init__(
        self, net, augmentor, agg_method="concat", 
        use_improved_loss=True
    ):
        super().__init__()
        self.augmentor = augmentor
        self.net = net
        self.agg_method = agg_method
        self.use_improved_loss = use_improved_loss
        
    def _aggregate_views(self, x1, x2):
        if self.agg_method == "mean":
            return (x1 + x2) / 2.
        elif self.agg_method == "sum":
            return x1 + x2
        return torch.cat([x1, x2], dim=-1)
    
    def _aggregate_hetro_views(self, x1, x2):
        out = {}
        for key in self.net.metadata[0]:
            if self.agg_method == "mean":
                out[key] = (x1[key] + x2[key]) / 2.
            elif self.agg_method == "sum":
                out[key] = x1[key] + x2[key]
            else:
                out[key] = torch.cat(
                    (x1[key], x2[key]), dim=-1
                )
        return out
        
    def aggregate_views(self, x1, x2):
        if hasattr(self.net, "metadata"):
            return self._aggregate_hetro_views(x1, x2)
        return self._aggregate_views(x1, x2)
    
    def forward(self, x, edge_index, edge_attr=None):
        
        view1, view2 = self.augmentor(x, edge_index=edge_index, edge_attr=edge_attr)
        if self.augmentor.name == "f": # if feature augmentor
            x1, x2 = view1, view2
            edge_index1 = edge_index2 = edge_index
            edge_attr1 = edge_attr2 = edge_attr
        elif self.augmentor.name == "t": # if topological augmentor
            x1 = x2 = view1["x"]
            edge_index1 = view1["edge_index"]
            edge_index2 = view2["edge_index"]
            edge_attr1 = view1["edge_attr"]
            edge_attr2 = view2["edge_attr"]
            self.hon = (edge_index2.detach().cpu(), edge_attr2.detach().cpu())
        
        z1 = self.net(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
        z2 = self.net(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
        return z1, z2
    
    @torch.no_grad()
    def infer(self, x, edge_index, edge_attr=None, loader=None):
        if loader is None:
            z1, z2 = self(x, edge_index, edge_attr)
        else:
            x1, x2 = self.augmentor.inference(x)
            torch.cuda.empty_cache()
            z1 = self.net.inference(x1, loader, desc="Inferring lattent representations for the first view").cpu()
            torch.cuda.empty_cache()
            z2 = self.net.inference(x2, loader, desc="Inferring lattent representations for the second view").cpu()
                
        torch.cuda.empty_cache()
        return self.aggregate_views(z1, z2)
