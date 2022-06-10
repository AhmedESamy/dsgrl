from torch_geometric.profile import get_gpu_memory_from_nvidia_smi
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor, transpose
from torch_sparse.matmul import matmul

import torch

import numpy as np

import dataclasses
import argparse
import os.path as osp
import os
import sys
import yaml

torch.manual_seed(0)


DATASET = 'dataset'
TRAIN_LOADER = 'train_loader'
SUBGRAPH_LOADER = 'subgraph_loader'


def parse_basic_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/workspace/data/dsgrl/")
    parser.add_argument("--name", type=str, default="Photo")
    parser.add_argument("--aug-dim", "-ad", type=int, default=128)
    parser.add_argument("--model-dim", '-md', type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--inv-w", type=float, default=1.)
    parser.add_argument("--var-w", type=float, default=1.)
    parser.add_argument("--cov-w", type=float, default=1.)
    parser.add_argument("--mod-w", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--aug-type", '-at', type=str, default="feature")
    parser.add_argument("--mini-batch", '-mb', dest="mini_batch", action="store_true")
    parser.add_argument("--full-batch", '-fb', dest="mini_batch", action="store_false")
    parser.add_argument("--num-aug-layers", '-nal', type=int, default=2)
    parser.add_argument("--num-gnn-layers", '-ngl', type=int, default=2)
    parser.add_argument("--edge-mul", type=int, default=2)
    parser.add_argument("--keep-edges", dest="keep_edges", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tune", dest="tune", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--wandb", '-wb', dest="wandb", action="store_true")
    parser.add_argument("--entity", type=str, default="zekarias")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=500)
    parser.set_defaults(keep_edges=False)
    parser.set_defaults(mini_batch=False)
    parser.set_defaults(tune=False)
    parser.set_defaults(verbose=False)
    parser.set_defaults(wandb=False)
    return parser.parse_args()
    
    
def load_params(path):
    config = {}
    if osp.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f)
    return config

def parse_args():
    basic_args = parse_basic_args()
    if basic_args.root.startswith("~"):
        basic_args.root = osp.expanduser(basic_args.root)
    os.makedirs(basic_args.root, exist_ok=True)
    if basic_args.name.lower() == "yelp":
        basic_args.task = "mlc"
    else:
        basic_args.task = "mcc"
    # basic_args.aug_type = "topology"
    best_param_path = f"./params/{basic_args.name.lower()}/{basic_args.aug_type.lower()}.yaml"
    params = load_params(best_param_path)
    for key, default_val in basic_args.__dict__.items():
        if key not in params:
            params[key] = default_val
    
    # For ablation
#     params["epochs"] = 1
    
#     params["inv_w"] = 1.
#     params["cov_w"] = 1.
#     params["mod_w"] = 1.
    params = {**basic_args.__dict__, **params}
    config = dataclasses.make_dataclass("Config", fields=list(params.items()))
    config = config(**params)
    
    return config


def log(msg, cr=False, verbose=False):
    if verbose:
        if cr:
            sys.stdout.write(f"\r{msg}")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"{msg}\n")
        


def index_mask(train_mask, val_mask=None, test_mask=None, index=0):
    train_mask = train_mask if len(train_mask.shape) == 1 else train_mask[:, index]
    val_mask = val_mask if val_mask is None or len(val_mask.shape) == 1 else val_mask[:, index]
    test_mask = test_mask if test_mask is None or len(test_mask.shape) == 1 else test_mask[:, index]
    return train_mask, val_mask, test_mask


def create_mask(data, train_rate=0.05, val_rate=0.15, num_splits=1):
    train_size = int(data.num_nodes * train_rate)
    val_size = int(data.num_nodes * val_rate)
    test_size = data.num_nodes - (train_size + val_size)
    num_classes = data.y.unique().shape[0]
    t = RandomNodeSplit(
        num_train_per_class=train_size // num_classes,
        num_val=val_size, num_test=test_size, 
        num_splits=num_splits, split="test_rest"
    )
    data = t(data.clone())
    return data.train_mask, data.val_mask, data.test_mask


def topk_edges(adj, k):
    """
    Retuns the top k indices (edge_index) along with their scores (edge_weight)
    from a given weighted adjacency matrix
    
    Args:
    
        adj (Tensor): A square adjacency matrix
        k (int): The top-k value
    """
    values, indices = torch.topk(adj.flatten(), k, sorted=False)
    shape = adj.shape
    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices / dim

    return torch.stack(coord[::-1], dim=-1).long().T, values


def compute_sim(a, batch):
    """
    Given an block diagonal feature matrix a = [X_1, ..., X_n], where each
    block matrix X_i is a two dimensional with arbitrary dimensions,
    the funnction computes the pairwise similarity between each feature
    vector within each block.
    Returns another block matrix [A_1, ..., A_n], where each block matrix is
    a square matrix containing the pairwise similarities.
    
    Args:
        a (Tensor): A block diagonal tensor 
        batch (1d Tensor): a batch index for each of the features indicating to which
                    block they belong.
    """
    if batch is None:
        return a@a.T
    counts = batch.unique(return_counts = True)[1]
    block = torch.block_diag(*a.split_with_sizes(counts.tolist()))
    b = SparseTensor.from_dense(block)
    index = block.nonzero().T
    values = block[index[0], index[1]]
    a = torch.sparse.FloatTensor(index, values, block.shape)
    indexA = a.coalesce().indices()
    valueA = a.coalesce().values()

    indexB, valueB = transpose(indexA, valueA, a.shape[0], a.shape[1])

    C = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(a.shape[1], a.shape[0]), is_sorted=False)
    return matmul(b, C).to_dense()


def create_dirs(root, name_list):
    for name in name_list:
        os.makedirs(osp.join(root, name), exist_ok=True)


def to_surgeon_input(batch, device, full_data=None):
    if not hasattr(batch, "x") and full_data is None: # Heterogeneous data
        x = {k: v.to(device) for k, v in batch.x_dict.items()}
        edge_index = {
            k: v.to(device) for k, v in batch.edge_index_dict.items()}
        edge_attr = None
    elif isinstance(batch, Data):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr
        edge_attr = edge_attr if edge_attr is None else edge_attr.to(device)
    else:
        batch_size, node_ids, adjs = batch
        x = full_data.x[node_ids].to(device)
        edge_index = [elem for elem in adjs]  # edge_index = (edge_index, e_id, size)
        edge_index[0].to(device)
        edge_index[1].to(device)
        edge_attr = None
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}


def get_device():
    gpu_0_free_space, _ = get_gpu_memory_from_nvidia_smi(device=0)
    gpu_1_free_space, _ = get_gpu_memory_from_nvidia_smi(device=1)
    if gpu_0_free_space < 2000 and gpu_1_free_space < 2000:
        return torch.device("cpu")
    else:
        device_id = 0 if gpu_0_free_space > gpu_1_free_space else 1
        print(f"Automatically selected device: gpu {device_id}")
        return torch.device(f"cuda:{device_id}")


class EarlyStop:
    
    def __init__(self, patience=20, mode="max"):
        assert mode.lower() in {"min", "max"}
        self._patience = patience
        self._mode = mode
        self._best_score = 0 if mode == "max" else np.inf
        self._best_epoch = -1
        
    def _track_epoch_score(self, score, epoch):
        if self._mode == "max":
            condition = score > self._best_score
        elif self._mode == "min":
            condition = score < self._best_score
            
        if condition:
            self._best_score = score
            self._best_epoch = epoch
        
    def stop(self, score, epoch):
        self._track_epoch_score(score, epoch)
        wait = epoch - self._best_epoch
        return wait > self._patience
        
        
class BestScoreTracker:
    
    def __init__(self, metric_to_track="f1macro"):
        self.set_metric_to_track(metric_to_track)
        
    def set_metric_to_track(self, metric_to_track):
        self._metric_to_track = metric_to_track
        self._best_val_scores = {
            metric_to_track: 0
        }
        self._best_test_scores = {
            metric_to_track: 0
        }
        
    def __str__(self):
        def fmt(scores):
            return " ".join(
                f" {key}: {np.mean(score)}, std: {np.std(score)}" 
                for key, score in scores.items()
            )
        
        val_fmt = fmt(self._best_val_scores)
        test_fmt = fmt(self._best_test_scores)
        return f"Validation: {val_fmt}\nTest: {test_fmt}"
    
    def __repr__(self):
        return str(self)
    
    def track(self, val_scores, test_scores):
        assert self._metric_to_track in val_scores
        new_best = val_scores[self._metric_to_track]
        old_best = self._best_val_scores[self._metric_to_track]
        if np.mean(new_best) > np.mean(old_best):
            self._best_val_scores = val_scores
            self._best_test_scores = test_scores
                    