from torch_geometric.profile import get_gpu_memory_from_nvidia_smi
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor, transpose
from torch_sparse.matmul import matmul

import dataclasses
import argparse
import os.path as osp
import os

import torch
import yaml

torch.manual_seed(0)


DATASET = 'dataset'
TRAIN_LOADER = 'train_loader'
SUBGRAPH_LOADER = 'subgraph_loader'


def parse_basic_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/workspace/data/dsgrl/")
    parser.add_argument("--name", type=str, default="Photo")
    parser.add_argument("--aug-dim", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--inv-w", type=float, default=1.)
    parser.add_argument("--cov-w", type=float, default=1.)
    parser.add_argument("--mod-w", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--aug-type", type=str, default="feature")
    parser.add_argument("--loader", type=str, default="full")
    parser.add_argument("--num-aug-layers", type=int, default=2)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--edge-mul", type=int, default=2)
    parser.add_argument("--keep-edges", dest="keep_edges", action="store_true")
    parser.add_argument("--task", type=str, default="mcc")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tune", dest="tune", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=500)
    parser.set_defaults(keep_edges=False)
    parser.set_defaults(tune=False)
    parser.set_defaults(verbose=False)
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
    
    best_param_path = f"./params/{basic_args.name.lower()}/{basic_args.aug_type.lower()}.yaml"
    params = load_params(best_param_path)
    for key, default_val in basic_args.__dict__.items():
        if key not in params:
            params[key] = default_val
        
    params = {**basic_args.__dict__, **params}
    config = dataclasses.make_dataclass("Config", fields=list(params.items()))
    config = config(**params)
    
    return config


def log(msg, stream=print, verbose=False):
    if verbose:
        stream(msg)


def index_mask(train_mask, val_mask=None, test_mask=None, index=0):
    train_mask = train_mask if len(train_mask.shape) == 1 else train_mask[:, index]
    val_mask = val_mask if val_mask is None or len(val_mask.shape) == 1 else val_mask[:, index]
    test_mask = test_mask if test_mask is None or len(test_mask.shape) == 1 else test_mask[:, index]
    return train_mask, val_mask, test_mask


# def create_mask_(data, train_rate=0.05, val_rate=0.15):
#     perm = torch.randperm(data.num_nodes)
#     train_size = int(data.num_nodes * train_rate)
#     val_size = int(data.num_nodes * val_rate)
#     train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#     val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#     train_mask[perm[:train_size]] = True
#     val_mask[perm[train_size:train_size + val_size]] = True
#     test_mask = ~(train_mask + val_mask)
#     return train_mask, val_mask, test_mask

def create_mask(data, train_rate=0.05, val_rate=0.15):
    train_size = int(data.num_nodes * train_rate)
    val_size = int(data.num_nodes * val_rate)
    test_size = data.num_nodes - (train_size + val_size)
    num_classes = data.y.unique().shape[0]
    t = RandomNodeSplit(
        num_train_per_class=train_size // num_classes,
        num_val=val_size, num_test=test_size, 
        num_splits=10, split="test_rest"
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


def to_surgeon_input(batch, full_data=None):
    if isinstance(batch, Data):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
    else:
        batch_size, node_ids, adjs = batch
        x = full_data.x[node_ids]
        edge_index = [elem for elem in adjs]  # edge_index = (edge_index, e_id, size)
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