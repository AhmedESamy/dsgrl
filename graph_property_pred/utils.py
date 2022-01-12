from dataset.dsutils import SupportedDatasets

from torch_geometric.profile import get_gpu_memory_from_nvidia_smi
from torch_sparse import SparseTensor, transpose
from torch_sparse.matmul import matmul
import argparse
import dataclasses
import argparse
import os.path as osp
import os
import yaml

import torch


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

supported_ds = SupportedDatasets()


def parse_basic_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/workspace/data/dsgrl/")
    parser.add_argument("--name", type=str, default="DD")
    parser.add_argument("--aug-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--inv-w", type=float, default=25.)
    parser.add_argument("--var-w", type=float, default=25.)
    parser.add_argument("--cov-w", type=float, default=1.)
    parser.add_argument("--mod-w", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--aug-type", type=str, default="feature")
    parser.add_argument("--num-aug-layers", type=int, default=2)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--edge-mul", type=int, default=2)
    parser.add_argument("--keep-edges", dest="keep_edges", action="store_true")
    parser.add_argument("--encode-atom", dest="encode_atom", action="store_true")
    parser.add_argument("--encode-bond", dest="encode_bond", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--tune", dest="tune", action="store_true")
    parser.add_argument("--degree-profile", dest="degree_profile", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--trials", type=int, default=500)
    parser.set_defaults(keep_edges=False)
    parser.set_defaults(encode_atom=False)
    parser.set_defaults(encode_bond=False)
    parser.set_defaults(tune=False)
    parser.set_defaults(degree_profile=False)
    parser.set_defaults(verbose=False)
    return parser.parse_args()


def load_params(path):
    config = {}
    print(path)
    if osp.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f)
    return config


def parse_args():
    basic_args = parse_basic_args()
    basic_args.name = supported_ds.datasets_name_mapping[basic_args.name.lower()]
    if basic_args.root.startswith("~"):
        basic_args.root = osp.expanduser(basic_args.root)
    os.makedirs(basic_args.root, exist_ok=True)
    
    best_param_path = f"./params/{basic_args.name.lower()}/{basic_args.aug_type.lower()}.yaml"
    params = load_params(best_param_path)
    
    for key, default_val in basic_args.__dict__.items():
        if key not in params:
            params[key] = default_val
    
    # For ablation study
    # params["epochs"] = 15
    # params["inv_w"] = 1.
    # params["var_w"] = 1.
    # params["cov_w"] = 1.
    # params["mod_w"] = 1.
            
    params = {**basic_args.__dict__, **params}
    config = dataclasses.make_dataclass("Config", fields=list(params.items()))
    config = config(**params)
    
    return config

class Aggregator:
    
    def __init__(self, name="cat"):
        self.name = name.lower()
        
    def __call__(self, z1, z2):
        if self.name in {"cat", "concat", "concatenate"}:
            return torch.cat([z1, z2], dim=-1)
        elif self.name in {"sum", "add"}:
            return z1 + z2
        elif self.name in {"mean", "avg", "average"}:
            return (z1 + z2) / 2.
        
    def __repr__(self):
        args = ",\n".join(f"{field}={value}" for field, value in self.__dict__.items())
        return f"{type(self).__name__}(\n\t{args}\n)"
    
def get_device():
    if not torch.cuda.is_available():
        print("WARNING: No GPU is available, using CPU!")
    free0, _ = get_gpu_memory_from_nvidia_smi(0)
    free1, _ = get_gpu_memory_from_nvidia_smi(1)
    if free0 < 2000 and free1 < 2000:
        print("WARNING: The available GPU memory is insufficient, falling back to CPU.")
        return torch.device("cpu")
    device_id = 0 if free0 > free1 else 1
    print(f"Auto select gpu: {device_id}")
    return torch.device(f"cuda:{device_id}")


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