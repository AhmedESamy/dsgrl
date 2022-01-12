from torch_geometric.data import ClusterData, ClusterLoader, InMemoryDataset
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, WikiCS, Flickr
from torch_geometric.datasets import Actor, Reddit, Yelp, FacebookPagePage, GitHub
from torch_geometric.datasets import DeezerEurope
from torch_geometric.loader import NeighborSampler

import torch

import utils

import os.path as osp
import os


class CompiledDataset:
    
    def __init__(self, args):
        self._args = args
        
    def compile(self):
        args = self._args
        compiled_data = {}
        if args.mini_batch:
            dataset = Dataset(args.root, args.name)
            data = dataset.data
            train_idx = data.train_mask.nonzero(as_tuple=True)[0]
            train_loader = NeighborSampler(
                data.edge_index, node_idx=train_idx, sizes=[32] * args.num_gnn_layers, 
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            subgraph_loader = NeighborSampler(
                data.edge_index, node_idx=None, sizes=[-1], batch_size=args.batch_size, 
                shuffle=False, num_workers=args.workers)
            compiled_data[utils.DATASET] = dataset
            compiled_data[utils.TRAIN_LOADER] = train_loader
            compiled_data[utils.SUBGRAPH_LOADER] = subgraph_loader
        else:
            dataset = Dataset(args.root, args.name)
            compiled_data[utils.DATASET] = dataset
            compiled_data[utils.TRAIN_LOADER] = [dataset.data]
        return compiled_data
            

class Dataset(InMemoryDataset):

    """
    A unified dataset object for PyTorch Geometric datasets
    """
    def __init__(self, root, name, transform=None, pre_transform=None):
        super(Dataset, self).__init__(osp.join(root, name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data)

    def download(self):
        if not osp.exists(self.processed_paths[0]):
            if self.root.startswith("~"):
                self.root = osp.expanduser(self.root)
            root, name = osp.split(self.root)
            fetch_data(root=root, name=name)

    def process(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]


def fetch_data(root, name):
    if root.startswith("~"):
        root = osp.expanduser(root)
        
    if name.lower() in {'cora', "pubmed", "dblp"}:
        dataset = CitationFull(root=root, name=name)
    elif name.lower() in {'computers', "photo"}:
        dataset = Amazon(root=root, name=name)
    elif name.lower() in {'cs',  'physics'}:
        dataset = Coauthor(root=root, name=name)
    elif name.lower() in {"wiki", "wikics"}:
        dataset = WikiCS(osp.join(root, name)) 
    elif name.lower() == "flickr":
        dataset = Flickr(osp.join(root, name))
    elif name.lower() == "actor":
        dataset = Actor(osp.join(root, name))
    elif name.lower() == "yelp":
        dataset = Yelp(osp.join(root, 'yelp'))
    elif name.lower() == "reddit":
        dataset = Reddit(osp.join(root, name))
    elif name.lower() == "facebook":
        dataset = FacebookPagePage(osp.join(root, name))
    elif name.lower() == "git":
        dataset = GitHub(osp.join(root, name))
    elif name.lower() == "deezer":
        dataset = DeezerEurope(osp.join(root, name))
    print(dataset.data)
    data_dir = osp.split(dataset.raw_dir)[0]
    result_dir = osp.join(data_dir,  "result")
    model_dir = osp.join(data_dir, "model")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return verify(dataset)

def verify(dataset):
    def save(data_list):
        data, slices = dataset.collate(data_list)
        torch.save((data, slices), dataset.processed_paths[0])
        return data
        
    data = dataset.data
    # Verify mask
    if not hasattr(data, "train_mask"):
        print("Creating masks")
        train_mask, val_mask, test_mask = utils.create_mask(data)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data = save([data])
            
    # Verify y for multi-labels
    if isinstance(data.y, torch.LongTensor):
        if data.y.ndim > 1 and data.y.sum(dim=-1) > 1:  # is multi-label
            print("Casting class labels to float tensor ...")
            data.y = data.y.float()
            data = save([data])
            
    return data
    
        
