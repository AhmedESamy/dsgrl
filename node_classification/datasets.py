from torch_geometric.datasets import (
    CitationFull, Coauthor, Amazon, WikiCS, Flickr,
    Actor, Reddit, Yelp, FacebookPagePage, GitHub,
    DeezerEurope, IMDB, DBLP, HGBDataset
)
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import (
    ClusterData, ClusterLoader, InMemoryDataset
)
from torch_geometric.utils import degree
import torch

import utils

import os.path as osp
import os


class DataModule:
    
    def __init__(self, args):
        self._args = args
        self._init_data()
        
    def _init_data(self):
        args = self._args
        self.dataset = Dataset(args.root, args.name)
        data = self.dataset.data
        # print(data)
        # exit(0)
        if args.mini_batch:
            # train_idx = data.train_mask.nonzero(as_tuple=True)[0]
            train_idx = torch.arange(data.num_nodes)
            self.train_loader = NeighborSampler(
                data.edge_index, node_idx=train_idx, sizes=[32] * args.num_gnn_layers, 
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            self.subgraph_loader = NeighborSampler(
                data.edge_index, node_idx=None, sizes=[-1], batch_size=args.batch_size, 
                shuffle=False, num_workers=args.workers)
        else:
            self.train_loader = [data]
            self.subgraph_loader = None

    def labeled_nodetype(self):
        if self.is_hetro:
            data = self.dataset.data
            for nodetype in data.metadata()[0]:
                if hasattr(data[nodetype], "y"):
                    return nodetype
                
    def get_classification_data(self):
        data = self.dataset.data
        if self.is_hetro:
            labeled_key = self.labeled_nodetype()
            data = data[labeled_key]
            
        return data
    
    @property
    def is_hetro(self):
        return hasattr(self.dataset.data, "metadata")
    
    @property
    def num_features(self):
        if self.is_hetro:
            return -1
        else:
            return self.dataset.num_features
    
    @property
    def num_classes(self):
        dataset = self.dataset
        if self.is_hetro:
            labeled_key = self.labeled_nodetype()
            return dataset.data[labeled_key].y.unique().shape[0]
        return dataset.num_classes
    
    @property
    def metadata(self):
        if self.is_hetro:
            return self.dataset.data.metadata()


class Dataset(InMemoryDataset):

    """
    A unified dataset object for PyTorch Geometric datasets
    """
    def __init__(self, root, name, transform=None, pre_transform=None):
        super(Dataset, self).__init__(osp.join(root, name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self._add_missing_info()
        
    def _add_missing_info(self):
        self._add_missing_features()
        self._add_val_mask()
        
    def _add_missing_features(self):
        root, name = osp.split(self.root)
        missing_keys = []
        if name.lower() == "acm":
            missing_keys = ["term"]
        elif name.lower() == "dblp":
            missing_keys = ["conference"]
        elif name.lower() == "freebase":
            missing_keys = [
                "book", "film", "music", "sports", "people", 
                "location", "organization", "business"
            ]
        add_missing_features(self.data, missing_keys)
            
    def _add_val_mask(self):
        root, name = osp.split(self.root)
        if name.lower() in {"acm", "freebase"}:
            if name.lower() == "acm":
                target, num_train = "paper", 400
            elif name.lower() == "freebase":
                target, num_train = "book", 1000
            add_val_mask(data=self.data, target=target, num_train=num_train)

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
        
    if name.lower() in {'cora', "pubmed"}:
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
    elif name.lower() == "imdb":
        dataset = IMDB(osp.join(root, name))
    elif name.lower() == "dblp":
        dataset = DBLP(osp.join(root, name))
    elif name.lower() in {"acm", "freebase"}:
        dataset = HGBDataset(root=root, name=name)
        
    data_dir = osp.split(dataset.raw_dir)[0]
    result_dir = osp.join(data_dir,  "result")
    model_dir = osp.join(data_dir, "model")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    if name.lower() in {"imdb", "acm", "dblp", "freebase"}:
        return dataset
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


def add_val_mask(data, target, num_train):
    target_data = data[target]
    train_indices = target_data.train_mask.nonzero().flatten()
    order = torch.randperm(train_indices.shape[0])
    train_idx = train_indices[order[:num_train]]
    val_idx = train_indices[order[num_train:]]
    target_data.train_mask[:] = False
    target_data.train_mask[train_idx] = True
    target_data.val_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.val_mask[val_idx] = True
    data[target].train_mask = target_data.train_mask
    data[target].val_mask = target_data.val_mask

    
def add_missing_features(data, missing_keys, features="const", edge_key=("conference", "to", "paper")):
    for key in missing_keys:
        if features == "const":
            data[key].x = torch.ones(
                data[key].num_nodes, 1
            )
        else:
            # edge_key = "conference", "to", "paper"
            index = data[edge_key]["edge_index"][0]
            d = degree(
                index, num_nodes=index.unique().shape[0]
            ).unsqueeze(-1)
            data[key].x = d / d.sum()