from .dsutils import *

from torch_geometric.datasets import TUDataset, ZINC, MoleculeNet
from torch_geometric.data import InMemoryDataset, Data
from ogb.graphproppred import PygGraphPropPredDataset
import torch
import os.path as osp
import os

supported_ds = SupportedDatasets()

class UnifiedGraphDataset(InMemoryDataset):
    
    def __init__(self, root, name, degree_profile=True, transform=None, pre_transform=None):
        self.degree_profile = degree_profile
        super().__init__(osp.join(root, name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.splits = self.load_splits()

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def splits_file_names(self):
        return ["splits.json"]
    
    @property
    def splits_dir(self):
        return osp.join(self.root, "splits")
    
    @property
    def splits_path(self):
        return [osp.join(self.splits_dir, name) 
                for name in self.splits_file_names]

    def load_splits(self):
        with open(self.splits_path[0]) as f:
            return json.load(f)
        
    def save_splits(self, splits):
        with open(self.splits_path[0], "w") as f:
                json.dump(splits, f)
    
    def prepare_splits(self, dataset):
        data_list = []
        name = osp.split(self.root)[1]
        if not osp.exists(self.splits_path[0]):
            splits = prepare_splits(dataset, name)
            if isinstance(splits, tuple):
                data_list, splits = splits
            self.save_splits(splits)
        return data_list
                
    def process(self):
        os.makedirs(self.splits_dir, exist_ok=True)
        root, name = osp.split(self.root)
        dataset = get_dataset(root, name)
        data_list = self.prepare_splits(dataset)
        if len(data_list) == 0:
            if supported_ds.is_mol_dataset(name):
                data_list = unify_atoms_and_bonds(dataset=dataset)
            else:
                data_list = create_verify_features(dataset, self.degree_profile)
            if data_list is not None:
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])
        else:
            if supported_ds.is_mol_dataset(name):
                data_list = unify_atoms_and_bonds(data_list=data_list)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

            
def get_zinc(root):
    """
    Returns all the splits of the ZINC dataset in a list
    """
    return [
        ZINC(root, split="train"),
        ZINC(root, split="val"),
        ZINC(root, split="test")
    ]
        
def get_dataset(root, name):
    """
    Returns a dataset with name from PyG or OGB datasets after downloading it
    to the root directory.
    """
    if supported_ds.tag[name] == supported_ds.tu_tag:
        return TUDataset(root=root, name=name)
    elif supported_ds.tag[name] == supported_ds.molnet_tag:
        return MoleculeNet(root, name)
    elif supported_ds.tag[name] == supported_ds.ogbg_tag:
        return PygGraphPropPredDataset(
            root=root, name=name)
    elif supported_ds.tag[name] == supported_ds.zinc_tag:
        return get_zinc(osp.join(root, name))