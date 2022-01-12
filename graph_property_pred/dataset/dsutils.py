from .transforms import SplitTransform

from torch_geometric.transforms import LocalDegreeProfile, NormalizeFeatures
import torch

from urllib.request import urlopen
import json


splits_base_url = "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits"
aug_type_map = {"f": "feature", "feature": "feature", 
                "t": "topology", "top": "topology", "topology": "topology"}

class SupportedDatasets:
    
    def __init__(self):
        self.ogb_datasets = {"ogbg-molhiv"}
        self.zinc_dataset = {"ZINC"}
        self.mol_datasets = {"ZINC", "NCI1", "Yeast"}
        self.molnet_datasets = {"freesolv", "esol", "lipo", "bace"}
        self.tu_chem_datasets_splits_na = {"Yeast"}
        self.tu_chem_datasets_splits_avilable = {"DD", "NCI1", "PROTEINS", "ENZYMES"}
        self.tu_chem_datasets = self.tu_chem_datasets_splits_na | self.tu_chem_datasets_splits_avilable
        self.tu_collab_datasets = {"COLLAB", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY"}
        
        self.tu_datasets = self.tu_chem_datasets | self.tu_collab_datasets
        self.pyg_datasets = self.tu_datasets | self.zinc_dataset | self.molnet_datasets

        self.all_datasets = self.tu_datasets | self.pyg_datasets | self.ogb_datasets
        self.name_mappings = {name.lower(): name for name in self.all_datasets}

        self.__init_tags()
        self.__init_split_urls()
        self.__init_split_sources()
        
    def __init_split_urls(self):
        split_urls_chem = {
            name: f"{splits_base_url}/CHEMICAL/{name}_splits.json" 
            for name in self.tu_chem_datasets if name.lower() != "proteins"
        }
        split_urls_chem["PROTEINS"] = f"{splits_base_url}/CHEMICAL/PROTEINS_full_splits.json" 
        split_urls_collab = {
            name: f"{splits_base_url}/COLLABORATIVE_1/{name}_splits.json"
            for name in self.tu_collab_datasets
        }
        self.split_urls = {**split_urls_chem, **split_urls_collab}
        self.split_sources = {}
    
    def __init_tags(self):
        self.zinc_tag = "zinc"
        self.tu_tag = "tu"
        self.molnet_tag = "molnet"
        self.ogbg_tag = "ogbg"
        self.split_source_url = "url"
        self.split_source_ogb = "ogb"
        self.split_source_pyg = "pyg"
        self.split_source_na = None
        self.tag_maps = {}
        for name in self.all_datasets:
            if name in self.tu_datasets:
                self.tag_maps[name] = self.tu_tag
            elif name in self.zinc_dataset:
                self.tag_maps[name] = self.zinc_tag
            elif name in self.molnet_datasets:
                self.tag_maps[name] = self.molnet_tag
            elif name in self.ogb_datasets:
                self.tag_maps[name] = self.ogbg_tag
                
    def __set_split_source(self, name, source):
        self.split_sources[name] = source
        self.split_sources[name.lower()] = source
                
    def __init_split_sources(self):
        datasets_with_split_url = (self.tu_chem_datasets_splits_avilable | 
                                   self.tu_collab_datasets)
        datasets_with_no_split = (self.tu_chem_datasets_splits_na | 
                                  self.molnet_datasets)
        datasets_with_pyg_splits = self.zinc_dataset
        for name in datasets_with_split_url:
            self.__set_split_source(name, self.split_source_url)
        for name in datasets_with_no_split:
            self.__set_split_source(name, self.split_source_na)
        for name in datasets_with_pyg_splits:
            self.__set_split_source(name, self.split_source_pyg)
        for name in self.ogb_datasets:
            self.__set_split_source(name, self.split_source_ogb)
            
    def is_mol_dataset(self, name):
        name = self.name_mappings[name.lower()]
        return name in self.mol_datasets
        
    @property
    def datasets_name_mapping(self):
        return self.name_mappings
    
    @property
    def tag(self):
        return self.tag_maps
    
    
def get_formated_splits(train_index, val_index, test_index):
    """
    Formats indices in a unified format for consistency with
    the format provided in:
        https://github.com/diningphil/gnn-comparison
        
    """
    return [{
        "test": test_index.cpu().numpy().tolist(),
        "model_selection": [{
            "train": train_index.cpu().numpy().tolist(),
            "validation": val_index.cpu().numpy().tolist()
        }]
    }]


def prepare_splits_from_datasets(train_dataset, val_dataset, test_dataset, is_mol=False):
    """
    Transforms a dataset with splits into a single dataset and 
    creates split indices from the training, validation, and
    test datasets so that it is compatible with the format used in:
            https://github.com/diningphil/gnn-comparison
    
    Returns a single list containing the data from all the datasets
    along with the split list
    Args:
        train_dataset (PyG Dataset): The training dataset
        val_dataset (PyG Dataset): The validation datset
        test_dataset (PyG Dataset): The test dataset
        
    """
    def to_list(dataset):
        data_list_ = []
        for data in dataset:
            data_list_.append(data)
        return data_list_
    
    print("Preparing splits ...")
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    num_test = len(test_dataset)
    train_indices = torch.arange(num_train)
    val_indices = torch.arange(num_train, num_train + num_val)
    test_indices = torch.arange(num_train + num_val, 
                                num_train + num_val + num_test)
    splits = get_formated_splits(
        train_indices, val_indices, test_indices)
    data_list = to_list(train_dataset)
    data_list += to_list(val_dataset)
    data_list += to_list(test_dataset)
    return data_list, splits


def create_splits(dataset):
    """
    Returns new splits if no split is available for the dataset
    
    Args:
        dataset (PyG Dataset): The dataset
    """
    print("Creating new splits")
    data = Data(y=dataset.data.y.squeeze(), 
                num_nodes=dataset.data.y.shape[0])
    transform = SplitTransform(
        num_nodes=dataset.data.y.shape[0], 
        num_classes=dataset.num_classes,
        num_splits=10)
    return transform(data)


def get_splits_from_url(split_url):
    """
    Returns publicly available splits from a specified url
    
    Args:
        split_url (str): The url
    """
    print(f"Grabbing splits from {split_url}")
    with urlopen(split_url) as url:
        splits = json.loads(url.read().decode())
    return splits


def prepare_splits(dataset, name):
    """
    Prepares splits according to their availablity.
    
    For TUDatasets, we grab publicly available splits from:
        https://github.com/diningphil/gnn-comparison
    
    which is prepared according to the proposed experimental 
    protocol in the following paper:
    
        A Fair Comparison of Graph Neural Networks for Graph Classification 
        (ICLR 2020)
        
        They provide splits in the following format
        [split_1, ..., split_10]
        where each split_i is a dictionary formated as:
        {
            "test": [<list_of_test_indices>], 
             "model_selection": [
                 {"train": [<list_of_train_indices>]}
                  "validation": [<list_of_validation_indices>]
             ]
        }
    All the other splits are unified to this format
        
    For OGB dataset, we use the splits from OGB and organize them as above.
    That is:
        [split_1]
        
        The list contains only a single split, since OGB provides a single split
    
    For PyG datasets with splitted datasets, that is, train_dataset, 
    val_dataset, and test_dataset we transform them into the above format
    
    If no split is available, a new one will be created
    Args:
        dataset (PyG or OGB Dataset): The dataset object
    """
    split_source = supported_ds.split_sources[name]
    if split_source == supported_ds.split_source_url:
        return get_splits_from_url(supported_ds.split_urls[name])
    elif split_source == supported_ds.split_source_ogb:
        splits = dataset.get_idx_split() 
        return get_formated_splits(splits["train"], splits["valid"], splits["test"])
    elif split_source is supported_ds.split_source_na:
        return create_splits(dataset)
    elif split_source == supported_ds.split_source_pyg:
        return prepare_splits_from_datasets(*dataset)
    
def unify_atoms_and_bonds(dataset=None, data_list=None):
    """
    Returns a unified represetation (BATCH_SIZE x 1 LongTensor) 
    for atoms and bonds, as different data sources use different schemes
    """
    print("Unifying atom and bond features")
    assert dataset is not None or data_list is not None
    container = dataset if dataset is not None else data_list
    data_list_ = []
    data = dataset[0] if dataset is not None else data_list[0]
    has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None
    for data in container:
        if data.x.ndim == 1:
            x = data.x.view(-1, 1)
        elif data.x.shape[1] > 1:
            x = data.x.nonzero()[:, 1:]
        else: 
            x = data.x
        data.x = x
        
        if has_edge_attr:
            if data.edge_attr.ndim == 1:
                edge_attr = data.edge_attr.view(-1, 1)
            elif data.edge_attr.shape[1] > 1:
                edge_attr = data.edge_attr.nonzero()[:, 1:]
            else:
                edge_attr = data.edge_attr
            data.edge_attr = edge_attr
            
        data_list_.append(data)
    return data_list_

def create_verify_features(dataset, degree_profile):
    """
    For a dataset with no node features, this routine creates degree profile features.
    otherwise verifies if all the features (node, edge) are of type float
    
    Returns a list of PyG Data objects, where each data is augmented with
    a degree profile object.
    
    Args:
        dataset: (PyG Dataset): A PyG dataset object
        
        
    """
    if not hasattr(dataset.data, "x") or dataset.data.x is None:
        print("Creating degree profile features")
        if degree_profile:
            ldp = LocalDegreeProfile()
            norm = NormalizeFeatures()
            data = dataset.data
            data = ldp(data.clone())
            data = norm(data.clone())
            x = data.x
        else:
            x = torch.ones((dataset.data.num_nodes, 1)).float()
        data_list = []
        start = 0
        for data in dataset:
            data.x = x[start:start + data.num_nodes]
            data_list.append(data)
            start += data.num_nodes
        return data_list
    else:
        data_list = []
        if not isinstance(dataset.data.x, torch.FloatTensor):
            x = dataset.data.x
            start = 0
            for data in dataset:
                data.x = x[start:start + data.num_nodes].float()
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr.float()
                data_list.append(data)
                start += data.num_nodes
            return data_list

supported_ds = SupportedDatasets()