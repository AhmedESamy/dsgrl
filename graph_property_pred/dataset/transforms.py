from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data


class SplitTransform:
    
    def __init__(self, num_nodes, num_classes, num_splits=1, train_rate=0.05, val_rate=0.15):
        assert num_splits > 0
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.num_splits = num_splits
        self.train_rate = train_rate
        self.val_rate = val_rate
        
    def init_transform_object(self):
        train_size = self.num_nodes * self.train_rate
        num_train_pc = int(train_size / self.num_classes)
        val_size = int(self.num_nodes * self.val_rate)
        test_size = int(self.num_nodes - (train_size + val_size))

        print("Applying random node splits, "
              f"\n\tnumber of splits: {self.num_splits} " 
              f"\n\tnumber of training samples per class: {num_train_pc} " 
              f"\n\tnumber of validation samples: {val_size} "
              f"\n\tnumber of test samples: {test_size}")
        self.split_transform = RandomNodeSplit(
            num_train_per_class=num_train_pc, 
            num_val=val_size, num_splits=self.num_splits,
            num_test=test_size, split="test_rest"
        )
    
    def __call__(self, data):
        self.init_transform_object()
        data = self.split_transform(data)
        splits = []
        for i in range(self.num_splits):
            test_indices = data.test_mask[:, i].nonzero().ravel().numpy().tolist()
            train_indices = data.train_mask[:, i].nonzero().ravel().numpy().tolist()
            val_indices = data.val_mask[:, i].nonzero().ravel().numpy().tolist()
            splits.append({
                "test": test_indices, 
                "model_selection": [{
                    "train": train_indices,
                    "validation": val_indices
                }]
            })
        return splits