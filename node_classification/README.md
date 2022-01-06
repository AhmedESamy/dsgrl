# Node classification experiment

### Example Usage

### Training

```sh
$ python main.py
```

### Tuning

```sh
$ python main.py --tune
```

### Options

`--root:`
A path to a root directory to put all the datasets. Default is `./data`

`--name:`
The name of the datasets. Default is `Photo`. Check the [`Supported dataset names`](#Supported-dataset-names)

`--aug-dim`: The number of dimensions used in the augmentors. Default is 128

`--model-dim`: The number of final output dimensions. Default is 64

`--dropout`: Dropout rate (probability). Default is 0.5

`--inv-w`: The weight for the invariance term. Default is 1

`--cov-w`: The weight for the covariance regularization term. Default is 1

`--mod-w`: The weight for the model regularization term. Default is 1

`--lr`: Learning rate. Default is 0.001

`--aug-type:` The type of augmentation (`feature` or `topology`). Default is `feature`

`--loader:` The type of batch loader (`full` or `mini` for full batch or mini batch training). Default is `full`

`--num-aug-layers:` The number of layers of the augmentor model. Default is 1

`--num-gnn-layers:` The number of layers of the GNN encoder. Default is 2

`--edge-mul:` The multiplicity of new edges to be added in case of `topology` augmentation. 
The number of edges in the high-order graph will become `--edge-mul * #edges`. Default is 4

`--keep-edges:` Whether to ensure that the original edges are kept when learning a high-order network
in the case of `topology` augmentation. Default is false.

`--task`: The type of node classification task (`bc`, `mcc`, or `mlc` for binary, multi-class or multi-label classification).
Default is `mcc`

`--workers:` The number of CPU workers

`--tune:` Enables model selection (hyper-parameter tuning) experiment. Default is false.

`--verbose:` Enables verbosity. Default is false.

`--epochs:` The number of training epochs

`--trials:` The number of trials for model selection, if `--tune` is activated. Default is 500.
