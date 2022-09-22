# H2GCN-Pytorch

This repo is a pytorch implementation of H2GCN raised in the
paper ["Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs"](https://arxiv.org/abs/2006.11468)
. Original tensorflow implementation can be found [here](https://github.com/GemsLab/H2GCN).

## Requirement

This project should be able to run without any modification after following packages installed.

```
pytorch
networkx
torch-sparse
torch-geometric
```

## Tutorial

### Run train.py

```
usage: train.py [-h] [--seed SEED] [--epochs EPOCHS] [--lr LR] [--k K]
                [--wd WD] [--hidden HIDDEN] [--dropout DROPOUT]
                [--patience PATIENCE] [--dataset DATASET] [--gpu GPU]
                [--split SPLIT]

optional arguments:
  -h, --help           show this help message and exit
  --seed SEED          seed
  --epochs EPOCHS      number of epochs to train
  --lr LR              learning rate
  --k K                number of embedding rounds
  --wd WD              weight decay value
  --hidden HIDDEN      embedding output dim
  --dropout DROPOUT    dropout rate
  --patience PATIENCE  patience for early stop
  --dataset DATASET    dateset name
  --gpu GPU            gpu id to use while training, set -1 to use cpu
  --split SPLIT        data split to use
```

### Custom dataset

All dataset used in this repo were forked from repo geom-gcn.
Custom dataset should fit following format :

```
PROJECT_ROOT/new_data/DATASET_NAME/
out1_graph_edges.txt            # format for each lines : SRC_NODE DST_NODE
out1_node_feature_label.txt     # format for each rows  : NODE_ID f0,f1,···
```

### Use model.py

If you only want to use model.py separately, you need to pass two matrix to forward function while training.

```
adj : torch.sparse.Tensor.
x : torch.FloatTensor.
```
