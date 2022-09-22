import os
import pickle as pkl
import random
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB

root = os.path.split(__file__)[0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_dataset(name: str, device=None):
    if device is None:
        device = torch.device('cpu')
    name = name.lower()
    if name in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid(root=root + "/dataset/Planetoid", name=name)
    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root=root + "/dataset/WikipediaNetwork", name=name)
    elif name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=root + "/dataset/WebKB", name=name)
    else:
        raise "Please implement support for this dataset in function load_dataset()."
    data = dataset[0].to(device)
    x, y = data.x, data.y
    n = len(x)
    edge_index = data.edge_index
    nfeat = data.num_node_features
    nclass = len(torch.unique(y))
    return x, y, nfeat, nclass, eidx_to_sp(n, edge_index), data.train_mask, data.val_mask, data.test_mask


def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)


def select_mask(i: int, train: torch.Tensor, val: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    indices = torch.tensor([i]).to(train.device)
    train_idx = torch.index_select(train, 1, indices).reshape(-1)
    val_idx = torch.index_select(val, 1, indices).reshape(-1)
    test_idx = torch.index_select(test, 1, indices).reshape(-1)
    return train_idx, val_idx, test_idx
