import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
from dgl.data.reddit import RedditDataset
from dgl.data.ppi import PPIDataset

from dataReader_utils.WikiCS import load_graph_data



def DGLDatasetReader(dataset_name,self_loops,device=None):

    data = load_data(dataset_name,self_loops)
    if dataset_name == 'reddit':
        # DGL doesn't support DirectML device ('privateuseone'), so keep the graph on CPU

        g = data.graph.int().to(device)
        
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)


    else:
        g = data[0].to(device)

        # Extract from g.ndata
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        num_labels = int(labels.max().item()) + 1

    # Normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    g.ndata['norm'] = norm.unsqueeze(1).to(device)

    # Add self-loops if needed
    if self_loops:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    return g, features, labels, num_labels, train_mask, test_mask, val_mask


def WikiCS_loader(self_loops=False,device=None):
    data = load_graph_data.load(self_loop=self_loops)
    g = data.graph.to(device)
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.to(device)
    g.ndata['norm'] = norm.unsqueeze(1)
    # add self loop
    if self_loops:
        # g.add_edges(g.nodes(), g.nodes())
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    return g, torch.FloatTensor(data.features), torch.LongTensor(data.labels), max(data.labels)+1, \
           data.train_masks, data.test_mask, data.val_masks


from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn import preprocessing
def ogbn_arxiv_reader(device='cpu',data_root_dir="/extra/sscala/Dataset/"):
    dataset = 'ogbn-arxiv'
    dataset = DglNodePropPredDataset(name=dataset,root=data_root_dir)
    g, labels = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    srcs, dsts = g.all_edges()
    g.add_edges(dsts, srcs)
    features = g.ndata['feat']
    min_max_scaler = preprocessing.MinMaxScaler()
    feat = min_max_scaler.fit_transform(features)
    # feat_smooth_matrix = calc_feat_smooth(adj, feat)
    n_classes = (labels.max() + 1).item()

    g = g.remove_self_loop().add_self_loop()
    old_g = DGLGraph(g.to_networkx())
    old_g = old_g.clone().to(device)

    evaluator = Evaluator(name="ogbn-arxiv")
    split_idxdx = dataset.get_idx_split()
    idx_train, idx_val, idx_test = split_idxdx["train"], split_idxdx["valid"], split_idxdx["test"]

    return g.to(device), torch.tensor(feat, dtype=torch.float32).to(device), labels[:,0].to(device), n_classes, idx_train.to(device), idx_val.to(device), idx_test.to(device), evaluator



def load_data(dataset_name,self_loops):
    if dataset_name == 'cora':
        return citegrh.load_cora()
    elif dataset_name == 'citeseer':
        return citegrh.load_citeseer()
    elif dataset_name== 'pubmed':
        return citegrh.load_pubmed()
    elif dataset_name== "PPI":
        return PPIDataset('test')
    elif dataset_name== "WikiCS":
        return WikiCS_loader(self_loops)

    elif dataset_name is not None and dataset_name.startswith('reddit'):
        return RedditDataset(self_loop=self_loops)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

