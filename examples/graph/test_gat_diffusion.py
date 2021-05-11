import torch
import argparse
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GAT
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PrePtbDataset
import torch_geometric.transforms as T
import numpy as np
import sys
from deeprobust.graph.global_attack import MetaApprox, Metattack

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.1,  help='perturbation rate:0.05, 0.1, 0.15, 0.2, 0.25')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

gat = GAT(nfeat=features.shape[1],
      nhid=8, heads=8,
      nclass=labels.max().item() + 1,
      dropout=0.5, device=device)
gat = gat.to(device)


# test on clean graph
print('==================')
print('=== train on clean graph ===')

data1 = Dpr2Pyg(data)
gat.fit(data1, verbose=True) # train with earlystopping
gat.test()

print('==================')
print('=== train on clean graph + diffusion===')

pyg_data = Dpr2Pyg(data)
data2 = pyg_data[0]
gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128,
                                       dim=0), exact=True)
data2 = gdc(data2)

gat.fit1(data2, verbose=True) # train with earlystopping
gat.test()


# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
print('==================')
print('=== load graph perturbed by Zugner metattack (under seed 15) ===')
perturbed_data = PrePtbDataset(root='/tmp/',
        name=args.dataset,
        attack_method='meta',
        ptb_rate=args.ptb_rate)

perturbed_adj = perturbed_data.adj

data1.update_edge_index(perturbed_adj) # inplace operation
gat.fit(data1, verbose=True) # train with earlystopping
gat.test()

# load pre-attacked graph by Zugner: https://github.com/danielzuegner/gnn-meta-attack
print('==================')
print('=== load graph perturbed + diffusion ===')

data2.edge_index = torch.LongTensor(perturbed_adj.nonzero())
# data2.idx_test = perturbed_data.target_nodes

gat.fit1(data2, verbose=True) # train with earlystopping
gat.test()