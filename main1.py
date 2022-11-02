
import pkg_resources
from gnn import GNN
import argparse
from itertools import product
#python /root/GASSL_code/main1.py > run.log
import torch
import logging
import torch
from torch_geometric.utils import degree

from torch_geometric.transforms import OneHotDegree, ToSparseTensor
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

import argparse
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from gassl import GASSL
from tqdm import tqdm
import sys
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

parser = argparse.ArgumentParser()
parser.add_argument('--gnn', type=str, default='gin',
                    help='GNN gin, or gcn (default: gin)')
parser.add_argument('--drop_ratio', type=float, default=0.0,
                    help='dropout ratio (default: 0.0)')
parser.add_argument('--decay', type=float, default=0.99,
                    help='moving_average_decay (default: 0.99)')
parser.add_argument('--num_layer', type=int, default=2,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=512,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="MUTAG",
                    help='dataset name (default: ogbg-molhiv, ogbg-molpcba)')
parser.add_argument('--pp', type=str, default="H",
                    help='perturb_position (default: X(feature), H(hidden layer))')

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--runs', type=int, default=1)

parser.add_argument('--step-size', type=float, default=8e-3)
parser.add_argument('--delta', type=float, default=8e-3)
parser.add_argument('--m', type=int, default=3)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--num_tasks', type=int, default=512)
parser.add_argument('--projection_size', type=int, default=512)
parser.add_argument('--prediction_size', type=int, default=512)
parser.add_argument('--projection_hidden_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

# layers = [1, 2, 3, 4, 5]
# hiddens = [16, 32, 64, 128]
# datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']  # , 'COLLAB']
# datasets = ['MUTAG', 'PTC_MR', 'IMDB-BINARY',
#             'IMDB-MULTI', 'COLLAB', 'NCI1']  # , 'COLLAB']
datasets = ['COLLAB']
datasets = ['NCI1']
# nets = [
#     GCNWithJK,
#     GraphSAGEWithJK,
#     GIN0WithJK,
#     GINWithJK,
#     Graclus,
#     TopK,
#     SAGPool,
#     DiffPool,
#     EdgePool,
#     GCN,
#     GraphSAGE,
#     GIN0,
#     GIN,
#     GlobalAttentionNet,
#     Set2SetNet,
#     SortPool,
#     ASAP,
# ]
nets = [GASSL]
PPs = ['H', 'X']
GNNs = ['gin', 'gcn']


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


device = torch.device("cuda:" + str(args.device)
                      ) if torch.cuda.is_available() else torch.device("cpu")
results = []
for dataset_name, Net, pp, gnn in product(datasets, nets, PPs, GNNs):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__} - {pp} - {gnn}')

    dataset = get_dataset(dataset_name, sparse=Net != False)
    feat_dim = dataset.num_features
    # if args.gnn == 'gin':
    #     gnnmodel = GNN(gnn_type='gin', num_layer=args.num_layer, emb_dim=args.emb_dim,
    #                    drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim, perturb_position=args.pp).to(device)
    # elif args.gnn == 'gcn':
    #     gnnmodel = GNN(gnn_type='gcn', num_layer=args.num_layer, emb_dim=args.emb_dim,
    #                    drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim, perturb_position=args.pp).to(device)
    # else:
    #     raise ValueError('Invalid GNN type')
    gnnmodel = GNN(gnn_type=gnn, num_layer=args.num_layer, emb_dim=args.emb_dim,
                   drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim, perturb_position=pp).to(device)
    model = GASSL(gnnmodel, emb_dim=args.emb_dim, projection_size=args.projection_size,
                  prediction_size=args.prediction_size, projection_hidden_size=args.projection_hidden_size,
                  moving_average_decay=args.decay)
    model.to(device)
    acc, std = cross_validation_with_val_set(
        dataset,
        model,
        dataset_name,
        gnn,
        pp,

        folds=10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_factor=0.5,
        lr_decay_step_size=50,
        weight_decay=0,
        args=args,
        logger=None,
    )
    # if loss < best_result[0]:
    #     best_result = (loss, acc, std)

#     desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
#     print(f'Best result - {desc}')
#     results += [f'{dataset_name} - {model}: {desc}']
# results = '\n'.join(results)
# print(f'--\n{results}')
