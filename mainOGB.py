'''
version 4.9    validation on linear evaluation
version 4.4    加入 linear evaluation
version 4.1    fix step-size， control delta, projection-hidden-size = 64 --bottleneck
version 2.2    输出loss
version 1.14   改回LiblinearSVC, 修改_base.py中的warning
version 1.13.2 修改了evalesp中的LinearSVC,换成SVC(kernel='linear')就不报错了
version 1.13 加上x的one_hot编码
version 1.11 加上x和edge_attr随机初始化
version 1.10 可以正常运行MUTAG, PTC-MR
'''
import logging
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from OGBgnn import GNN
import os
import argparse
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

### importing OGB
#from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from gassl import GASSL
import pdb
from tqdm import tqdm
from tqdm import trange
import sys
### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
sys.path.insert(0, '../..')


parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
parser.add_argument('--gnn', type=str, default='gcn',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--drop_ratio', type=float, default=0,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--decay', type=float, default=0.99,
                    help='moving_average_decay (default: 0.99)')
parser.add_argument('--num_layer', type=int, default=2,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=128,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-molbbbp",
                    help='dataset name (default: ogbg-molhiv, ogbg-molpcba)')
parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
parser.add_argument('--pp', type=str, default="H",
                    help='perturb_position (default: X(feature), H(hidden layer))')

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--runs', type=int, default=1)

parser.add_argument('--step_size', type=float, default=8e-3)
parser.add_argument('--delta', type=float, default=8e-3)
parser.add_argument('--m', type=int, default=3)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--attack', type=str, default='flag')
parser.add_argument('--num_tasks', type=int, default=512)
#parser.add_argument('--projection_size', type=int, default=512)
#parser.add_argument('--prediction_size', type=int, default=512)
parser.add_argument('--projection_hidden_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=312)
parser.add_argument('--projection_size', type=int, default=300)
parser.add_argument('--prediction_size', type=int, default=300)

parser.add_argument('--ps', type=str, default="", help='personal notes')


args = parser.parse_args()
print(args)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)




def trainesp(model, device, loader, optimizer, task_type, args):
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            model.train()
            optimizer.zero_grad()
            # forward = lambda perturb: model(batch, perturb).to(torch.float32)
            perturb_shape = (batch.x.shape[0], args.emb_dim)
            perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.delta, args.delta).to(device)
            perturb.requires_grad_()

            loss = model(batch, perturb)
            # loss = forward(perturb)

            for _ in range(args.m - 1):
                loss.backward()
                perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0

                #    loss = forward(perturb)
                loss = model(batch, perturb)
                loss /= args.m

            total_loss += loss.item()
            loss.backward()
            model.update_moving_average()
            optimizer.step()
    return total_loss / len(loader)



def train_linear_perepoch(model, logreg, linear_optimizer, device, train_loader, valid_loader, test_loader, evaluator, metric):
    optimizer = linear_optimizer
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    best_train = best_val = best_test = 0
    with trange(100) as t:
        for epoch in t:
            t.set_description('epoch %i' % epoch)

            for step, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                is_labeled = batch.y == batch.y

                batch_embed = model.embed(batch)
                logits = logreg(batch_embed)
                loss = cls_criterion(logits.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                train_perf = eval(model, logreg, device, train_loader, evaluator)
                valid_perf = eval(model, logreg, device, valid_loader, evaluator)
                test_perf = eval(model, logreg, device, test_loader, evaluator)

                tra, val, tst = (train_perf[metric], valid_perf[metric], test_perf[metric])

                if val > best_val:
                    best_train, best_val, best_test = tra, val, tst

                print(f'epoch:{epoch} Train:{tra:9.5f} val:{val:9.5f}, test:{tst:9.5f}')
                t.set_postfix(tra=best_train, val=best_val, test=best_test)

    print(f'Train:{best_train:9.5f} val:{best_val:9.5f}, test:{best_test:9.5f}')
    return (best_train, best_val, best_test)


def get_one_hot(dataset):
    g_idx = 0
    total_node = 0
    for i in dataset.data.num_nodes:
        total_node += i
    total_degree = np.zeros(total_node)
    node_start = 0
    node_end = 0
    for i in dataset.data.num_nodes:
        node_end += i
        edge_start = dataset.slices['edge_index'][g_idx]
        edge_end = dataset.slices['edge_index'][g_idx+1]
        edges = dataset.data.edge_index[:, edge_start:edge_end]
        in_degree = out_degree = np.zeros(i)

        for ee in edges:
            in_degree[ee] += 1
            out_degree[ee] += 1
        tot_degree = in_degree + out_degree
        total_degree[node_start:node_end] = tot_degree
        node_start = node_end
        g_idx += 1


    total_degree = total_degree.astype(np.int64)
    return torch.nn.functional.one_hot(torch.tensor(torch.from_numpy(total_degree))).float()



def main():
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if dataset.data.x is None:
        dataset.data.x = get_one_hot(dataset)

    all_idx = torch.tensor(range(0, len(dataset)))  # train on all data
    all_loader = DataLoader(dataset[all_idx], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    if 'x' not in all_loader.dataset.slices:
        tmp = torch.LongTensor(len(dataset.data.num_nodes)+1)
        accum_node = 0
        tmp[0] = 0
        for i in range(len(dataset.data.num_nodes)):
            accum_node += dataset.data.num_nodes[i]
            tmp[i+1] = accum_node
        all_loader.dataset.slices['x'] = tmp

    if dataset.data.x is not None:
        feat_dim = dataset.data.x.shape[-1]

    best_result = -1
    all_results = []
    seeds = [args.seed]

    best_acc = -1
    best_std = -1
    best_results = []
    trains, vals, tests = [], [], []
    for run in range(len(seeds)):
        best_train, best_val, best_test = 0, 0, 0
        seed = seeds[run]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.gnn == 'gin':
            gnnmodel = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                           drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim, perturb_position=args.pp).to(device)
        elif args.gnn == 'gcn':
            gnnmodel = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                           drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim, perturb_position=args.pp).to(device)
        else:
            raise ValueError('Invalid GNN type')

        model = GASSL(gnnmodel, num_tasks=dataset.num_tasks, emb_dim = args.emb_dim, projection_size=args.projection_size,
                     prediction_size=args.prediction_size, projection_hidden_size=args.projection_hidden_size,
                     moving_average_decay = args.decay
                     )
        
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



        logreg = LogisticRegression(args.emb_dim, dataset.num_tasks)
        logreg = logreg.to(device)


        split_idx = dataset.get_idx_split()
        ### automatic evaluator. takes dataset name as input
        evaluator = Evaluator(args.dataset)


        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)


        cnt_wait = 0
        best = 1e9
        patience = 20
        train_loss = np.zeros(args.epochs+1)

        for epoch in tqdm(range(1, args.epochs+1)):
            loss= trainesp(model, device, train_loader, optimizer, None, args)

            if epoch % args.test_freq == 0 or epoch == args.epochs:
                linear_optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
                result = train_linear_perepoch(model, logreg, linear_optimizer, device, train_loader, valid_loader, test_loader, evaluator, dataset.eval_metric)
                tra, val, tst = result
                if val > best_val:
                    best_train, best_val, best_test = result

                print(f'Train:{best_train:9.5f} val:{best_val:9.5f} test:{best_test:9.5f}')

        trains.append(best_train)
        vals.append(best_val)
        tests.append(best_test)

    print('')
    print(f"Average train accuracy: {np.mean(trains)}  {np.std(trains)}")
    print(f"Average val accuracy: {np.mean(vals)}  {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)}  {np.std(tests)}")


if __name__ == "__main__":
    main()
