'''
version 5.13   1. 更正扰动位置，增加第一个隐藏层的扰动。
version 4.1    fix step-size， control delta, projection-hidden-size = 64 --bottleneck
version 2.2    输出loss
version 1.14   改回LiblinearSVC, 修改_base.py中的warning
version 1.13.2 修改了evalesp中的LinearSVC,换成SVC(kernel='linear')就不报错了
version 1.13 加上x的one_hot编码
version 1.11 加上x和edge_attr随机初始化
version 1.10 可以正常运行MUTAG, PTC-MR IMDB-BiNARY IMDB-MULTI
'''
import logging
import torch
from torch_geometric.utils import degree
import pkg_resources
from torch_geometric.transforms import OneHotDegree, Constant
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from gnn import GNN
import argparse
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from gassl import GASSL
from tqdm import tqdm
import sys

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
sys.path.insert(0, '../..')


parser = argparse.ArgumentParser(
    description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
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


def train(model, device, loader, optimizer, args):
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x is None or batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            print("Training passed")

        else:
            model.train()
            optimizer.zero_grad()
            perturb_shape = (batch.x.shape[0], args.emb_dim)
            # 均匀分布
            perturb = torch.FloatTensor(
                *perturb_shape).uniform_(-args.delta, args.delta).to(device)
            perturb.requires_grad_()

            loss = model(batch, perturb)

            for _ in range(args.m - 1):
                loss.backward()
                perturb_data = perturb.detach() + args.step_size * \
                    torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0

                loss = model(batch, perturb)
                loss /= args.m

            total_loss += loss.item()
            loss.backward()
            model.update_moving_average()
            optimizer.step()

    return total_loss / len(loader)


def eval(model, device, loader):
    model.eval()
    all_embed = []
    y_true = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                batch_embed = model.embed(batch)
                all_embed.append(batch_embed)
                y_true.append(batch.y.detach().cpu())

    all_embed = torch.cat(all_embed, dim=0)
    y_true = torch.cat(y_true, dim=0)
    x = all_embed.cpu().detach().numpy()
    y = y_true.cpu().detach().numpy()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(
            LinearSVC(dual=False), params, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return (np.mean(accuracies), np.std(accuracies))


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
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")
    useNodeAttr = False
    dataset_with_no_x = ['IMDB-BINARY', 'COLLAB']
    if args.dataset in dataset_with_no_x:
        useNodeAttr = True
    dataset = TUDataset('', args.dataset, use_node_attr=useNodeAttr)

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        # dataset.data.x = get_one_hot(dataset)
        try:
            dataset.data = OneHotDegree(
                max_degree=-2, cat=False)(dataset.data)
        except:
            dataset.data = Constant(
            )(dataset.data)

    else:
        dataset.data.x = dataset.data.x.float()

    all_idx = torch.tensor(range(0, len(dataset)))  # train on all data
    all_loader = DataLoader(dataset[all_idx], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)

    if 'x' not in all_loader.dataset.slices:
        tmp = torch.LongTensor(len(dataset.data)+1)
        accum_node = 0
        tmp[0] = 0
        i = 0
        for data in dataset:
            accum_node += data.num_nodes
            # tmp[i] = accum_node
            # TODO
            print(i)
            i += 1
        # all_loader.dataset.slices['x'] = tmp
        all_loader.dataset.slices['x'] = tmp

    if dataset.data.x is not None:
        feat_dim = dataset.data.x.shape[-1]

    best_result = -1
    all_results = []
    seeds = [args.seed]

    best_acc = -1
    best_std = -1
    for run in range(len(seeds)):
        results = []
        seed = seeds[run]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.gnn == 'gin':
            gnnmodel = GNN(gnn_type='gin', num_layer=args.num_layer, emb_dim=args.emb_dim,
                           drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim, perturb_position=args.pp).to(device)
        elif args.gnn == 'gcn':
            gnnmodel = GNN(gnn_type='gcn', num_layer=args.num_layer, emb_dim=args.emb_dim,
                           drop_ratio=args.drop_ratio, virtual_node=False, feat_dim=feat_dim, perturb_position=args.pp).to(device)
        else:
            raise ValueError('Invalid GNN type')

        model = GASSL(gnnmodel, emb_dim=args.emb_dim, projection_size=args.projection_size,
                      prediction_size=args.prediction_size, projection_hidden_size=args.projection_hidden_size,
                      moving_average_decay=args.decay)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        cnt_wait = 0
        best = 1e9
        patience = 20
        train_loss = np.zeros(args.epochs+1)
        for epoch in tqdm(range(1, args.epochs + 1)):
            loss = train(model, device, all_loader, optimizer, args)
            train_loss[epoch] = loss

            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait > patience:
                break

            if epoch % args.test_freq == 0:
                acc, std = eval(model, device, all_loader)
                results.append([seed, epoch, acc, std])
                if acc > best_acc:
                    best_acc, best_std = acc, std
                print(
                    f'acc mean {acc:.5f}, std {std:.5f}, best acc mean {best_acc:.5f}, std {best_std:.5f}, loss {loss:.5f}')
        for r in results:
            print(f'seed{r[0]}, epoch{r[1]} acc:{r[2]:.5f} std:{r[3]:.5f}')
            best_result = r[2] if r[2] > best_result else best_result
        all_results.append(results)

    print(f'best acc = {best_result}')


if __name__ == "__main__":
    main()
