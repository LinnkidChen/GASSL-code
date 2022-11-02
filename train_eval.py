import time

import torch
import torch.nn.functional as F

from torch import tensor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def cross_validation_with_val_set(dataset, model, dataset_name, gnn, pp, folds, epochs, batch_size, lr, lr_decay_factor, lr_decay_step_size, weight_decay, args, logger=None):

    results, accs, durations = [], [], []
    best_acc = -1
    best_std = -1
    # train_dataset = dataset[train_idx]
    # test_dataset = dataset[test_idx]
    # val_dataset = dataset[val_idx]

    if 'adj' in dataset[0]:
        train_loader = DenseLoader(dataset, batch_size, shuffle=True)
        # val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
        # test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader, args)
        # val_losses.append(eval_loss(model, val_loader))
        # accs.append(eval_acc(model, train_loader))
        # eval_info = {
        #     'epoch': epoch,
        #     'train_loss': train_loss,
        #     # 'val_loss': val_losses[-1],
        #     'test_acc': accs[-1],
        # }

        # if logger is not None:
        #     logger(eval_info)
        if epoch % args.test_freq == 0:
            acc, std = eval(model, device, train_loader)
            results.append([args.seed, epoch, acc, std])
            if acc > best_acc:
                best_acc, best_std = acc, std
            print(
                f'acc mean {acc:.5f}, std {std:.5f}, best acc mean {best_acc:.5f}, std {best_std:.5f}, loss {train_loss:.5f}')
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    # loss, acc, duration = tensor(train_loss), tensor(accs), tensor(durations)
    # loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    # loss, argmin = loss.min(dim=1)
    # acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    # loss_mean = loss.mean().item()

    # acc_mean = acc.mean().item()
    # acc_std = acc.std().item()
    # duration_mean = duration.mean().item()
    print(f'{dataset_name} - {gnn} --{pp}:')
    print(f' Test Accuracy: {best_acc:.3f} '
          f'± {best_std:.3f}')

    return best_acc, best_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


# def train(model, optimizer, loader):
#     model.train()

#     total_loss = 0
#     for data in loader:
#         optimizer.zero_grad()
#         data = data.to(device)
#         out = model(data)
#         loss = F.nll_loss(out, data.y.view(-1))
#         loss.backward()
#         total_loss += loss.item() * num_graphs(data)
#         optimizer.step()
#     return total_loss / len(loader.dataset)

def train1(model, device, loader, optimizer, args):
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x is None or batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            print("Training passed")

        else:
            model.train()
            optimizer.zero_grad()
            perturb_shape = (batch.x.shape[0], args.emb_dim)
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


def train(model, optimizer, loader, args):
    loss = train1(model, "cuda:0", loader, optimizer, args)
    return loss


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)
