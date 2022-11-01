import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, args, logger=None):

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader, args)
            # val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                # 'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, acc_mean, acc_std


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