from gnn import *
import torch
from torch import nn
import torch.nn.functional as F
from functools import wraps
import copy
import random
from functools import wraps


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def reset_weight(l):
    if hasattr(l, 'reset_parameters'):
        l.reset_parameters()
class MLP(nn.Module):
    def __init__(self, dim, hidden_size, projection_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            # nn.PReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        self.net.apply(reset_weight)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


class GASSL(nn.Module):
    def __init__(self, net, emb_dim=300, projection_hidden_size=2048, projection_size=512, prediction_size=256, num_tasks=256, moving_average_decay=0.99):
        super().__init__()

        self.projection_hidden_size = projection_hidden_size
        self.moving_average_decay = moving_average_decay

        self.online_encoder = net
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_projector = MLP(
            emb_dim, projection_hidden_size, projection_size)
        # predict_size)
        self.predictor = MLP(
            projection_size, projection_hidden_size, prediction_size)


    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_parameters(self):
        self.online_encoder.reset_parameters()
        self.reset_moving_average()
        self.target_ema_updater = EMA(self.moving_average_decay)

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder, self.online_encoder)

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, batch, perturb=None):
        online_proj_one = self.online_encoder(batch, None)
        online_proj_two = self.online_encoder(batch, perturb)

        # self.online_predictor(online_proj_one)
        online_pred_one = self.online_projector(online_proj_one)
        # self.online_predictor(online_proj_two)
        online_pred_two = self.online_projector(online_proj_two)

        # self.online_predictor(online_proj_one)
        online_pred_one = self.predictor(online_pred_one)
        # self.online_predictor(online_proj_two)
        online_pred_two = self.predictor(online_pred_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(batch, None)
            target_proj_two = target_encoder(batch, perturb)

            # self.online_predictor(online_proj_one)
            target_pred_one = self.online_projector(target_proj_one)
            # self.online_predictor(online_proj_two)
            target_pred_two = self.online_projector(target_proj_two)

        loss_one = self.loss_fn(online_pred_one,
                                target_pred_two.detach())  # self.loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = self.loss_fn(online_pred_two,
                                target_pred_one.detach())  # self.loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two  # [bs, num_node]
        return loss.mean()

    def embed(self, batch_data):
        online_l_one = self.online_encoder(batch_data, None)
        return online_l_one.detach()
