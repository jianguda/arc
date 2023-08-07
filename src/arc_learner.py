from collections import defaultdict

from einops import rearrange
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch import optim, Tensor
from torch.nn import Parameter
from torch.utils import data

from arc_shared import DEVICE, CFG_MODE_INSPECT
from arc_util import get_attr, inspect


class CoupleDataset(data.Dataset):
    def __init__(self, embeds, labels):
        self.length = len(embeds)
        self.embeds = embeds
        self.labels = labels

    def __getitem__(self, item):
        return self.embeds[item], self.labels[item]

    def __len__(self):
        return self.length


class ADA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # channel
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=dim, out_features=dim, bias=False)
        )

    def forward(self, x):
        # logger.warning(f'{x.shape=}')
        b, l, c = x.shape
        x = x.transpose(-2, -1)

        # channel
        avg_out = self.avg_pool(x)
        avg_out = self.mlp(avg_out.view(b, -1))
        max_out = self.max_pool(x)
        max_out = self.mlp(max_out.view(b, -1))
        channel_out = avg_out + max_out
        channel_scale = torch.sigmoid(channel_out)
        channel_scale = channel_scale.unsqueeze(-1).expand_as(x)
        x = x * channel_scale

        x = x.transpose(-2, -1)
        # logger.warning(f'{x.shape=}')
        # logger.debug(f'{x=}')
        return x


class SAN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        heads = 8
        dim_head = 64
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, w)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=dim, out_features=2 * dim, bias=False)
        self.fc2 = nn.Linear(in_features=2 * dim, out_features=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class RFC(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        x = x + out
        out = self.fc2(x)
        x = x + out
        return x


class Refiner(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        # # ...
        # # self.focus = ADA(dim)
        # # self.en_san = SAN(dim)
        # self.en_rfc = RFC(dim)
        self.en_mlp = MLP(dim)

        # # self.de_san = SAN(dim)
        self.de_mlp = MLP(dim)
        # self.de_rfc = RFC(dim)
        # # self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128))
        # # x = torch.mean(x, dim=1) if self.pool == 'mean' else x[:, 0]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x = self.focus(x)
        # crate encoder
        # t = x + self.en_san(x)
        z = self.en_mlp(x)
        # z = self.en_rfc(x)

        # crate decoder
        # x = self.de_rfc(z)
        x = self.de_mlp(z)

        # x = t - self.de_san(t)
        return z, x

    def inverse(self, z: Tensor) -> Tensor:
        # x = self.de_rfc(z)
        x = self.de_mlp(z)
        return x


class Learner(LightningModule):
    def __init__(self, model, attrs, dim_size, cluster_num, identifier):
        super().__init__()
        self.model = model
        self.attrs = attrs
        self.class_num = cluster_num
        self.refiner = Refiner(dim=dim_size)
        self.identifier = identifier

        # loss
        self.simi_fn = nn.CosineSimilarity(dim=-1)
        # important to let CE be sum and MSE be mean
        self.criterion1 = nn.CrossEntropyLoss(reduction='sum')
        self.criterion2 = nn.MSELoss(reduction='mean')

        # cache
        self.epoch_idx = 1
        self.cache_epoch_embeds = list()
        self.cache_epoch_labels = list()

    def build_actual_matrix(self):
        # lm_head_matrix
        lm_head_matrix = get_attr(self.model, self.attrs['lm_head'])
        lm_head_matrix = lm_head_matrix.weight
        lm_head_matrix = lm_head_matrix.transpose(0, 1).to(DEVICE)
        return lm_head_matrix

    def build_virtual_matrix(self):
        # lm_head_matrix
        lm_head_matrix = get_attr(self.model, self.attrs['lm_head'])
        lm_head_matrix = lm_head_matrix.weight
        lm_head_matrix = lm_head_matrix.transpose(0, 1).to(DEVICE)

        # # it seems that only kaiming uniform could work well
        # init_noise = torch.zeros_like(lm_head_matrix)
        # init_noise = nn.init.kaiming_uniform_(init_noise, mode='fan_in', nonlinearity='relu')
        # noised_matrix = lm_head_matrix + init_noise

        # NOTE use `pinv` instead of `transpose`
        # matrix = noised_matrix.transpose(0, 1)
        matrix = torch.linalg.pinv(lm_head_matrix.to(dtype=torch.float32).cpu()).to(DEVICE)
        return matrix

    def compute_anchor_embed(self, matrix, token_label):
        # embeddings_matrix
        embeddings_matrix = get_attr(self.model, self.attrs['embedding'])
        embeddings_matrix = embeddings_matrix.weight

        # one-hot tensor
        vocab_size = embeddings_matrix.shape[0]
        # print(f'{vocab_size=}')
        # print(f'{embeddings_matrix.shape=}')
        one_hot_tensor = torch.zeros([1, vocab_size], device=DEVICE)
        one_hot_tensor[:, token_label] = 1.0
        # label_tensor = torch.tensor([token_label], dtype=torch.int64).to(device=DEVICE)
        # one_hot_tensor = torch.zeros(len(label_tensor), vocab_size, device=DEVICE)
        # one_hot_tensor = one_hot_tensor.scatter_(1, label_tensor.unsqueeze(1), 1.)
        # one_hot_tensor = torch.zeros(len(label_tensor), vocab_size, device=DEVICE) + 0.5 / self.class_num
        # one_hot_tensor = one_hot_tensor.scatter_(1, label_tensor.unsqueeze(1), 0.5 + 0.5 / self.class_num)
        one_hot_tensor = one_hot_tensor.clone().requires_grad_(True)
        # print(f'{one_hot_tensor.shape=}')
        heuristic_dist = one_hot_tensor
        heuristic_repr = torch.matmul(heuristic_dist, matrix)
        heuristic_repr = heuristic_repr.flatten()
        return heuristic_repr

    def adapt_anchors(self, anchors):
        ada_anchors = self.refiner.inverse(anchors)
        return ada_anchors

    def init_anchors(self, embeds=None, labels=None):
        if embeds is None or labels is None:
            self.anchors = Parameter(self.init_abs_anchors())
        else:
            self.anchors = Parameter(self.init_rel_anchors(embeds, labels))

    def init_abs_anchors(self):
        matrix = self.build_virtual_matrix()

        anchors = list()
        # absolute_anchor
        for label in range(self.class_num):
            anchor = self.compute_anchor_embed(matrix, label)
            anchors.append(anchor)
        anchors = torch.stack(anchors, dim=0)
        return anchors

    def init_rel_anchors(self, embeds, labels):
        anchors = list()
        # relative_anchor
        embeds_by_label = defaultdict(list)
        for embed, label in zip(embeds, labels):
            embeds_by_label[label].append(embed)
        for _label, _embeds in embeds_by_label.items():
            # logger.warning(f'{_embeds=}')
            _embeds = torch.cat(_embeds, dim=0)
            _anchor = torch.mean(_embeds, dim=0)
            # logger.critical(f'{_anchor=}')
            anchors.append(_anchor)
        anchors = torch.stack(anchors, dim=0)
        return anchors

    def forward(self, embeds: Tensor) -> tuple[Tensor, Tensor]:
        # dists = list()
        # drift_loss = 0
        # for x, length in zip(xs, lengths):
        #     x = x.unsqueeze(0)
        #     x, scale = self.filter(x)
        #     dist = self.embeds_to_logits(x)
        #     dists.append(dist)
        # dists = torch.cat(dists, 0)
        # return dists, drift_loss

        z, embeds = self.refiner(embeds)
        # TODO whether we need normalize???
        # embeds = normalize(embeds, dim=1)
        return z, embeds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.refiner.parameters(), lr=1e-4)
        return optimizer

    def compute_loss(self, embeds, embeds_z, embeds_hat, labels):
        simis = self.simi_fn(embeds_z, self.anchors)
        logits = nn.functional.softmax(simis, dim=-1).to(DEVICE)
        truths = labels.to(DEVICE)
        clustering_loss = self.criterion1(logits, truths)
        reconstruct_loss = self.criterion2(embeds_hat, embeds)
        loss = clustering_loss + reconstruct_loss
        return loss

    def on_train_epoch_end(self):
        if CFG_MODE_INSPECT:
            embeds = self.cache_epoch_embeds
            labels = self.cache_epoch_labels

            # if CFG_ANCHOR_INIT == 'rel':
            #     self.init_anchors(embeds, labels)

            # have to do prediction here
            embeds = torch.stack(embeds, 0)
            embeds_z, embeds_hat = self(embeds)
            # logger.debug(f'{embeds.shape=}')
            # logger.debug(f'{embeds_hat.shape=}')
            labels = [label.tolist() for label in labels]

            inspect(self.epoch_idx, self.identifier, embeds_z, labels)
            self.epoch_idx += 1
            self.cache_epoch_embeds.clear()
            self.cache_epoch_labels.clear()

    def training_step(self, batch, batch_idx):
        embeds, labels = batch
        embeds_z, embeds_hat = self(embeds)
        train_loss = self.compute_loss(embeds, embeds_z, embeds_hat, labels)
        if CFG_MODE_INSPECT:
            self.cache_epoch_embeds.extend(embeds)
            self.cache_epoch_labels.extend(labels)
        return train_loss

    def validation_step(self, batch, batch_idx):
        embeds, labels = batch
        embeds_z, embeds_hat = self(embeds)
        val_loss = self.compute_loss(embeds, embeds_z, embeds_hat, labels)
        return val_loss

    def test_step(self, batch, batch_idx):
        embeds, labels = batch
        embeds_z, embeds_hat = self(embeds)
        test_loss = self.compute_loss(embeds, embeds_z, embeds_hat, labels)
        return test_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        embeds, labels = batch
        embeds_z, embeds_hat = self(embeds)
        # score, _ = clustering_metric(Y, X, cluster_num)
        # print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'
        #       .format(score['NMI'], score['ARI'], score['f_measure'], score['accuracy']))
        return embeds_z.detach()
