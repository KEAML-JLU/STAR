import torch
import torch.nn as nn
import torch.nn.functional as F
from set_model import get_set_model
from torch_geometric.nn import GCNConv, SGConv
# this version use different projectors for different-level loss

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels, cached=True)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels, cached=True))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

    
class Decoder(nn.Module):
    def __init__(self, name: str, out_channels: int, num_heads: int):
        super(Decoder, self).__init__()
        self.set_func = get_set_model(name, out_channels, num_heads)

    def forward(self, x: torch.Tensor):

        return self.set_func(x)

class Projector(nn.Module):
    def __init__(self, n_in, n_out):
        super(Projector, self).__init__()
        
        self.w1 = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.w2 = nn.Linear(n_out, n_in)
    
    def forward(self, x):
        x = F.elu(self.w1(x))
        x = self.w2(x)
        return x
        
        
class OpTA(nn.Module):
    def __init__(
            self,
            regularization: float,
            max_iter: int,
            stopping_criterion: float,
            device: str = "cpu"):

        super(OpTA, self).__init__()
        self.sinkhorn = Sinkhorn(
            eps=regularization,
            max_iter=max_iter,
            thresh=stopping_criterion,
            eps_parameter=False,
            device=device)

    def forward(self, z_support: torch.Tensor, z_query: torch.Tensor):
        """
        Applies Optimal Transport between support and query features.

        Arguments:
            - z_support (torch.Tensor): support prototypes (or features)
            - z_query (torch.Tensor): query features

        Returns:
            - tuple(transported support prototypes, unchanged query features)
        """
        cost, transport_plan, _ = self.sinkhorn(z_support, z_query)

        z_support_transported = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True), z_query
        )

        return z_support_transported, z_query
    
class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.tau: float = tau

        self.proj1 = Projector(num_hidden, num_proj_hidden)
        self.proj2 = Projector(num_hidden, num_proj_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)
    
    def set_forward(self, x: torch.Tensor):
        return self.decoder(x)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z)) # wheter to add batchnorm1d?
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes)#.to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0, loss_type: str='ins'):
        if loss_type == 'ins':
            h1 = self.proj1(z1)
            h2 = self.proj1(z2)

        else:
            h1 = self.proj2(z1)
            h2 = self.proj2(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
    
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = torch.mean(loss.view(anchor_count, batch_size))

        return loss


class Classifier(nn.Module):
    def __init__(self, n_in, n_out):
        super(Classifier, self).__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.w = nn.Linear(n_in, n_out)
    
    def forward(self, x):
        x = F.relu(self.bn(x))
        x = self.w(x)