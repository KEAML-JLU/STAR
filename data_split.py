import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CoraFull, Reddit2, Coauthor, Planetoid, Amazon, EmailEUCore, Reddit, WikiCS, CitationFull
import random
import yaml
from yaml import SafeLoader
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from sklearn import preprocessing
import scipy.sparse as sp
import scipy.io as sio
from args import get_args

class_split = {
    "CoraFull": {"train": 40, 'dev': 15, 'test': 15},  # Sufficient number of base classes
    "ogbn-arxiv": {"train": 20, 'dev': 10, 'test': 10},
    "coauthor-cs": {"train": 5, 'dev': 5, 'test': 5},
    "Amazon-Computer": {"train": 4, 'dev': 3, 'test': 3},
    "Cora": {"train": 3, 'dev': 2, 'test': 2},
    "CiteSeer": {"train": 2, 'dev': 2, 'test': 2},
    "Reddit": {"train": 21, 'dev': 10, 'test': 10},
    "Amazon_clothing": {"train": 40, 'dev': 17, 'test': 20},
    "Amazon_eletronics": {"train": 90, 'dev': 37, 'test': 40},
    "dblp": {"train": 80, 'dev': 27, 'test': 30},
    "Email": {"train": 20 , 'dev': 10, 'test': 12},
    "WikiCS": {"train": 4, 'dev': 3, 'test': 3},
    "Cora_ML": {"train": 3, 'dev': 2, 'test': 2}
    
}


args = get_args()
config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    
    
def split(dataset_name):
    
    if dataset_name == 'Cora':
        dataset = Planetoid(root='./dataset/' + dataset_name, name="Cora")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Cora_ML':
        dataset = CitationFull(root='./dataset/' + dataset_name, name="Cora_ML")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='./dataset/' + dataset_name, name="CiteSeer")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'WikiCS':
        dataset = WikiCS(root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Amazon-Computer':
        dataset = Amazon(root='./dataset/' + dataset_name, name="Computers")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'coauthor-cs':
        dataset = Coauthor(root='./dataset/' + dataset_name, name="CS")
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'CoraFull':
        dataset = CoraFull(root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    elif dataset_name == 'Email':
        dataset_pr = EmailEUCore(root='./dataset/' + dataset_name)
        num_nodes = dataset_pr.data.num_nodes
        dataset = Data(x=torch.eye(num_nodes), edge_index=dataset_pr[0].edge_index, y=dataset_pr[0].y)
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name = dataset_name, root='./dataset/' + dataset_name)
        num_nodes = dataset.data.num_nodes
    else:
        print("Dataset not support!")
        exit(0)
    if dataset_name != 'Email':
        data = dataset.data
        class_list = [i for i in range(dataset.num_classes)]

    else:
        data = dataset
        class_list = [i for i in range(len(set(data.y.numpy())))]
    print("********" * 10)

    train_num = class_split[dataset_name]["train"]
    dev_num = class_split[dataset_name]["dev"]
    test_num = class_split[dataset_name]["test"]
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    random.shuffle(class_list)
    train_class = class_list[: train_num]
    dev_class = class_list[train_num : train_num + dev_num]
    test_class = class_list[train_num + dev_num :]
    print("train class: {}; dev class: {}; test class: {}".format(train_class, dev_class, test_class))

    print("train_num: {}; dev_num: {}; test_num: {}".format(train_num, dev_num, test_num))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(torch.squeeze(data.y).tolist()):
        id_by_class[cla].append(id)

    train_idx = []
    for cla in (train_class + dev_class):
        train_idx.extend(id_by_class[cla])

    return dataset, np.array(train_idx), id_by_class, train_class, dev_class, test_class#, degree_inv


def test_task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected


class Sinkhorn(nn.Module):

    def __init__(self, eps, eps_parameter, max_iter, thresh, reduction="none", device="cpu"):
        super(Sinkhorn, self).__init__()
        self.device = device
        self.eps_parameter = eps_parameter

        self.eps = eps
        if self.eps_parameter:
            self.eps = nn.Parameter(torch.tensor(self.eps))

        self.max_iter = max_iter
        self.thresh = thresh
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        cost_normalization = C.max()
        C = (
            C / cost_normalization
        )  # Needs to normalize the matrix to be consistent with reg

        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(
            1.0 / x_points).squeeze().to(self.device)

        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(
            1.0 / y_points).squeeze().to(self.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u,
                                      v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < self.thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


@torch.no_grad()
def distributed_sinkhorn(input, epsilon, iterations):
    # ensure that input has not infinite values
    np_input = input.cpu().numpy()
    input[input == float("Inf")] = np.nanmax(np_input[np_input != np.inf]) + 10

    while True:
        # Q is K-by-B
        Q = torch.exp(input / epsilon).t()
        B = Q.shape[1]   # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        if not torch.isnan(Q).any():
            break
        epsilon += 0.1

    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()