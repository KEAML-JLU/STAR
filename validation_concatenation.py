import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from model import Classifier
from data_split import test_task_generator, Sinkhorn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim


# this version use all support nodes to evaluate the model performance, it is the final version

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    if not isinstance(x, torch.FloatTensor):
        x = torch.FloatTensor(x)
    if not isinstance(y, torch.FloatTensor):
        y = torch.FloatTensor(y)

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).float()
    correct = correct.sum()
    return correct / len(labels)


def distributed_sinkhorn(out, epsilon=0.05, num_iterations=10, normalization='col'):
    if not isinstance(out, torch.FloatTensor):
        out = torch.FloatTensor(out)

    Q = torch.exp(out / epsilon)  # Q is NS-by-NQ (B = batch size, K = queue size)
    B = Q.shape[0]
    K = Q.shape[1]

    # make the matrix sums to 1
    Q /= torch.sum(Q)

    if normalization == 'col':
        for it in range(num_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= K

        Q *= K  # the colomns must sum to 1 so that Q is an assignment
    else:
        for it in range(num_iterations):
            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= K

            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q


def fs_dev(model, x, edge_index, y, test_num, id_by_class, test_class, n_way, k_shot, m_qry):
    model.eval()
    z = model(x, edge_index)
    z = F.normalize(z)
    z = z.detach().cpu().numpy()

    test_acc_all = []

    for i in range(test_num):
        test_id_support, test_id_query, test_class_selected = \
            test_task_generator(id_by_class, test_class, n_way, k_shot, m_qry)

        train_z = z[test_id_support]
        test_z = z[test_id_query]

        train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_support]])
        test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_query]])

        clf = LogisticRegression(solver='lbfgs', max_iter=1000,
                                 multi_class='auto').fit(train_z, train_y)

        test_acc = clf.score(test_z, test_y)
        test_acc_all.append(test_acc)

    final_mean_clf = np.mean(test_acc_all)
    final_std_clf = np.std(test_acc_all)

    return final_mean_clf, final_std_clf


def fs_test(model, x, edge_index, y, train_idx, test_num, id_by_class, test_class, n_way, k_shot, m_qry, device, args):
    model.eval()
    z = model(x, edge_index)
    z = F.normalize(z)
    
    z_np = z.cpu()
    sim = z_np @ z_np.T
    sim.fill_diagonal_(0)
    
    topnk_idx = torch.topk(sim, k=args.topk, dim=1, largest=True).indices # N x k
    topk_idx = topnk_idx[:, torch.randperm(topnk_idx.size(1))[:args.topk]]
    set_emb = z_np[topk_idx] # N x k x d
    
    set_emb = set_emb.to(device)
    zs = model.set_forward(set_emb)
    z = torch.cat([z, zs], dim=1)
    z = z.detach().cpu().numpy()

    test_acc_all = []
    for i in range(test_num):
        test_id_support, test_id_query, test_class_selected = \
            test_task_generator(id_by_class, test_class, n_way, k_shot, m_qry)

        train_z = z[test_id_support]
        test_z = z[test_id_query]
        
        sinkhorn = Sinkhorn(eps=0.05,
            max_iter=1000,
            thresh=1e-4,
            eps_parameter=False,
            device="cpu")
        
        train_z, test_z = torch.FloatTensor(train_z), torch.FloatTensor(test_z)
        cost, transport_plan, _ = sinkhorn(train_z, test_z) # 

        z_support_transported = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True), test_z)

        train_z = z_support_transported.numpy()
        train_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_support]])
        test_y = np.array([test_class_selected.index(i) for i in torch.squeeze(y)[test_id_query]])

        clf = LogisticRegression(solver='lbfgs', max_iter=1000,
                multi_class='auto').fit(train_z, train_y)

        test_acc = clf.score(test_z, test_y)
        test_acc_all.append(test_acc)

    final_mean = np.mean(test_acc_all)
    final_std = np.std(test_acc_all)

    return final_mean, final_std