import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# MeanPooling --------------------------------------------------------
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, X):
        return X.mean(dim=1)

# DeepSet -------------------------------------------------------------
class DeepSet(nn.Module):
    def __init__(self, dim):
        super(DeepSet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.layers(X.sum(dim=1))

# RepSet --------------------------------------------------------------
class RepSet(torch.nn.Module):    
    def __init__(self, dim, n_elements):
        super(RepSet, self).__init__()
        self.dim = dim
        self.n_elements = n_elements
        
        self.Wc = nn.Parameter(torch.FloatTensor(dim, n_elements*dim))
        self.Wc.data.uniform_(-1, 1)
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU()

    def forward(self, X):
        # B = # of set
        # N = # of set elements
        # K = # of hidden units
        
        # B x N x (K x dim)
        t = self.relu(torch.matmul(X, self.Wc))
        B, N, _ = t.shape
        t = t.view(B, N, self.n_elements, self.dim).contiguous()
        t = torch.max(t, dim=2)[0]
        t = torch.sum(t, dim=1)
        return self.layers(t)

# MAB ---------------------------------------------------------------------
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(dim_split), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)        
        return O

# SAB ---------------------------------------------------------------------
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

# PMA1 ---------------------------------------------------------------------
class PMA1(nn.Module):
    def __init__(self, dim, num_heads, ln=False):
        super(PMA1, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        X = self.mab(self.S.repeat(X.size(0), 1, 1), X)
        return X.squeeze(1)

# DeepPooler ----------------------------------------------------------------
class DeepPooler(nn.Module):
    def __init__(self, dim):
        super(DeepPooler, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, X):
        return self.mlp(torch.cat([
            X.mean(dim=1), 
            X.max(dim=1)[0],
            X.min(dim=1)[0],
            X.std(dim=1, unbiased=False)
        ], dim=-1))

def get_set_model(set_model, last_hidden_size, num_heads):
    # ours ---------------------------------------------------------------------
    # DeepPooler
    if set_model == 'DeepPooler':
        decoder = nn.Sequential(
            DeepPooler(last_hidden_size)
        )#.to(device)

    # SAB(ln=True),DeepPooler
    elif set_model == 'SLD':
        decoder = nn.Sequential(
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=True),
            DeepPooler(last_hidden_size)
        )#.to(device)
    # SAB(ln=False),DeepPooler
    elif set_model == 'SD':
        decoder = nn.Sequential(
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=False),
            DeepPooler(last_hidden_size)
        )#.to(device)
    # SABx2(ln=True),DeepPooler
    elif set_model == 'S2LD':
        decoder = nn.Sequential(
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=True),
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=True),
            DeepPooler(last_hidden_size)
        )#.to(device)
    # SABx2(ln=False),DeepPooler
    elif set_model == 'S2D':
        decoder = nn.Sequential(
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=False),
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=False),
            DeepPooler(last_hidden_size)
        )#.to(device)
    # SABx3(ln=True),DeepPooler
    elif set_model == 'S3LD':
        decoder = nn.Sequential(
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=True),
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=True),
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=True),
            DeepPooler(last_hidden_size)
        )#.to(device)
    # SABx3(ln=False),DeepPooler
    elif set_model == 'S3D':
        decoder = nn.Sequential(
            SAB(last_hidden_size, last_hidden_size, args.num_heads, ln=False),
            SAB(last_hidden_size, last_hidden_size, args.num_heads, ln=False),
            SAB(last_hidden_size, last_hidden_size, args.num_heads, ln=False),
            DeepPooler(last_hidden_size)
        )#.to(device)
    
    # baselines --------------------------------------------------------------------- 
    # MeanPooling
    elif set_model == 'MeanPooling':    
        decoder = MeanPooling()#.to(device)
    # DeepSet
    elif set_model == 'DeepSet':    
        decoder = DeepSet(last_hidden_size)#.to(device)
    # RepSet
    elif set_model == 'RepSet':    
        decoder = RepSet(last_hidden_size, 4)#.to(device)
    # SetTransformer
    elif set_model == 'SetTransformer':    
        decoder = nn.Sequential(
            SAB(last_hidden_size, last_hidden_size, num_heads, ln=False),
            PMA1(last_hidden_size, num_heads, ln=False),
            nn.Linear(last_hidden_size, last_hidden_size)
        )#.to(device)
    else:
        raise NotImplementedError        

    return decoder