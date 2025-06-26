import torch as th
from torch import nn
import numpy as np
from dgl import function as fn
from dgl.base import DGLError


class LGConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False,
                 p=2.0,
                 ):  # ✅ NEW ARG, i primi esperimenti erano con True
        super(LGConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.p = p

        self._alpha = nn.ParameterList()
        for i in range(self._k + 1):
            self._alpha.append(nn.Parameter(th.Tensor(1)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.out_feats)
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        stdvk = 1. / np.sqrt(self._k+1)
        for i in range(self._k+1):
            self._alpha[i].data.uniform_(-stdvk, stdvk)



    def forward(self, g, feat):
        with g.local_scope():
            if self._cached and self._cached_h is not None:
                return self._cached_h

            h = feat

            hs = self._propagate(g, h)

            h = 0
            for i, h_i in enumerate(hs):
                h += self.fc(self._alpha[i] * h_i)

            #prima la linearizzazione veniva svolta su tutta la sommatoria 
            # dopo il for, h = self.fc(h)
            # e nel for di sopra h += self._alpha[i] * h_i

            if self._cached:
                self._cached_h = h
            return h

    def _propagate(self, g, feat):
        deg_norm = None
        # degree normalization
        degs = g.in_degrees().float().clamp(min=1)
        deg_norm = th.pow(degs, -0.5).unsqueeze(1).to(feat.device)
        hs = [feat]
        for _ in range(self._k):
            h_current = hs[-1]
            # pre normalization
            h_current = h_current * deg_norm
            
            g.ndata['h'] = h_current  # set node features

            def custom_message(edges):
                p = self.p
                fi = edges.dst['h']  # f(i) - destination node's feature
                fj = edges.src['h']  # f(j) - source node's feature
                diff = (fi - fj)
                # Add a small constant for numerical stability

                norm_diff = th.abs(diff) + 1e-9  # shape [num_edges, out_feats]
                scale = (norm_diff ** (p - 2))
                msg = scale * diff
                msg = fi-msg
                return {'m': msg}

            g.update_all(custom_message, fn.sum('m', 'h'))

            # message passing (sum meighbors)
            h_new = g.ndata.pop('h')  # update new features  prima era h_new = g.ndata['h']
            h_new = h_new * deg_norm  # post norm

            if self.norm == "both":
                degs = g.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -self.p).unsqueeze(1).to(h_new.device)  # D^{-p} normalization added if bot
                h_new = h_new * norm
            hs.append(h_new)
        return hs

