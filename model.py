# DGL's module --- HeteroGraphConv
from dgl.nn.pytorch import HeteroGraphConv

import math
import dgl.nn.pytorch as dglnn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import init


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        edge_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        linear=True,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        #self.fc_edge=nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)


        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_edge=nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))


        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if linear:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)  # resnet
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")            # match the var of relu activation function
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            # feat[0]   embedding of source node           embedding  --dropout-->  hidden state  --fully connnected--> embedding
            # feat[1]   embedding of destination node
            # feat[2]   embedding of edge
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            h_edge = self.feat_drop(graph.edata['feature'])


            if not hasattr(self, "fc_src"):
                self.fc_src, self.fc_dst = self.fc, self.fc
            feat_src, feat_dst,feat_edge= h_src, h_dst,h_edge
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            feat_edge = self.fc_edge(h_edge).view(-1, self._num_heads, self._out_feats)


            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))


            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            graph.edata.update({"e": graph.edata["e"]+ee})

            #e = self.leaky_relu(graph.edata.pop("e"))
            e = self.leaky_relu(graph.edata["e"])





            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                #print(graph.number_of_edges(), perm)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                # print(e.shape, eids.shape)
                a = torch.zeros_like(e)
                # print(e[eids])
                # print("============", a)
                a[eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
                # print(a)
                graph.edata.update({"a": a})
                # graph.edata["a"] = torch.zeros_like(e)
                # graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
                # print(graph.edata["a"])
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]





            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -1)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm


            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval


            #rst = self._activation(rst)

            return rst


class RGAT(nn.Module):
   def __init__(
       self,
       in_feats,
       hid_feats,
       out_feats,
       num_heads,
       rel_names,
   ):
       super().__init__()

       self.conv1 = HeteroGraphConv({
           rel: GATConv(in_feats, hid_feats, in_feats, num_heads=num_heads) for rel in rel_names},
           aggregate='sum')
       self.conv2 = HeteroGraphConv({
           rel: GATConv(hid_feats, out_feats, in_feats, num_heads=num_heads) for rel in rel_names},
           aggregate='sum')

   def forward(self,graph,inputs):
       # inputs are features of nodes
       h = self.conv1(graph, inputs)
       h = {k: F.relu(v) for k, v in h.items()}
       h = self.conv2(graph, h)
       return h



class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']



class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, rel_names):
        super().__init__()
        self.rgat = RGAT(in_features, hidden_features, out_features, num_heads, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        h = self.rgat(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


















