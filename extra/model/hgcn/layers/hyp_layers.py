from collections import defaultdict
import torch
import torch.nn as nn
# from torch.nn.modules.module import Module

from recbole.model.init import xavier_normal_initialization

from ..manifolds import PoincareBall

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, c_in, out_features, network, num_layers)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        # output = h, adj
        return h


class HRCFHyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer for HRCF.
    """

    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers, entity2num):
        super(HRCFHyperbolicGraphConvolution, self).__init__()
        self.agg = HRCFHypAgg(manifold, c_in, out_features, network, num_layers, entity2num)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        # output = h, adj
        return h


class StackGCNs(nn.Module):

    def __init__(self, num_layers):
        super(StackGCNs, self).__init__()

        self.num_gcn_layers = num_layers - 1

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        if self.num_gcn_layers == 0:
            return x_tangent
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

    def resAddGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        return output[-1]

    def denseGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]


class HypAgg(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, network, num_layers):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        output = self.stackGCNs((x_tangent, adj))
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HRCFHypAgg(nn.Module):
    """
    Hyperbolic aggregation layer for HRCF.
    """

    def __init__(self, manifold, c, in_features, network, num_layers, entity2num):
        super(HRCFHypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers), network)
        self.mask = []
        padding_idx = 0
        for num in entity2num.values():
            self.mask.extend(list(range(padding_idx + 1, padding_idx + num)))
            padding_idx += num

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        output = self.stackGCNs((x_tangent, adj))
        output_mean = output[self.mask].mean(dim=0)
        output = output - output_mean
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
    

class MultiHyperbolicGraphConvolution(nn.Module):
    """
    Multi Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, c_list, pro, config, space2entities):
        super().__init__()
        self.aggs = nn.ModuleList([MultiHypFusionAgg(manifold, c, pro, config['agg_mode'], space2entities) for c in c_list])

    def forward(self, x: dict, adj: dict, entity2space_weights: dict):
        if len(self.aggs) == 0:
            return x
        h_list = [x]
        for i, agg in enumerate(self.aggs):
           h_list.append(agg(h_list[i], adj, entity2space_weights))
        # Add the outputs of each gcn layer of the entity
        # e.g.: [layer1: [A1, B1], layer2: [A2, B2], ...] -> {A: A1 + A2 + ..., B: B1 + B2 + ...}
        output = dict((entity, sum([h[entity] for h in h_list[1:]])) for entity in x.keys())
        return output
        # return self.agg(x, adj, space_weight)

    # def change_to_hyper_agg(self):
    #     self.agg.agg_fuc = self.agg.hyper_agg

    # def change_to_eucli_agg(self):
    #     self.agg.agg_fuc = self.agg.eucli_agg


class MultiHypFusionAgg(nn.Module):
    """
    Multiple Hyperbolic aggregation layer using space fusion.
    """
    def __init__(self, manifold: PoincareBall, c, pro: torch.nn.ModuleDict, agg_mode, space2entities):
        super().__init__()
        self.manifold = manifold
        # self.emb_size = config['embedding_size']
        # self.fusion_type = config['fusion_type']
        # if self.fusion_type is not None:
        #     if config['fusion_type'] == 'gcn':
        #         self.W = nn.Linear(self.emb_size, self.emb_size)
        #     elif config['fusion_type'] == 'graphsage':
        #         self.W = nn.Linear(self.emb_size * 2, self.emb_size)
        #     elif config['fusion_type'] == 'bi':
        #         self.W1 = nn.Linear(self.emb_size, self.emb_size)
        #         self.W2 = nn.Linear(self.emb_size, self.emb_size)
        #     else:
        #         raise ValueError('config[space_fusion] should be chose in {gcn, graphsage, bi}')
        #     self.activation = nn.LeakyReLU()
        #     self.apply(xavier_normal_initialization)

        self.c = c
        self.pro = pro
        # self.num_layers = config['n_layers']
        # self.agg_fuc = getattr(self, agg_mode)
        # self.agg_fuc = {'inter': self.hyper_agg, 'net': self.hyper_agg, 'nei': self.eucli_agg}
        # self.agg_fuc = dict((space, getattr(self, fuc_str)) for space, fuc_str in agg_mode.items())
        self.agg_fuc = dict((space, getattr(self, agg_mode.get(space, None)) if agg_mode.get(space, None) is not None else self.hyper_agg) \
                            for space in c.keys())
        # self.space_emb_weight = space_emb_weight
        self.space2entities = space2entities

    def forward(self, x: dict, adj: dict, entity2space_weights: dict):
        # if self.num_layers == 0:
        #     return x
        # output = [x]
        # for i in range(self.num_layers):
            # # trainable space weight 
            # main_embeddings = self.hyper_agg(output[i], self.c['inter'], adj['inter'])
            # if self.fusion_type is not None:
            #     side_embeddings = 0
            #     for key in self.c.keys():
            #         if key == 'inter':
            #             continue
            #         side_embeddings += self.hyper_agg(output[i], self.c[key], adj[key])
            #     # mean
            #     side_embeddings /= (len(self.c) - 1)
            #     # spase fusion
            #     if self.fusion_type == 'gcn':
            #         agg_embeddings = self.activation(self.W(main_embeddings + side_embeddings))
            #     elif self.fusion_type == 'graphsage':
            #         agg_embeddings = self.activation(self.W(torch.cat([main_embeddings, side_embeddings], dim=1)))
            #     elif self.fusion_type == 'bi':
            #         add_embeddings = main_embeddings + side_embeddings
            #         sum_embeddings = self.activation(self.W1(add_embeddings))
            #         bi_embeddings = torch.mul(main_embeddings, side_embeddings)
            #         bi_embeddings = self.activation(self.W2(bi_embeddings))
            #         agg_embeddings = bi_embeddings + sum_embeddings
            # else:
            #     agg_embeddings = main_embeddings
            # output.append(agg_embeddings)

        agg_embeddings_dict = defaultdict(int)
        for space in self.c.keys():
            # pro = None
            # if self.pro is not None:
            #     pro = self.pro[space]
            e1, e2 = self.space2entities[space]
            embeddings_e1 = x[e1]
            x_input = embeddings_e1
            if e1 != e2:
                x_input = torch.cat([x_input, x[e2]], dim=0)
            x_agg = self.agg_fuc[space](x_input, self.c[space], adj[space])
            # every aggregation result will mutiply a space weight
            agg_embeddings_dict[e1] += entity2space_weights[e1][space] * x_agg[:len(embeddings_e1), :]
            if e1 != e2:
                agg_embeddings_dict[e2] += entity2space_weights[e2][space] * x_agg[len(embeddings_e1):, :]

        # debug code
        # if torch.any(torch.isnan(agg_embeddings_dict)) or torch.any(torch.isinf(agg_embeddings_dict)):
        #     raise ValueError('The result of the calculation is nan / inf.')
        return agg_embeddings_dict

    def hyper_agg(self, x, c, adj, pro=None):
        """
        Hyperbolic aggregation.
        First, map ``x`` to hyperbolic space by ``c``. Then aggregate ``x`` depending on ``adj``. 
        Lastly, map aggregation result back to euclidian space.
        """
        if pro is not None:
            x = torch.tanh(pro(x)) + x
        x_hyp = self.manifold.expmap0(x, c)
        x_hyp_agg = self.manifold.weighted_midpoint_spmm(x_hyp, c, adj)
        x_euc_agg = self.manifold.logmap0(x_hyp_agg, c)
        return x_euc_agg

    def eucli_agg(self, x, c, adj, pro=None):
        """
        Euclidian aggregation.
        Aggregate ``x`` depending on ``adj`` in euclidian space directly.
        """
        if pro is not None:
            x = torch.tanh(pro(x)) + x
        x_euc_agg = torch.sparse.mm(adj, x)
        return x_euc_agg

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypWeightAgg(nn.Module):
    """
    Hyperbolic aggregation layer using space fusion.
    """
    def __init__(self, manifold: PoincareBall, c, agg_mode):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.agg_fuc = getattr(self, agg_mode)

    def forward(self, x, adj):
        output = {}
        for key in self.c.keys():
            agg_embeddings = self.agg_fuc(x, self.c[key], adj[key])
            output[key] = agg_embeddings
        # agg_embeddings = self.agg_fuc(x, self.c, adj)

        # debug code
        # if torch.any(torch.isnan(agg_embeddings)) or torch.any(torch.isinf(agg_embeddings)):
        #     raise ValueError('The result of the calculation is nan / inf.')
        return output

    def hyper_agg(self, x, c, adj):
        x_hyp = self.manifold.expmap0(x, c)
        x_hyp_agg = self.manifold.weighted_midpoint_spmm(x_hyp, c, adj)
        x_euc_agg = self.manifold.logmap0(x_hyp_agg, c)
        return x_euc_agg

    def eucli_agg(self, x, c, adj, pro=None):
        if pro is not None:
            x = torch.tanh(pro(x)) + x
        x_euc_agg = torch.sparse.mm(adj, x)
        return x_euc_agg

    def extra_repr(self):
        return 'c={}'.format(self.c)
