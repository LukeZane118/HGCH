from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
# from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_uniform_initialization


import extra.model.hgcn.layers.hyp_layers as hyp_layers
import extra.model.hgcn.manifolds as manifolds
from extra.model.hgcn.utils.model_utils import *

class HGCCEncoder(nn.Module):
    r"""HGCCEncoder is a module for MCHG to encode embeddings.
    """
    def __init__(self, config, manifold, c_list, entity2space_weights, space2entities):
        super().__init__()
        self.latent_dim = config['embedding_size']
        # gate prior is for training a better fusion weight
        self.entity2space_weights_prior = entity2space_weights
        self.fusion_method = config['fusion_method']
        self.layers = hyp_layers.MultiHyperbolicGraphConvolution(manifold, c_list, None, config, space2entities)
        self.count = 1
        if self.fusion_method.startswith('gate'):
            # every entity in every space has a unique gate to compute its embedding in correspond space weight
            # if an entity only exist in one space, it won't have gate
            self.entity2gate_kernels = nn.ModuleDict()
            for entity, space_weights in entity2space_weights.items():
                if len(space_weights) == 1:
                    continue
                self.entity2gate_kernels[entity] = nn.ModuleDict()
                for space in space_weights.keys():
                    self.entity2gate_kernels[entity][space] = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
                    self.entity2gate_kernels[entity][space].apply(xavier_uniform_initialization)
        if self.fusion_method == 'gate':
            self.encode_method = self.encode_by_gate
        elif self.fusion_method == 'prior':
            self.encode_method = self.encode_by_prior
        elif self.fusion_method == 'gate&prior':
            self.encode_method = self.encode_by_gate_with_prior
        elif self.fusion_method == 'gate&decay_prior':
            self.encode_method = self.encode_by_gate_with_decay_prior
        else:
            raise ValueError('config[fusion_method] should be choose in {prior, gate, gate&prior}.')
    
    def change_to_hyper_agg(self):
        r"""Change all aggregation function to hyperbolic aggregation.
        """
        for l in self.layers.aggs:
            l.agg_fuc = l.hyper_agg

    def change_to_eucli_agg(self):
        r"""Change all aggregation function to euclidian aggregation.
        """
        for l in self.layers.aggs:
            l.agg_fuc = l.eucli_agg

    def encode_by_gate(self, x: dict, adj: dict):
        # use prior weight and trainable gate to fuse space output embeddings
        entity2space_weights = defaultdict(dict)
        entity2denom = defaultdict(int)
        # get gate output
        for entity, space_weights_prior in self.entity2space_weights_prior.items():
            if len(space_weights_prior) == 1:
                entity2space_weights[entity][list(space_weights_prior.keys())[0]] = 1
                continue
            for space, weights_prior in space_weights_prior.items():
                gate = torch.sigmoid(self.entity2gate_kernels[entity][space](x[entity]))    # N x D
                entity2space_weights[entity][space] = gate
                entity2denom[entity] += entity2space_weights[entity][space]     # N x D
        # normalization
        for entity, space_weights in entity2space_weights.items():
            if len(space_weights) == 1:
                continue
            for space, weight in space_weights.items():
                space_weights[space] = weight / entity2denom[entity]
        output = self.layers(x, adj, entity2space_weights)
        return output
    
    def encode_by_prior(self, x: dict, adj: dict):
        # only use prior weight to fuse space output embeddings
        output = self.layers(x, adj, self.entity2space_weights_prior)
        return output

    def encode_by_gate_with_prior(self, x: dict, adj: dict):
        # use prior weight and trainable gate to fuse space output embeddings
        entity2space_weights = defaultdict(dict)
        entity2denom = defaultdict(int)
        # combine prior and gate output
        for entity, space_weights_prior in self.entity2space_weights_prior.items():
            if len(space_weights_prior) == 1:
                entity2space_weights[entity][list(space_weights_prior.keys())[0]] = 1
                continue
            for space, weights_prior in space_weights_prior.items():
                gate = torch.sigmoid(self.entity2gate_kernels[entity][space](x[entity]))    # N x D
                # logit = self.entity2gate_kernels[entity][space](x[entity])
                # logit = torch.squeeze(F.adaptive_avg_pool1d(logit.unsqueeze(0), 1), 0)
                # gate = torch.sigmoid(logit)
                entity2space_weights[entity][space] = gate * weights_prior
                entity2denom[entity] += entity2space_weights[entity][space]     # N x D
        # normalization
        for entity, space_weights in entity2space_weights.items():
            if len(space_weights) == 1:
                continue
            for space, weight in space_weights.items():
                space_weights[space] = weight / entity2denom[entity]
        output = self.layers(x, adj, entity2space_weights)
        return output

    def encode_by_gate_with_decay_prior(self, x: dict, adj: dict):
        # use prior weight and trainable gate to fuse space output embeddings
        entity2space_weights = defaultdict(dict)
        entity2denom = defaultdict(int)
        # combine prior and gate output
        for entity, space_weights_prior in self.entity2space_weights_prior.items():
            if len(space_weights_prior) == 1:
                entity2space_weights[entity][list(space_weights_prior.keys())[0]] = 1
                continue
            for space, weights_prior in space_weights_prior.items():
                gate = torch.sigmoid(self.entity2gate_kernels[entity][space](x[entity]))    # N x D
                entity2space_weights[entity][space] = gate * weights_prior ** (1. / (np.log10(self.count) + 1))
                entity2denom[entity] += entity2space_weights[entity][space]     # N x D
        # normalization
        for entity, space_weights in entity2space_weights.items():
            if len(space_weights) == 1:
                continue
            for space, weight in space_weights.items():
                space_weights[space] = weight / entity2denom[entity]
        output = self.layers(x, adj, entity2space_weights)
        return output

    def update_decay(self):
        self.count += 1

    def encode(self, x: dict, adj: dict):
        return self.encode_method(x, adj)


class Curvature:
    r"""Curvature is a class for multiple curvature gradient backpropagation.
    """
    def __init__(self, c_dict: torch.nn.ParameterDict):
        self._c_dict = c_dict
        self._cf_dict = {}
        for k in c_dict.keys():
            if self._c_dict[k].requires_grad:
                self._cf_dict[k] = lambda x=k: self._c_dict[x]

    def __getattr__(self, item):
        if item in self._cf_dict:
            # return torch.clamp(torch.relu(self._cf_dict[item]()), min=1e-8, max=1e8)
            return torch.clamp(F.softplus(self._cf_dict[item]()), min=1e-8, max=1e8)
        if item in self._c_dict:
            return self._c_dict[item]
        raise AttributeError(f"'Curvature' object has no attribute '{item}'")

    def __getitem__(self, index):
        # return torch.clamp(torch.relu(self._cf_dict[index]()), min=1e-8, max=1e8)
        if index in self._cf_dict:
            return torch.clamp(F.softplus(self._cf_dict[index]()), min=1e-8, max=1e8)
        else:
            return self._c_dict[index]

    def keys(self):
        return self._c_dict.keys()


class HGCC(GeneralRecommender):
    r"""HGCC is a Hyperbolic GCN-based recommender model.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.space2entities = config['spaces']  # e.g.: {inter: [user, item], ...}
        self.latent_dim = config['embedding_size'] # int type: the embedding size of HGCF
        self.n_layers = config['n_layers'] # int type: the layer num of HGCF
        self.reg_weight = config['reg_weight'] # float32 type: the weight decay for l2 normalization
        # self.reg_c_weight = config['reg_c_weight']
        self.margin = config['margin']
        # self.require_pow = config['require_pow']

        # load dataset info and build extral info
        self.adj_matrix_dict = {}
        self.entity2num = {}
        self.entity2num['user'] = self.n_users
        self.entity2num['item'] = self.n_items
        for space, (e1, e2) in self.space2entities.items():    # space name is the same as postfix of file name
            if space == 'inter':
                self.adj_matrix_dict[space] = dataset.inter_matrix(form='coo').astype(np.float32)
            elif space == 'geon':   # building geon data from scratch and saving as file is too expensive, so we build it here since filtered data is much smaller
                self.adj_matrix_dict[space] = dataset.nei_matrix(config['geo_threshold'], form='coo').astype(np.float32)
            else:
                self.adj_matrix_dict[space] = dataset.extral_matrix(space, form='coo').astype(np.float32)
            if self.entity2num.get(e1, None) is None:
                self.entity2num[e1] = self.adj_matrix_dict[space].shape[0]
            if self.entity2num.get(e2, None) is None:
                self.entity2num[e2] = self.adj_matrix_dict[space].shape[1]
        self.entity2spaces = defaultdict(list)
        for space, (e1, e2) in self.space2entities.items():
            self.entity2spaces[e1].append(space)
            if e2 != e1:
                self.entity2spaces[e2].append(space)

        # define manifold
        self.manifold = manifolds.PoincareBall()

        # define embeddings, curvature
        self.embedding_dict = torch.nn.ModuleDict()
        for entity, num in self.entity2num.items():
            self.embedding_dict[entity] = torch.nn.Embedding(num_embeddings=num, embedding_dim=self.latent_dim, padding_idx=0)
        if config['curvature'] is not None:
            self._c_list = torch.nn.ModuleList()    # _c_list is original tensor data of curvature
            self.c_list = []    # c_list is for calling, which process _c_list then output
            for i in range(self.n_layers + 1): # last layer is for computing score
                _c = torch.nn.ParameterDict()
                for key, value in config['curvature'].items():
                    if isinstance(value, list):
                        assert len(value) == self.n_layers + 1
                        c_v = value[i]
                    else:
                        c_v = value
                    if c_v is None:
                        _c[key] = torch.nn.Parameter(torch.tensor([1.]), requires_grad=True)
                    else:
                        _c[key] = torch.nn.Parameter(torch.tensor([c_v]), requires_grad=False)
                self._c_list.append(_c)
                self.c_list.append(Curvature(_c))
        else:
            raise ValueError('config[curvature] must be set.')

        # init embeddings 
        # init user and item embeddings
        if config['init_mode'] == 'uni':
            self.embedding_dict['user'].weight = get_uni_init_euc_weights(self.embedding_dict['user'].weight, True, config['scale'], padding_idx=0)
            self.embedding_dict['item'].weight = get_uni_init_euc_weights(self.embedding_dict['item'].weight, True, config['scale'], padding_idx=0)
        elif config['init_mode'] == 'pop' or config['init_mode'] == 'pop_xavier':
            # get popularity
            act_count = torch.from_numpy(self.adj_matrix_dict['inter'].sum(1).A)
            pop_count = torch.from_numpy(self.adj_matrix_dict['inter'].sum(0).A.T)
            min_count = min(act_count[1:].min(), pop_count[1:].min())
            act_count[0] = torch.tensor([min_count])
            pop_count[0] = torch.tensor([min_count])
            if config['init_mode'] == 'pop':
                self.embedding_dict['user'].weight = get_pop_init_euc_weights(self.embedding_dict['user'].weight, act_count, True, config['scale'], padding_idx=0)
                self.embedding_dict['item'].weight = get_pop_init_euc_weights(self.embedding_dict['item'].weight, pop_count, True, config['scale'], padding_idx=0)
            else:
                self.embedding_dict['user'].weight = get_pop_xavier_init_euc_weights(self.embedding_dict['user'].weight, act_count, True, padding_idx=0)
                self.embedding_dict['item'].weight = get_pop_xavier_init_euc_weights(self.embedding_dict['item'].weight, pop_count, True, padding_idx=0)
        else:
            raise ValueError('config[init_mode] should be choose in {uni, pop, pop_xavier}.')
        # init extral embeddings
        for entity, embeddings in self.embedding_dict.items():
            if entity == 'user' or entity == 'item':
                continue
            # embeddings.weight = get_uni_init_euc_weights(embeddings.weight, True, config['scale'], padding_idx=0)
            embeddings.weight = get_uni_init_euc_weights(embeddings.weight, True, 0.01, padding_idx=0)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix_dict = {}
        for space in self.space2entities.keys():
            self.norm_adj_matrix_dict[space] = self.get_norm_adj_mat(space).to(self.device)
        # self.norm_adj_matrix['inter'] = self.get_norm_adj_mat('all').to(self.device)

        # get prior space embedding fusion weight
        self.entity2space_weights = self.get_space_emb_weight()
        # self.space_emb_weight = None
        
        # define encoder
        self.encoder = HGCCEncoder(config, self.manifold, self.c_list[:-1], self.entity2space_weights, self.space2entities)

        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.use_fermi_dirac = False

        # define auxiliary data
        self.aux_tasks = config['aux_tasks']
        self.edges = {}
        for task in self.aux_tasks:
            if task == 'inter':
                continue
            self.edges[task] = np.stack(self.adj_matrix_dict[task].nonzero()).T

    def get_norm_adj_mat(self, space='inter'):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
        A_{hat} = D^{-1} \times A

        Returns:
        Sparse tensor of the normalized interaction matrix.
        """
        e1, e2 = self.space2entities[space]
        offset = 0
        n_entities = self.entity2num[e1]
        if e1 != e2:
            offset = n_entities
            n_entities += self.entity2num[e2]
        A = sp.dok_matrix((n_entities, n_entities), dtype=np.float32)
        # build adj matrix
        mat_M = self.adj_matrix_dict[space]
        mat_M_t = mat_M.transpose()
        data_dict = dict(zip(zip(mat_M.row, mat_M.col + offset), [1] * mat_M.nnz))
        data_dict.update(dict(zip(zip(mat_M_t.row + offset, mat_M_t.col), [1] * mat_M_t.nnz)))
        A._update(data_dict)
        # self loop
        if space == 'inter':
            A += sp.eye(A.shape[0])
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        D = sp.diags(diag)
        L = D * A
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_space_emb_weight(self):
        assert hasattr(self, 'norm_adj_matrix_dict')

        entity2space_weights = defaultdict(dict)
        entity2n_neibours = defaultdict(int)
 
        for space, adj in self.norm_adj_matrix_dict.items():
            index = adj.coalesce().indices()
            data = (adj.coalesce().values() > 0).long()
            A = torch.sparse.FloatTensor(index, data, torch.Size(adj.shape))
            n_neibours = torch.sparse.sum(A, dim=1).to_dense()
            e1, e2 = self.space2entities[space]
            
            n_neibours_e1 = n_neibours[:self.entity2num[e1], None]
            entity2space_weights[e1][space] = n_neibours_e1
            entity2n_neibours[e1] += n_neibours_e1
            if e1 != e2:
                n_neibours_e2 = n_neibours[self.entity2num[e1]:, None]
                entity2space_weights[e2][space] = n_neibours_e2
                entity2n_neibours[e2] += n_neibours_e2

        for entity, space_weights in entity2space_weights.items():
            for space, weight in space_weights.items():
                space_weights[space] = weight / entity2n_neibours[entity]
        return entity2space_weights
    
    def hyp_dist(self, u_embeddings, i_embeddings, c, square=True):
        sqdist = self.manifold.sqdist(u_embeddings, i_embeddings, c, square)
        return sqdist

    def hyp_dist_for_mat(self, u_embeddings, i_embeddings, c, square=True):
        sqdist = self.manifold.sqdist_for_mat(u_embeddings, i_embeddings, c, square)
        return sqdist

    def fermi_dirac(self, dist):
        # equation: sim (x, y) = σ(t(r − dist(x, y)))
        r = 0.5
        t = 10
        return torch.sigmoid(t * (r - dist))

    def get_batch_auxiliary_edges(self, task, batch_size, head_all_embeddings, tail_all_embeddings, negative_method='uni', reverse=False):
        r"""Sample training samples for auxiliary task.

        Args:
            task (str): training task.
            batch_size (int): batch size.
            head_all_embeddings (torch.tensor): all head node embeddings in euclidean space.
            tail_all_embeddings (torch.tensor): all tail node embeddings in euclidean space.
            negative_method (torch.tensor): method of sampling negative samples.

        Returns:
            (torch.tensor): head node indices.
            (torch.tensor): tail node indices.
            (torch.tensor): negative tail indices.
        """
        all_edges = self.edges[task]
        # n_heads, n_tails = self.adj_matrix_dict[task].shape
        n_heads, n_tails = len(head_all_embeddings), len(tail_all_embeddings)
        # positive
        idxs_pos = np.random.randint(0, len(all_edges), batch_size)
        edges = all_edges[idxs_pos]
        if reverse:
            head = torch.tensor(edges[:, 1], device=self.device, dtype=torch.long)
            tail = torch.tensor(edges[:, 0], device=self.device, dtype=torch.long)
        else:
            head = torch.tensor(edges[:, 0], device=self.device, dtype=torch.long)
            tail = torch.tensor(edges[:, 1], device=self.device, dtype=torch.long)
        # negative
        if negative_method == 'uni':
            idxs_neg = np.random.randint(1, n_tails, batch_size) # index 0 is padding
        elif negative_method == 'dynamic':
            n_candidates = 10
            self.eval()
            with torch.no_grad():
                idxs_can = np.random.randint(1, n_tails, batch_size * n_candidates) # index 0 is padding
                h_embeddings = head_all_embeddings[head.repeat(n_candidates)]
                t_embeddings = tail_all_embeddings[idxs_can]
                scores = -self.hyp_dist(h_embeddings, t_embeddings, self.c_list[-1][task], square=False).reshape(n_candidates, -1)
                indices = torch.max(scores, dim=0)[1].detach().cpu()
                idxs_can = idxs_can.reshape(n_candidates, -1)
                idxs_neg = idxs_can[indices, [i for i in range(idxs_can.shape[1])]]
            self.train()
        neg = torch.tensor(idxs_neg, device=self.device, dtype=torch.long)
        return head, tail, neg

    def get_loss(self, h_embeddings, t_embeddings, n_embeddings, c):
        r"""Compute margin loss and regularion loss by given euclidean embeddings and curvature of hyperbolic space.

        Args:
            h_embeddings (torch.tensor): head node embeddings in euclidean space.
            t_embeddings (torch.tensor): tail node embeddings in euclidean space.
            n_embeddings (torch.tensor): negative tail node embeddings in euclidean space.
            c (torch.tensor): curvature of hyperbolic space to predict.

        Returns:
            (torch.tensor): total loss.
        """
        h_embeddings_hyp = self.manifold.expmap0(h_embeddings, c)
        t_embeddings_hyp = self.manifold.expmap0(t_embeddings, c)
        n_embeddings_hyp = self.manifold.expmap0(n_embeddings, c)
        if self.use_fermi_dirac:
            pos_scores = self.fermi_dirac(self.hyp_dist(h_embeddings_hyp, t_embeddings_hyp, c, square=False))
            neg_scores = self.fermi_dirac(self.hyp_dist(h_embeddings_hyp, n_embeddings_hyp, c, square=False))
        else:
            pos_scores = -self.hyp_dist(h_embeddings_hyp, t_embeddings_hyp, c, square=True)
            neg_scores = -self.hyp_dist(h_embeddings_hyp, n_embeddings_hyp, c, square=True)
        
        # mf loss
        mf_loss = -(pos_scores - neg_scores) + self.margin
        mf_loss[mf_loss < 0] = 0
        non_zero = torch.count_nonzero(mf_loss).item()
        mf_loss = torch.sum(mf_loss)
        if non_zero > 0:
            mf_loss = mf_loss / non_zero

        # embedding reg_loss
        if self.reg_weight == 0:
            reg_loss = torch.tensor([0.], device=self.device)
        else:
            reg_loss = self.manifold.dist_from_ori(h_embeddings_hyp, c, True).sum() + self.manifold.dist_from_ori(t_embeddings_hyp, c, True).sum() + \
                    self.manifold.dist_from_ori(n_embeddings_hyp, c, True).sum()
            reg_loss /= h_embeddings_hyp.shape[0]
        reg_loss *= self.reg_weight
        return mf_loss, reg_loss

    def forward(self):

        embedding_weights_dict = dict((entity, embedding.weight) for entity, embedding in self.embedding_dict.items())

        mchg_embedding_dict = self.encoder.encode(embedding_weights_dict, self.norm_adj_matrix_dict)

        return mchg_embedding_dict

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        embedding_dict = self.forward()
        # user_all_embeddings, item_all_embeddings = self.forward(self.task)

        # target task embeddings
        u_embeddings = embedding_dict['user'][user]
        pos_embeddings = embedding_dict['item'][pos_item]
        neg_embeddings = embedding_dict['item'][neg_item]
        
        # calculate loss
        loss = []
        # inter
        mf_loss, reg_loss = self.get_loss(u_embeddings, pos_embeddings, neg_embeddings, self.c_list[-1]['inter'])
        loss.append(mf_loss + reg_loss)
        # auxiliary
        for task in self.aux_tasks:
            e1, e2 = self.space2entities[task]
            emb1, emb2 = embedding_dict[e1], embedding_dict[e2]
            # h2t
            h, t, n = self.get_batch_auxiliary_edges(task, len(user), emb1, emb2, negative_method='dynamic')
            h_embeddings = emb1[h]
            t_embeddings = emb2[t]
            n_embeddings = emb2[n]
            loss.append(0.01 * sum(self.get_loss(h_embeddings, t_embeddings, n_embeddings, self.c_list[-1][task])))
        return tuple(loss)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        embedding_dict = self.forward()

        u_embeddings = embedding_dict['user'][user]
        i_embeddings = embedding_dict['item'][item]
        c = self.c_list[-1]['inter']

        u_embeddings_hyp = self.manifold.expmap0(u_embeddings, c)
        i_embeddings_hyp = self.manifold.expmap0(i_embeddings, c)

        scores = -self.hyp_dist(u_embeddings_hyp, i_embeddings_hyp, c, square=False)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        c = self.c_list[-1]['inter']
        if self.restore_user_e is None or self.restore_item_e is None:
            embedding_dict = self.forward()
            restore_user_e, restore_item_e = embedding_dict['user'], embedding_dict['item']
            self.restore_user_e = self.manifold.expmap0(restore_user_e, c)
            self.restore_item_e = self.manifold.expmap0(restore_item_e, c)
        # get user embedding from storage variable
        u_embeddings_hyp = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = -self.hyp_dist_for_mat(u_embeddings_hyp, self.restore_item_e.transpose(0, 1), c, square=False)

        return scores.view(-1)
