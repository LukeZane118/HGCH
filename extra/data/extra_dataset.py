import numpy as np
# import pandas as pd
# import torch
# from sklearn.cluster import *
from sklearn.metrics.pairwise import haversine_distances
from scipy.sparse import coo_matrix
from multiprocessing import Pool

from recbole.data.dataset.dataset import Dataset
from recbole.utils import FeatureSource, FeatureType
from recbole.data.interaction import Interaction


class ExtraDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)

    def _data_filtering(self):
        """Data filtering

        - Filter missing user_id or item_id
        - Remove duplicated user-item interaction
        - Value-based data filtering
        - Remove interaction by user or item
        - K-core data filtering

        Note:
            After filtering, feats(``DataFrame``) has non-continuous index,
            thus :meth:`~recbole.data.dataset.dataset.Dataset._reset_index` will reset the index of feats.
        """
        self._filter_nan_user_or_item()
        self._remove_duplication()
        self._filter_by_field_value()
        self._filter_inter_by_user_or_item()
        self._filter_by_inter_num()
        self._filter_additional_feat_by_inter()  # Newly added
        self._reset_index()

    def _filter_additional_feat_by_inter(self):
        """Remove interaction in additional_feat which user_id or item_id is not in inter.
        """
        if self.config['filter_additional_feat_by_inter'] is None or self.config['filter_additional_feat_by_inter'] is not True:
            return
        for alia_name, alias in self.alias.items():
            for alia in alias:
                if alia == self.uid_field or alia == self.iid_field:
                    continue
                source = self.field2source[alia]
                df = getattr(self, source + '_feat')
                key = self.uid_field if alia_name == 'user_id' else self.iid_field

                remained_ids = set(self.inter_feat[key].values)
                remained_row = df[alia].isin(remained_ids)
                df.drop(df.index[~remained_row], inplace=True)
    
    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
            return datasets

        # ordering
        ordering_args = self.config['eval_args']['order']
        if ordering_args == 'RO':
            self.shuffle()
        elif ordering_args == 'TO':
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(f'The ordering_method [{ordering_args}] has not been implemented.')

        # splitting & grouping
        split_args = self.config['eval_args']['split']
        if split_args is None:
            raise ValueError('The split_args in eval_args should not be None.')
        if not isinstance(split_args, dict):
            raise ValueError(f'The split_args [{split_args}] should be a dict.')

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config['eval_args']['group_by']
        if split_mode == 'RS':
            if not isinstance(split_args['RS'], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == 'none':
                datasets = self.split_by_ratio(split_args['RS'], group_by=None)
            elif group_by == 'user':
                datasets = self.split_by_ratio(split_args['RS'], group_by=self.uid_field)
            else:
                raise NotImplementedError(f'The grouping method [{group_by}] has not been implemented.')
        elif split_mode == 'LS':
            datasets = self.leave_one_out(group_by=self.uid_field, leave_one_mode=split_args['LS'])
        elif split_mode == 'HS':
            assert group_by == 'user'
            ratio = split_args['HS']['ratio']
            num = split_args['HS']['num']
            assert self.user_num >= num * 2
            datasets = self.split_by_held_out(ratio, num, group_by=self.uid_field)
        else:
            raise NotImplementedError(f'The splitting_method [{split_mode}] has not been implemented.')

        return datasets
    
    def split_by_held_out(self, ratios, num, group_by=None):
        """Split interaction records by held-out.

        Args:
            ratios (list): List of split ratios. No need to be normalized.
            num (int): Number of held-out user for validation and test.
            group_by (str, optional): Field name that interaction records should grouped by before splitting.
                Defaults to ``None``

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been split.

        Note:
            Other than the first one, each part is rounded down.
        """
        self.logger.debug(f'split by held-out of ratios [{ratios}] and number [{num}], group_by=[{group_by}]')
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        grouped_inter_feat_index = list(self._grouped_index(self.inter_feat[group_by].numpy()))
        
        grouped_train_inter_feat_idx = grouped_inter_feat_index[:-num*2]
        grouped_valid_inter_feat_idx = grouped_inter_feat_index[-num*2:-num]
        grouped_test_inter_feat_idx = grouped_inter_feat_index[-num:]
        
        def split_by_ratio(grouped_inter_feat_idx, rs):
            next_index = [[] for _ in range(len(rs))]
            for grouped_index in grouped_inter_feat_idx:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=rs)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start:end])
            return next_index
        
        train_index = [[]]
        for index in grouped_train_inter_feat_idx:
            train_index[0].extend(index)    # [train]
        valid_index = split_by_ratio(grouped_valid_inter_feat_idx, ratios)  # [valid_tr, valid_te]
        test_index = split_by_ratio(grouped_test_inter_feat_idx, ratios)    # [test_tr, test_te]
        
        next_index = train_index + valid_index + test_index

        self._drop_unused_col()
        next_df = [self.inter_feat[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        next_ds = [next_ds[0], (next_ds[1], next_ds[2]), (next_ds[3], next_ds[4])]
        return next_ds

    def net_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe friend between user_id and friend_id.

        Sparse matrix has shape (user_num, user_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.net_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if hasattr(self, 'net_feat') is not True:
            raise ValueError('dataset does not exist net, thus can not converted to sparse matrix.')
        return self._create_sparse_matrix(self.net_feat, self.net_feat.columns[0], self.net_feat.columns[1], form, value_field)

    def nei_matrix(self, geo_threshold, form='coo'):
        """Get sparse matrix that describe neighbour relationship between location based on ``config['geo_threshold']``.

            Sparse matrix has shape (item_num, item_num).

            For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if metric of ``haversine_distances`` 
            between location src and location tgt less than ``config['geo_threshold']``.

            Args:
                form (str, optional): Sparse matrix format. Defaults to ``coo``.

            Returns:
                scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if hasattr(self, 'geo_feat') is not True:
            raise ValueError('dataset does not exist geo, thus can not build sparse matrix.')
        location_field = self.config['GEO_FIELD']['LOCATION_ID']
        latitude_field = self.config['GEO_FIELD']['LATITUDE_FIELD']
        longitude_field = self.config['GEO_FIELD']['LONGITUDE_FIELD']

        location_id = self.geo_feat[location_field]
        location_feat = np.stack([self.geo_feat[latitude_field], self.geo_feat[longitude_field]]).T
        location_feat_rad = np.radians(location_feat)
        
        nei_batch_size = 512
        src = []
        tgt = []
        data = []

        def handler(res):
            idx, nei_mask = res
            # src = location_id[idx]
            _src, _tgt = np.nonzero(nei_mask)
            _src += idx[0] # offset
            # map back to id
            _src, _tgt = location_id[_src], location_id[_tgt]
            src.extend(_src)
            tgt.extend(_tgt)
            data.extend([1] * len(_src))

        p = Pool()
        for i in range(0, len(location_id), nei_batch_size):
            start, end = i, min(i + nei_batch_size, len(location_id))
            p.apply_async(find_neighbour, (range(start, end), location_feat_rad, geo_threshold,), \
                callback=handler, error_callback=print_error)
        p.close()
        p.join()

        # result = haversine_distances(location_feat_rad)
        # result *= 6371.393  # multiply by Earth radius to get kilometers
        # mask = (result <= geo_threshold)

        # src, tgt = mask.nonzero()
        # src, tgt = location_id[src], location_id[tgt]
        # data = np.ones_like(src)

        mat = coo_matrix((data, (src, tgt)), shape=(self.item_num, self.item_num))
        mat_self = coo_matrix((np.ones(self.item_num - 1), (range(1, self.item_num), range(1, self.item_num))), shape=(self.item_num, self.item_num))
        mat -= mat_self

        if form == 'coo':
            return mat.tocoo()
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    # def geo_matrix(self, n_clusters=128, form='coo', value_field=None):
    #     """Create region data from ``geo_feat`` if ``geo_feat`` exists, 
    #     and get sparse matrix that describe the regional affiliation between subregion_id and region_id.

    #     Args:
    #         n_clusters (int): Number of cluster for K-means.
    #         form (str, optional): Sparse matrix format. Defaults to ``coo``.
    #         value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
    #             Defaults to ``None``.

    #     Returns:
    #         scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
    #     """
    #     self._create_region_aff_feat_from_geo(n_clusters)
        
    #     src = self.region_aff_feat['subregion']
    #     tgt = self.region_aff_feat['region']
    #     if value_field is None:
    #         data = np.ones(len(self.region_aff_feat))
    #     else:
    #         if value_field not in self.region_aff_feat:
    #             raise ValueError(f'Value_field [{value_field}] should be one of `df_feat`\'s features.')
    #         data = self.region_aff_feat[value_field]
    #     n_items_regions = tgt.max() + 1
    #     mat = coo_matrix((data, (src, tgt)), shape=(n_items_regions, n_items_regions))

    #     if form == 'coo':
    #         return mat
    #     elif form == 'csr':
    #         return mat.tocsr()
    #     else:
    #         raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')
        
    # def _create_region_aff_feat_from_geo(self, n_clusters):
    #     """Create ``region_aff_feat`` from ``geo_feat`` if ``geo_feat`` exists.
    #     """
    #     if hasattr(self, 'geo_feat') is not True:
    #         raise ValueError('dataset does not exist geo, thus can not create region from geo.')
    #     # if self.config['geo_cluster_method'] is None:
    #     #     raise ValueError('the geo cluster method is not specified, so the regional affiliation cannot be constructed.')
        
    #     location_field = self.config['GEO_FIELD']['LOCATION_ID']
    #     latitude_field = self.config['GEO_FIELD']['LATITUDE_FIELD']
    #     longitude_field = self.config['GEO_FIELD']['LONGITUDE_FIELD']

    #     subregion_list = []
    #     region_list = []
    #     region_feat = torch.stack([self.geo_feat[latitude_field], self.geo_feat[longitude_field]]).T
    #     region_id = self.geo_feat[location_field]
    #     id_offset = int(region_id.max()) + 1
        
    #     while region_feat.shape[0] > 1:
    #         km = KMeans(n_clusters)
    #         dis = km.fit_transform(region_feat)
    #         center_id = dis.argmin(axis=-1) + id_offset
    #         center_feat = km.cluster_centers_

    #         subregion_list.append(region_id)
    #         region_list.append(center_id)
    #         region_feat = center_feat
    #         region_id = np.arange(n_clusters) + id_offset

    #         id_offset += region_feat.shape[0]
    #         n_clusters >>= 1

    #     subregion = np.concatenate(subregion_list)
    #     region = np.concatenate(region_list)

    #     self.region_aff_feat = Interaction({'subregion': subregion, 'region': region})

    def extral_matrix(self, file_postfix, form='coo', value_field=None):
        """Get sparse matrix that describe link between entity a and entity b.

        Sparse matrix has shape (user_num, user_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.[file_postfix]_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if hasattr(self, f'{file_postfix}_feat') is not True:
            raise ValueError(f'dataset does not exist {file_postfix}, thus can not converted to sparse matrix.')
        feat = getattr(self, f'{file_postfix}_feat')
        return self._create_sparse_matrix(feat, feat.columns[0], feat.columns[1], form, value_field)

# Multiprocessing
def find_neighbour(idx, loc_feat, threshold):
    X = loc_feat[idx]
    result = haversine_distances(X, loc_feat)
    result *= 6371.393  # multiply by Earth radius to get kilometers
    nei_mask = (result <= threshold)
    return idx, nei_mask

def print_error(value):
    print("subprocess error: ", value)
        

        