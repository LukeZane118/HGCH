import os
import pickle

from recbole.data.dataloader import *
from recbole.utils import set_color
from recbole.utils.argument_list import dataset_arguments

from extra.data.extra_dataset import ExtraDataset


def create_extra_dataset(config):
    """Create dataset for extra recommendation.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_class = ExtraDataset

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-dataset.pth')
    file = config['dataset_save_path'] or default_file
    logger = getLogger()
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['filter_additional_feat_by_inter', 'seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                logger.info('Current config ' + set_color(f'[{arg}]', 'red') + ' is different from the saved one.')
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            dataset.config = config
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset
    else:
        logger.info('File ' + set_color(f'{file}', 'red') + ' does not exist, so the data file will be built from scratch.')

    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset

def load_split_extra_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """
    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-for-{config["model"]}-dataloader.pth')
    dataloaders_save_path = config['dataloaders_save_path'] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
    for arg in dataset_arguments + ['seed', 'repeatable', 'eval_args']:
        if config[arg] != train_data.config[arg]:
            return None

    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    
    logger = getLogger()
    logger.info(set_color('Load split dataloaders from', 'pink') + f': [{dataloaders_save_path}]')
    return train_data, valid_data, test_data

def save_emb_as_pre_trained(model, dataset, config):
    output_path = os.path.join(config['data_path'])

    uid = dataset.field2id_token[dataset.uid_field][1:]
    iid = dataset.field2id_token[dataset.iid_field][1:]
    u_embs = model.user_embedding.weight[1:].detach().cpu().numpy()
    i_embs = model.item_embedding.weight[1:].detach().cpu().numpy()

    file_u = os.path.join(output_path, config['dataset'] + '.useremb')
    file_i = os.path.join(output_path, config['dataset'] + '.itememb')
    with open(file_u, 'w') as f:
        f.write('uid:token\tuser_emb:float_seq\n')
        for id, emb in zip(uid, u_embs):
            f.write(f'{id}\t{" ".join(map(str, emb))}\n')
    with open(file_i, 'w') as f:
        f.write('iid:token\titem_emb:float_seq\n')
        for id, emb in zip(iid, i_embs):
            f.write(f'{id}\t{" ".join(map(str, emb))}\n')


