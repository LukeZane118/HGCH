import os
import argparse

from logging import getLogger
from recbole.utils import init_seed
from recbole.config import Config
from recbole.data import data_preparation
from recbole.model.general_recommender import *

from extra.model import *
from extra.utils.data_utils import *
from extra.utils.environment_utils import *
from extra.trainer.extra_trainer import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, default='hgcc', help='model name')
    parser.add_argument('-dn', '--dataset_name', type=str, default='gowalla', help='dataset')
    parser.add_argument('--config_path', type=str, default='./extra/configs', help='config file path')

    args, _ = parser.parse_known_args()
    model_name = args.model_name.lower()
    model_class = name2class[model_name]
    dataset_name = args.dataset_name.lower()

    config_file_list = [
        os.path.join(args.config_path, 'env.yaml'),
        os.path.join(args.config_path, 'dataset_config', dataset_name + '.yaml'),
        os.path.join(args.config_path, model_name + '.yaml'),
        os.path.join(args.config_path, 'eval.yaml'),
        ]

    # Build config
    config = Config(model=model_class, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # Logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    try:
        # Load dataloader
        train_data, valid_data, test_data = load_split_extra_dataloaders(config)
        logger.info("Training data info:")
        logger.info(train_data.dataset)

    except Exception:
        # Dataset creating
        dataset = create_extra_dataset(config)
        logger.info(dataset)

        # Dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    if model_name == 'hgcc':
        trainer = HGCCTrainer(config, model)
    else:
        trainer = ExtraTrainer(config, model)

    # Model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # Model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))