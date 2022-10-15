"""
recbole.data.dataloader.ae_user_dataloader
################################################
"""
import torch

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.interaction import Interaction


class HOUserDataLoader(AbstractDataLoader):
    """:class:`UserDataLoader` will return a batch of data which only contains user-id when it is iterated.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        shuffle (bool): Whether the dataloader will be shuffle after a round.
            However, in :class:`UserDataLoader`, it's guaranteed to be ``True``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        if shuffle is False:
            shuffle = True
            self.logger.warning('UserDataLoader must shuffle the data.')

        self.uid_field = dataset.uid_field
        # self.user_list = Interaction({self.uid_field: torch.arange(dataset.user_num)})
        user_avail_list = [i for i, u2ids in enumerate(sampler.used_ids) if len(u2ids) > 0]
        # self.ru2u = dict(((ru, u) for ru, u in enumerate(self.user_avail_list)))
        self.user_list = Interaction({self.uid_field: torch.tensor(user_avail_list)})

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.user_list)

    def _shuffle(self):
        self.user_list.shuffle()

    def _next_batch_data(self):
        cur_data = self.user_list[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data
