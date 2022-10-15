from time import time
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage

from .optim import *

class ExtraTrainer(Trainer):
    r"""This class add extral function for recommender task.
    """
    def __init__(self, config, model):
        self.momentum = config['momentum']
        super().__init__(config, model)
        # Warm up and scheduler
        self._build_scheduler()
                
    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)
        momentum = kwargs.pop('momentum', self.momentum)


        if self.config['reg_weight'] and weight_decay and weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        elif learner.lower() == 'rsgd':
            optimizer = RiemannianSGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        elif learner.lower() == 'radam':
            optimizer = RiemannianAdam(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _build_scheduler(self):
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return
        scheduler_after = None
        if self.config['warm_up'] is not None:
            if self.config['warm_up'].get('after_scheduler') is not None:
                key, value = zip(*self.config['warm_up']['after_scheduler'].items())
                if hasattr(optim.lr_scheduler, key[0]):
                    scheduler_after = getattr(optim.lr_scheduler, key[0])(self.optimizer, **value[0])
                else:
                    raise ValueError('The scheduler set in config[after_scheduler] does not exist in [torch.optim.lr_scheduler].')
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=self.config['warm_up']['multiplier'], \
                total_epoch=self.config['warm_up']['total_epoch'], after_scheduler=scheduler_after)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, list):
                    for sch in self.scheduler:
                        sch.step()
                else:
                    self.scheduler.step()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self._add_tracked_info_to_tensorboard(epoch_idx)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')
            # if hasattr(self.model, 'encoder'):
            #     self.model.encoder.update_decay()
            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step+=1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        if self.config['track_gradient']:
            self.gradient_dict = {}
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            if self.config['track_gradient']:
                for name, para in self.model.named_parameters():
                    if para.grad is not None:
                        if self.gradient_dict.get(name) is None:
                            self.gradient_dict[name] = [para.grad.norm()]
                        else:
                            self.gradient_dict[name].append(para.grad.norm())
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        if self.config['track_gradient']:
            for name, grad_list in self.gradient_dict.items():
                self.gradient_dict[name] = sum(grad_list) / len(grad_list)
        return total_loss

    def _add_tracked_info_to_tensorboard(self, epoch_idx):
        if self.config['track_weight']:
            for name, para in self.model.named_parameters():
                if para.requires_grad:
                    self.tensorboard.add_histogram(f'Param/{name}', para, epoch_idx)
        if self.config['track_gradient']:
            for name, grad in self.gradient_dict.items():
                self.tensorboard.add_scalar(f'Param_grad/{name}', grad, epoch_idx)


class HGCCTrainer(ExtraTrainer):
    r"""This class is designed for HGCC model to train.
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        # rebuild optimizer for gate
        if config['fusion_method'].startswith('gate'):
            params = []
            params_gate = []
            for name, p in model.named_parameters():
                if 'gate' in name:
                    params_gate.append(p)
                else:
                    params.append(p)
            self.optimizer = self._build_optimizer(params=[
                {"params": params},
                {"params": params_gate, "weight_decay": config['weight_decay_gate']},
            ]
            )
            self._build_scheduler()