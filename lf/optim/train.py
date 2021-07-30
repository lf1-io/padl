"""Training functionality for optim"""
from collections import defaultdict
import os
import sys
import torch.utils.data
import torch


class TrainingFinished(Exception):
    ...


class TimeToSave(Exception):
    ...


class TimeToSaveAndStop(TrainingFinished, TimeToSave):
    ...


class SimpleTrainer:
    """
    SimpleTrainer is a module to train a model using the giving optimizer

    :param model: Model to train
    :param optimizer: Optimizer to use
    :param loss: Loss function for the training
    :param train_samples: Training data
    :param valid_samples: Validation data
    :param valid_loss: Loss function for validation
    :param save: Function to save checkpoints of model
    :param horovod: use multi machines
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss=None,
                 train_samples=None,
                 valid_samples=None,
                 valid_loss=None,
                 save=lambda: None,
                 horovod=False):

        self.m = model
        self.o = optimizer
        self.l = loss
        if train_samples is None:
            with self.m.set_stage('train'):
                train_samples = range(len(self.m))
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.save = save
        self.it = 0
        self.vl = valid_loss if valid_loss is not None else loss
        self.train_loss = None

        self.train_load = {
            'batch_size': 100,
            'verbose': False,
            'flatten': False,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': True,
        }

        self.valid_load = {
            'batch_size': 100,
            'verbose': False,
            'flatten': False,
            'shuffle': False,
            'num_workers': 0,
            'drop_last': True,
        }

        self._callbacks = defaultdict(list)

        self.horovod = (
                horovod
                or (os.environ.get('USE_HOROVOD', 'false').lower() == 'true')
        )
        self.rank = 0
        if horovod:
            print('initialising horovod!')
            self._init_hvd()
        else:
            if torch.cuda.is_available():
                self.m.to('cuda')
        if torch.cuda.is_available():
            try:
                self.l.to('cuda')
            except AttributeError:
                pass
            try:
                self.vl.to('cuda')
            except AttributeError:
                pass

    def _init_hvd(self):
        """Initialize horovod"""
        import horovod.torch as hvd
        hvd.init()
        self.hvd = hvd
        torch.manual_seed(1)
        print(f'local rank = {hvd.local_rank()}')
        torch.cuda.set_device(hvd.local_rank())
        self.m.to('cuda')
        for layer in self.m.layers.values():
            self.hvd.broadcast_parameters(
                layer.state_dict(),
                root_rank=0
            )
        self.rank = hvd.rank()
        self.o = hvd.DistributedOptimizer(
            self.o,
        )

    def take_step(self, batch_data):
        """Take a step in training"""
        self._call_callbacks('loss')
        if self.l is not None:
            loss = self.l(batch_data)
        else:
            loss = batch_data

        self._call_callbacks('back')
        self.o.zero_grad()
        loss.backward()
        self.train_loss = loss.item()

        self._call_callbacks('step')
        self.o.step()

        self.m.eval()
        self._call_callbacks('eval')
        self.m.train()

        self.it += 1

    def do_epoch(self):
        """Run a epoch"""
        self.m.train()
        for batch_data in self.m.preprocess(self.train_samples, **self.train_load,
                                            horovod=self.horovod):
            self.take_step(batch_data)

    def eval(self):
        """Evaluate the validation loss"""
        losses = []
        for batch_data in self.m.preprocess(self.valid_samples, **self.valid_load):
            losses.append(self.vl(batch_data).item())
        return sum(losses) / len(losses)

    def train(self):
        """Perform training"""
        if torch.cuda.is_available():
            self.m.to('cuda')
        while True:
            self.do_epoch()

    def _get_callbacks(self, phase):
        """Get the callbacks"""
        callbacks = {}
        for k in self._callbacks:

            if k.startswith(phase):
                if k == 'finished':
                    callbacks[k] = self[k]
                    continue

                if '!' in k:
                    when = int(k.split('!')[-1])
                    repeat = False
                else:
                    when = int(k.split('/')[-1])
                    repeat = True

                if (repeat and self.it % when == 0) or \
                        (not repeat and self.it == when):
                    callbacks[k] = self[k]
        return callbacks

    def _execute_single_callback(self, callback):
        """Execute a single callbacks and handle exceptions"""
        try:
            callback()
        except TrainingFinished:
            raise TrainingFinished
        except TimeToSave:
            self.save()

    def _execute_list_of_callbacks(self, callbacks):
        """Execute list of callbacks and handle exceptions"""
        for a_func in callbacks:
            try:
                a_func()
            except TimeToSaveAndStop:
                self.save()
                self._call_callbacks('finished')
                raise TimeToSaveAndStop
            except TrainingFinished:
                self._call_callbacks('finished')
                raise TrainingFinished
            except TimeToSave:
                self.save()

    def _call_callbacks(self, phase):
        """Call the callbacks"""
        if self.rank != 0:
            return

        if self.it == 0:
            return

        callbacks = self._get_callbacks(phase)

        if not callbacks:
            return

        for k in callbacks:
            if callable(callbacks[k]):
                self._execute_single_callback(callbacks[k])
            else:
                assert isinstance(callbacks[k], list)
                self._execute_list_of_callbacks(callbacks[k])

        sys.stdout.flush()
        sys.stderr.flush()

    def __setitem__(self, key, value):
        self._callbacks[key] = value

    def __getitem__(self, key):
        return self._callbacks[key]


Trainer = SimpleTrainer


class DefaultTrainer:
    """
    Default trainer

    :param tm: training model
    :param experiment: name of folder
    :param monitor: monitor
    :param m: model
    :param optimizer:
    :param lr: learning rate
    :param lr_schedule: learning rate schedule
    :param lr_schedule_arguments: arguments to pass to learning rate schedule
    :param optimizer_arguments: arguments for the optimizer
    :param validation_interval: interval at which to calculate the validation loss
    :param verbosity_interval:
    :param batch_size: batch size
    :param save:
    :param num_workers: number of workers to use to load data
    :param horovod:
    """
    def __init__(self,
                 tm,
                 experiment,
                 monitor=None,
                 m=None,
                 optimizer='Adam',
                 lr=0.0001,
                 lr_schedule='constant',
                 lr_schedule_arguments=None,
                 optimizer_arguments=None,
                 validation_interval=1000,
                 verbosity_interval=1,
                 batch_size=100,
                 save=None,
                 num_workers=0,
                 horovod=False):

        if optimizer_arguments is None:
            optimizer_arguments = {}

        mod_parameters = tm.parameters() if m is None else m.parameters()
        mod_parameters = [p for p in mod_parameters if p.requires_grad]
        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)(mod_parameters, lr=lr, **optimizer_arguments)

        if isinstance(lr_schedule, str) and lr_schedule != 'constant':
            from lf.optim.lr import lr_schedules
            assert lr_schedule_arguments is not None
            lr_schedule = lr_schedules[lr_schedule](
                param_groups=tm.param_groups,
                min_lr=lr,
                **lr_schedule_arguments,
            )

        from lf.monitor.monitor import DefaultMonitor

        def saving(model, mon):
            def _save():
                """Save the model and monitor"""
                model.save(f'{experiment}/model.tabc')
                if isinstance(mon, DefaultMonitor):
                    mon.save(f'{experiment}/monitor')
                model.to(tm.device)
            return _save

        if m is None and save is None:
            save = saving(tm, monitor)
        elif save is None:
            save = saving(m, monitor)
        else:
            assert callable(save)

        if monitor is None:
            from lf.monitor.monitor import Monitor
            monitor = Monitor([])

        # get length of data from model
        self.t = SimpleTrainer(
            model=tm,
            optimizer=optimizer,
            save=save,
            loss=None,
            horovod=horovod,
        )
        self.t.train_load['batch_size'] = batch_size
        self.t.train_load['num_workers'] = num_workers
        self.t.valid_load['batch_size'] = batch_size
        self.t.valid_load['num_workers'] = num_workers
        self.t.it = monitor.it

        # check the monitor isn't broken
        self.t['eval/!1'].append(lambda: monitor(self.t.it))
        self.t[f'eval/{validation_interval}'].append(lambda: monitor(self.t.it))
        if lr_schedule != 'constant':
            self.t['step/1'].append(lambda: lr_schedule(self.t.it))
        self.t[f'step/{verbosity_interval}'].append(
            lambda: print('TRAIN iteration: {}; loss: {};'.format(self.t.it, self.t.train_loss))
        )

    def reload(self, model):
        """Reload"""
        self.t.m.load_stat_dict(model)

    def __setitem__(self, key, value):
        self.t[key] = value

    def train(self):
        """Train"""
        return self.t.train()

    def take_step(self, x):
        """Take a step"""
        return self.t.take_step(x)
