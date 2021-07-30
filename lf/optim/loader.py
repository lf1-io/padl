"""Loading functionality for optim"""
import importlib
import os

from lf.optim.train import SimpleTrainer, DefaultTrainer
from lf.transforms import Transform
from lf.monitor.monitor import DefaultMonitor
from lf import load


class Loader:
    """Loader

    :param experiment: path to config file
    :param config: name of config file
    """
    def __init__(self, experiment, config='config'):
        self.experiment = experiment
        self.config = config

    def _import(self):
        """Import config module"""
        return importlib.import_module(
            f'{self.experiment.replace("/", ".")}.{self.config}'
        )

    @staticmethod
    def _check_precompiled_type(cf):
        """Check precompiled type"""
        return (
            hasattr(cf, 't') and (
                isinstance(cf.t, SimpleTrainer)
                or isinstance(cf.t, DefaultTrainer)
            )
        )

    @staticmethod
    def _check_default_type(cf):
        """Check default type"""
        okay = True
        okay = okay and (hasattr(cf, 'm') and isinstance(cf.m, Transform))
        okay = okay and (hasattr(cf, 'tm') and isinstance(cf.tm, Transform))
        try:
            _ = len(cf.tm)
        except AttributeError:
            okay = False
        return okay

    @staticmethod
    def _check_default_function_type(cf):
        """Check default function type"""
        return (
            (hasattr(cf, 'model') and callable(cf.model))
            and (hasattr(cf, 'data') and callable(cf.data))
        )

    @staticmethod
    def _check_predefined_function_type(cf):
        """Check pre-defined function type"""
        return hasattr(cf, 'trainer') and callable(cf.trainer)

    def _build_default_function_trainer(self, cf, **kwargs):
        """Build default function trainer"""
        train_data, valid_data, ground_truth = cf.data()
        mod, train_mod = cf.model()
        return self.build_trainer(
            self.experiment, train_mod, mod, valid_data, ground_truth, td=train_data,
            metrics=cf.metrics() if hasattr(cf, 'metrics') else {},
            **kwargs
        )

    def _build_default_trainer(self, cf, **kwargs):
        """Build default trainer"""
        metrics = {
            k: getattr(cf, k)
            for k in cf.__dict__
            if hasattr(getattr(cf, k), 'is_metric')
            and getattr(cf, k).is_metric == True
        }
        return self.build_trainer(
            self.experiment, cf.tm, cf.m, cf.vd, cf.gt, metrics=metrics, **kwargs
        )

    @staticmethod
    def build_trainer(
        experiment,
        tm,
        m,
        vd,
        gt,
        td=None,
        variables=None,
        metrics=None,
        validation_interval=1000,
        verbose=False,
        watch='loss',
        retries=5,
        num_workers=0,
        batch_size=100,
        learning_rate=0.0001,
        lr_schedule='constant',
        lr_schedule_arguments=None,
        optimizer='Adam',
        optimizer_arguments=None,
        horovod=False,
        continued=True,
        overwrite=False,
    ):
        """
        Invoke loader.

        :param experiment: experiment directory
        :param tm: training model
        :param m: model to save
        :param vd: validation data
        :param gt: ground truth data
        :param td: training data - if None then assumed in tm
        :param variables: variables dictionary
        :param metrics: metrics dictionary
        :param validation_interval: interval of validation
        :param verbose: toggle for more output
        :param watch: key from metrics/ loss to watch in stopping
        :param retries: no. of retries after 0 improvement
        :param num_workers: number of workers
        :param batch_size: batch-size
        :param learning_rate: learning-rate
        :param lr_schedule: learning-schedule from `lf.optim.lr`
        :param lr_schedule_arguments: arguments
        :param optimizer: optimizer from `torch.optim`
        :param optimizer_arguments: arguments
        :param horovod: toggle on to use horovod server
        :param continued: toggle on to load previous checkpoint
        :param overwrite: toggle on to force overwrite previous checkpoint

        :returns: trainer object
        """

        if td is not None:
            tm = td >> tm

        if os.path.exists(f'{experiment}/model.tabc'):
            print(f'found existing checkpoint {experiment}/model.tabc')
            if continued:
                loaded_model = load(f'{experiment}/model.tabc', var=variables)
                m.load_state_dict(loaded_model)
                tm.load_state_dict(loaded_model)
            else:
                if not overwrite:
                    raise Exception(
                        'You have specified to not continue, '
                        'but there is already a checkpoint present. '
                        'This could lead to loss of results. '
                        'To force specify overwrite=True'
                    )

        if lr_schedule_arguments is None:
            lr_schedule_arguments = {}
        if metrics is None:
            metrics = {}
        if optimizer_arguments is None:
            optimizer_arguments = {}

        if continued and os.path.exists(f'{experiment}/monitor.tabc'):
            monitor = DefaultMonitor.load(
                f'{experiment}/monitor',
                m=m,
                tm=tm,
                verbose=verbose,
                batch_size=batch_size,
                watch=watch,
                retries=retries,
            )
        else:
            monitor = DefaultMonitor(
                m=m,
                validation_data=vd,
                ground_truth=gt,
                tm=tm,
                verbose=verbose,
                batch_size=batch_size,
                watch=watch,
                retries=retries,
                **metrics,
            )
        a_trainer = DefaultTrainer(
            tm=tm,
            experiment=experiment,
            m=m,
            monitor=monitor,
            validation_interval=validation_interval,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=learning_rate,
            lr_schedule=lr_schedule,
            lr_schedule_arguments=lr_schedule_arguments,
            optimizer=optimizer,
            optimizer_arguments=optimizer_arguments,
            horovod=horovod,
        )
        if continued:
            a_trainer.it = monitor.it
        return a_trainer

    def __call__(self, **kwargs):
        """
        Invoke loader.

        :param validation_interval: interval of validation
        :param verbose: toggle for more output
        :param watch: key from metrics/ loss to watch in stopping
        :param retries: no. of retries after 0 improvement
        :param num_workers: number of workers
        :param batch_size: batch-size
        :param learning_rate: learning-rate
        :param lr_schedule: learning-schedule from `lf.optim.lr`
        :param lr_schedule_arguments: arguments
        :param optimizer: optimizer from `torch.optim`
        :param optimizer_arguments: arguments
        :param horovod: toggle on to use horovod server
        :param continued: toggle on to load previous checkpoint
        :param overwrite: toggle on to force overwrite previous checkpoint

        :returns: trainer object
        """
        config = self._import()
        if self._check_precompiled_type(config):
            return config.t

        if self._check_default_type(config):
            return self._build_default_trainer(config, **kwargs)

        if self._check_predefined_function_type(config):
            return config.trainer()

        if self._check_default_function_type(config):
            return self._build_default_function_trainer(config, **kwargs)

        raise Exception('couldnt determine a trainer from the provided setup.')
