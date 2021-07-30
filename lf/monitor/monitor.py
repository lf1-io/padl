"""Monitor"""
import json
import inspect
from collections import defaultdict

from lf.transforms import Transform, load as load_transform, NamedRollout, \
    NamedParallel
from lf.transforms.data import DataSet
from lf.optim.train import TrainingFinished, TimeToSave, TimeToSaveAndStop
from lf.utils import stage


class Monitor:
    """
    Class for monitoring training process with a series of callbacks.
    These callbacks are passed in a list `glances` and the callbacks are
    called sequentially. If a callback returns a dictionary, the values in
    the dictionary are saved in an internal state. Subsequence callbacks can
    receive these as inputs.

    Example:

    >>> glances = [lambda: {'progress': 'is good'},
    ... lambda progress: {'metric': float(progress == 'is good')}]
    >>> m = Monitor(glances, save=lambda: False)
    >>> m(0)
    VALID iteration: 0; metric: 1.0000000000;
    GLANCE 0 iteration: 0; data: {"progress": "is good"};
    """

    def __init__(self, glances, stop=lambda: False,
                 save=lambda: True, exclude=()):
        """
        :param glances: dictionary of functions to be called periodically,
                        potentially outputting data
        :param stop: function with returns True when training should stop
        :param save: function which returns True when model should be saved
        :param exclude: list of values to exclude from output
        """
        self.glances = glances
        self.stop = stop
        self.save = save
        self._past_results = defaultdict(list)
        self.exclude = exclude

    @property
    def past_results(self):
        """Return past results"""
        return self._past_results

    @past_results.setter
    def past_results(self, value):
        """Set past results"""
        self._past_results = defaultdict(list)
        self._past_results.update(value)

    @property
    def it(self):
        """Return current iteration"""
        try:
            return self.past_results['iteration'][-1]
        except (KeyError, IndexError):
            return 0

    def __add__(self, other):
        return Monitor(
            self.glances + other.glances,
            stop=self.stop,
            save=self.save,
            exclude=tuple(list(self.exclude) + list(other.exclude)),
        )

    def do(self, it):
        """Update iteration tracking by *it* and run callbacks in glances"""
        result = {}
        for a_glance in self.glances:
            params = inspect.signature(a_glance).parameters
            input_ = {k: v for k, v in {**result}.items() if k in params}
            if 'iteration' in params:
                input_['iteration'] = it
            out = a_glance(**input_)
            if out is not None:
                result.update(out)

        result['iteration'] = it
        for k in result:
            if k not in self.exclude:
                self.past_results[k].append(result[k])

        return self.format(result)

    def format(self, result):
        """Format the results into log and glance"""
        log = {}
        glance = {}
        for k in result:
            if k in self.exclude:
                continue
            if isinstance(result[k], float) or isinstance(result[k], int):
                log[k] = result[k]
            else:
                glance[k] = result[k]
        return log, glance

    def _glance(self, it, glance, keys):
        """Format the glance string and print"""
        if glance:
            for i, k in enumerate(keys):
                str_ = (
                        f'GLANCE {i} iteration: {it}; data: '
                        + json.dumps({k: glance[k]}) + ';'
                )
                print(str_)

    def _do_save(self):
        save_keys = inspect.signature(self.save).parameters.keys()
        save_d = {k: v for k, v in self.past_results.items() if k in save_keys}
        if save_d or not save_keys:
            if self.save(**save_d):
                return True
        else:
            if self.save(self.past_results):
                return True
        return False

    def _do_stop(self):
        stopping_keys = inspect.signature(self.stop).parameters.keys()
        stopping_d = {k: v for k, v in self.past_results.items() if k in stopping_keys}
        if stopping_d or not stopping_keys:
            if self.stop(**stopping_d):
                return True
        else:
            if self.stop(self.past_results):
                return True
        return False

    def __call__(self, it=0):

        log, glance = self.do(it)
        keys = sorted(glance.keys())

        if log:
            str_ = (
                f'VALID iteration: {it};'
                + ''.join([f' {k}: {v:.10f};' for k, v in log.items() if k != 'iteration'])
            )
            print(str_)

        self._glance(it, glance, keys)

        do_save = self._do_save()
        do_stop = self._do_stop()

        if do_save and do_stop:
            raise TimeToSaveAndStop
        elif do_save:
            raise TimeToSave
        elif do_stop:
            raise TrainingFinished

        return glance


class DefaultMonitor:
    """
    Default monitor for the standard lf design pattern.

    :param m: inference model
    :param validation_data: validation-data
    :param ground_truth: ground-truth data
    :param tm: training model (optional)
    :param reducers: reducers (optional)
    :param verbose: toggle for verbosity (optional)
    :param batch_size: batch size for calculating inferences (optional)
    :param watch: metric to watch in stopping/ saving (optional)
    :param retries: how often to wait for no improvement (optional)
    :param stop: explicit stopping criterion (optional)
    :param save: explicit saving criterion (optional)
    :param num_workers: number of parallel workers (optional)
    :param glances: list of functions to be called periodically,
                    potentially outputting data (optional)
    :param **kwargs: key-value pairs of metric transforms (optional)
    """
    def __init__(
            self,
            m=None,
            validation_data=None,
            ground_truth=None,
            tm=None,
            verbose=True,
            batch_size=100,
            watch=None,
            retries=5,
            stop=None,
            save=None,
            num_workers=0,
            glances=None,
            **kwargs,
    ):

        for k in kwargs:
            assert isinstance(kwargs[k], Transform)

        if glances is None:
            glances = []

        if tm is not None:
            def validate():
                """Calculate validation loss"""
                with stage(tm, 'eval'):
                    a_list = list(tm(range(len(tm)), flatten=False, num_workers=0,
                                  verbose=verbose, batch_size=batch_size))
                    a_list = sum([x.item() for x in a_list]) / len(a_list)
                return {'loss': a_list}
            glances.append(validate)

        if not isinstance(validation_data, Transform):
            validation_data = DataSet(validation_data)
        if not isinstance(ground_truth, Transform):
            ground_truth = DataSet(ground_truth)
        self.ground_truth = ground_truth
        self.validation_data = validation_data
        self.metrics = NamedRollout(**kwargs) if kwargs else None
        self.evaluator = self.validation_data >> m
        self.batch_size = batch_size
        self.verbose = verbose
        self.retries = retries
        self.num_workers = num_workers
        if self.metrics is not None:
            glances.append(self.evaluate_metrics)

        if watch is None and tm is not None:
            watch = 'loss'
        elif watch is None:
            watch = list(kwargs.keys())[0]

        stop = self._stop(stop, watch)
        save = self._save(save, watch)

        self.monitor = Monitor(glances, stop=stop, save=save)

    def _stop(self, stop, watch):
        """Return the stop function"""
        if stop is None:
            if watch == 'loss':
                def stop(loss):
                    """Stop condition"""
                    return sum([x > min(loss) for x in loss[-self.retries:]]) == self.retries
            else:
                source = f'lambda {watch}: sum([x < max({watch}) ' \
                         f'for x in {watch}[-{self.retries}:]]) == {self.retries}'
                stop = eval(source)
        return stop

    def _save(self, save, watch):
        """Return the save function"""
        if save is None:
            if watch == 'loss':
                def save(loss):
                    """Save condition"""
                    return loss[-1] == min(loss)
            else:
                source = f'lambda {watch}: {watch}[-1] == max({watch})'
                save = eval(source)
        return save

    def evaluate_metrics(self):
        """Evaluate metrics"""
        ground_truth = []
        for i in range(len(self.ground_truth)):
            ground_truth.append(self.ground_truth.do(i))

        with stage(self.evaluator, 'eval'):
            predictions = list(self.evaluator(
                range(len(self.validation_data)),
                batch_size=self.batch_size,
                verbose=self.verbose,
                flatten=True,
                num_workers=self.num_workers,
            ))
        with stage(self.metrics, 'infer'):
            metrics = self.metrics((predictions, ground_truth))
        return metrics

    def __call__(self, it=0):
        return self.monitor(it=it)

    @property
    def it(self):
        return self.monitor.it

    def save(self, path_):
        if self.metrics is not None:
            to_save = NamedParallel(
                metrics=self.metrics,
                validation_data=self.validation_data,
                ground_truth=self.ground_truth,
            )
        else:
            to_save = NamedParallel(
                validation_data=self.validation_data,
                ground_truth=self.ground_truth,
            )
        to_save.save(path_ + '.tabc')
        with open(f'{path_}.json', 'w') as file1:
            json.dump(self.monitor.past_results, file1)

    @staticmethod
    def load(path_, m, var=None, **kwargs):
        """Load the monitor"""
        if var is None:
            var = {}
        mon = DefaultMonitor.loader(path_)(m, **kwargs, var=var)
        with open(path_ + '.json') as file1:
            past_results = json.load(file1)
        mon.monitor.past_results = past_results
        return mon

    @staticmethod
    def loader(path_):
        """Loader for the monitor"""

        def loader(m, var=None, **kwargs):
            """Loader for the monitor"""
            if var is None:
                var = {}
            loaded = load_transform(path_ + '.tabc', **var)
            if hasattr(loaded, 'metrics'):
                metrics = dict(zip(loaded.metrics.keys, loaded.metrics.trans_list))
            else:
                metrics = {}
            validation_data = loaded.validation_data
            ground_truth = loaded.ground_truth
            mon = DefaultMonitor(
                m,
                validation_data=validation_data,
                ground_truth=ground_truth,
                **metrics,
                **kwargs,
            )
            return mon

        return loader


def metric(t):
    """
    :param t: trainer
    :return:
    """
    t.metric = True
    return t
