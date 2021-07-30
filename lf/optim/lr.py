"""Learning rate functionality for optim"""
from collections import OrderedDict
import numpy as np
from lf.optim.train import TrainingFinished


class LrSchedule:
    """
    Scheduler for learning rate

    :param param_groups: parameter groups
    :param end: max number of iterations for training
    """
    def __init__(self, param_groups, end=float('inf')):

        self.end = end
        self.param_groups = param_groups
        self.lr = None

    def calc_lr(self, iteration):
        """Calculate the learning rate"""
        raise NotImplementedError

    def __call__(self, iteration):
        if iteration > self.end:
            raise TrainingFinished
        self.lr = self.calc_lr(iteration)
        for param_group in self.param_groups:
            param_group['lr'] = self.lr


class TestLrSchedule(LrSchedule):
    """
    :param min_lr: minimum learning rate
    :param max_lr: maximum learning rate
    :param n_its: number of iterations it takes to scale up to max learning rate
    :param args: additional arguments
    :param kwargs: additional keyword arguments
    """
    freq = '1/it'

    def __init__(self, min_lr, max_lr, n_its, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_its = n_its

    def calc_lr(self, iteration):
        return self.min_lr + iteration / self.n_its * (self.max_lr - self.min_lr)


class TriangleLrSchedule(LrSchedule):
    """
    :param min_lr: minimum learning rate
    :param max_lr: maximum learning rate
    :param n_its: number of iterations used to determine the cycle
    :param args: additional arguments
    :param kwargs: additional keyword arguments
    """
    freq = '1/it'

    def __init__(self, min_lr, max_lr, n_its, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = min_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_its = n_its

    def calc_lr(self, iteration):
        cycle_r = (iteration % self.n_its) / float(self.n_its)
        if cycle_r < 0.5:
            learning_rate = self.min_lr + cycle_r * 2 * (self.max_lr - self.min_lr)
        else:
            learning_rate = self.max_lr - (cycle_r - 0.5) * 2 * (self.max_lr - self.min_lr)
        return learning_rate


class CosineLrSchedule(LrSchedule):
    """Cosine lr with resets, starts high, drops like half a cosine, then resets to the initial
    value.
    """
    freq = '1/it'

    def __init__(self, min_lr, max_lr, n_its, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_its = n_its

    def calc_lr(self, iteration):
        return ((np.cos(np.pi * (iteration % self.n_its) / self.n_its) + 1)
                * (self.max_lr - self.min_lr) / 2
                + self.min_lr)


class OneCycleLrSchedule(LrSchedule):
    """One triangular cycle: Rises linearly from *min_lr* to *max_lr* in *n_its_up* iterations,
    then drops linearly back to *min_lr* (*n_its_down*). After that, there is an additional
    linear drop to *min_lr* / 100 in *n_its_after* iterations. Then it ends by raising an Exception.
    """
    freq = '1/it'

    def __init__(self, min_lr, max_lr, n_its=None, n_its_up=None, n_its_down=None, n_its_after=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = min_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        assert(n_its is not None or all([x is not None
                                         for x in [n_its_up, n_its_down, n_its_after]]))
        self.n_its_up = n_its_up or n_its / 2
        self.n_its_down = n_its_down or n_its / 2
        self.n_its_after = n_its_after or n_its / 3
        self.n_its = self.n_its_up + self.n_its_down + self.n_its_after

    def calc_lr(self, iteration):
        if iteration < self.n_its_up:
            return self.min_lr + iteration / (self.n_its_up) * (self.max_lr - self.min_lr)
        elif iteration < self.n_its_up + self.n_its_down:
            return self.max_lr - (iteration - (self.n_its_up)) / (self.n_its_down) \
                   * (self.max_lr - self.min_lr)
        elif iteration < self.n_its_up + self.n_its_down + self.n_its_after:
            return self.min_lr - (iteration - self.n_its_up - self.n_its_down) / (self.n_its_after) \
                   * (self.min_lr - self.min_lr / 100)
        else:
            raise TrainingFinished


class ChainedLrSchedule(LrSchedule):
    """
    :param sub_schedules: dictionary of schedules to chain together
    :param args: additional arguments
    :param kwargs: additional keyword arguments
    """
    freq = '1/it'

    def __init__(self, sub_schedules, *args, **kwargs):

        super().__init__(*args, **kwargs)
        try:
            sub_schedules = {
                int(it): lr_schedules[s['type']](param_groups=self.param_groups, **s['args'])
                for it, s in sub_schedules.items()
            }
        except KeyError:
            pass
        self.sub_schedules = OrderedDict(sorted(sub_schedules.items(), key=lambda x: -x[0]))

    def calc_lr(self, iteration):
        for s_it, sched in self.sub_schedules.items():
            if s_it <= iteration:
                return sched.calc_lr(iteration - s_it)
        return None


lr_schedules = {
    'lr_test': TestLrSchedule,
    'triangle': TriangleLrSchedule,
    'cosine': CosineLrSchedule,
    'one_cycle': OneCycleLrSchedule,
    'chain': ChainedLrSchedule
}
