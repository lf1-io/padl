# pylint: disable=no-member
"""Main"""
import os
import re

import importlib
import json
import click
import numpy
import torch

from lf import load
from lf.monitor.monitor import DefaultMonitor
from lf.quantization.quantize import quantize_layer
from lf.server.server import Server
from lf.optim.lr import lr_schedules
from lf.optim.loader import Loader
from lf.optim.train import TimeToSave

optimizers = [x for x in torch.optim.__dict__ if not x.startswith('_') and x != 'lr_scheduler']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class KeyValuePairs(click.ParamType):
    """Convert to key value pairs"""
    name = "key-value-pairs"

    def convert(self, value, param, ctx):
        """
        Convert to key value pairs

        :param value: value
        :param param: parameter
        :param ctx: context
        """
        if not value.strip():
            return {}
        try:
            my_dict = dict([x.split('=') for x in value.split(',')])
            for k, val in my_dict.items():
                if val.isnumeric():
                    my_dict[k] = eval(val)
            return my_dict
        except TypeError:
            self.fail(
                "expected string for key-value-pairs() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid key-value-pair", param, ctx)


class Tuple(click.ParamType):
    """Convert to tuple"""
    name = "tuple"

    def convert(self, value, param, ctx):
        """
        Convert to tuple

        :param value: value
        :param param: parameter
        :param ctx: context
        """
        try:
            return value.split(',')
        except TypeError:
            self.fail(
                "expected string for tuple() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid tuple", param, ctx)


@click.group()
def cli():
    """CLI"""
    ...


@cli.command()
@click.option('-m', '--model', type=str)
def check(model):
    """
    Check the model by generating inputs and testing on these.
    """
    _check(model)


def _check(model: str):
    """
    Check loaded model

    :param model: model
    """
    if model.endswith('.tabc'):
        loaded_model = load(model)
    else:
        config = importlib.import_module('.'.join(model.split('.')[:-1]))
        loaded_model = getattr(config, model.split('.')[-1])
    print('checking...')
    loaded_model.check()


@cli.command()
@click.option('-m', '--model', help='model to compute')
@click.option('-i', '--inputs', help='inputs to load')
@click.option('-o', '--outputs', help='path to outputs to dump')
@click.option('-v', '--variables', type=KeyValuePairs(), default='',
              help='dictionary of variables, key-values',
              show_default=True)
@click.option('--quiet/--no-quiet', default=True,
              help='toggle on to get a progress bar',
              show_default=True)
@click.option('-p', '--parser',
              default='json',
              help='method to load inputs into the program',
              show_default=True)
@click.option('-s', '--serializer',
              default='json',
              help='method to dump outputs from the program',
              show_default=True)
@click.option('-bs', '--batch-size',
              type=int,
              default=100,
              help='batch-size for computing outputs',
              show_default=True)
@click.option('-nw', '--num-workers',
              type=int,
              default=0,
              help='number of workers in computing outputs',
              show_default=True)
@click.option('--sep',
              type=str,
              default='\t',
              help='separator for csv exports',
              show_default=True)
@click.option('--input-col',
              type=str,
              default='item_id',
              help='col 1 of dataframe',
              show_default=True)
@click.option('--output-col',
              type=str,
              default='output',
              help='col 2 of dataframe',
              show_default=True)
@click.option('--cleaner',
              type=str,
              default='->',
              help='col 2 of dataframe',
              show_default=True)
def compute_outputs(model, inputs, outputs, variables, quiet, parser, serializer, batch_size,
                    num_workers, sep, input_col, output_col, cleaner):
    """
    Compute outputs of the model on an input file in .eval() mode and save to output file

    :param model: model
    :param inputs: inputs
    :param outputs: outputs
    :param variables: variables
    :param quiet: If True do not print progress bars
    :param parser:
    :param serializer:
    :param batch_size: batch size
    :param num_workers: number of workers to load data
    :param sep: separation token
    :param input_col: input column label
    :param output_col: output column label
    :param cleaner: cleaner
    """

    inputs = _get_inputs(parser, inputs)
    variables = _get_variables(variables)

    model = load(model, **variables)
    model.eval()
    model.to(device)
    computed = list(
        model(inputs, batch_size=batch_size, flatten=True, num_workers=num_workers,
              verbose=not quiet)
    )
    if isinstance(computed[0], torch.Tensor):
        computed = torch.stack(computed).to('cpu').numpy()
    elif isinstance(computed[0], numpy.ndarray):
        computed = numpy.stack(computed)

    if serializer in {'json', 'csv', 'txt'}:
        computed = [list(computed[i]) for i in range(computed.shape[0])]

    if serializer == 'json':
        with open(outputs, 'w') as file1:
            json.dump(computed, file1)
    elif serializer == 'numpy':
        numpy.save(outputs, computed)
    elif serializer == 'txt':
        with open(outputs, 'w') as file1:
            file1.write('\n'.join([str(x) for x in computed]))
    elif serializer == 'csv':
        cleaner = cleaner.split('->')
        with open(outputs, 'w') as file1:
            file1.write(f'{input_col}{sep}{output_col}\n')
        with open(outputs, 'a') as file1:
            for i, line in enumerate(computed):
                line = str(line)
                if cleaner[0]:
                    line = re.sub(cleaner[0], cleaner[1], line)
                file1.write(f'{inputs[i]}{sep}{line}\n')
    else:
        raise NotImplementedError


def _get_inputs(_parser, _inputs):
    """
    Get inputs

    :param _parser: parser
    :param _inputs: inputs
    """
    if _parser == 'json':
        with open(_inputs) as file1:
            _inputs = json.load(file1)
    elif _parser == 'txt':
        with open(_inputs) as file1:
            _inputs = [x for x in file1.read().split('\n')]
    elif _parser == 'numpy':
        _inputs = numpy.load(_inputs)
    elif _parser == 'directory':
        _inputs = [f'{_inputs}/{x}' for x in os.listdir(_inputs)]
    else:
        raise NotImplementedError
    return _inputs


def _get_variables(_variables):
    """
    Get variables

    :param _variables: variables
    """
    for key, val in _variables.items():
        if not '%' in val:
            continue
        action, type_, val = val.split('%')
        if action == 'file':
            if type_ == 'json':
                with open(val) as file1:
                    val = json.load(file1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        _variables[key] = val
    return _variables

@cli.command()
@click.option('-m', '--model', default=None, help='model to validate')
@click.option('-mo', '--monitor', default=None, help='monitor with which to validate')
@click.option('-v', '--variables', type=KeyValuePairs(), default='',
              help='dictionary of variables, key-values',
              show_default=True)
@click.option('-q', '--quiet/--no-quiet', default=True,
              help='toggle on to get a progress bar',
              show_default=True)
def validate(model, monitor, variables, quiet):
    """
    Validate a model using a monitor.

    :param model: model
    :param monitor: monitor
    :param variables: variables to pass to _validate
    :param quiet: If True do not print progress bars
    """
    try:
        _validate(model, monitor, variables, quiet)
    except TimeToSave:
        pass


def _validate(model, monitor, variables, quiet):
    """
    Validate a model using a monitor.

    :param model: model to validate
    :param monitor: monitor
    :param variables: variables to pass to loaded_model and monitor
    :param quiet: If True do not print progress bars
    """
    loader = DefaultMonitor.loader(monitor)
    loaded_model = load(model, var=variables)
    monitor = loader(loaded_model, var=variables, verbose=not quiet)
    monitor()


@cli.command()
@click.option('-e', '--experiment', required=True, help='folder for content (configs/ checkpoints)')
@click.option('-c', '--config', default='config', help='name of .py config file', show_default=True)
@click.option('-vi', '--validation-interval', type=int, default=1000,
              help='batch iterations between validation', show_default=True)
@click.option('-v', '--variables', type=KeyValuePairs(), default='',
              help='dictionary of variables, key-values',
              show_default=True)
@click.option('--verbose/--no-verbose', default=True, help='toggle on for more verbosity',
              show_default=True)
@click.option('-w', '--watch', type=str, default=None, help='metric to watch for stopping/ saving',
              show_default=True)
@click.option('-rt', '--retries', type=int, default=5,
              help='how often to continue after no metric improvement',
              show_default=True)
@click.option('-nw', '--num-workers', type=int, default=0, help='number of workers in data loading',
              show_default=True)
@click.option('-bs', '--batch-size', type=int, default=100, help='size of batch', show_default=True)
@click.option('-lr', '--learning-rate', type=float, default=0.0001, help='learning rate',
              show_default=True)
@click.option('-ls', '--lr-schedule', type=click.Choice(['constant'] + list(lr_schedules.keys())),
              default='constant', help='type of learning rate schedule', show_default=True)
@click.option('-lsa', '--lr-schedule-arguments', type=KeyValuePairs(), default='',
              show_default=True)
@click.option('-o', '--optimizer', type=click.Choice(optimizers), default='Adam', show_default=True)
@click.option('-oa', '--optimizer-arguments', type=KeyValuePairs(), default='', show_default=True)
@click.option('--horovod/--no-horovod', default=False, show_default=True)
@click.option('--continued/--no-continued', default=True,
              help='toggle off to not continue training but start again', show_default=True)
@click.option('--overwrite/--no-overwrite', default=False,
              help='toggle on to overwrite present results', show_default=True)
def train(
    experiment,
    config,
    validation_interval,
    variables,
    verbose,
    watch,
    retries,
    num_workers,
    batch_size,
    learning_rate,
    lr_schedule,
    lr_schedule_arguments,
    optimizer,
    optimizer_arguments,
    horovod,
    continued,
    overwrite,
):
    """
    Train a model using default trainer and monitor.
    """
    _train(
        experiment,
        config,
        validation_interval,
        variables,
        verbose,
        watch,
        retries,
        num_workers,
        batch_size,
        learning_rate,
        lr_schedule,
        lr_schedule_arguments,
        optimizer,
        optimizer_arguments,
        horovod,
        continued,
        overwrite,
    )


def _train(
    experiment,
    config,
    validation_interval,
    variables,
    verbose,
    watch,
    retries,
    num_workers,
    batch_size,
    learning_rate,
    lr_schedule,
    lr_schedule_arguments,
    optimizer,
    optimizer_arguments,
    horovod,
    continued,
    overwrite,
):
    """
    Train a model using default trainer and monitor.
    """

    loader = Loader(experiment, config=config)
    a_trainer = loader(
        variables=variables,
        validation_interval=validation_interval,
        verbose=verbose,
        watch=watch,
        retries=retries,
        num_workers=num_workers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        lr_schedule_arguments=lr_schedule_arguments,
        optimizer=optimizer,
        optimizer_arguments=optimizer_arguments,
        horovod=horovod,
        continued=continued,
        overwrite=overwrite,
    )
    a_trainer.train()


@cli.command()
@click.option('-e', '--experiment')
@click.option('-l', '--layer')
@click.option('-v', '--variables', type=KeyValuePairs(), default={})
@click.option('-b', '--batch_size', type=int, default=100)
def quantize(experiment, layer, variables, batch_size):
    """Add validation data to the model"""
    _quantize(experiment, layer, variables, batch_size)


def _quantize(experiment, layer, variables, batch_size):
    """Add validation data to the model"""
    loaded_model = load(f'{experiment}/model.tabc')
    validation_data = \
        DefaultMonitor.load(loaded_model, f'{experiment}/monitor', var=variables).validation_data
    quantize_layer(
        validation_data >> loaded_model,
        layer_name=layer,
        data=range(len(validation_data)),
        batch_size=batch_size,
    )
    loaded_model.save(f'{experiment}/quantized.tabc')


@cli.command()
@click.option('-m', '--model', default=None, help='model/ slice of model to serve')
@click.option('-s', '--signature', type=Tuple(), help='argument signature in frontend')
@click.option('-c', '--config', default=None, help='name of config file')
@click.option('-v', '--variables', default=None, type=KeyValuePairs(),
              help='variables to instantiate on loading')
@click.option('-p', '--port', type=int, default=80, help='serving port')
def serve(model, signature, config, variables, port):
    """
    :param model: model
    :param signature: signature in frontend
    :param config: name of config file
    :param variables: variables to instantiate on loading
    :param port: port
    :return:
    """
    _serve(model, signature, config, variables, port)


def _serve(model, signature, config, variables, port):
    """
        :param model: model
        :param signature: signature in frontend
        :param config: name of config file
        :param variables: variables to instantiate on loading
        :param port: port
        :return:
    """
    if config is not None:
        with open(config) as file1:
            config = json.load(file1)
    else:
        assert model is not None
        config = {
            'models': {
                '/': {
                    'path': model,
                    'variables': variables,
                    'signature': signature,
                }
            },
            'serve_args': {
                'port': port,
            }
        }
    Server(**config).serve()


if __name__ == '__main__':
    cli.main()
