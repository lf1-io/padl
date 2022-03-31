import click
import collections
import json
import os
import re
import padl


def build(f, build_name=None, **kwargs):
    if build_name is None:
        build_name = f.__name__
    if build_name in padl.build_args:
        kwargs.update(padl.build_args[build_name])
    return f(**kwargs)


def parse_inputs(x):
    groups = re.finditer('"([^\']+)"', x)
    reference = {}
    for i, g in enumerate(groups):
        x = x.replace(g.group(), f'#{i}')
        reference[f'#{i}'] = g.groups()[0]
    my_dict = dict([x.split('=') for x in x.split(',')])
    for k, val in my_dict.items():
        if val.isnumeric():
            my_dict[k] = eval(val)
        elif val.startswith('#'):
            my_dict[k] = reference[val]
        elif val in {'true', 'True', 'false', 'False'}:
            my_dict[k] = val.lower() == 'true'
        elif '+' in val:
            val = val.split('+')
            val = [x for x in val if x]
            val = [eval(x) if x.isnumeric() else x for x in val]
            my_dict[k] = val
    return my_dict


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
            my_dict = parse_inputs(value)
            for k, v in my_dict.items():
                if isinstance(v, str) and '$' in v:
                    group = re.match('.*\$([A-Za-z\_0-9]+)', v).groups()[0]
                    try:
                        my_dict[k] = v.replace(f'${group}', os.environ[group])
                    except KeyError:
                        raise Exception('key values referred to environment variable which did'
                                        ' not exist')
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


def postprocess_kwargs(kwargs):
    output = collections.defaultdict(lambda: {})
    for k in kwargs:
        key, subkey = k.split('.')
        output[key][subkey] = kwargs[k]
    return dict(output)


def main(path, **kwargs):
    padl.build_args = kwargs
    if path.endswith('.py'):
        path = path[:-3].replace('/', '.')
    dir_ = '/'.join(path.replace('.', '/').split('/')[:-1])
    with open(dir_ + '/padl.build.json', 'w') as f:
        json.dump(kwargs, f, indent=2)
    exec(f'import {path}')
    padl.build_args = {}


@click.command()
@click.argument('path')
@click.option('--kwargs', '-kw', default=None, type=KeyValuePairs())
def _main(path, kwargs):
    if kwargs is None:
        kwargs = {}
    kwargs = postprocess_kwargs(kwargs)
    main(path, **kwargs)


if __name__ == '__main__':
    _main()