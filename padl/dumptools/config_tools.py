import ast

from padl.dumptools import var2mod, sourceget, ast_utils, symfinder


def param(name, val, description=None, use_default=True):
    """Helper function for marking parameters.

    Parameters can be overridden when loading. See also :func:`padl.load`.

    :param name: The name of the parameter.
    :param val: The default value of the parameter / the value before saving.
    :param description: Description of the parameter.
    :param use_default: If True, will use *val* when loading without specifying a different value.
    :returns: *val*
    """
    return val


def params(name, *, use_defaults=True, allow_free=False, **kwargs):
    """Helper function for marking parameters.

    Parameters can be overridden when loading. See also :func:`padl.load`.

    :param name: The name of the parameter.
    :param description: Description of the parameter.
    :param use_defaults: If True, use passed keyword args when loading without specifying
        different values.
    """
    return kwargs


def apply_params(source, parsed_kwargs, scope):
    """Apply param replacement in *source*.

    :param source: The sourcecode in which to replace the params.
    :param parsed_kwargs: The parsed keyword argument with params to replace.
    :param scope: The scope.
    """
    param_dicts = extract_param_s(source)
    params_dicts = extract_params_s(source)

    code_graph = var2mod.CodeGraph()

    done = set()

    for k, v in parsed_kwargs.items():
        if k in param_dicts:
            code_graph_here = var2mod.CodeGraph().from_source(f'PARAM_{k} = {v}', scope, name=k)
            if len(code_graph_here) > 1:
                code_graph.update(code_graph_here)
                source = change_param(source, k, f'PARAM_{k}')
            else:
                source = change_param(source, k, v)
            done.add(k)
        elif k in params_dicts:
            value_dict = parse_dict(v)
            repl = {}
            for sub_k, sub_v in value_dict.items():
                code_graph_here = var2mod.CodeGraph().from_source(f'PARAM_{k}_{sub_k} = {sub_v}',
                                                                  scope, name=k)
                if len(code_graph_here) > 1:
                    code_graph.update(code_graph_here)
                    repl[sub_k] = f'PARAM_{k}_{sub_k}'
                else:
                    repl[sub_k] = sub_v
                done.add((k, sub_k))

            source = change_params(source, k, **repl)
        else:
            raise ValueError(f'Extra parameter: {k}')
    print(done)

    missing = []
    for k, v in param_dicts.items():
        if not eval(v['use_default']) and k not in done:
            missing.append(k)
    for k, v in params_dicts.items():
        for sub_k in v['kwargs']:
            if not eval(v['use_defaults']) and (k, sub_k) not in done:
                missing.append(f'{k}.{sub_k}')

    if missing:
        raise ValueError(f'Missing mandatory parameter(s): {missing}')

    return code_graph.dumps().strip('\n') + source


def extract_param_s(source: str) -> dict:
    """Extract :func:`params` calls from *source*.

    Example:

    >>> source = '''a = param('param_a', x, 'a parameter')
    ... b = param('b', 2)'''
    >>> extract_param_s(source)  # doctest: +NORMALIZE_WHITESPACE
    {'param_a': {'name': "'param_a'", 'val': 'x', 'description': "'a parameter'",
                 'use_default': 'True',
                 'position': Position(lineno=1, end_lineno=1, col_offset=21, end_col_offset=22)},
     'b': {'name': "'b'", 'val': '2', 'description': 'None', 'use_default': 'True',
           'position': Position(lineno=2, end_lineno=2, col_offset=15, end_col_offset=16)}}

    :param source: The source code from which to extract the :func:`param`-calls.
    :return: A dict mapping :func:`param` arguments (*name*, *val*, *description*, *use_default*)
        to their string values. Also adds a "position" entry with the position of the parameter's
        value in the code.
    """
    res = {}
    for call in var2mod.Finder(ast.Call).find(ast.parse(source)):
        if ast_utils.get_source_segment(source, call.func) not in ('param', 'padl.param'):
            continue
        position = ast_utils.get_position(source, call)
        call_source = sourceget.cut(source, *position, one_indexed=True)
        args, keys, _, _ = symfinder._get_call_signature(call_source)
        signature = symfinder.Signature(argnames=['name', 'val', 'description', 'use_default'],
                                        defaults={'description': 'None', 'use_default': 'True'})
        assignments = signature.get_call_assignments(args, keys)
        if call.args:
            # value provided positionally
            assignments['position'] = ast_utils.get_position(source, call.args[1])
        else:
            # value provided as keyword
            assignments['position'] = [ast_utils.get_position(source, x.value)
                                       for x in call.keywords if x.arg == 'val'][0]
        res[assignments['name'].strip("'\"")] = assignments
    return res


def extract_params_s(source):
    """Extract :func:`params` calls from *source*.

    Example:

    >>> source = '''x = X(**params('params_x', a=1, b=2))'''
    >>> extract_params_s(source)  # doctest: +NORMALIZE_WHITESPACE
    {'params_x':
        {'name': "'params_x'",
         'kwargs': {'a': '1', 'b': '2'},
         'use_defaults': 'True',
         'allow_free': 'False',
         'positions': {'a': Position(lineno=1, end_lineno=1, col_offset=29, end_col_offset=30),
                       'b': Position(lineno=1, end_lineno=1, col_offset=34, end_col_offset=35)},
         'end_position': Position(lineno=1, end_lineno=1, col_offset=35, end_col_offset=35)}}


    :param source: The source code from which to extract the :func:`params`-calls.
    :return: A dict mapping :func:`params` arguments (*name*, *use_defaults* and *allow_free*)
        to their string values. Also adds a "kwargs" entry and a "positions" entry with the position
        of the parameter's values in the code.
    """
    res = {}
    for call in var2mod.Finder(ast.Call).find(ast.parse(source)):
        if ast_utils.get_source_segment(source, call.func) not in ('params', 'padl.params'):
            continue
        position = ast_utils.get_position(source, call)
        call_source = sourceget.cut(source, *position, one_indexed=True)
        args, keys, _, _ = symfinder._get_call_signature(call_source)
        signature = symfinder.Signature(argnames=['name', 'use_defaults', 'allow_free'],
                                        kwarg='kwargs',
                                        defaults={'use_defaults': 'True', 'allow_free': 'False'})
        assignments = signature.get_call_assignments(args, keys, dump_kwargs=False)
        if 'kwargs' not in assignments:
            assignments['kwargs'] = {}
        assignments['positions'] = {
            ke.arg: ast_utils.get_position(source, ke.value)
            for ke in call.keywords
            if ke.arg not in ('name', 'use_defaults', 'allow_free')
        }
        arg_end_lineno = max(pos.end_lineno for pos in assignments['positions'].values())
        arg_end_col_offset = max(pos.end_col_offset for pos in assignments['positions'].values()
                                 if pos.end_lineno == arg_end_lineno)
        assignments['end_position'] = ast_utils.Position(arg_end_lineno, arg_end_lineno,
                                                         arg_end_col_offset, arg_end_col_offset)
        res[assignments['name'].strip("'\"")] = assignments
    return res


def change_param(source: str, param_name: str, value: str) -> str:
    """Change the value of a param with name *param_name* in *source* to *value*.

    Note that *value* is the string representation of the value.

    Example:

    >>> source = "x = param('myparam', 1)"
    >>> change_param(source, "myparam", "100")
    "x = param('myparam', 100)"
    """
    param_dict = extract_param_s(source)[param_name]
    return sourceget.replace(source, value, *param_dict['position'], one_indexed=True)


def change_params(source, param_name, **kwargs):
    """Change the value of params with name *param_name* in *source* to values provided as *kwargs*.

    Note that values must be the string representation of the values.

    Example:

    >>> source = "x = params('myparam', a=1, b=2)"
    >>> change_params(source, "myparam", a="100")
    "x = params('myparam', a=100, b=2)"
    """
    params_dict = extract_params_s(source)[param_name]
    kwargs_ = {**params_dict['kwargs'], **kwargs}
    for k, v in kwargs_.items():
        params_dict = extract_params_s(source)[param_name]
        if k in params_dict['kwargs']:
            source = sourceget.replace(source, v, *params_dict['positions'][k], one_indexed=True)
        else:
            if not eval(params_dict['allow_free']):
                raise ValueError(f'You provided param "{param_name}.{k}" which doesn\'t exist.')
            source = sourceget.replace(source, f', {k}={v}', *params_dict['end_position'],
                                       one_indexed=True)
    return source


def parse_dict(source):
    """Parse a string containing a python dict.

    Example:

    >>> parse_dict("{'a': 1, 'b': '2', 'c': random.rand(1000)}")
    {'a': '1', 'b': "'2'", 'c': 'random.rand(1000)'}

    :param source: A string containing a python dict.
    :return: The dict. Note that the values are strings of the values.
    """
    dict_node = ast.parse(source).body[0].value
    return {
        ast_utils.get_source_segment(source, k).strip("'\""): ast_utils.get_source_segment(source, v)
        for k, v in zip(dict_node.keys, dict_node.values)
    }
