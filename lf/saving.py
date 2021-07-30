"""Saving utilities"""
import io
import json
import os
import pickle
import torch

from lf.variables import Variable


class DeprecationError(Exception):
    ...


class PickleObject:
    """Object to wrap pickle-able objects with."""

    def __init__(self, value):
        self.value = value


class SaveObject:
    """Object to wrap recursively encoded dictionaries with."""

    def __init__(self, value):
        self.value = value


class EncoderDecoder:
    """
    Encoder/ decoder object for saving dictionaries with certain important objects inside.

    :param encoder_dictionary: dictionary containing mapping from types to byte encoders
    :param decoder_dictionary: dictionary containing mapping from types to byte decoders
    :param type_dictionary: dictionary containing mapping from types to type strings
    """
    def __init__(self, encoder_dictionary, decoder_dictionary, type_dictionary):
        self.encoder_dictionary = encoder_dictionary
        self.decoder_dictionary = decoder_dictionary
        self.type_dictionary = type_dictionary
        self.type_lookup = dict(zip(type_dictionary.values(), type_dictionary.keys()))

    def _encode(self, x):
        """Encode *x* with determined type"""
        type_ = self._determine_type(x)
        return self.encoder_dictionary[type_](x)

    def _decode(self, x, t):
        """Decode *x* with type *t*"""
        return self.decoder_dictionary[t](x)

    def _encode_one(self, k, v):
        if isinstance(v, SaveObject):
            out = (
                self._encode(k),
                self._encode('SaveObject'),
                self.encode(v.value),
            )
        else:
            out = (
                self._encode(k),
                self._encode(self.type_dictionary[self._determine_type(v)]),
                self._encode(v),
            )
        return out

    def _determine_type(self, o):
        """Determine the type of *o*"""
        type_ = None
        for k in self.type_dictionary:
            if isinstance(o, k):
                type_ = k
                break
        if type_ is None:
            raise Exception(
                f'Unknown type {type_} to this converter set. Did you forget to specify '
                f'`dont_store=True` in your variable?'
            )
        return type_

    def _decode_one(self, k, t, v):
        k = self._decode(k, str)
        if self._decode(t, str) == 'SaveObject':
            t = SaveObject
        else:
            t = self.type_lookup[self._decode(t, str)]
        if t == SaveObject:
            v = self.decode(v)
        else:
            v = self._decode(v, t)
        return k, v

    def encode(self, d):
        _k = []
        _t = []
        _v = []
        keys = list(d.keys())
        for k in keys:
            _kk, _tt, _vv = self._encode_one(k, d[k])
            _k.append(_kk)
            _t.append(_tt)
            _v.append(_vv)
        hl_ = self._encode(len(_k) * 3)
        sofar = 0
        _h = b''
        _b = b''
        for i in range(len(_k)):
            _tmp = _k[i]
            sofar += len(_tmp)
            _h += self._encode(sofar)
            _b += _tmp
        for i in range(len(_t)):
            _tmp = _t[i]
            sofar += len(_tmp)
            _h += self._encode(sofar)
            _b += _tmp
        for i in range(len(_v)):
            _tmp = _v[i]
            sofar += len(_tmp)
            _h += self._encode(sofar)
            _b += _tmp
        return hl_ + _h + _b

    def decode_header(self, bytes_):
        header_len = self.decoder_dictionary[int](bytes_[:64])
        bytes_ = bytes_[64:]
        header = []
        for i in range(header_len):
            header.append(self.decoder_dictionary[int](bytes_[i * 64: (i + 1) * 64]))
        return header, bytes_[64 * header_len:]

    def decode(self, bytes_):
        out = {}
        header, bytes_ = self.decode_header(bytes_)
        assert len(header) % 3 == 0
        if not header:
            return {}

        l_ = len(header) // 3
        header = [0, *header]
        for i in range(len(header) // 3):
            k = bytes_[header[i]: header[i + 1]]
            t = bytes_[header[l_ + i]: header[l_ + i + 1]]
            v = bytes_[header[2 * l_ + i]: header[2 * l_ + i + 1]]
            k, v = self._decode_one(k, t, v)
            out[k] = v
        return out

    def get_header_only(self, fp):
        if os.path.isfile(fp):
            with open(fp, 'rb') as file1:
                a_line = self.decoder_dictionary[int](file1.read(64))
                file1.seek(64)
                content = file1.read(a_line * 64)
        elif str(fp).startswith('s3://'):
            from s3fs import S3FileSystem
            with S3FileSystem().open(fp, 'rb') as file1:
                a_line = self.decoder_dictionary[int](file1.read(64))
                file1.seek(64)
                content = file1.read(a_line * 64)
        else:
            raise FileNotFoundError(f'couldnt find {fp}')
        header = [
            self.decoder_dictionary[int](content[i * 64: (i + 1) * 64])
            for i in range(a_line)
        ]
        return header, (a_line + 1) * 64

    def get_keys_only(self, fp):
        header, start = self.get_header_only(fp)
        assert (len(header) % 3) == 0
        if os.path.isfile(fp):
            with open(fp, 'rb') as file1:
                file1.seek(start)
                last_header = len(header) // 3
                content = file1.read(header[last_header])
        elif str(fp).startswith('s3://'):
            from s3fs import S3FileSystem
            with S3FileSystem().open(fp, 'rb') as file1:
                file1.seek(start)
                last_header = len(header) // 3
                content = file1.read(header[last_header])
        else:
            raise FileNotFoundError(f'cant find {fp}')
        keys = []
        header = [0, *header]
        for i in range(len(header) // 3 - 1):
            keys.append(
                content[header[i]:header[i + 1]].decode('utf-8')
            )
        return keys

    def get_single_value(self, fp, key):
        header, start = self.get_header_only(fp)
        keys = self.get_keys_only(fp)
        try:
            position = [i for i, k in enumerate(keys) if k == key][0]
        except IndexError:
            raise Exception('key not found in the file.')
        type_position = (len(header) // 3) + position

        if os.path.isfile(fp):
            with open(fp, 'rb') as file1:
                file1.seek(start + header[type_position - 1])
                _tmp = file1.read(header[type_position] - header[type_position - 1])
        elif str(fp).startswith('s3://'):
            from s3fs import S3FileSystem
            with S3FileSystem().open(fp, 'rb') as file1:
                file1.seek(start + header[type_position - 1])
                _tmp = file1.read(header[type_position] - header[type_position - 1])
        else:
            raise FileNotFoundError(f'couldnt find {fp}')

        type_ = _tmp.decode('utf-8')
        type_ = self.type_lookup[type_]
        value_position = 2 * (len(header) // 3) + position
        if os.path.isfile(fp):
            with open(fp, 'rb') as file1:
                file1.seek(start + header[value_position - 1])
                _tmp = file1.read(header[value_position] - header[value_position - 1])
        elif str(fp).startswith('s3://'):
            from s3fs import S3FileSystem
            with S3FileSystem().open(fp, 'rb') as file1:
                file1.seek(start + header[value_position - 1])
                _tmp = file1.read(header[value_position] - header[value_position - 1])
        else:
            raise FileNotFoundError(f'couldnt find {fp}')
        return self.decoder_dictionary[type_](_tmp)


def _get_torch_buffer(x):
    # for some reason you do it like this
    buffer = io.BytesIO()
    torch.jit.save(x, buffer)
    # get the bytes
    return buffer.getvalue()


encoder_decoders_1_0 = EncoderDecoder(
    encoder_dictionary={
        str: lambda x: bytes(x, 'utf-8'),
        int: lambda x: x.to_bytes(64, 'big'),
        bytes: lambda x: x,
        dict: lambda x: bytes(json.dumps(x), 'utf-8'),
        list: lambda x: bytes(json.dumps(x), 'utf-8'),
        torch.jit.ScriptModule: lambda x: _get_torch_buffer(x),
        PickleObject: lambda x: pickle.dumps(x.value),
    },
    decoder_dictionary={
        str: lambda x: x.decode('utf-8'),
        int: lambda x: int.from_bytes(x, 'big'),
        bytes: lambda x: x,
        dict: lambda x: json.loads(x.decode('utf-8')),
        list: lambda x: json.loads(x.decode('utf-8')),
        torch.jit.ScriptModule: lambda x: torch.jit.load(io.BytesIO(x)),
        PickleObject: lambda x: pickle.loads(x),
    },
    type_dictionary={
        str: 'str',
        int: 'int',
        bytes: 'bytes',
        dict: 'dict',
        list: 'list',
        torch.jit.ScriptModule: 'torch.jit.ScriptModule',
        PickleObject: 'PickleObject',
    }
)

basic_encoders_decoders = EncoderDecoder(
    encoder_dictionary={
        str: lambda x: bytes(x, 'utf-8'),
        int: lambda x: x.to_bytes(64, 'big'),
        bytes: lambda x: x,
        dict: lambda x: bytes(json.dumps(x), 'utf-8'),
        list: lambda x: bytes(json.dumps(x), 'utf-8'),
    },
    decoder_dictionary={
        str: lambda x: x.decode('utf-8'),
        int: lambda x: int.from_bytes(x, 'big'),
        bytes: lambda x: x,
        dict: lambda x: json.loads(x.decode('utf-8')),
        list: lambda x: json.loads(x.decode('utf-8')),
    },
    type_dictionary={
        str: 'str',
        int: 'int',
        bytes: 'bytes',
        dict: 'dict',
        list: 'list',
    }
)


def save(file_, d, v, metadata=None, samples=None, version='1.0'):
    """
    Save definition and variables

    :param file_: path to file to save
    :param d: definition dictionary
    :param v: variable dictionary
    :param metadata: metadata dictionary
    :param samples: input/ output samples
    :param version: version of saver with which to save
    :return:
    """
    if metadata is None:
        metadata = {}
    if version == '1.0':
        content = encoder_decoders_1_0.encode({
            'meta': metadata,
            'version': '1.0',
            'd': d,
            'v': SaveObject(v),
            'samples': PickleObject(samples),
        })
    else:
        raise NotImplementedError

    if str(file_).startswith('s3://'):
        from s3fs import S3FileSystem
        with S3FileSystem().open(file_, 'wb') as file1:
            file1.write(content)

    with open(file_, 'wb') as file1:
        file1.write(content)


def load(file_):
    """
    Load nested/ typed dictionary from file using the versioned converters.

    :param file_: file from which to read bytes

    :return: nested dictionary
    """
    try:
        version = basic_encoders_decoders.get_single_value(file_, 'version')
        print(f'found saver/ loader version {version}')
    except Exception:
        raise DeprecationError('couldnt find a versioned loader.')

    if os.path.isfile(file_):
        with open(file_, 'rb') as file1:
            bytes_ = file1.read()
    elif str(file_).startswith('s3://'):
        from s3fs import S3FileSystem
        with S3FileSystem().open(file_, 'rb') as file1:
            bytes_ = file1.read()
    else:
        raise FileNotFoundError(f'cant find "{file_}"')

    if version == '1.0':
        out = encoder_decoders_1_0.decode(bytes_)
        return out
    raise NotImplementedError
