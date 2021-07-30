import os
import pytest
import random

from lf import transforms as lf
from lf import saving as s
from tests.unittests.transforms.test_core import DummyModel


class TestEncoderDecoder:
    @pytest.fixture(scope='class')
    def m(self):
        return (
            lf.to_tensor
            >> lf.GPU(True)
            >> lf.Identity()
            >> lf.CPU(True)
            >> lf.x.item()
        )

    @pytest.fixture(scope='class')
    def to_save(self, m):
        d, v = m.to_dict()
        versions = m.__versions__
        samples = m._samples
        return {
            'metadata': {'__versions__': versions},
            'd': d,
            'v': s.SaveObject(v),
            'samples': s.PickleObject(samples),
        }

    @pytest.fixture(scope='class')
    def big_m(self, m):
        return (
            lf.to_tensor
            >> lf.GPU(True)
            >> lf.Layer(DummyModel(26, 32, 26), layer_name='test')
            >> lf.CPU(True)
        )

    @pytest.fixture
    def tmp_file(self, tmp_path):
        return tmp_path / '_tmp.tabc'

    def test__encode(self):
        s.encoder_decoders_1_0._encode('test')
        s.encoder_decoders_1_0._encode(2)

    def test__decode(self):
        assert (
            s.encoder_decoders_1_0._decode(
                s.encoder_decoders_1_0._encode('test'),
                str,
            ) == 'test'
        )
        assert (
            s.encoder_decoders_1_0._decode(
                s.encoder_decoders_1_0._encode(123),
                int,
            ) == 123
        )

    def test__encode_one(self):
        s.encoder_decoders_1_0._encode_one(
            'test',
            2
        )

    def test__decode_one(self):
        round_trip = s.encoder_decoders_1_0._decode_one(
            *s.encoder_decoders_1_0._encode_one(
                'test',
                2
            )
        )
        assert round_trip == ('test', 2)

    def test_encode(self):
        s.encoder_decoders_1_0.encode({'test': 2})

    def test_decode_header(self):
        h, b = s.encoder_decoders_1_0.decode_header(
            (2).to_bytes(64, 'big') + (3).to_bytes(64, 'big') + (4).to_bytes(64, 'big')
        )
        assert h == [3, 4]
        assert not b
        h, b = s.encoder_decoders_1_0.decode_header(
            (1).to_bytes(64, 'big') + (3).to_bytes(64, 'big') + (4).to_bytes(64, 'big')
        )
        assert h == [3]
        assert b

    def test_decode(self):
        decoded = s.encoder_decoders_1_0.decode(
            s.encoder_decoders_1_0.encode({'test': 25})
        )
        assert decoded == {'test': 25}
        decoded = s.encoder_decoders_1_0.decode(
            s.encoder_decoders_1_0.encode({'test': 25, 'bla': 'foo'})
        )
        assert decoded == {'test': 25, 'bla': 'foo'}
        print(decoded)

    def test_encode_model(self, to_save):
        print(s.encoder_decoders_1_0.encode(to_save))

    def test_decode_model(self, to_save):
        encoded = s.encoder_decoders_1_0.encode(to_save)
        decoded = s.encoder_decoders_1_0.decode(encoded)
        print(decoded)

    def test_get_header_only(self, to_save, tmp_file):
        encoded = s.encoder_decoders_1_0.encode(to_save)
        with open(tmp_file, 'wb') as f:
            f.write(encoded)
        h, _ = s.encoder_decoders_1_0.get_header_only(tmp_file)

    def test_get_keys_only(self, to_save, tmp_file):
        encoded = s.encoder_decoders_1_0.encode(to_save)
        with open(tmp_file, 'wb') as f:
            f.write(encoded)
        keys = s.encoder_decoders_1_0.get_keys_only(tmp_file)
        assert keys == ['metadata', 'd', 'v']

    def test_get_single_value(self, to_save, tmp_file):
        encoded = s.encoder_decoders_1_0.encode(to_save)
        with open(tmp_file, 'wb') as f:
            f.write(encoded)
        meta = s.encoder_decoders_1_0.get_single_value(tmp_file, 'metadata')
        assert (
            next(iter(meta.keys()))
            == '__versions__'
        )

    def test_big_m(self, big_m):
        big_m.infer()
        output = big_m([random.random() for _ in range(26)])
        print(output)

    def test_proper_model(self, big_m, tmp_file):
        d, v = big_m.to_dict()
        x = [random.random() for _ in range(26)]
        y = big_m(x)
        s.save(tmp_file, d=d, v=v)
        output = s.load(tmp_file)
        reloaded = lf.loads(output['d'], output['v'])
        assert (y == reloaded(x)).sum() == len(y)

    def test_original_function(self, big_m, tmp_file):
        big_m.save(tmp_file)
        reloaded = lf.load(tmp_file)
        x = [random.random() for _ in range(26)]
        y = big_m(x)
        assert (y == reloaded(x)).sum() == len(y)

    def test_meta_one(self, big_m, tmp_file):
        big_m.meta_info['mymeta'] = 'yes'
        big_m.save(tmp_file)
        reloaded = lf.load(tmp_file)
        assert reloaded.meta_info['mymeta'] == 'yes'
