# pylint: disable=no-member

import pytest

from lf.testing import TransformTest
from lf.transforms import random


class TestShuffle(TransformTest):
    @pytest.fixture(autouse=True)
    def init(self):
        self.transform = random.shuffle
        self.inputs = [[1, 2, 3], ['a', 'b', 'c']]
        self.outputs = self.inputs.copy()

    def test_apply(self):
        """
        Test if the
        Transform
        output is as expected.
        """
        self.transform.infer()
        for input_, output in zip(self._inputs, self._outputs):
            assert sorted(self.transform(input_)) == sorted(output)

    def test_clone(self):
        """Test if cloning works. """
        self.transform.infer()
        a_clone = self.transform.clone()
        assert self.transform == a_clone
        for input_, output in zip(self._inputs, self._outputs):
            assert sorted(a_clone(input_)) == sorted(output)

    def test_loads(self):
        """Test if loading works as expected. """
        pass

    def test_save_load(self, tmp_path):
        """Test if saving works as expected. """
        pass

    def test_trans_forward_post(self):
        """Test if splitting the transforms into its stages works as expected. """
        self.transform.infer()
        split_transforms = (
                self.transform.trans
                >> self.transform.forward
                >> self.transform.postprocess
        )

        for input_, output in zip(self._inputs, self._outputs):
            assert sorted(split_transforms(input_)) == sorted(output)

    def test_eval(self):
        """Test if the eval mode works as expected. """
        pass

