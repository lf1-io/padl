"""Testing"""
import os
from typing import List, Dict, Any
from warnings import warn

from numpy.testing import assert_allclose

from lf import transforms as lf


def assert_close(a, b, rtol=1e-7, atol=0):
    """
    Assert that all elements of *a* and *b* are close

    :param a: list, tuple, or array
    :param b: list, tuple, or array
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    """

    if isinstance(a, list) or isinstance(a, tuple):
        for item1, item2 in zip(a, b):
            assert_close(item1, item2, rtol=rtol, atol=atol)
        return
    try:
        assert_allclose(a, b, rtol=rtol, atol=atol)
    except TypeError:
        assert a == b


class TransformTest:
    """A template for testing a Transform. To test a given Transform, inherit from this and set
    the following attributes

    - transforms: The Transform that should be tested.
    - inputs: A list of sample inputs.
    - input_: One sample input (use instead of `inputs` if there's only one sample).
    - outputs: A list of expected outputs when applying the Transform to `inputs`.
    - output: One sample output (use instead of `outputs` if there's only one sample).
    - vars: A dictionary of variables needed for loading the transform.
    - strictly_needs_type: If this is *True*, will test if the input- and output-types of the
        transform are different from lf.Any. If it is *False*, will still check, but raise a warning
        instead of failing a test.

    This will then run a variety of tests on the transforms (different modes, saving loading,
    composition etc.)
    """
    # these need to be defined in inheriting test cases
    transform: lf.Transform = None
    inputs: List = None
    input_: Any = None
    outputs: List = None
    output: Any = None
    vars: Dict = {}
    strictly_needs_type: bool = False

    @property
    def _inputs(self):
        if self.inputs is None:
            return [self.input_]
        return self.inputs

    @property
    def _outputs(self):
        if self.outputs is None:
            return [self.output]
        return self.outputs

    def test_has_input_type(self):
        """Test if the transform has an input type that is not Any. """
        if self.strictly_needs_type:
            assert self.transform.type.x.type.val is not lf.Any
        elif self.transform.type.x.type.val is lf.Any:
            warn('No input type.')

    def test_has_output_type(self):
        """Test if the transform has an output type that is not Any. """
        if self.strictly_needs_type:
            assert self.transform.type.y.type.val is not lf.Any
        elif self.transform.type.y.type.val is lf.Any:
            warn('No output type.')

    def test_apply(self):
        """Test if the Transform output is as expected. """
        self.transform.infer()
        for input_, output in zip(self._inputs, self._outputs):
            assert_close(self.transform(input_, type_check=True), output, rtol=1e-3)

    def test_clone(self):
        """Test if cloning works. """
        self.transform.infer()
        a_clone = self.transform.clone()
        assert self.transform == a_clone
        for input_, output in zip(self._inputs, self._outputs):
            assert_close(a_clone(input_), output, rtol=1e-3)

    def test_to_dict(self):
        """Test if to_dict works as expected. """
        self.transform.infer()
        self.transform.to_dict()

    def test_loads(self):
        """Test if loading works as expected. """
        self.transform.infer()
        my_dict = self.transform.to_dict()
        fd_ = lf.loads(*my_dict)
        fd_.clone()
        for input_, output in zip(self._inputs, self._outputs):
            assert_close(fd_(input_), output, rtol=1e-3)

    def test_save_load(self, tmp_path):
        """Test if saving works as expected. """
        self.transform.infer()
        path = f'{tmp_path}/test.tabc'
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        # save
        lf.save(self.transform, path)
        # load
        tl_ = lf.load(path, **self.vars)
        assert tl_ == self.transform
        for input_, output in zip(self._inputs, self._outputs):
            assert_close(tl_(input_), output, rtol=1e-3)

    def test_trans_forward_post(self):
        """Test if splitting the transforms into its stages works as expected. """
        self.transform.infer()
        split_transforms = (
            self.transform.trans
            >> self.transform.forward
            >> self.transform.postprocess
        )

        for input_, output in zip(self._inputs, self._outputs):
            assert_close(split_transforms(input_), output, rtol=1e-3)

    def test_eval(self):
        """Test if the eval mode works as expected. """
        self.transform.eval()
        for input_, output in zip(self._inputs, self._outputs):
            eval_out = list(self.transform([input_, input_, input_], num_workers=0))
            assert len(eval_out) == 3
            for a_val in eval_out:
                assert_close(a_val, output, rtol=1e-3)
