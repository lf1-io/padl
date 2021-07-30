"""Quantize"""
import torch
from tqdm import tqdm
from lf.utils import stage


def quantize_layer(t, layer_name, data, batch_size=100):
    """
    Quantizes layer *layer_name* in transform *t*. Some input data is needed to gather stats for
    quantization.

    Note that not all layers can be quantized and some preparation is needed.

    Have a look at https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html and
    https://pytorch.org/docs/stable/quantization.html.

    :param t: Transform.
    :param layer_name: Name of layer to be quantized.
    :param data: Data for quantization initialization.
    :param batch_size: Batch size.
    """
    layer = t.layers[layer_name]

    assert t.device == 'cpu'

    # fuse layers if possible
    try:
        layer.fuse_model()
    except AttributeError:
        pass

    # prepare for quantization
    layer.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(layer, inplace=True)

    # pass data through the model to gather quantization stats
    print('Feeding data through model. This might take a while...')
    with stage(t, 'eval'):
        for _ in tqdm(t.preprocess(data, batch_size=batch_size, shuffle=True, flatten=False),
                      total=len(data) // batch_size):
            pass

    print('Converting layer...')
    torch.quantization.convert(layer, inplace=True)
    print('Done!')
    return layer
