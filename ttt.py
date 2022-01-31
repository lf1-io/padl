import numpy
from padl import transform


@transform
def rescale_image_tensor(x):
    return (255 * (x * 0.5 + 0.5)).numpy().astype(numpy.uint8)


_pd_main = rescale_image_tensor

