from padl import transform


@transform
class MyTransform:
    def __init__(self, x):
        self.x = x

    def __call__(self, y):
        return self.x + y


def make(z):
    return MyTransform(x=z)


def getclass():
    return MyTransform
