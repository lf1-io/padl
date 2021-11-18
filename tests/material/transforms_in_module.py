from padl import transform


@transform
def function(x):
    return x


@transform
class Class:
    def __init__(self, x):
        self.x = x

    def __call__(self, y):
        return self.x + y


var = 100
obj = Class(var)

compound = obj >> obj + function >> function / function
