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

pipeline = obj >> obj + function >> function / function


def makefunction():
    @transform
    def z(x):
        return x
    return z

CONST = 123


def k(y):
    return y


@transform
def y(y):
    return CONST + k(y)


def makeclass():
    @transform
    class MyClassTransform:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def calculate(self, input_):
            return self.a * self.b + input_

        def __call__(self, input_):
            return self.calculate(input_) + y(input_) + self.c(input_)

    return MyClassTransform


def makeclasstransform(a, b, c):
    @transform
    class MyClassTransform:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def calculate(self, input_):
            return self.a * self.b + input_

        def __call__(self, input_):
            return self.calculate(input_) + y(input_) + self.c(input_)

    return MyClassTransform(a, b, c)


def makefunction_squared():
    return makefunction()


def makeclass_squared():
    return makeclasstransform(1, 2, 3)


def wrap_transform():
    return transform(k)


def makelambda():
    return transform(lambda x: x)


@transform
class DeviceCheckInInit:
    def __init__(self, t):
        self.t = t
        self.t.pd_forward_device_check()
