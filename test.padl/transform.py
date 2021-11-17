from padl import transform


GLOBAL_1 = 0
GLOBAL_1 = GLOBAL_1 + 5


@transform
def plus_global(x):
    return x + GLOBAL_1


_pd_main = plus_global
