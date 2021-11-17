from padl import transform, value, save

x = [1, 2, 3]
LABELS = value(x)


@transform
def test(x):
    return LABELS[x]

