import padl


def create_transform(a, b):
    return padl.transform(lambda x: x * a + b)


def combine_transforms(t1, t2):
    return t1 >> t2
