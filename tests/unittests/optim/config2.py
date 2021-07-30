from tests.unittests.optim import config1 as cf


def data():
    return cf.td, cf.vd, cf.gt


def model():
    return cf.m, cf.tm[1:]


def metrics():
    return {'precision': cf.precision}
