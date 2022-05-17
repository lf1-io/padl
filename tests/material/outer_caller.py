from padl.dumptools import inspector


def here():
    return inspector.outer_caller_frameinfo(__name__)


def caller_module():
    return inspector.caller_module()


def caller_frame():
    return inspector.caller_frame()
