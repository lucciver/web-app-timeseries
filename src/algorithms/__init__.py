import importlib
import os.path as osp


def load(name, directory):
    spec = importlib.util.spec_from_file_location(name, directory)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    return foo
