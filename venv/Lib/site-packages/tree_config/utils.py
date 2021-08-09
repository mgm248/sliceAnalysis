"""Utils
========

"""
from inspect import isclass
from io import StringIO
from ruamel.yaml import YAML
from typing import Dict, Any

__all__ = (
    'get_yaml', 'yaml_dumps', 'yaml_loads', 'get_class_bases',
    'get_class_annotations', 'class_property')


def get_class_bases(cls):
    """Gets all the base-classes of the class.
    :param cls:
    :return:
    """
    for base in cls.__bases__:
        if base.__name__ == 'object':
            break
        for cbase in get_class_bases(base):
            yield cbase
        yield base


def get_class_annotations(obj_or_cls) -> Dict[str, Any]:
    cls = obj_or_cls
    if not isclass(obj_or_cls):
        cls = obj_or_cls.__class__

    annotations = {}
    for c in [cls] + list(get_class_bases(cls)):
        annotations.update(getattr(c, '__annotations__', {}))
    return annotations


def get_yaml():
    yaml = YAML(typ='safe')
    return yaml


def yaml_dumps(value, get_yaml_obj=get_yaml):
    yaml = get_yaml_obj()
    s = StringIO()
    yaml.dump(value, s)
    return s.getvalue()


def yaml_loads(value, get_yaml_obj=get_yaml):
    yaml = get_yaml_obj()
    return yaml.load(value)


class class_property(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()
