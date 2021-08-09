"""Configuration
=================

:mod:`tree-config` provides the tools to configure applications where the
objects to be configured are nested in a tree-like fashion such that they
can be represented as nested dicts in e.g. a yaml file.
"""

__version__ = '0.1.0'


from typing import Tuple, List, Dict, Any
import os
from inspect import isclass

from .utils import get_class_bases, yaml_dumps, yaml_loads, \
    get_class_annotations, class_property

__all__ = (
    'Configurable', 'read_config_from_object', 'apply_config',
    'read_config_from_file', 'load_config', 'dump_config',
    'load_apply_save_config', 'get_config_prop_names', 'get_config_prop_items',
    'get_config_children_names', 'get_config_children_items')


class Configurable:

    _config_props_: Tuple[str] = ()

    _config_props_cache: List[str] = None

    _config_children_: Dict[str, str] = {}

    _config_children_cache: Dict[str, str] = None

    @class_property
    def _config_props(cls) -> List[str]:
        props = cls.__dict__.get('_config_props_cache', None)
        if props is None:
            cls._config_props_cache = props = _get_config_prop_names(cls)
        return props

    @class_property
    def _config_children(cls) -> Dict[str, str]:
        children = cls.__dict__.get('_config_children_cache', None)
        if children is None:
            cls._config_children_cache = children = _get_config_children_names(
                cls)
        return children

    def apply_config_child(self, name, prop, obj, config):
        """If it configures the object, it must either call apply_config that
        implicitly dispatches :meth:`post_config_applied`, or it must be called
        directly.

        """
        apply_config(obj, config)

    def get_config_property(self, name):
        return getattr(self, name)

    def apply_config_property(self, name, value):
        setattr(self, name, value)

    def post_config_applied(self):
        pass


def get_config_prop_items(obj_or_cls, get_attr=getattr) -> Dict[str, Any]:
    get_config_property = getattr(obj_or_cls, 'get_config_property', None)
    if get_config_property is None or isclass(obj_or_cls):
        return {prop: get_attr(obj_or_cls, prop)
                for prop in get_config_prop_names(obj_or_cls)}

    return {prop: get_config_property(prop)
            for prop in get_config_prop_names(obj_or_cls)}


def get_config_prop_names(obj_or_cls) -> List[str]:
    props = getattr(obj_or_cls, '_config_props', None)
    if props is not None:
        return props
    return _get_config_prop_names(obj_or_cls)


def _get_config_prop_names(obj_or_cls) -> List[str]:
    cls = obj_or_cls
    if not isclass(obj_or_cls):
        cls = obj_or_cls.__class__

    props = {}
    for c in [cls] + list(get_class_bases(cls)):
        for prop in c.__dict__.get('_config_props_', ()):
            if prop in props:
                continue

            if not hasattr(cls, prop):
                raise Exception('Missing attribute <{}> in <{}>'.
                                format(prop, cls.__name__))
            props[prop] = None

    return list(props)


def get_config_children_items(
        obj_or_cls, get_attr=getattr) -> List[Tuple[str, str, Any]]:
    return [
        (name, prop, get_attr(obj_or_cls, prop))
        for name, prop in get_config_children_names(obj_or_cls).items()
    ]


def get_config_children_names(obj_or_cls) -> Dict[str, str]:
    children = getattr(obj_or_cls, '_config_children', None)
    if children is not None:
        return children
    return _get_config_children_names(obj_or_cls)


def _get_config_children_names(obj_or_cls) -> Dict[str, str]:
    cls = obj_or_cls
    if not isclass(obj_or_cls):
        cls = obj_or_cls.__class__

    children = {}
    for c in [cls] + list(get_class_bases(cls)):
        for name, prop in c.__dict__.get('_config_children_', {}).items():
            if name in children:
                continue

            if not hasattr(cls, prop):
                raise Exception('Missing attribute <{}> in <{}>'.
                                format(prop, cls.__name__))
            children[name] = prop

    return children


def _fill_config_from_children(children, config: dict, get_attr):
    for name, prop, obj in children:
        if obj is None:
            continue

        if isinstance(obj, (list, tuple)):
            config[name] = [read_config_from_object(o, get_attr) for o in obj]
        else:
            config[name] = read_config_from_object(obj, get_attr)


def read_config_from_object(obj, get_attr=getattr):
    # TODO: break infinite cycle if obj is listed in its nested config classes
    config = {}

    # get all the configurable classes used by the obj
    children = get_config_children_items(obj, get_attr)
    _fill_config_from_children(children, config, get_attr)

    used_keys = {s for name, prop, obj in children for s in (name, prop)}

    for prop, value in get_config_prop_items(obj, get_attr).items():
        if prop not in used_keys:
            config[prop] = value
    return config


def _apply_config_to_children(root_obj, children, config, get_attr, set_attr):
    for name, prop, obj in children:
        if obj is None or name not in config:
            continue

        # todo: handle when obj is a list/dict
        method = getattr(root_obj, 'apply_config_child', None)
        if method is None:
            apply_config(obj, config[name], get_attr, set_attr)
        else:
            method(name, prop, obj, config[name])


def apply_config(obj, config: dict, get_attr=getattr, set_attr=setattr):
    """Takes the config data read with :func:`read_config_from_object`
    or :func:`read_config_from_file` and applies
    them to any existing class instances listed in classes.

    Calls :func:`post_config_applied` after object is configured.
    """
    # get all the configurable classes used by the obj
    children = get_config_children_items(obj, get_attr)
    _apply_config_to_children(obj, children, config, get_attr, set_attr)

    used_keys = {s for name, prop, obj in children for s in (name, prop)}

    method = getattr(obj, 'apply_config_property', None)
    if method is None:
        for k, v in config.items():
            if k not in used_keys:
                set_attr(obj, k, v)
    else:
        for k, v in config.items():
            if k not in used_keys:
                method(k, v)

    post_config_applied = getattr(obj, 'post_config_applied', None)
    if post_config_applied is not None:
        post_config_applied()


def read_config_from_file(filename, yaml_load_str=yaml_loads):
    """Reads the config file and loads all the config data.

    The config data is returned as a dict. If there's an error, an empty dict
    is returned.
    """
    with open(filename) as fh:
        opts = yaml_load_str(fh.read())
    if opts is None:
        opts = {}
    return opts


def load_config(
        obj, filename, get_attr=getattr, yaml_dump_str=yaml_dumps,
        yaml_load_str=yaml_loads):
    if not os.path.exists(filename):
        dump_config(
            filename, read_config_from_object(obj, get_attr),
            yaml_dump_str=yaml_dump_str)

    return read_config_from_file(filename, yaml_load_str=yaml_load_str)


def dump_config(filename, data, yaml_dump_str=yaml_dumps):
    with open(filename, 'w') as fh:
        fh.write(yaml_dump_str(data))


def load_apply_save_config(
        obj, filename, get_attr=getattr, set_attr=setattr,
        yaml_dump_str=yaml_dumps, yaml_load_str=yaml_loads):
    config = load_config(
        obj, filename, get_attr, yaml_dump_str=yaml_dump_str,
        yaml_load_str=yaml_load_str)
    apply_config(obj, config, get_attr, set_attr)

    config = read_config_from_object(obj, get_attr)
    dump_config(filename, config, yaml_dump_str=yaml_dump_str)
    return config
