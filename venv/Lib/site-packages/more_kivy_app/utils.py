"""Utilities
=============
"""
from kivy.properties import ObservableDict, ObservableList, Property
from functools import partial
from tree_config.utils import yaml_loads, get_yaml as orig_get_yaml, \
    yaml_dumps as orig_yaml_dumps

__all__ = ('get_yaml', 'yaml_dumps', 'yaml_loads')


def represent_property(representer, data: Property):
    return representer.represent_data(data.defaultvalue)


def get_yaml():
    yaml = orig_get_yaml()
    yaml.default_flow_style = False

    yaml.representer.add_multi_representer(
        ObservableList, yaml.representer.__class__.represent_list)
    yaml.representer.add_multi_representer(
        ObservableDict, yaml.representer.__class__.represent_dict)

    yaml.representer.add_multi_representer(Property, represent_property)
    return yaml


yaml_dumps = partial(orig_yaml_dumps, get_yaml_obj=get_yaml)
