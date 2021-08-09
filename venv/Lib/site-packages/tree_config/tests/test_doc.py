import pytest
import subprocess
import os
import pathlib
from ..utils import yaml_loads

conf_contents = '''
import os
from functools import partial

from tree_config.doc_gen import create_doc_listener, write_config_props_rst

project = 'TestProject'
extensions = ['sphinx.ext.autodoc']


def setup(app):
    fname = os.environ['TREE_CONFIG_DOC_YAML_PATH']
    rst_filename = os.environ['TREE_CONFIG_DOC_RST_PATH']
    create_doc_listener(app, 'tree_config', fname)
    {}

    app.connect(
        'build-finished', partial(
            write_config_props_rst, obj, 'TestProject', filename=fname,
            rst_filename=rst_filename)
    )

'''

doc_index_contents = '''
Welcome to tree-config's documentation!
=======================================

Contents:

{}
'''

doc_autodoc_item = '''
.. automodule:: {}
   :members:
'''


class ClassWithSometimesNiceProps:
    """A class"""
    _config_props_ = ('prop_a', 'prop_b')
    prop_a = 1
    """A very nice property.
    """
    prop_b = 2
    """Can be a nice property.
    Sometimes it isn't nice though.
    """


class ClassWithChildrenThatHaveSometimesNiceProperties:
    """A class"""
    _config_props_ = ('prop_c', )
    _config_children_ = {'a child': 'child_a', 'b child': 'child_b'}
    prop_c = 3
    """This is truly a nicely done property"""
    child_a: ClassWithSometimesNiceProps = None
    child_b: ClassWithSometimesNiceProps = None

    def __init__(self):
        self.child_a = ClassWithSometimesNiceProps()
        self.child_b = ClassWithSometimesNiceProps()


def config_doc_files(
        tmp_path: pathlib.Path, conf_obj: list, auto_doc_mods: list):
    source = tmp_path / 'source'
    build = tmp_path / 'build'
    yaml_file = tmp_path / 'config_props.yaml'
    rst_file = tmp_path / 'config_props.rst'

    if not source.exists():
        source.mkdir()

    conf = conf_contents.format('\n    '.join(conf_obj))
    (source / 'conf.py').write_text(conf)

    items = [doc_autodoc_item.format(mod) for mod in auto_doc_mods]
    auto_doc_items = '\n\n'.join(items)
    index_rst = doc_index_contents.format(auto_doc_items)
    (source / 'index.rst').write_text(index_rst)

    env = os.environ.copy()
    env['TREE_CONFIG_DOC_YAML_PATH'] = str(yaml_file)
    env['TREE_CONFIG_DOC_RST_PATH'] = str(rst_file)

    try:
        subprocess.check_output(
            ['sphinx-build', str(source), str(build)],
            stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        print(e.output.decode('utf8'))
        raise

    assert yaml_file.exists()
    assert rst_file.exists()
    return yaml_file, rst_file


rst_file_contents_class_width_children = '''
TESTPROJECT Config
==================

The following are the configuration options provided by the app. It can \
be configured by changing appropriate values in the ``config.yaml`` settings \
file. The options default to the default value of the classes for each of the \
options.

`prop_c`: 3
 This is truly a nicely done property


b child
-------

`prop_a`: 1
 A very nice property.

`prop_b`: 2
 Can be a nice property.
 Sometimes it isn't nice though.


a child
-------

`prop_a`: 1
 A very nice property.

`prop_b`: 2
 Can be a nice property.
 Sometimes it isn't nice though.
 '''.strip()


@pytest.mark.parametrize(
    'obj', [
        ['from tree_config.tests.test_doc import '
         'ClassWithChildrenThatHaveSometimesNiceProperties',
         'obj = ClassWithChildrenThatHaveSometimesNiceProperties'],
        ['from tree_config.tests.test_doc import '
         'ClassWithChildrenThatHaveSometimesNiceProperties',
         'obj = ClassWithChildrenThatHaveSometimesNiceProperties()']])
def test_single_mod_doc(tmp_path, obj):
    yaml_file, rst_file = config_doc_files(
        tmp_path, obj, ['tree_config.tests.test_doc'])
    yaml_doc = yaml_loads(yaml_file.read_text())

    assert yaml_doc == {
        "tree_config.tests.test_doc."
        "ClassWithChildrenThatHaveSometimesNiceProperties": {
            "prop_c": [
                "This is truly a nicely done property",
                ""
            ]
        },
        "tree_config.tests.test_doc.ClassWithSometimesNiceProps": {
            "prop_a": [
                "A very nice property.",
                "    ",
                ""
            ],
            "prop_b": [
                "Can be a nice property.",
                "Sometimes it isn't nice though.",
                ""
            ]
        }
    }
    rst_doc = rst_file.read_text().strip()
    assert rst_doc == rst_file_contents_class_width_children

    # now update the yaml file by adding a new module
    yaml_file, rst_file = config_doc_files(
        tmp_path, obj, ['tree_config.tests.doc_dummy_class'])

    yaml_doc = yaml_loads(yaml_file.read_text())
    assert yaml_doc == {
        "tree_config.tests.doc_dummy_class.RootClass": {
            "prop_z": [
                "This is the very last property ever.",
                "    ",
                ""
            ]
        },
        "tree_config.tests.test_doc."
        "ClassWithChildrenThatHaveSometimesNiceProperties": {
            "prop_c": [
                "This is truly a nicely done property",
                ""
            ]
        },
        "tree_config.tests.test_doc.ClassWithSometimesNiceProps": {
            "prop_a": [
                "A very nice property.",
                "    ",
                ""
            ],
            "prop_b": [
                "Can be a nice property.",
                "Sometimes it isn't nice though.",
                ""
            ]
        }
    }

    # but rst should be unchanged because the object passed to conf is the same
    rst_doc = rst_file.read_text().strip()
    assert rst_doc == rst_file_contents_class_width_children
