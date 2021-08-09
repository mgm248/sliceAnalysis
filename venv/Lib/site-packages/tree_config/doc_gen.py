"""Documentation generation
============================

Overview
---------

Configuration works as follows. Each class that has configuration attributes
must list these attributes in a list in the class ``__config_props__``
attribute. Each of the properties listed there must be "Kivy" properties of
that class.

When generating docs, the documentation of these properties are dumped to
a yaml file using :func:`create_doc_listener`.

Each app instance defines a application class based on
:class:`~base_kivy_app.app.BaseKivyApp`. Using this classe's
:meth:`~base_kivy_app.app.BaseKivyApp.get_config_classes` method we get a list
of all classes used in the current app that requires configuration
and :func:`write_config_props_rst` is used to combine all these docs
and display them in a single place in the generated html pages.

Similarly, when the app is run, a single yaml file is generated with all these
config values and is later read and is used to configure the app by the
user. :attr:`~base_kivy_app.app.BaseKivyApp.app_settings` is where it's stored
after reading. Each class is responsible for reading its configuration
from there.

Usage
-----

When creating an app, ensure that the app
inherited from :class:`~base_kivy_app.app.BaseKivyApp` overwrites the
:meth:`~base_kivy_app.app.BaseKivyApp.get_config_classes` method
returning all the classes that need configuration.

Then, in the sphinx conf.py file do::

    def setup(app):
        import package
        from package import MyApp
        fname = os.environ.get(
            'BASEKIVYAPP_CONFIG_DOC_PATH', 'config_attrs.yaml')
        create_doc_listener(app, package, fname)
        if MyApp.get_running_app() is not None:
            classes = MyApp.get_running_app().get_config_instances()
        else:
            classes = MyApp.get_config_classes()

        app.connect(
            'build-finished', partial(
                write_config_props_rst, classes, package, filename=fname)
        )

and run `make html` twice. This will create the ``config_attrs.yaml`` file
and the config.rst file under source/. This
config.rst should have been listed in the sphinx index so on the second run
this file will be converted to html containing all the config tokens.

You should similarly run doc generation for all packages the your package
relies on so you get all their config options in ``config_attrs.yaml``
so they can be included in the docs.
"""

from inspect import isclass
import os.path
from typing import Tuple, List, Dict, Any
import operator
import urllib.request
from collections import deque

from .utils import get_class_bases, get_class_annotations, yaml_loads, \
    yaml_dumps
from tree_config import get_config_children_names

__all__ = (
    'create_doc_listener', 'write_config_props_rst', 'download_merge_yaml_doc',
    'merge_yaml_doc',
)


def _get_config_children_objects(
        obj_or_cls, get_attr=getattr) -> List[Tuple[str, str, Any]]:
    annotations = get_class_annotations(obj_or_cls)
    objects = []
    for name, prop in get_config_children_names(obj_or_cls).items():
        obj = get_attr(obj_or_cls, prop)

        if obj is None:
            # try searching annotations for type
            obj = annotations.get(prop, None)
            if obj is not None:
                # only accept typing generic that is a list/tuple
                args = getattr(obj, '__args__', None)
                if getattr(obj, '__origin__', None) in (list, tuple) and \
                        args is not None and isinstance(args, (list, tuple)) \
                        and len(args) == 1:
                    obj = args[0]

        if obj is not None:
            objects.append((name, prop, obj))
    return objects


def _get_config_prop_items_class(
        obj_or_cls, get_attr=getattr) -> Dict[str, List[Tuple[str, Any]]]:
    cls = obj_or_cls
    if not isclass(obj_or_cls):
        cls = obj_or_cls.__class__

    classes = {}
    for c in [cls] + list(get_class_bases(cls)):
        cls_name = f'{c.__module__}.{c.__name__}'
        cls_props = {}
        for prop in c.__dict__.get('_config_props_', ()):
            if prop in cls_props:
                continue

            if not hasattr(cls, prop):
                raise Exception('Missing attribute <{}> in <{}>'.
                                format(prop, cls.__name__))

            cls_props[prop] = get_attr(cls, prop)

        if cls_props:
            classes[cls_name] = list(cls_props.items())

    return classes


def create_doc_listener(
        sphinx_app, package_name, filename, yaml_dump_str=yaml_dumps):
    """Creates a listener for the ``__config_props__`` attributes and dumps
    the docs of any props listed to ``filename``. If the file
    already exists, it extends it with new data and overwrites any exiting
    properties that we see again in this run.

    To use, in the sphinx conf.py file do::

        def setup(app):
            import package
            create_doc_listener(
                app, package,
                os.environ.get(
                    'BASEKIVYAPP_CONFIG_DOC_PATH', 'config_attrs.yaml')
            )

    where ``package`` is the module for which the docs are generated.
    """
    try:
        with open(filename) as fh:
            data = yaml_loads(fh.read())
    except IOError:
        data = {}

    def config_attrs_doc_listener(app, what, name, obj, options, lines):
        if not name.startswith(package_name):
            return

        if what == 'class':
            props = obj.__dict__.get('_config_props_', ())
            # ew haven't seen this class before
            if name not in data:
                # just add items for all props
                data[name] = {n: [] for n in props}
            else:
                # we have seen this class, add only missing props
                cls_data = data[name]
                for n in props:
                    if n not in cls_data:
                        cls_data[n] = []
        elif what == 'attribute':
            # now that we saw the class, we fill in the prop docs
            parts = name.split('.')  # parts of the prop path x.Class.prop
            cls = '.'.join(parts[:-1])  # full class path x.Class
            # we have seen the class and prop?
            if cls in data and parts[-1] in data[cls]:
                data[cls][parts[-1]] = lines

    def dump_config_attrs_doc(app, exception):
        # dump config docs
        filtered_data = {
            name: {n: lines for n, lines in props.items() if lines}
            for name, props in data.items()}
        filtered_data = {
            name: props for name, props in filtered_data.items() if props}

        with open(filename, 'w') as fh_:
            fh_.write(yaml_dump_str(filtered_data))

    sphinx_app.connect('autodoc-process-docstring', config_attrs_doc_listener)
    sphinx_app.connect('build-finished', dump_config_attrs_doc)


def _walk_config_classes_flat(obj, get_attr=getattr):
    classes_flat = []  # stores all the configurable classes
    stack = deque([(-1, '', obj)])

    while stack:
        level, name, obj = stack.popleft()
        # now we "visited" obj
        classes_flat.append((level, name, obj, {}, {}))

        children = _get_config_children_objects(obj, get_attr=get_attr)

        for child_name, child_prop, child_obj in sorted(
                children, key=lambda x: x[0]):
            if isinstance(child_obj, (list, tuple)):
                for i, o in enumerate(child_obj):
                    stack.appendleft((
                        level + 1, '{} --- {}'.format(child_name, i), o))
            else:
                stack.appendleft((level + 1, child_name, child_obj))

    assert len(classes_flat) >= 1
    return classes_flat


def _get_config_attrs_doc(obj, filename, get_attr=getattr):
    """Objects is a dict of object (class) paths and keys are the names of the
    config attributes of the class.
    """
    classes_flat = _walk_config_classes_flat(obj, get_attr=get_attr)

    # get the modules associated with each of the classes
    for _, _, obj, classes_props, _ in classes_flat:
        # get all the parent classes of the class and their props
        for cls, props in _get_config_prop_items_class(obj).items():
            classes_props[cls] = {prop: [val, None] for prop, val in props}

    # get the saved docs
    with open(filename) as fh:
        docs = yaml_loads(fh.read())

    # mapping of class name to a mapping of class props to their docs
    for level, name, obj, classes_props, props_docs in classes_flat:
        for cls, props in classes_props.items():
            cls_docs = docs.get(cls, {})
            for prop in props:
                props[prop][1] = cls_docs.get(prop, [])
                props_docs[prop] = props[prop]
    return classes_flat


def write_config_props_rst(
        obj, project, app, exception, filename, rst_filename,
        get_attr=getattr):
    """Walks through all the configurable classes of this package
    (should be gotten from
    :meth:`~base_kivy_app.app.BaseKivyApp.get_config_classes` or
    :meth:`~base_kivy_app.app.BaseKivyApp.get_config_instances`) and loads the
    docs of those properties and generates a rst output file with all the
    tokens.

    For example in the sphinx conf.py file do::

        def setup(app):
            app.connect('build-finished', partial(write_config_props_rst, \
ProjectApp.get_config_classes(), project_name))

    where project_name is the project module and ProjectApp is the App of the
    package.
    """
    headings = [
        '-', '`', ':', "'", '"', '~', '^', '_', '*', '+', '#', '<', '>']
    n = len(headings) - 1

    # get the docs for the props
    classes_flat = _get_config_attrs_doc(obj, filename, get_attr)

    header = '{} Config'.format(project.upper())
    lines = [
        header, '=' * len(header), '',
        'The following are the configuration options provided by the app. '
        'It can be configured by changing appropriate values in the '
        '``config.yaml`` settings file. The options default to the default '
        'value of the classes for each of the options.', '']

    for level, name, _, _, props_docs in classes_flat:
        if level >= 0:
            lines.append(name)
            lines.append(headings[min(level, n)] * len(name))
            lines.append('')
        for prop, (default, doc) in sorted(
                props_docs.items(), key=operator.itemgetter(0)):
            if isinstance(default, str):
                lines.append('`{}`: "{}"'.format(prop, default))
            else:
                lines.append('`{}`: {}'.format(prop, default))
            while doc and not doc[-1].strip():
                del doc[-1]

            lines.extend([' ' + d for d in doc if d])
            lines.append('')
        lines.append('')

    lines = '\n'.join(lines)
    try:
        with open(rst_filename) as fh:
            if fh.read() == lines:
                return
    except IOError:
        pass

    with open(rst_filename, 'w') as fh:
        fh.write(lines)


def download_merge_yaml_doc(filename, url, out_filename):
    with urllib.request.urlopen(url) as f:
        remote = yaml_loads(f.read())

    local = {}
    if filename and os.path.exists(filename):
        with open(filename) as f:
            local = yaml_loads(f)

    local.update(remote)
    with open(out_filename, 'w') as f:
        f.write(yaml_dumps(local))


def merge_yaml_doc(filename1, filename2, out_filename):
    with open(filename1) as f:
        data = yaml_loads(f.read())

    with open(filename2) as f:
        data.update(yaml_loads(f.read()))

    with open(out_filename, 'w') as f:
        f.write(yaml_dumps(data))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Config Docs generation')
    subparsers = parser.add_subparsers(dest='command')

    download = subparsers.add_parser('download')
    download.add_argument('-f', '--filename', default=None)
    download.add_argument('-u', '--url', required=True)
    download.add_argument('-o', '--out_filename', required=True)

    download = subparsers.add_parser('merge')
    download.add_argument('-f1', '--filename1', required=True)
    download.add_argument('-f2', '--filename2', required=True)
    download.add_argument('-o', '--out_filename', required=True)

    args = parser.parse_args()

    if args.command == 'download':
        download_merge_yaml_doc(args.filename, args.url, args.out_filename)
    elif args.command == 'merge':
        merge_yaml_doc(args.filename1, args.filename2, args.out_filename)
