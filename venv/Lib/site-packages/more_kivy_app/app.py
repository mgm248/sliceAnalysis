"""App
========

The base App class.
"""

import os
import inspect
import traceback
import sys
import logging
from functools import wraps
from os.path import dirname, join, isdir
from more_kivy_app.config import apply_config, read_config_from_file, \
    read_config_from_object, dump_config


if not os.environ.get('KIVY_DOC_INCLUDE', None):
    from kivy.config import Config
    Config.set('kivy', 'exit_on_escape', 0)

from kivy import resources
from kivy.modules import inspector
from kivy.resources import resource_add_path
from kivy.base import ExceptionManager, ExceptionHandler
from kivy.app import App
from kivy.logger import Logger
from kivy.clock import Clock


__all__ = ('MoreKivyApp', 'run_app', 'run_app_async', 'app_error',
           'app_error_async', 'report_exception_in_app')


def report_exception_in_app(e, exc_info=None, threaded=True):
    """Takes the error and reports it to :meth:`MoreKivyApp.handle_exception`.

    :param e: The error
    :param exc_info: If not None, the return value of ``sys.exc_info()`` or
        a stringified version of it.
    :param threaded: If the app should be called in a thread safe manner,
        e.g. if called from another thread.
    """
    def report_exception(*largs):
        if App.get_running_app() is not None:
            App.get_running_app().handle_exception(
                e, exc_info=exc_info or sys.exc_info())
        else:
            logging.error(e)
            if exc_info is not None:
                if isinstance(exc_info, str):
                    logging.error(exc_info)
                else:
                    logging.error(
                        ''.join(traceback.format_exception(*exc_info)))

    if threaded:
        Clock.schedule_once(report_exception)
    else:
        report_exception()


def app_error(app_error_func=None, threaded=True):
    """A decorator which wraps the function in `try...except` and calls
    :meth:`MoreKivyApp.handle_exception` when a exception is raised.

    E.g.::

        @app_error
        def do_something():
            do_something
    """
    def inner_decorator(f):
        @wraps(f)
        def safe_func(*largs, **kwargs):
            try:
                return f(*largs, **kwargs)
            except Exception as e:
                exc_info = sys.exc_info()
                stack = traceback.extract_stack()
                tb = traceback.extract_tb(exc_info[2])
                full_tb = stack[:-1] + tb
                exc_line = traceback.format_exception_only(*exc_info[:2])

                err = 'Traceback (most recent call last):'
                err += "".join(traceback.format_list(full_tb))
                err += "".join(exc_line)
                report_exception_in_app(e, exc_info=err, threaded=threaded)
        return safe_func

    if app_error_func is None:
        return inner_decorator
    return inner_decorator(app_error_func)


def app_error_async(app_error_func=None, threaded=True):
    """A decorator which wraps the async function in `try...except` and calls
    :meth:`MoreKivyApp.handle_exception` when a exception is raised.

    E.g.::

        @app_error
        async def do_something():
            do_something
    """
    def inner_decorator(f):
        @wraps(f)
        async def safe_func(*largs, **kwargs):
            try:
                return await f(*largs, **kwargs)
            except Exception as e:
                exc_info = sys.exc_info()
                stack = traceback.extract_stack()
                tb = traceback.extract_tb(exc_info[2])
                full_tb = stack[:-1] + tb
                exc_line = traceback.format_exception_only(*exc_info[:2])

                err = 'Traceback (most recent call last):'
                err += "".join(traceback.format_list(full_tb))
                err += "".join(exc_line)
                report_exception_in_app(e, exc_info=err, threaded=threaded)
        return safe_func

    if app_error_func is None:
        return inner_decorator
    return inner_decorator(app_error_func)


class MoreKivyApp(App):
    """The base app.
    """

    _config_props_ = ('inspect', )

    yaml_config_path = None
    '''The full path to the config file used for the app.

    If it's `None` when ``__init__`` of the base class is called, it is
    set to ``ClassName_config.yaml``, where ``ClassName`` is the name of the
    App class.
    '''

    app_settings = {}
    '''A dict that contains the :mod:`tree-config` settings for the
    app for all the configurable classes. See that module for details.

    The keys in the dict are configuration names for a class and its
    values are property values or nested dicts whose keys are class attributes
    names and values are their values. These attributes are the ones listed in
    ``_config_props_``. See :mod:`tree-config` for how configuration works.
    '''

    inspect = False
    '''Enables GUI inspection. If True, it is activated by hitting ctrl-e in
    the GUI.
    '''

    _data_path = ''

    def __init__(self, **kw):
        if self.yaml_config_path is None:
            self.yaml_config_path = '{}_config.yaml'.format(
                self.__class__.__name__)

        super(MoreKivyApp, self).__init__(**kw)

        self.init_load()

    def init_load(self):
        """Creates any config files and adds :attr:`data_path` to the kivy
        resource PATH when the app is instantiated.
        """
        d = self.data_path
        if isdir(d):
            resource_add_path(d)

        self.ensure_file(self.yaml_config_path)

    def ensure_file(self, filename):
        """Returns the full path to ``filename`` by searching for it in kivy's
         resource directories if it exists. Otherwise, it create the empty file
         and returns the path to it.
        """
        if not resources.resource_find(filename):
            with open(join(self.data_path, filename), 'w') as fh:
                pass
        return resources.resource_find(filename)

    @property
    def data_path(self):
        """The install dependent path to the persistent config data directory.
        It is automatically added to the kivy resource PATH.

        if this is running under a PyInstaller installation, it is the path
        that contains the executable, it's the data folder next to the app
        class's python file.
        """
        if self._data_path:
            return self._data_path

        if hasattr(sys, '_MEIPASS'):
            return os.path.abspath(os.path.dirname(sys.executable))
        return join(dirname(inspect.getfile(self.__class__)), 'data')

    def load_app_settings_from_file(self):
        """Reads the config from the :attr:`yaml_config_path` file and saves
        it to :attr:`app_settings`. See
        :func:`tree_config.read_config_from_file`.
        """
        self.app_settings = read_config_from_file(
            self.ensure_file(self.yaml_config_path))

    def apply_app_settings(self):
        """Applies the app config stored in :attr:`app_settings` to the
        application instance. See :func:`tree_config.apply_config`.
        """
        apply_config(self, self.app_settings)

    def dump_app_settings_to_file(self):
        """Saves the app config from the app to the config file at
        :attr:`yaml_config_path`. See :func:`tree_config.dump_config`.
        """
        dump_config(
            self.ensure_file(self.yaml_config_path),
            read_config_from_object(self))

    def build(self, root=None):
        """Similar to App's build, but it takes the root widget if provided
        and if :attr:`inspect` is True, it activates kivy's inspector.

        :param root: The root widget instance.
        :return: The root widget
        """
        if root is not None and self.inspect:
            from kivy.core.window import Window
            inspector.create_inspector(Window, root)
        return root

    def ask_cannot_close(self, *largs, **kwargs):
        """Called by kivy when the user tries to close the app. It only closes
        if this returns False.

        This is only used when the app is started with :func:`run_app` or
        :func:`run_app_async`.
        """
        return False

    def handle_exception(self, msg, exc_info=None, level='error', *largs):
        """Should be called whenever an exception is caught in the app.

        If the app is started with :func:`run_app` or :func:`run_app_async`,
        this is called if kivy encounters an error. Similarly,
        :func:`app_error`, :func:`app_error_async`, and
        :func:`report_exception_in_app` call this on the app upon an exception.

        It logs the message to the logger using :meth:`get_logger`.

        :parameters:

            `exception`: string or exception object
                The caught exception (i.e. the ``e`` in
                ``except Exception as e``)
            `exc_info`: stack trace
                If not None, the return value of ``sys.exc_info()`` or a
                stringified version of it. It is used to log the stack trace.
            `level`: string
                The log level to use on the message. Can be ``'error'``,
                ``'exception'`` (with the same meaning), or any other python
                log level.
        """

        if isinstance(exc_info, str):
            self.get_logger().error(msg)
            self.get_logger().error(exc_info)
        elif level in ('error', 'exception'):
            self.get_logger().error(msg)
            if exc_info is not None:
                if isinstance(exc_info, str):
                    self.get_logger().error(exc_info)
                else:
                    self.get_logger().error(
                        ''.join(traceback.format_exception(*exc_info)))
        else:
            getattr(self.get_logger(), level)(msg)

        error_indicator = self.error_indicator
        if not error_indicator:
            return

        error_indicator.add_item('{}'.format(msg))

    def get_logger(self):
        """Returns the logger to use in :meth:`handle_exception` to log
        messages.

        Defaults to returning the Kivy Logger.
        """
        return Logger

    def clean_up(self):
        """Called by :func:`run_app` or :func:`run_app_async` after the app
        closes to clean up any remaining resources.

        By default, if :attr:`inspect` is enabled, it cleans up after it.
        """
        if self.inspect and self.root:
            from kivy.core.window import Window
            inspector.stop(Window, self.root)


class _MoreKivyAppHandler(ExceptionHandler):

    def handle_exception(self, inst):
        app = App.get_running_app()
        if app:
            app.handle_exception(inst, exc_info=sys.exc_info())
            return ExceptionManager.PASS
        return ExceptionManager.RAISE


def run_app(cls_or_app):
    """Entrance method used to start the App. It runs, or instantiates and runs
    a :class:`MoreKivyApp` type instance.
    """
    from kivy.core.window import Window
    handler = _MoreKivyAppHandler()
    ExceptionManager.add_handler(handler)

    app = cls_or_app() if inspect.isclass(cls_or_app) else cls_or_app
    Window.fbind('on_request_close', app.ask_cannot_close)
    try:
        app.run()
    except Exception as e:
        app.handle_exception(e, exc_info=sys.exc_info())

    try:
        app.clean_up()
    except Exception as e:
        app.handle_exception(e, exc_info=sys.exc_info())

    Window.funbind('on_request_close', app.ask_cannot_close)
    ExceptionManager.remove_handler(handler)
    return app


async def run_app_async(cls_or_app):
    """Entrance method used to start the App. It runs, or instantiates and runs
    a :class:`MoreKivyApp` type instance.
    """
    from kivy.core.window import Window
    handler = _MoreKivyAppHandler()
    ExceptionManager.add_handler(handler)

    app = cls_or_app() if inspect.isclass(cls_or_app) else cls_or_app
    Window.fbind('on_request_close', app.ask_cannot_close)
    try:
        await app.async_run()
    except Exception as e:
        app.handle_exception(e, exc_info=sys.exc_info())

    try:
        app.clean_up()
    except Exception as e:
        app.handle_exception(e, exc_info=sys.exc_info())

    Window.funbind('on_request_close', app.ask_cannot_close)
    ExceptionManager.remove_handler(handler)
