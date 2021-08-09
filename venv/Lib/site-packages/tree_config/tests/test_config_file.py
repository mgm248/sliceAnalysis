from tree_config import read_config_from_file, load_config, dump_config, \
    load_apply_save_config, read_config_from_object
from tree_config.utils import yaml_loads


def config_class():
    class SomeClass:
        _config_props_ = ('prop_a', 'prop_b')
        prop_a = 1
        prop_b = 2

    class ClassWithChildren:
        _config_props_ = ('prop_c', )
        _config_children_ = {'a child': 'child_a', 'b child': 'child_b'}

        prop_c = 3
        child_a = None
        child_b = None

        def __init__(self):
            self.child_a = SomeClass()
            self.child_b = SomeClass()

    return ClassWithChildren()


def test_load_default_config(tmp_path):
    obj = config_class()
    obj.child_a.prop_a = 95
    obj.prop_c = 463

    p = tmp_path / 'config.yaml'
    f = str(p)

    assert not p.exists()
    config = load_config(obj, f)
    assert p.exists()
    assert config == {
        'a child': {'prop_a': 95, 'prop_b': 2},
        'b child': {'prop_a': 1, 'prop_b': 2},
        'prop_c': 463,
    }

    assert yaml_loads(p.read_text()) == config


def test_write_config(tmp_path):
    obj = config_class()
    obj.child_a.prop_a = 87
    obj.prop_c = 91

    p = tmp_path / 'config.yaml'
    f = str(p)

    assert not p.exists()
    dump_config(f, read_config_from_object(obj))
    assert p.exists()

    assert yaml_loads(p.read_text()) == {
        'a child': {'prop_a': 87, 'prop_b': 2},
        'b child': {'prop_a': 1, 'prop_b': 2},
        'prop_c': 91,
    }


def test_load_exists_config(tmp_path):
    obj = config_class()
    obj.child_a.prop_a = 88
    obj.prop_c = 91

    p = tmp_path / 'config.yaml'
    f = str(p)

    assert not p.exists()
    dump_config(f, read_config_from_object(obj))
    assert p.exists()

    obj.child_a.prop_a = 254
    obj.prop_c = 546

    config = load_config(obj, f)
    assert config == {
        'a child': {'prop_a': 88, 'prop_b': 2},
        'b child': {'prop_a': 1, 'prop_b': 2},
        'prop_c': 91,
    }


def test_read_config_file(tmp_path):
    obj = config_class()
    obj.child_a.prop_a = 43
    obj.prop_c = 56

    p = tmp_path / 'config.yaml'
    f = str(p)

    assert not p.exists()
    dump_config(f, read_config_from_object(obj))
    assert p.exists()

    assert read_config_from_file(f) == {
        'a child': {'prop_a': 43, 'prop_b': 2},
        'b child': {'prop_a': 1, 'prop_b': 2},
        'prop_c': 56,
    }


def test_load_apply_save_config(tmp_path):
    obj = config_class()
    obj.child_a.prop_a = 554
    obj.prop_c = 856

    p = tmp_path / 'config.yaml'
    f = str(p)

    assert not p.exists()
    config = load_apply_save_config(obj, f)
    assert p.exists()
    assert config == {
        'a child': {'prop_a': 554, 'prop_b': 2},
        'b child': {'prop_a': 1, 'prop_b': 2},
        'prop_c': 856,
    }

    obj = config_class()
    assert obj.child_a.prop_a == 1
    assert obj.prop_c == 3

    load_apply_save_config(obj, f)
    assert obj.child_a.prop_a == 554
    assert obj.prop_c == 856
