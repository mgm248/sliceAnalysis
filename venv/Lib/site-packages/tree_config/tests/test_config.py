import pytest
from tree_config import get_config_children_names, get_config_prop_names, \
    read_config_from_object, apply_config, get_config_prop_items, \
    get_config_children_items, Configurable


@pytest.fixture(params=[object, Configurable])
def simple_prop_class(request):
    class SomeClass(request.param):
        _config_props_ = ('prop_a', 'prop_b')
        prop_a = 1
        prop_b = 2

    yield SomeClass


@pytest.fixture(params=[object, Configurable])
def class_with_children(request, simple_prop_class):
    class ClassWithChildren(request.param):
        _config_children_ = {'a child': 'child_a', 'b child': 'child_b'}
        child_a = None
        child_b = None

        def __init__(self):
            self.child_a = simple_prop_class()
            self.child_b = simple_prop_class()

    yield ClassWithChildren


def test_config_props_only_cls(simple_prop_class):
    assert get_config_prop_names(simple_prop_class) == ['prop_a', 'prop_b']
    assert get_config_prop_items(simple_prop_class) == {
        'prop_a': 1, 'prop_b': 2}

    assert get_config_children_names(simple_prop_class) == {}
    assert get_config_children_items(simple_prop_class) == []

    assert read_config_from_object(simple_prop_class) == {
        'prop_a': 1, 'prop_b': 2}


def test_config_props_only_obj(simple_prop_class):
    obj = simple_prop_class()
    assert get_config_prop_names(obj) == ['prop_a', 'prop_b']
    assert get_config_prop_items(obj) == {'prop_a': 1, 'prop_b': 2}

    assert get_config_children_names(obj) == {}
    assert get_config_children_items(obj) == []

    assert read_config_from_object(obj) == {'prop_a': 1, 'prop_b': 2}

    apply_config(obj, {'prop_a': 'home', 'prop_b': 43})
    assert obj.prop_a == 'home'
    assert obj.prop_b == 43


def test_config_children_only_cls(class_with_children):
    assert get_config_prop_names(class_with_children) == []
    assert get_config_prop_items(class_with_children) == {}

    assert get_config_children_names(class_with_children) == {
        'a child': 'child_a', 'b child': 'child_b'}
    assert get_config_children_items(class_with_children) == [
        ('a child', 'child_a', None), ('b child', 'child_b', None)
    ]

    assert read_config_from_object(class_with_children) == {}


def test_config_children_only_obj(class_with_children):
    obj = class_with_children()
    assert get_config_prop_names(obj) == []
    assert get_config_prop_items(obj) == {}

    assert get_config_children_names(obj) == {
        'a child': 'child_a', 'b child': 'child_b'}
    assert get_config_children_items(obj) == [
        ('a child', 'child_a', obj.child_a),
        ('b child', 'child_b', obj.child_b)
    ]

    assert read_config_from_object(obj) == {
        'a child': {'prop_a': 1, 'prop_b': 2},
        'b child': {'prop_a': 1, 'prop_b': 2}
    }

    apply_config(obj, {
        'a child': {'prop_a': 11, 'prop_b': 13},
        'b child': {'prop_a': 12, 'prop_b': 14}
    })
    assert obj.child_a.prop_a == 11
    assert obj.child_a.prop_b == 13
    assert obj.child_b.prop_a == 12
    assert obj.child_b.prop_b == 14
