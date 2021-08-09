from tree_config.tests.test_doc import \
    ClassWithChildrenThatHaveSometimesNiceProperties


class RootClass:
    """A root class"""
    _config_props_ = ('prop_z', )
    _config_children_ = {'first child': 'first_child'}
    prop_z = 4
    """This is the very last property ever.
    """
    first_child: ClassWithChildrenThatHaveSometimesNiceProperties = None

    def __init__(self):
        self.first_child = ClassWithChildrenThatHaveSometimesNiceProperties()
