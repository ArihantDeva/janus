"""The Glimmer text tower lives at full.model.language_model.

transformers 5.18.0: MuseGlimmerForConditionalGeneration holds
self.model = MuseGlimmerModel, and the text tower is
MuseGlimmerModel.language_model. Amarel jobs 63525469/63525539 died on
`full.language_model` with AttributeError. This pins the nesting.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _Tower:
    pass


class _Inner:
    def __init__(self):
        self.language_model = _Tower()


class _Outer:
    """Mimics MuseGlimmerForConditionalGeneration nesting."""
    def __init__(self):
        self.model = _Inner()


def resolve_encoder(full):
    """The one correct path: full.model.language_model."""
    return full.model.language_model


def test_encoder_resolves_through_model_nesting():
    assert isinstance(resolve_encoder(_Outer()), _Tower)


def test_bare_language_model_attr_is_not_the_path():
    assert not hasattr(_Outer(), "language_model")


def test_sources_use_nested_path():
    root = os.path.join(os.path.dirname(__file__), "..", "src")
    for name in ("train_glimmer_freeze.py", "extract_glimmer_features.py"):
        src = open(os.path.join(root, name)).read()
        assert "full.model.language_model" in src, name
        assert "enc = full.language_model" not in src, name
