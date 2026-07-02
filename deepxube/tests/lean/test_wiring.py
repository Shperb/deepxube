def test_importing_subpackage_registers_everything():
    import importlib
    importlib.import_module("deepxube.domains.lean")
    from deepxube.factories.domain_factory import domain_factory
    from deepxube.factories.heuristic_factory import heuristic_factory
    from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys
    assert "lean" in domain_factory.get_all_class_names()
    assert "leanheur" in heuristic_factory.get_all_class_names()
    assert any(key[1] == "lean_text_sg" for key in get_domain_nnet_input_keys("lean"))


def test_no_heavy_deps_imported_on_registration():
    import sys
    import importlib
    # Drop any cached heavy modules so the guard is meaningful even if they get installed later.
    for m in list(sys.modules):
        if m in ("lean_dojo", "transformers") or m.startswith("lean_dojo.") or m.startswith("transformers."):
            del sys.modules[m]
    importlib.import_module("deepxube.domains.lean")
    assert "lean_dojo" not in sys.modules
    assert "transformers" not in sys.modules
