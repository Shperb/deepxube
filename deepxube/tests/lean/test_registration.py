import deepxube.domains.lean.domain  # noqa: F401  (import triggers registration)
from deepxube.factories.domain_factory import domain_factory


def test_lean_domain_is_registered():
    assert "lean" in domain_factory.get_all_class_names()


def test_parser_exposes_k_and_seed_arguments():
    parser = domain_factory.get_parser("lean")
    assert parser is not None
    assert parser.parse("4k") == {"k": 4}
    assert parser.parse("8k_3seed") == {"k": 8, "seed": 3}
