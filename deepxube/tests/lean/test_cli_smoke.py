def test_heur_nnet_par_builds_for_lean_v():
    # The V-heuristic factory must pair leanheur with the lean_text_sg nnet input,
    # and the resulting par must convert states/goals to the 5-array nnet input.
    import deepxube.domains.lean  # noqa: F401  (ensure registration)
    from deepxube.factories.heuristic_factory import build_heur_nnet_par
    from deepxube.domains.lean.domain import LeanDomain
    from deepxube.domains.lean.backend import FakeLeanBackend
    from deepxube.domains.lean.policy import FakeTacticGenerator
    from deepxube.domains.lean.corpus import Corpus, CorpusEntry
    from deepxube.domains.lean.dojo_backend import TheoremRef
    from deepxube.domains.lean.types import LeanState, LeanGoal

    corpus = Corpus([CorpusEntry(TheoremRef("u", "c", "F.lean", "t"), proof_len=1)])
    d = LeanDomain(FakeLeanBackend({"u@c:F.lean:t": {"__init__": "⊢ p"}}),
                   FakeTacticGenerator({}), corpus, k=2, model_name="m", seed=0)
    par = build_heur_nnet_par(d, "lean", "leanheur", {"mlp_hidden": 16}, "V")
    arrays = par.to_np([LeanState("t", (), "⊢ p", False)], [LeanGoal(None)])
    assert len(arrays) == 5
