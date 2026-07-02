""" Regression test: heuristic/policy functions must handle an empty batch (0 states) gracefully.

This case arises in search when a frontier goes empty (e.g. an instance is solved, or all children
are dead-ends). Before the fix, an empty batch reached nnet_batched / the shared-memory parallel path
and raised (IndexError on nnet_batched(...)[0], or SharedMemory 'size must be positive'). The fix
short-circuits empty input in the returned heuristic/policy closures.
"""
import torch

import deepxube.domains.grid  # noqa: F401  (registers the grid domain)
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.heuristic_factory import build_heur_nnet_par


def test_heur_v_fn_returns_empty_for_empty_batch() -> None:
    domain = domain_factory.build_class("grid", {})
    par = build_heur_nnet_par(domain, "grid", "gridnet", {}, "V")
    nnet = par.get_nnet()
    fn = par.get_nnet_fn(nnet, None, torch.device("cpu"), None)

    # empty input -> empty output, with no nnet / shared-memory machinery invoked
    assert fn([], []) == []

    # sanity: non-empty still works and returns one value per state
    states, goals = domain.sample_problem_instances([1, 1, 1])
    out = fn(states, goals)
    assert len(out) == len(states)
