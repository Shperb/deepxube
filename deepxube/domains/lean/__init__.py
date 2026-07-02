""" Lean 4 theorem-proving domain. Heavy deps (lean_dojo, transformers) are imported lazily. """
from deepxube.domains.lean import domain as _domain     # noqa: F401  registers "lean" + parser
from deepxube.domains.lean import nnet as _nnet          # noqa: F401  registers nnet input + "leanheur"
