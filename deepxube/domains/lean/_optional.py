from types import ModuleType
import importlib


def require(module_name: str, extra: str) -> ModuleType:
    """ Import an optional dependency, raising a helpful error if it is missing.

    :param module_name: importable module name
    :param extra: the pip extra that provides it (for the error message)
    :return: the imported module
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"'{module_name}' is required for the Lean domain. Install it with: pip install deepxube[{extra}]"
        ) from e
