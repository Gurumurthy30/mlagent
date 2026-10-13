"""Prints the signature and documentation of a Python object.

Run as: python -m mlagent.library_lookup lightgbm.LGBMClassifier.fit
It runs in its own process, so a heavy library is never imported into the agent itself.
"""

import importlib
import inspect
import sys

from mlagent.settings import LIBRARY_HELP_CHARS

MAX_PUBLIC_NAMES_SHOWN = 60


def find_python_object(dotted_name: str) -> object:
    """Find 'package.module.Class.method' by importing the longest module part first."""
    name_parts = dotted_name.split(".")
    for module_part_count in range(len(name_parts), 0, -1):
        module_name = ".".join(name_parts[:module_part_count])
        try:
            found_object = importlib.import_module(module_name)
        except ImportError:
            continue
        try:
            for attribute_name in name_parts[module_part_count:]:
                found_object = getattr(found_object, attribute_name)
        except AttributeError:
            continue
        return found_object
    raise LookupError(f"Not found: {dotted_name}")


def describe_python_object(dotted_name: str) -> str:
    found_object = find_python_object(dotted_name)
    lines = []
    try:
        lines.append(f"{dotted_name}{inspect.signature(found_object)}")
    except (TypeError, ValueError):
        pass
    documentation = inspect.getdoc(found_object) or "No documentation."
    if inspect.ismodule(found_object):
        documentation = documentation[: LIBRARY_HELP_CHARS // 2]
    lines.append(documentation)
    if inspect.ismodule(found_object):
        public_names = [name for name in dir(found_object) if not name.startswith("_")]
        lines.append("Public names: " + ", ".join(public_names[:MAX_PUBLIC_NAMES_SHOWN]))
    description = "\n".join(lines)
    return description[:LIBRARY_HELP_CHARS]


def main() -> None:
    print(describe_python_object(sys.argv[1]))


if __name__ == "__main__":
    main()
