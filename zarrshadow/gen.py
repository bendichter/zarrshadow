"""Lazy evaluation of kerchunk "gen" entries.

A gen entry describes many references with templates over integer dimensions,
for example chunk i of a raw binary file at offset header + i * chunk_nbytes:

    {"key": "data/c/{{i}}/0", "url": "{{u0}}", "offset": "{{12 + i * 256000}}",
     "length": "256000", "dimensions": {"i": {"stop": 200000}}}

fsspec expands every entry when the references are loaded, rendering each one
with jinja. Here a key is matched against each entry's key pattern when it is
requested, and only that entry's offset and length are computed. Templates may
use names (dimension variables and "templates" entries) combined with integer
arithmetic: + - * // % and parentheses. Anything else is refused.
"""

from __future__ import annotations

import ast
import itertools
import operator
import re
from collections.abc import Iterator
from typing import Any

_PLACEHOLDER = re.compile(r"{{\s*(.*?)\s*}}")
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def evaluate(expression: str, variables: dict[str, Any]) -> Any:
    """Evaluate a restricted arithmetic expression over named variables."""
    tree = ast.parse(expression, mode="eval")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown name {node.id!r} in gen expression {expression!r}")
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"unsupported gen expression {expression!r}")

    return ev(tree)


def render(template: str, variables: dict[str, Any]) -> str:
    """Replace each {{ expression }} in a template with its value."""
    return _PLACEHOLDER.sub(lambda m: str(evaluate(m[1], variables)), template)


class Generator:
    """One gen entry, answering lookups for the keys it describes."""

    def __init__(self, entry: dict, templates: dict[str, str]) -> None:
        self.entry = entry
        self.templates = templates
        self.dimensions: dict[str, range | list[int]] = {
            name: dim if isinstance(dim, list) else range(dim.get("start", 0), dim["stop"], dim.get("step", 1))
            for name, dim in entry["dimensions"].items()
        }
        self._members = {name: set(dim) if isinstance(dim, list) else dim for name, dim in self.dimensions.items()}
        # The key template may only substitute plain names, so it can be inverted
        pattern, last = [], 0
        for m in _PLACEHOLDER.finditer(entry["key"]):
            pattern.append(re.escape(entry["key"][last : m.start()]))
            name = m[1]
            if name in self.dimensions:
                pattern.append(f"(?P<{name}>-?\\d+)")
            elif name in templates:
                pattern.append(re.escape(templates[name]))
            else:
                raise ValueError(f"gen key {entry['key']!r} may only use dimension or template names")
            last = m.end()
        pattern.append(re.escape(entry["key"][last:]))
        self._key_pattern = re.compile("".join(pattern))
        literal = _PLACEHOLDER.split(entry["key"])[0]
        self.static_prefix = literal  # key text before the first placeholder

    def lookup(self, key: str) -> list | None:
        """Return [url, offset, length] (or [url]) for key, or None if it is not generated here."""
        m = self._key_pattern.fullmatch(key)
        if m is None:
            return None
        values = {name: int(v) for name, v in m.groupdict().items()}
        if any(v not in self._members[name] for name, v in values.items()):
            return None
        return self._ref(values)

    def _ref(self, values: dict[str, int]) -> list:
        variables = {**self.templates, **values}
        url = render(self.entry["url"], variables)
        if "offset" in self.entry and "length" in self.entry:
            return [url, int(render(self.entry["offset"], variables)), int(render(self.entry["length"], variables))]
        return [url]

    def items(self) -> Iterator[tuple[str, list]]:
        """Every (key, ref) this entry describes. Used for listing and expansion."""
        names = list(self.dimensions)
        for combo in itertools.product(*self.dimensions.values()):
            values = dict(zip(names, combo))
            yield render(self.entry["key"], {**self.templates, **values}), self._ref(values)
