"""Tiny rule engine: match a `when` dict against a context dict, propose `do`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from repo2rocm.learning.kb_store import KBStore, Rule


@dataclass
class RuleMatch:
    rule: Rule
    do: dict[str, Any]


class RuleEngine:
    def __init__(self, kb: KBStore):
        self.kb = kb

    def evaluate(self, ctx: dict[str, Any]) -> list[RuleMatch]:
        out: list[RuleMatch] = []
        for r in self.kb.list_rules():
            if self._match(r.when, ctx):
                out.append(RuleMatch(rule=r, do=r.do))
        return sorted(out, key=lambda m: -m.rule.confidence)

    def _match(self, when: dict[str, Any], ctx: dict[str, Any]) -> bool:
        for k, v in when.items():
            cv = ctx.get(k)
            if isinstance(v, list):
                if cv not in v:
                    return False
            elif isinstance(v, dict) and isinstance(cv, dict):
                if not all(cv.get(k2) == v2 for k2, v2 in v.items()):
                    return False
            elif cv != v:
                return False
        return True
