"""Human-readable staleness warnings for old memories.

Per Ch. 11 of the book: 'today' / 'yesterday' / 'N days ago' format triggers the right
model reasoning more reliably than raw ISO timestamps.
"""
from __future__ import annotations


def staleness_warning(age_days: float) -> str:
    if age_days < 1:
        return ""  # today: no warning
    if age_days < 2:
        return (
            "> [memory recorded yesterday; verify code citations against current state]"
        )
    return (
        f"> [memory is {int(age_days)} days old — code behavior and file:line "
        "citations may be outdated; verify against current code]"
    )
