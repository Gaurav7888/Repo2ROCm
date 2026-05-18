---
name: py312_compat
description: Python 3.12 breakage patterns and how to fix them.
when_to_use: Use when the container's python is 3.12 and the repo was written for 3.8-3.10.
---

# Python 3.12 compatibility

## Removed stdlib modules

| Removed | Replacement |
|---|---|
| `distutils` | `setuptools` (`from setuptools import ...`) |
| `imp` | `importlib` |
| `cgi`, `cgitb` | `urllib` or framework-specific |
| `crypt`, `spwd`, `nis` | (no replacement) |
| `aifc`, `audioop`, `chunk`, `sunau`, `sndhdr` | external libs (`audiofile`, `soundfile`) |
| `imghdr` | `python-magic`, `filetype` |
| `mailcap` | (no replacement) |
| `pipes` | `subprocess` |
| `telnetlib` | `telnetlib3` |
| `uu`, `xdrlib` | `base64`, struct-based encoding |

## ABC import paths

Replace bare-collections imports:

```python
# old (3.9 still works, 3.10+ removed)
from collections import Mapping, Sequence, Iterable
# new
from collections.abc import Mapping, Sequence, Iterable
```

## Common errors

- `ModuleNotFoundError: No module named 'distutils'` → `pip install setuptools` then patch
  imports to `from setuptools._distutils import ...` if needed.
- `AttributeError: module 'asyncio' has no attribute 'coroutine'` → remove
  `@asyncio.coroutine` (removed in 3.11+).
- `TypeError: ... got an unexpected keyword argument 'loop'` → remove `loop=` from asyncio calls.

## Fast-fail probe

```bash
python -c "import distutils, imp" 2>&1 | head -5
```

If both fail, the repo will need patching.
