---
name: pin_hazards
description: Known-bad version pins on Python 3.12 / modern PyTorch where the pinned wheel doesn't exist or build
when_to_use: When a pip install fails with "no matching distribution" or compilation errors on a pinned dependency
paths: ["**/requirements*.txt", "**/pyproject.toml", "**/setup.py"]
---

# Old-pin hazards on Python 3.12 and modern ROCm PyTorch

These pins are known to have NO Python 3.12 wheel. They will either fall back to
source build (often requiring Rust/C toolchains we don't ship) or fail outright.

| Package | Bad pin (below) | Recommended fix |
|---|---|---|
| `transformers` | `<4.36.0` | Drop pin; use `pip install transformers` (latest) |
| `tokenizers` | `<0.15.0` | Drop pin; built by `transformers` install |
| `scipy` | `<1.11.0` | Drop pin; use latest |
| `scikit-learn` | `<1.3.0` | Drop pin; use latest |
| `pandas` | `<2.1.0` | Drop pin; use latest |
| `numpy` | `<1.26.0` | Drop pin; rocm/pytorch already pins a working numpy |
| `pillow` | `<10.0.0` | Drop pin; use latest |
| `grpcio` | `<1.58.0` | Drop pin; use latest |

## How to handle a pin hazard

1. Don't fight the build — strip or relax the pin before `Download` runs.
2. After install, run the project's actual entrypoint to confirm there's no API
   incompatibility introduced by the version bump. If there is, search for the
   specific API rename and apply an `Edit` patch.
3. If the project's tests fail because of a breaking change in the new version,
   prefer upgrading the project code (one or two small edits) rather than pinning
   the old version back.

## Python 3.12 stdlib breakages

If the agent must run on Python 3.12 and the project uses removed/moved stdlib:

- `import imp` → `import importlib as imp`
- `import distutils` → `from setuptools import distutils as ...`
- `from collections import Mapping` → `from collections.abc import Mapping`
- `import cgi` / `cgitb` / `nntplib` / `telnetlib` / `aifc` / `audioop` / `imghdr` /
  `mailcap` / `crypt` — these are gone; refactor or pin a backport.
