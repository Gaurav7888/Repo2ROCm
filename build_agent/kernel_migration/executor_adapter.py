"""
Sandbox executor adapters for the kernel migration scaffold.

The scaffold defines an ``Executor = Callable[[str], CommandResult]`` interface.
The two adapters here let the scaffold (and the new ``KernelConverterAgent``)
run either:

- inside the project's pexpect-backed Docker sandbox via ``SandboxExecutor``
  wrapping ``Sandbox.get_session().execute_simple(...)`` (or any object that
  exposes a similar ``execute(...)`` / ``execute_simple(...)`` method), or
- entirely in-memory via ``DryRunExecutor`` for unit tests, recording every
  command without touching disk.

Both adapters return a populated ``CommandResult``. They never raise on
sandbox-level errors (timeouts, missing toolchain, broken pipes); failures are
surfaced as a non-zero ``return_code`` so the converter agent can mark the
migration as ``unsupported`` and continue without breaking the outer loop.
"""

from __future__ import annotations

import shlex
from typing import Any, Callable, List, Optional

from .scaffold import CommandResult, Executor


class DryRunExecutor:
    """In-memory executor that records every issued command.

    Useful for unit tests and the converter's dry-run path. The default
    behaviour returns ``return_code=0`` so the scaffold treats every step as a
    no-op success, but ``stdout_factory`` lets a test fake structured output
    (e.g. simulated hipify warnings) for individual commands.
    """

    def __init__(
        self,
        stdout_factory: Optional[Callable[[str], str]] = None,
        return_code: int = 0,
        stderr: str = "",
    ):
        self._stdout_factory = stdout_factory
        self._return_code = int(return_code)
        self._stderr = stderr
        self.commands: List[str] = []
        self.results: List[CommandResult] = []

    def __call__(self, command: str) -> CommandResult:
        self.commands.append(command)
        stdout = ""
        if self._stdout_factory is not None:
            try:
                stdout = self._stdout_factory(command) or ""
            except Exception as exc:  # pragma: no cover - defensive
                stdout = f"[DryRunExecutor] stdout_factory error: {exc}"
        result = CommandResult(
            command=command,
            return_code=self._return_code,
            stdout=stdout,
            stderr=self._stderr,
        )
        self.results.append(result)
        return result


class SandboxExecutor:
    """Adapter that drives a real ``utils.sandbox.Sandbox`` session.

    The Sandbox session API used by ``Configuration`` exposes
    ``execute_simple(command, timeout)`` returning ``(success_bool, output)``.
    We wrap that into the ``CommandResult`` shape expected by the scaffold and
    catch every failure mode so the kernel converter never crashes the outer
    loop.

    Parameters
    ----------
    session:
        A live sandbox session (anything with ``execute_simple`` /
        ``execute`` / ``get_returncode``). The default behaviour calls
        ``execute_simple`` to avoid the ``Configuration``-only waiting/conflict
        plumbing.
    timeout:
        Per-command timeout in seconds. Hipify on a single small file is
        usually fast; we default to 600s and let callers override.
    sandbox_factory:
        Optional callable returning a fresh session if ``session`` becomes
        unusable mid-run (e.g. on timeout). When omitted we just return a
        non-zero ``CommandResult`` and let the caller decide.
    """

    def __init__(
        self,
        session: Any,
        timeout: int = 600,
        sandbox_factory: Optional[Callable[[], Any]] = None,
    ):
        self.session = session
        self.timeout = int(timeout)
        self._sandbox_factory = sandbox_factory

    def _execute_simple(self, command: str) -> CommandResult:
        try:
            ok, output = self.session.execute_simple(command, timeout=self.timeout)
            return_code = 0 if bool(ok) else 1
            try:
                rc = self.session.get_returncode()
                if isinstance(rc, int):
                    return_code = rc
            except Exception:
                pass
            stdout = output if isinstance(output, str) else str(output or "")
            return CommandResult(
                command=command,
                return_code=return_code,
                stdout=stdout,
                stderr="",
            )
        except Exception as exc:
            # Replace the broken session if a factory was supplied.
            if self._sandbox_factory is not None:
                try:
                    self.session = self._sandbox_factory()
                except Exception:
                    pass
            return CommandResult(
                command=command,
                return_code=124,  # convention: 124 = timeout/abort
                stdout="",
                stderr=f"SandboxExecutor: {type(exc).__name__}: {exc}",
            )

    def __call__(self, command: str) -> CommandResult:
        return self._execute_simple(command)


def make_executor(session: Any, timeout: int = 600) -> Executor:
    """Convenience factory returning a ready-to-use ``Executor`` callable."""
    adapter = SandboxExecutor(session, timeout=timeout)
    return adapter


def shell_quote(path: str) -> str:
    """Quote a path so the scaffold's bash command builders stay safe."""
    return shlex.quote(path)


__all__ = [
    "DryRunExecutor",
    "SandboxExecutor",
    "make_executor",
    "shell_quote",
]
