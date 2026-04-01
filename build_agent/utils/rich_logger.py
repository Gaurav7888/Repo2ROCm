"""
Centralized rich-based logger for Repo2Run.
Provides structured, color-coded logging for every phase of the build agent.
Also writes a complete plain-text log to a file for debugging.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.markup import escape as rich_escape
import time
import os
from datetime import datetime

console = Console(width=140, force_terminal=True)

# ── File logging state ──
_log_file = None
_log_path = None
_verbose = False


def set_verbose(enabled=True):
    global _verbose
    _verbose = enabled


def is_verbose():
    return _verbose


def init_file_log(output_dir):
    """Initialize file logging. Call once after output dir is created."""
    global _log_file, _log_path
    os.makedirs(output_dir, exist_ok=True)
    _log_path = os.path.join(output_dir, "agent_debug_log.txt")
    _log_file = open(_log_path, "w", encoding="utf-8", buffering=1)
    _fwrite(f"=== Repo2Run Agent Log ===")
    _fwrite(f"Started: {datetime.now().isoformat()}")
    _fwrite(f"Log file: {_log_path}")
    _fwrite("")


def close_file_log():
    """Close the log file. Call at the end of the run."""
    global _log_file
    if _log_file:
        _fwrite(f"\n=== Log ended: {datetime.now().isoformat()} ===")
        _log_file.close()
        _log_file = None


def _fwrite(text):
    """Write a line to the log file if open."""
    if _log_file:
        _log_file.write(text + "\n")


def _fwrite_separator(char="=", width=120):
    """Write a separator line."""
    _fwrite(char * width)


def log_header(title, subtitle=None):
    """Log a major section header."""
    text = f"[bold white]{title}[/]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/]"
    console.print(Panel(text, style="bold cyan", box=box.DOUBLE, expand=True))
    _fwrite_separator("=")
    _fwrite(f"  {title}")
    if subtitle:
        _fwrite(f"  {subtitle}")
    _fwrite_separator("=")


def log_phase(phase_name, detail=None):
    """Log a phase transition."""
    msg = f"[bold yellow]{phase_name}[/]"
    if detail:
        msg += f"  [dim]{detail}[/]"
    console.print(Rule(msg, style="yellow"))
    _fwrite("")
    _fwrite_separator("-")
    line = f"  PHASE: {phase_name}"
    if detail:
        line += f"  ({detail})"
    _fwrite(line)
    _fwrite_separator("-")


def log_turn(turn_number, max_turns, model_name):
    """Log the start of a new LLM turn."""
    console.print(
        f"\n[bold magenta]{'='*60}[/]"
        f"\n[bold magenta]  TURN {turn_number}/{max_turns}  |  Model: {model_name}[/]"
        f"\n[bold magenta]{'='*60}[/]"
    )
    _fwrite("")
    _fwrite_separator("=", 80)
    _fwrite(f"  TURN {turn_number}/{max_turns}  |  Model: {model_name}  |  Time: {datetime.now().strftime('%H:%M:%S')}")
    _fwrite_separator("=", 80)


def log_prompt_sent(messages, label="MESSAGES TO LLM"):
    """Log what prompt/context is being sent to the LLM."""
    table = Table(title=f"[bold]{label}[/]", box=box.SIMPLE, expand=True, show_lines=True)
    table.add_column("Role", style="cyan", width=10)
    table.add_column("Content", style="white", overflow="fold")
    for msg in messages:
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))
        table.add_row(role, rich_escape(content))
    console.print(table)
    # File version
    _fwrite(f"\n  [{label}] ({len(messages)} messages)")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))
        _fwrite(f"    msg[{i}] role={role}  len={len(content)}")
        _fwrite(f"    preview: {content}")
        _fwrite("")


def log_llm_response(response_text, llm_time, usage, multi_action_count=0):
    """Log the raw LLM response and metadata."""
    console.print(Panel(
        f"[bold green]LLM Response[/]  |  "
        f"Time: [cyan]{llm_time:.1f}s[/]  |  "
        f"Tokens: [cyan]{usage.get('total_tokens', '?')}[/]  |  "
        f"Bash blocks found: [{'red' if multi_action_count > 1 else 'green'}]{multi_action_count}[/]",
        style="green",
        box=box.ROUNDED,
    ))
    console.print(f"[dim]{rich_escape(response_text)}[/]")
    # File version
    _fwrite(f"\n  [LLM RESPONSE]  time={llm_time:.1f}s  tokens={usage.get('total_tokens', '?')}  bash_blocks={multi_action_count}")
    _fwrite(f"  --- FULL RESPONSE ({len(response_text)} chars) ---")
    _fwrite(response_text)
    _fwrite(f"  --- END RESPONSE ---")


def log_multi_action_warning(total_blocks, executed_block):
    """Log when multiple action blocks were detected but only one executed."""
    console.print(Panel(
        f"[bold red]MULTI-ACTION DETECTED[/]\n"
        f"LLM generated [bold]{total_blocks}[/] bash blocks but only the "
        f"[bold]FIRST[/] was executed.\n"
        f"Executed: [green]{rich_escape(executed_block.strip())}[/]",
        style="red",
        box=box.HEAVY,
    ))
    _fwrite(f"\n  *** MULTI-ACTION WARNING: {total_blocks} bash blocks detected, only FIRST executed ***")
    _fwrite(f"  Executed: {executed_block.strip()}")


def log_action(command, action_index=0, total_actions=1):
    """Log an action being executed."""
    console.print(
        f"\n  [bold blue]ACTION [{action_index+1}/{total_actions}][/]  "
        f"[white on blue] {rich_escape(str(command))} [/]"
    )
    _fwrite(f"\n  >> ACTION [{action_index+1}/{total_actions}]: {command}")


def log_observation(observation_text, return_code=None):
    """Log the observation/result from an action."""
    rc_color = "green" if return_code == 0 or return_code == 'unknown' else "red"
    rc_str = f"  [bold {rc_color}]rc={return_code}[/]" if return_code is not None else ""
    console.print(Panel(
        rich_escape(observation_text),
        title=f"[bold]OBSERVATION{rc_str}[/]",
        style="dim",
        box=box.ROUNDED,
    ))
    _fwrite(f"  << OBSERVATION (rc={return_code}):")
    _fwrite(observation_text)
    _fwrite(f"  << END OBSERVATION")


def log_rocm_no_test_block():
    """Log when ROCm mode blocks the 'no tests = success' shortcut."""
    console.print(Panel(
        "[bold red]ROCm MODE: Blocked 'no tests = success' shortcut.[/]\n"
        "The agent must verify by running actual project scripts.\n"
        "runtest is disabled in ROCm mode.",
        style="red",
        box=box.DOUBLE,
    ))
    _fwrite(f"\n  *** ROCm: BLOCKED 'runtest' -- must verify via actual scripts ***")


def log_rocm_success():
    """Log when ROCm mode declares success after script verification."""
    console.print(Panel(
        "[bold green]ROCm VERIFICATION COMPLETE[/]\n"
        "The agent has verified imports and script execution.\n"
        "Environment is configured successfully.",
        style="bold green",
        box=box.DOUBLE,
    ))
    _fwrite(f"\n  *** ROCm: ENVIRONMENT VERIFIED SUCCESSFULLY ***")


def log_revert(command, return_code, reason=""):
    """Log when the sandbox reverts due to a failed command."""
    console.print(Panel(
        f"[bold red]REVERT[/]  cmd: {rich_escape(str(command))}  rc={return_code}\n"
        f"Reason: {rich_escape(str(reason))}",
        style="red",
        box=box.HEAVY,
    ))
    _fwrite(f"\n  !!! REVERT: cmd={command}  rc={return_code}  reason={reason}")


def log_skip_revert(command, return_code, reason=""):
    """Log when a revert is SKIPPED (e.g., timeout on non-destructive command)."""
    console.print(Panel(
        f"[bold yellow]REVERT SKIPPED[/]  cmd: {rich_escape(str(command))}  rc={return_code}\n"
        f"Reason: {rich_escape(str(reason))}",
        style="yellow",
        box=box.ROUNDED,
    ))
    _fwrite(f"\n  --- REVERT SKIPPED: cmd={command}  rc={return_code}  reason={reason}")


def log_container_op(op_name, detail=""):
    """Log a Docker container operation."""
    console.print(f"  [bold cyan]CONTAINER[/] {op_name}  [dim]{detail}[/]")
    _fwrite(f"  [CONTAINER] {op_name}  {detail}")


def log_success(message):
    """Log a success message."""
    console.print(Panel(f"[bold green]{message}[/]", style="green", box=box.DOUBLE))
    _fwrite(f"  [SUCCESS] {message}")


def log_error(message):
    """Log an error message."""
    console.print(Panel(f"[bold red]{message}[/]", style="red", box=box.HEAVY))
    _fwrite(f"  [ERROR] {message}")


def log_info(message):
    """Log an informational message."""
    console.print(f"  [blue]INFO[/] {message}")
    _fwrite(f"  [INFO] {message}")


def log_warning(message):
    """Log a warning message."""
    console.print(f"  [yellow]WARNING[/] {message}")
    _fwrite(f"  [WARNING] {message}")


def log_context_summary(current_dir, image_name, turns_left, success_cmds):
    """Log the context being sent back to the LLM."""
    table = Table(title="[bold]CONTEXT SENT BACK TO LLM[/]", box=box.SIMPLE, expand=True)
    table.add_column("Key", style="cyan", width=20)
    table.add_column("Value", style="white", overflow="fold")
    table.add_row("Current Directory", current_dir)
    table.add_row("Container Image", image_name)
    table.add_row("Turns Remaining", str(turns_left))
    table.add_row("Successful Commands", str(len(success_cmds)))
    if success_cmds:
        cmds_preview = "\n".join(success_cmds[-10:])
        if len(success_cmds) > 10:
            cmds_preview = f"... ({len(success_cmds)-10} earlier commands)\n" + cmds_preview
        table.add_row("Recent Commands", cmds_preview)
    console.print(table)
    # File version
    _fwrite(f"\n  [CONTEXT -> LLM]")
    _fwrite(f"    cwd={current_dir}  image={image_name}  turns_left={turns_left}")
    _fwrite(f"    successful_cmds ({len(success_cmds)}):")
    for cmd in success_cmds:
        _fwrite(f"      - {cmd}")


def log_finish_summary(total_turns, total_time, total_tokens, success):
    """Log the final summary when the agent finishes."""
    status_str = "SUCCESS" if success else "INCOMPLETE"
    status = f"[bold green]{status_str}[/]" if success else f"[bold red]{status_str}[/]"
    console.print(Panel(
        f"Status: {status}\n"
        f"Total Turns: [cyan]{total_turns}[/]\n"
        f"Total Time: [cyan]{total_time:.1f}s[/]\n"
        f"Total Tokens: [cyan]{total_tokens}[/]",
        title="[bold]AGENT FINISHED[/]",
        style="bold",
        box=box.DOUBLE,
    ))
    _fwrite_separator("=")
    _fwrite(f"  AGENT FINISHED")
    _fwrite(f"  Status: {status_str}")
    _fwrite(f"  Total Turns: {total_turns}")
    _fwrite(f"  Total Time: {total_time:.1f}s")
    _fwrite(f"  Total Tokens: {total_tokens}")
    _fwrite_separator("=")
