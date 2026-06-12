# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 


import re
import shlex
from .parse_dialogue import extract_dialogue_warnings

BASH_FENCE = ['```bash', '```']

def extract_commands(text):
    pattern = rf'{BASH_FENCE[0]}([\s\S]*?){BASH_FENCE[1]}'
    matches = re.findall(pattern, text)

    commands = []
    for command_text in matches:
        if command_text:
            commands.extend(list(filter(None, command_text.strip().split('\n'))))
    
    return commands

def extract_commands_warnings(text):
    thought, action = extract_dialogue_warnings(text)
    if thought and action:
        commands = extract_commands(action)
        if len(commands) == 0:
            print(f'''Wrong! Please note that the Action part of your response does not contain any actionable commands surrounded by {BASH_FENCE[0]} and {BASH_FENCE[1]} that can be executed. Please regenerate the action.
*Note*: Each Action part of your responses must contain and only contain one actionable command surrounded by {BASH_FENCE[0]} and {BASH_FENCE[1]}.\n''')
            return -1
        if len(commands) > 1:
            command_msg = '\n'.join(commands)
            print(f'''Please note that the Action part of your response contains more than one actionable command surrounded by {BASH_FENCE[0]} and {BASH_FENCE[1]} , including:
{command_msg}
Please regenerate to ensure that the Action part contains exactly one command surrounded by {BASH_FENCE[0]} and {BASH_FENCE[1]}.
*Note*: If you want to execute multiple actions, you can write them all within one command block surrounded by {BASH_FENCE[0]} and {BASH_FENCE[1]}, or consider executing one action and then performing the next one in the subsequent round of the conversation.\n''')    
            return -1
        else:
            print(f"Successfully extracted the command `{commands[0]}`, about to execute...")
            return commands[0]
    else:
        return -1

# 匹配`download`指令，如果是这个指令，则返回True，否则返回False
def match_download(text):
    # 正则表达式
    pattern = re.compile(r'^\s*download\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(text)
    return bool(match)

# 匹配`runpipreqs`指令，如果是这个指令，则返回True，否则返回False
def match_runpipreqs(text):
    # 正则表达式
    pattern = re.compile(r'^\s*runpipreqs\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(text)
    return bool(match)

def match_runtest(text):
    # 正则表达式
    pattern = re.compile(r'^\s*runtest\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(text)
    return bool(match)

def match_poetryruntest(text):
    # 正则表达式
    pattern = re.compile(r'^\s*poetryruntest\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(text)
    return bool(match)

def match_conflict_solve(text):
    # 正则表达式模式
    pattern = re.compile(
        # r'\s*conflictlist\s+solve\s*(?:(-v| -V)\s*["\']([<>=!]=?\d+\.\d+)["\']\s*|\s*(-u\s*))?',
        r'\s*conflictlist\s+solve\s*(?:(-v| -V)\s*["\']([<>=!]=?\d+(\.\d+)*?)["\']\s*|\s*(-u\s*))?',
        re.IGNORECASE
    )
    
    match = pattern.fullmatch(text.strip())
    
    if not match:
        return -1
    
    args = {
        'conflictlist_solve': True,
        'version_constraint': None,
        'unchanged': False
    }
    
    # 检查-v的匹配段
    if match.group(1) and match.group(2):
        args['version_constraint'] = match.group(2)
    elif match.group(3) is not None:
        args['unchanged'] = True
    
    return args

# def match_errorformatlist_sovle(command):
#     # 正则表达式模式
#     pattern = re.compile(
#         # r'\s*errorformatlist\s+solve\s*(["\'].*?["\']\s*)*',
#         r'\s*errorformatlist\s+solve\s*(?:["\'].*?["\']\s*)*',
#         re.IGNORECASE
#     )

#     match = pattern.fullmatch(command.strip())

#     if not match:
#         return -1
    
#     # 提取所有以单引号或双引号包裹的条目
#     entries = re.findall(r'["\'](.*?)["\']', command)
    
#     args = {
#         'errorformatlist_solve': True,
#         'entries': entries
#     }

#     return args

def match_waitinglist_add(command):
    # Normalize the command by converting to lowercase and removing extra spaces
    command = re.sub(r'\s+', ' ', command.strip().lower())
    
    # Define the pattern to match the command format
    pattern = r"waitinglist add -p ([^\s]+)( -v ([^\s]+))? -t ([^\s]+)"
    
    # Match the command against the pattern
    match = re.match(pattern, command)
    
    if match:
        # Extract package_name, version_constraints, and tool
        package_name = match.group(1)
        version_constraints = match.group(3) if match.group(3) else None
        tool = match.group(4)
        return {
            "package_name": package_name,
            "version_constraints": version_constraints,
            "tool": tool
        }
    else:
        return -1

def match_waitinglist_addfile(command):
    # Normalize the command by converting to lowercase and removing extra spaces
    command = re.sub(r'\s+', ' ', command.strip().lower())
    
    # Define the pattern to match the command format
    pattern = r"waitinglist addfile ([^\s]+)"
    
    # Match the command against the pattern
    match = re.match(pattern, command)
    
    if match:
        # Extract file_path
        file_path = match.group(1)
        return {
            "file_path": file_path
        }
    else:
        return -1

def match_waitinglist_show(command):
    pattern = re.compile(r'^\s*waitinglist show\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(command)
    return bool(match)

def match_waitinglist_clear(command):
    pattern = re.compile(r'^\s*waitinglist clear\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(command)
    return bool(match)

# def match_errorformatlist_clear(command):
#     pattern = re.compile(r'^\s*errorformatlist clear\s*$', re.IGNORECASE | re.MULTILINE)
#     match = pattern.match(command)
#     return bool(match)

def match_conflictlist_show(command):
    pattern = re.compile(r'^\s*conflictlist show\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(command)
    return bool(match)

def match_conflictlist_clear(command):
    pattern = re.compile(r'^\s*conflictlist clear\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(command)
    return bool(match)

def match_clear_configuration(command):
    pattern = re.compile(r'^\s*clear_configuration\s*$', re.IGNORECASE | re.MULTILINE)
    match = pattern.match(command)
    return bool(match)

# ── Stage 5b: in-loop retrieval tools ────────────────────────────────────────


def _split_tool_command(command: str, tool_name: str):
    """Tokenize a tool command using shell rules and verify the tool name."""
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        return None
    if not tokens or tokens[0].lower() != tool_name.lower():
        return None
    return tokens[1:]


def _parse_known_flags(tokens, value_flags=(), boolean_flags=(), multi_value_flags=()):
    """Parse a small shell-style flag set without regexes."""
    value_flags = set(value_flags)
    boolean_flags = set(boolean_flags)
    multi_value_flags = set(multi_value_flags)

    positionals = []
    options = {flag: [] for flag in multi_value_flags}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in value_flags:
            if i + 1 >= len(tokens):
                return None
            options[token] = tokens[i + 1]
            i += 2
        elif token in multi_value_flags:
            if i + 1 >= len(tokens):
                return None
            options[token].append(tokens[i + 1])
            i += 2
        elif token in boolean_flags:
            options[token] = True
            i += 1
        elif token.startswith("--"):
            return None
        else:
            positionals.append(token)
            i += 1
    return positionals, options


def _parse_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def match_mem_recall(command: str):
    """
    Parse:  mem_recall "<question>" [--rooms r1,r2] [--budget N] [--global]
    Returns dict with keys: question, rooms (list|None), budget (int), use_global (bool)
    on success; -1 on no match.
    """
    tokens = _split_tool_command(command, "mem_recall")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(
        tokens,
        value_flags=("--rooms", "--budget"),
        boolean_flags=("--global",),
    )
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    rooms_raw = options.get("--rooms")
    rooms = [r.strip() for r in rooms_raw.split(",") if r.strip()] if rooms_raw else None
    budget = _parse_int(options.get("--budget"), default=1500)
    if budget is None:
        return -1
    return {
        "question": positionals[0],
        "rooms": rooms,
        "budget": budget,
        "use_global": bool(options.get("--global")),
    }


def match_paper_recall(command: str):
    """
    Parse:  paper_recall "<question>" [--budget N] [--global]
    Returns dict with keys: question, budget (int), use_global (bool)
    on success; -1 on no match.
    """
    tokens = _split_tool_command(command, "paper_recall")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(
        tokens,
        value_flags=("--budget",),
        boolean_flags=("--global",),
    )
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    budget = _parse_int(options.get("--budget"), default=1500)
    if budget is None:
        return -1
    return {
        "question": positionals[0],
        "budget": budget,
        "use_global": bool(options.get("--global")),
    }


def match_graphify_query(command: str):
    """
    Parse:  graphify_query "<question>" [--scope paper|code|both] [--budget N]
    Returns dict with keys: question, scope, budget on success; -1 on no match.
    `scope` defaults to "code" to preserve the historical behaviour of this tool.
    """
    tokens = _split_tool_command(command, "graphify_query")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(tokens, value_flags=("--scope", "--budget"))
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    budget = _parse_int(options.get("--budget"), default=1500)
    if budget is None:
        return -1
    scope = str(options.get("--scope") or "code").lower()
    if scope not in ("paper", "code", "both"):
        return -1
    return {
        "question": positionals[0],
        "scope": scope,
        "budget": budget,
    }


def match_verify_paper_result(command: str):
    """
    Parse:  verify_paper_result --log <path>
                                [--metric NAME=VALUE]...
                                [--tolerance RULE]
                                [--direction higher_is_better|lower_is_better|equal]

    Returns dict {log_path, metrics: [{name, expected_value}], tolerance, direction}
    on success; -1 on no match.

    `--metric` may be repeated. The expected value is parsed as a float when
    possible and otherwise kept verbatim so qualitative claims still get a
    deterministic record. If no `--metric` is supplied, the dispatcher will
    fall back to the chosen experiment's `expected_metric_name` /
    `expected_metric_value` (and any `primary_metrics` list).
    """
    tokens = _split_tool_command(command, "verify_paper_result")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(
        tokens,
        value_flags=("--log", "--tolerance", "--direction"),
        multi_value_flags=("--metric",),
    )
    if parsed is None:
        return -1
    positionals, options = parsed
    if positionals:
        return -1
    log_path = options.get("--log")
    if not log_path:
        return -1

    metrics: list = []
    for metric_spec in options.get("--metric", []):
        if "=" not in metric_spec:
            return -1
        name, raw_val = metric_spec.split("=", 1)
        name = name.strip()
        raw_val = raw_val.strip()
        if not name:
            return -1
        try:
            val: object = float(raw_val)
        except (TypeError, ValueError):
            val = raw_val
        metrics.append({"name": name, "expected_value": val})

    tolerance = options.get("--tolerance") or ""
    direction = str(options.get("--direction") or "").lower()
    if direction and direction not in ("higher_is_better", "lower_is_better", "equal"):
        return -1

    return {
        "log_path": log_path,
        "metrics": metrics,
        "tolerance": tolerance,
        "direction": direction,
    }


# ── PR-A: external lookups (PyPI versions, Docker Hub tags) ──────────────────

def match_pypi_versions(command: str):
    """
    Parse:  pypi_versions <package_name> [--limit N]
    Package name allows letters, digits, dots, hyphens, underscores.
    Returns dict with keys: package, limit on success; -1 on no match.
    """
    tokens = _split_tool_command(command, "pypi_versions")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(tokens, value_flags=("--limit",))
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    limit = _parse_int(options.get("--limit"), default=12)
    if limit is None:
        return -1
    return {
        "package": positionals[0],
        "limit": limit,
    }


def match_dockerhub_tags(command: str):
    """
    Parse:  dockerhub_tags <image> [--limit N]
    Image is `repo/name` or just `name`.
    Returns dict with keys: image, limit on success; -1 on no match.
    """
    tokens = _split_tool_command(command, "dockerhub_tags")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(tokens, value_flags=("--limit",))
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    limit = _parse_int(options.get("--limit"), default=12)
    if limit is None:
        return -1
    return {
        "image": positionals[0],
        "limit": limit,
    }


# ── PR-B: web search + URL fetcher ───────────────────────────────────────────

def match_web_search(command: str):
    """
    Parse:  web_search "<query>" [--max-results N]
    Returns dict {query, max_results} on success; -1 on no match.
    """
    tokens = _split_tool_command(command, "web_search")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(tokens, value_flags=("--max-results",))
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    max_results = _parse_int(options.get("--max-results"), default=5)
    if max_results is None:
        return -1
    return {
        "query": positionals[0],
        "max_results": max_results,
    }


def match_visit_url(command: str):
    """
    Parse:  visit_url <url> [--max-chars N]
    Returns dict {url, max_chars} on success; -1 on no match.
    """
    tokens = _split_tool_command(command, "visit_url")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(tokens, value_flags=("--max-chars",))
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    max_chars = _parse_int(options.get("--max-chars"), default=8000)
    if max_chars is None:
        return -1
    return {
        "url": positionals[0],
        "max_chars": max_chars,
    }


# ── PR-C: deep_research sub-agent ────────────────────────────────────────────

def match_deep_research(command: str):
    """
    Parse:  deep_research "<question>" [--max-turns N] [--budget-s S] [--no-cache]
    Returns dict {question, max_turns, budget_s, use_cache} on success; -1 on no match.
    """
    tokens = _split_tool_command(command, "deep_research")
    if tokens is None:
        return -1
    parsed = _parse_known_flags(
        tokens,
        value_flags=("--max-turns", "--budget-s"),
        boolean_flags=("--no-cache",),
    )
    if parsed is None:
        return -1
    positionals, options = parsed
    if len(positionals) != 1:
        return -1
    max_turns = _parse_int(options.get("--max-turns"), default=6)
    budget_s = _parse_float(options.get("--budget-s"), default=90.0)
    if max_turns is None or budget_s is None:
        return -1
    return {
        "question": positionals[0],
        "max_turns": max_turns,
        "budget_s": budget_s,
        "use_cache": not bool(options.get("--no-cache")),
    }

if __name__ == '__main__':
    print(extract_commands_warnings('''
### Thought: hello
### Action:
```bash
waitinglist solve
```
sdfldfks
'''))
    print('*'*100)
    print(extract_commands_warnings('''
### Thought:
```bash
waitinglist solve
```
### Action:
hello
'''))
    # 测试match_download
    commands = [
        'download',
        ' Download ',
        'DownlOad   ',
        'download -'
    ]
    for cmd in commands:
        print(f'Command: {cmd}')
        print(f'Parsed: {match_download(cmd)}')
        print('---')
    
    # 测试match_conflict_solve
    commands = [
        'conflictlist solve',
        'conflictlist    solve   -v  "==2.0"',
        "conflictlist solve -V '>3.0'",
        "Conflictlist   solve  -u",
        "cOnflictlist   solvE  -v '>=1.2'",
        'conflictlist  solve  -u',
        'conflict solve -v ">s"',
        'conflictlist solve -v "torch==2.5.0"'
    ]
    print('@'*100)
    for cmd in commands:
        print(f'Command: {cmd}')
        print(f'Parsed: {match_conflict_solve(cmd)}')
        print('---')

    # # 测试match_errorfomatlist_solve
    # commands = [
    #     'errorformatlist solve',
    #     'errorformatlist  Solve  "numpy==1.2.0"',
    #     'errorformatlist solve   "numpy" \'matplotlib>=2.0\'',
    #     "ErrorFormatList Solve 'pandas<=1.0'   'scipy' 'clash<1.2' \"sos.s > 1\"",
    #     'errorformatlist   solve',
    #     'errorformatlist solve "text>1"',
    #     "errorformat solve \"sss\" pandas<=1.0"
    # ]
    # for cmd in commands:
    #     print(f'Command: {cmd}')
    #     print(f'Parsed: {match_errorformatlist_sovle(cmd)}')
    #     print('---')

    # Test cases
    commands = [
        "waitinglist add -p package_name1 -v >=1.0.0 -t pip",
        "waitinglist add -p package_name2 -t pip",
        "waitinglist add -p package_name3 -v ==2.0.0 -t pip",
        "waitingList add -p package_name4 -t pip",
        "waitinglist add   -p package_name5 -t apt"
    ]

    for cmd in commands:
        print(f"Command: {cmd}")
        print(f"Version Constraints: {match_waitinglist_add(cmd)}\n")

    # Test cases for waiting_list_add_file
    file_commands = [
        "waitinglist addfile /path/to/file",
        "waitingList addfile  anotherfile.txt",
        "waitinglist addfilE  /path/with spaces/file.txt",
        "waitinglist add /sss"
    ]

    for cmd in file_commands:
        print(f"Command: {cmd}")
        print(f"Details: {match_waitinglist_addfile(cmd)}\n")