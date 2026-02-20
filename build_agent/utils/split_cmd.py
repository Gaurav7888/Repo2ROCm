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


def _contains_heredoc(cmd):
    """Check if a command contains a heredoc (<<EOF, << 'EOF', << "EOF", etc.)."""
    return bool(re.search(r'<<-?\s*[\'"]?\w+[\'"]?', cmd))


def _contains_multiline_python(cmd):
    """Check if a command contains a multiline python -c with embedded newlines."""
    if 'python' not in cmd and 'python3' not in cmd:
        return False
    # Detect python -c "...\n..." or python -c '...\n...'
    if re.search(r'python[3]?\s+-c\s+["\']', cmd) and '\n' in cmd:
        return True
    return False


def split_cmd_statements(cmd):
    # If the command contains a heredoc or multiline python -c,
    # preserve it as a single command without collapsing newlines.
    if _contains_heredoc(cmd) or _contains_multiline_python(cmd):
        # Only strip leading/trailing whitespace, keep internal newlines
        stripped = cmd.strip()
        if stripped:
            return [stripped]
        return []

    # For simple commands: remove line-continuation backslashes
    cmd = re.sub(r'\\\s*\n', '', cmd)

    # Replace plain newlines with spaces (safe for single-line commands)
    cmd = re.sub(r'\n', ' ', cmd)

    # Split on && that is NOT inside quotes
    # Simple approach: split on && and rejoin if inside an unmatched quote
    parts = []
    current = []
    in_single = False
    in_double = False
    i = 0
    chars = cmd
    while i < len(chars):
        c = chars[i]
        # Track quote state
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
            i += 1
        elif c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
            i += 1
        elif c == '\\' and i + 1 < len(chars):
            current.append(c)
            current.append(chars[i + 1])
            i += 2
        elif c == '&' and i + 1 < len(chars) and chars[i + 1] == '&' and not in_single and not in_double:
            # Split point
            parts.append(''.join(current).strip())
            current = []
            i += 2
        else:
            current.append(c)
            i += 1
    # Don't forget the last segment
    if current:
        parts.append(''.join(current).strip())

    return [p for p in parts if p]

if __name__ == "__main__":
    # 示例输入
    # cmd = "echo Hello World\\\necho This is\\ a test && echo Another command"
    cmd = '''waitinglist add -p crawlerdetect -v "~0.1.7" -t pip && \
waitinglist add -p fastapi -v "~0.110" -t pip && \
waitinglist add -p fuzzywuzzy -v "~0.18" -t pip && \
waitinglist add -p gitignore-parser -v "==0.1.11" -t pip && \
waitinglist add -p imy[docstrings] -v ">=0.4.0" -t pip && \
waitinglist add -p introspection -v "~1.9.2" -t pip && \
waitinglist add -p isort -v "~5.13" -t pip && \
waitinglist add -p keyring -v "~24.3" -t pip && \
waitinglist add -p langcodes -v ">=3.4.0" -t pip && \
waitinglist add -p narwhals -v ">=1.12.1" -t pip && \
waitinglist add -p ordered-set -v ">=4.1.0" -t pip && \
waitinglist add -p path-imports -v ">=1.1.2" -t pip && \
waitinglist add -p pillow -v "~10.2" -t pip && \
waitinglist add -p python-levenshtein -v "~0.23" -t pip && \
waitinglist add -p python-multipart -v "~0.0.6" -t pip && \
waitinglist add -p pytz -v "~2024.1" -t pip && \
waitinglist add -p revel -v "~0.9.1" -t pip && \
waitinglist add -p timer-dict -v "~1.0" -t pip && \
waitinglist add -p tomlkit -v "~0.12" -t pip && \
waitinglist add -p typing-extensions -v ">=4.5" -t pip && \
waitinglist add -p unicall -v "~0.1.5" -t pip && \
waitinglist add -p uniserde -v "~0.3.14" -t pip && \
waitinglist add -p uvicorn[standard] -v "~0.29.0" -t pip && \
waitinglist add -p watchfiles -v "~0.21" -t pip && \
waitinglist add -p yarl -v ">=1.9" -t pip
'''
    # 调用函数拆分子语句
    result = split_cmd_statements(cmd)
    print(result)

    # 打印结果
    for i, statement in enumerate(result, 1):
        print(f"Statement {i}: {statement}")