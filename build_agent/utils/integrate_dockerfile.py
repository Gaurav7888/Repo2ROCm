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


import os
import json
import subprocess
import argparse
import shlex
import re

def find_package_version(package_name, dependencies):
    """
    遍历依赖树，查找指定包的实际安装版本
    :param package_name: 要查找的包名
    :param dependencies: 依赖树列表
    :return: 包的实际安装版本或 None 如果未找到
    """
    for package in dependencies:
        if package["key"].lower().replace('-', '_').replace('.', '_') == package_name.lower().replace('-', '_').replace('.', '_'):
            return package["installed_version"]
        # 递归查找子依赖
        sub_version = find_package_version(package_name, package["dependencies"])
        if sub_version:
            return sub_version
    return None

# 用于提取package_name
def extract_package_info(package_with_constraints):
    """
    从形式如 'requests==2.25.1' 的字符串中提取 package_name 和 version_constraints
    """
    pattern = re.compile(r'(?P<package_name>^[^=<>!~]+)(?P<version_constraints>.*)')
    match = pattern.match(package_with_constraints)
    
    if not match:
        raise ValueError(f"Invalid package string: {package_with_constraints}")
    
    package_name = match.group('package_name').strip()
    version_constraints = match.group('version_constraints').strip()
    
    return package_name

# 解析python /home/tools/pip_download.py指令参数
def parse_arguments(command):
    """
    解析包含命令行参数的字符串，提取参数值
    """
    # 使用 shlex.split 分割命令字符串
    args = shlex.split(command)
    # 创建解析器
    parser = argparse.ArgumentParser(description='Install a Python package with pip.')
    parser.add_argument('-p', '--package_name', required=True, type=str, help='The name of the package to install.')
    parser.add_argument('-v', '--version_constraints', type=str, default='', nargs='?', help='The version constraints of the package.')
    # 解析分割后的参数
    parsed_args = parser.parse_args(args[2:])  # 跳过第一个参数（脚本名）
    return parsed_args

# 解析pip install指令参数
def parse_pip_install_arguments(command):
    """
    解析包含 pip install 命令行参数的字符串，提取参数值
    """
    # 使用 shlex.split 分割命令字符串以处理引号和特殊字符
    args = shlex.split(command)

    # 创建解析器
    parser = argparse.ArgumentParser(description='Parse pip install command arguments.')

    # 定义位置参数（包名或者是 requirements）
    parser.add_argument(
        'requirements',
        nargs=argparse.REMAINDER,  # 支持多个包名或要求，获取所有剩余参数
        help='The packages or requirements to install.'
    )

    # 定义常见的 pip install 参数
    parser.add_argument(
        '-r', '--requirement',
        action='append',
        help='Install from the given requirements file. This option can be used multiple times.'
    )
    parser.add_argument(
        '-e', '--editable',
        action='append',
        help='Install a project in editable mode (i.e. setuptools "develop mode"). This option can be used multiple times.'
    )
    parser.add_argument(
        '--no-deps',
        action='store_true',
        help='Do not install package dependencies.'
    )
    parser.add_argument(
        '-t', '--target',
        help='Install packages into <dir>.'
    )
    parser.add_argument(
        '-U', '--upgrade',
        action='store_true',
        help='Upgrade all specified packages to the newest available version.'
    )
    parser.add_argument(
        '--force-reinstall',
        action='store_true',
        help='Reinstall all packages even if they are already up-to-date.'
    )
    parser.add_argument(
        '--no-cache-dir',
        action='store_true',
        help='Disable the cache.'
    )
    parser.add_argument(
        '--user',
        action='store_true',
        help='Install to the Python user install directory for your platform.'
    )
    parser.add_argument(
        '--prefix',
        help='Installation prefix.'
    )
    parser.add_argument(
        '--src',
        help='Directory to check out editable projects into. The default in a virtualenv is "<venv path>/src".'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='count',
        default=0,
        help='Give less output. Option can be used multiple times to increase verbosity.'
    )

    parser.add_argument(
        '-qq', '--quitequiet',
        action='count',
        default=0,
        help='Give less output. Option can be used multiple times to increase verbosity.'
    )
    # 解析分割后的参数
    parsed_args = parser.parse_args(args[1:])  # 跳过第一个参数（pip install）

    return parsed_args


safe_cmd = [
    "cd", "ls", "cat", "echo", "pwd", "whoami", "who", "date", "cal", "df", "du",
    "free", "uname", "uptime", "w", "ps", "pgrep", "top", "htop", "vmstat", "iostat",
    "dmesg", "tail", "head", "more", "less", "grep", "find", "locate", "whereis", "which",
    "file", "stat", "cmp", "diff", "md5sum", "sha256sum", "gzip", "gunzip", "bzip2", "bunzip2",
    "xz", "unxz", "sort", "uniq", "wc", "tr", "cut", "paste", "tee", "awk", "sed", "env", "printenv",
    "hostname", "ping", "traceroute", "ssh", "journalctl","lsblk", "blkid", "uptime",
    "lscpu", "lsusb", "lspci", "lsmod", "dmidecode", "ip", "ifconfig", "netstat", "ss", "route", "nmap",
    "strace", "ltrace", "time", "nice", "renice", "killall", "printf"
    ]

# 替换没有版本号的包名成带版本号的函数
def replace_versions(command, pipdeptree_data):
    # print(command)
    args = parse_pip_install_arguments(command)
    new_requirements = []

    for requirement in args.requirements[1:]:
        if '==' not in requirement:
            # 未指定版本号的包名替换为 `<package_name>==<version>`
            package_name = requirement
            package_version = find_package_version(package_name, pipdeptree_data)
            if package_version:
                new_requirements.append(f'{package_name}=={package_version}')
            else:
                new_requirements.append(requirement)
        else:
            new_requirements.append(requirement)
    
    # print(new_requirements)
    # 根据解析的内容重构命令
    if len(new_requirements) > 0:
        new_command = f'pip install {" ".join(new_requirements)}'
    else:
        new_command = 'pip install'
    
    if args.requirement:
        new_command += ' ' + ' '.join(f'-r {req}' for req in args.requirement)
    if args.editable:
        new_command += ' ' + ' '.join(f'-e {edit}' for edit in args.editable)
    if args.no_deps:
        new_command += ' --no-deps'
    if args.target:
        new_command += f' -t {args.target}'
    if args.upgrade:
        new_command += ' -U'
    if args.force_reinstall:
        new_command += ' --force-reinstall'
    if args.no_cache_dir:
        new_command += ' --no-cache-dir'
    if args.user:
        new_command += ' --user'
    if args.prefix:
        new_command += f' --prefix {args.prefix}'
    if args.src:
        new_command += f' --src {args.src}'

    return new_command

def generate_statement(inner_command, pipdeptree_data):
    # print(inner_command)
    command = inner_command['command']
    dir = inner_command['dir'] if 'dir' in inner_command else '/'
    returncode = inner_command['returncode']
    action_name = command.split(' ')[0].strip()
    if str(returncode).strip() != '0':
        return -1
    if action_name in ['pipdeptree']:
        return -1
    if action_name in safe_cmd and '>' not in command:
        return -1
    if command == 'python /home/tools/runtest.py' or command == 'python /home/tools/poetryruntest.py' or command == 'python /home/tools/runpipreqs.py' or command == 'python /home/tools/generate_diff.py':
        return -1
    if action_name == 'change_python_version':
        return f'FROM python:{command.split(" ")[1].strip()}'
    if action_name == 'change_base_image':
        return f'FROM {command.split(" ")[1].strip()}'
    if action_name == 'clear_configuration':
        return 'FROM python:3.10'
    if action_name == 'export':
        return f'ENV {command.split("export ")[1]}'
    
    if command.startswith('python /home/tools/pip_download.py'):
        # print(command)
        args = parse_arguments(command)
        # print(args.package_name)
        package_name = args.package_name
        package_version = find_package_version(package_name, pipdeptree_data)
        if package_version is None:
            return -1
        else:
            return f'RUN pip install {package_name}=={package_version}'
    # requirements = list()
    # if command.startswith('pip install'):
    #     args = parse_pip_install_arguments(command)
    #     for requirement in args.requirements:
    #         package_name = extract_package_info(requirement)
    #         package_version = find_package_version(package_name, pipdeptree_data)
    #         if package_version is None:
    #             # return -1c
    #             continue
    #         else:
    #             # return f'RUN pip install {package_name}=={package_version}'
    #             requirements.append(f'{package_name}=={package_version}')
    if command.startswith('pip install'):
        if dir != '/':
            return f'RUN cd {dir} && {replace_versions(command, pipdeptree_data)}'
        else:
            return f'RUN {replace_versions(command, pipdeptree_data)}'
    if dir != '/':
        return f'RUN cd {dir} && {command}'
    else:
        return f'RUN {command}'

def _is_rocm_base_image(base_image_st):
    """Check if the FROM statement uses a ROCm image."""
    image_name = base_image_st.replace('FROM ', '').strip().lower()
    return image_name.startswith('rocm/') or 'rocm' in image_name.split(':')[0]


def _get_pre_download(base_image_st):
    """Generate pre_download commands based on the base image type."""
    if _is_rocm_base_image(base_image_st):
        return (
            'RUN apt-get update && apt-get install -y curl git\n'
            'RUN curl -sSL https://install.python-poetry.org | python3 -\n'
            'ENV PATH="/root/.local/bin:$PATH"\n'
            'RUN pip install pytest pytest-xdist pipdeptree pipreqs'
        )
    else:
        return (
            'RUN apt-get update && apt-get install -y curl\n'
            'RUN curl -sSL https://install.python-poetry.org | python -\n'
            'ENV PATH="/root/.local/bin:$PATH"\n'
            'RUN pip install pytest pytest-xdist\n'
            'RUN pip install pipdeptree'
        )


def _normalize_run(line: str) -> str:
    """Loose-normalize a `RUN ...` line for duplicate detection.

    Strips `RUN`, an optional leading `cd <dir> &&`, runs of whitespace, and
    trailing redirection/tee suffixes (`2>&1`, `| tee /tmp/..`, `| tail`).
    Equality after normalization implies the two lines do equivalent work.
    """
    s = line.strip()
    if s.startswith('RUN '):
        s = s[4:].strip()
    s = re.sub(r'^cd\s+\S+\s*&&\s*', '', s, count=1)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*\|\s*tee\s+\S+\s*$', '', s)
    s = re.sub(r'\s*\|\s*tail(\s+-\w+\s+\d+)?\s*$', '', s)
    s = re.sub(r'\s*2>&1\s*$', '', s)
    return s.strip()


def _is_training_like(line: str) -> bool:
    """Heuristic for "long python entry-script invocation" — the kind that
    typically produces multiple near-duplicate retries differing in one flag."""
    s = line.strip()
    if not s.startswith('RUN '):
        return False
    if 'python ' not in s and 'python3 ' not in s and 'accelerate.commands.launch' not in s:
        return False
    if len(s) < 200:
        return False
    # crude entry-script signal
    return bool(re.search(r'\.py\b', s)) and not s.endswith('--help')


def _dedupe_runs(run_lines: list) -> list:
    """De-duplicate Dockerfile RUN lines.

    1) Exact normalized duplicates → keep last.
    2) Training-like near-duplicates that share the entry script → keep last.
    """
    if not run_lines:
        return run_lines

    # Pass 1: drop earlier exact duplicates (by normalized form).
    seen_norm: dict = {}
    for i, ln in enumerate(run_lines):
        seen_norm[_normalize_run(ln)] = i  # last index wins
    pass1 = [ln for i, ln in enumerate(run_lines) if seen_norm[_normalize_run(ln)] == i]

    # Pass 2: collapse training-like retries (same entry script .py + same dataset).
    def _entry_signature(s: str):
        m_script = re.search(r'(?:python3?\s+|accelerate\.commands\.launch[^\s]*\s+)([\S]+\.py)', s)
        if not m_script:
            return None
        script = m_script.group(1)
        m_ds = re.search(r'\+\+?dataset_name=(\S+)', s)
        ds = m_ds.group(1) if m_ds else ''
        return (script, ds)

    keep_idx_for_sig: dict = {}
    for i, ln in enumerate(pass1):
        if _is_training_like(ln):
            sig = _entry_signature(ln)
            if sig is not None:
                keep_idx_for_sig[sig] = i

    pass2 = []
    for i, ln in enumerate(pass1):
        if _is_training_like(ln):
            sig = _entry_signature(ln)
            if sig is not None and keep_idx_for_sig.get(sig) != i:
                continue
        pass2.append(ln)

    return pass2


# root_path must be absolute path
def integrate_dockerfile(root_path, runtime_base_image: str = ""):
    """
    Build the final Dockerfile from inner_commands.json.

    runtime_base_image (new): the base image the agent ACTUALLY ran in (passed
    in by main.py from the Sandbox / planner / CLI). Honored as the initial
    FROM line, so we don't fall back to `python:3.10` when neither
    change_base_image nor change_python_version was explicitly invoked.
    """
    dockerfile = list()
    root_path = os.path.normpath(root_path)
    author_name = root_path.split('/')[-2]
    repo_name = root_path.split('/')[-1]
    if runtime_base_image and runtime_base_image.strip():
        base_image_st = f'FROM {runtime_base_image.strip()}'
    else:
        base_image_st = 'FROM python:3.10'
    workdir_st = f'WORKDIR /'

    copy_st = f'COPY search_patch /search_patch'
    copy_edit_st = f'COPY code_edit.py /code_edit.py'

    git_st = f'RUN git clone https://github.com/{author_name}/{repo_name}.git'
    mkdir_st = 'RUN mkdir /repo'
    git_save_st = 'RUN git config --global --add safe.directory /repo'
    mv_st = f'RUN cp -r /{repo_name}/. /repo && rm -rf /{repo_name}/'
    rm_st = f'RUN rm -rf /{repo_name}'
    with open(f'{root_path}/sha.txt', 'r') as r1:
        sha = r1.read().strip()
    checkout_st = f'RUN cd /repo && git checkout {sha}'
    container_run_set = list()
    if not (os.path.exists(f'{root_path}/inner_commands.json')):
        subprocess.run('touch ERROR', cwd=root_path, shell=True)
    with open(f'{root_path}/inner_commands.json', 'r') as r1:
        commands_data = json.load(r1)
    with open(f'{root_path}/pipdeptree.json', 'r') as r2:
        pipdeptree_data = json.load(r2)
    diff_no = 1
    for command in commands_data:
        res = generate_statement(command, pipdeptree_data)
        if res == -1:
            continue
        if res.startswith('FROM'):
            base_image_st = res
            container_run_set = list()
        elif command['command'].startswith('python /home/tools/code_edit.py'):
            if diff_no == 1:
                container_run_set.append(f'RUN cd /repo && git apply --reject /patch/patch_{diff_no}.diff --allow-empty')
            else:
                container_run_set.append(f'RUN cd /repo && git apply -R --reject /patch/patch_{diff_no - 1}.diff --allow-empty')
                container_run_set.append(f'RUN cd /repo && git apply --reject /patch/patch_{diff_no}.diff --allow-empty')
            diff_no += 1
        else:
            container_run_set.append(res)
    
    pre_download = _get_pre_download(base_image_st)

    dockerfile.append(base_image_st)
    dockerfile.append(workdir_st)
    if os.path.exists(f'{root_path}/patch'):
        dockerfile.append(copy_st)

    has_code_edits = any(cmd.get('command', '').startswith('python /home/tools/code_edit.py') for cmd in commands_data)
    if has_code_edits:
        dockerfile.append(copy_edit_st)
    dockerfile.extend(pre_download.splitlines())
    
    dockerfile.append(git_st)
    dockerfile.append(mkdir_st)
    dockerfile.append(git_save_st)
    dockerfile.append(mv_st)
    dockerfile.append(rm_st)

    deduped = _dedupe_runs(container_run_set)
    if len(deduped) != len(container_run_set):
        print(
            f"[integrate_dockerfile] de-duplicated "
            f"{len(container_run_set) - len(deduped)} redundant RUN line(s) "
            f"({len(container_run_set)} → {len(deduped)})."
        )
    dockerfile.extend(deduped)
    with open(f'{root_path}/Dockerfile', 'w') as w1:
        w1.write('\n'.join(dockerfile))
