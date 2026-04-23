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

import os, sys, json
import subprocess
from agents.agent import Agent
from utils.llm import get_llm_response
from utils.agent_util import safe_cmd, extract_commands, append_trajectory, TIME_OUT_LABEL, extract_diffs, save_diff_description, DIFF_FENCE, BASH_FENCE, INIT_PROMPT, EDIT_PROMPT, HEAD, DIVIDER, UPDATED
from utils.tools_config import Tools
from utils.split_cmd import split_cmd_statements
from knowledge.rocm_knowledge import generate_rocm_prompt_section, generate_rocm_prompt_section_with_plan
from utils.rich_logger import (
    log_header, log_phase, log_turn, log_prompt_sent, log_llm_response,
    log_multi_action_warning, log_action, log_observation, log_rocm_no_test_block,
    log_rocm_success, log_revert, log_context_summary, log_finish_summary,
    log_success, log_error, log_info, console,
)
import re
import time

def res_truncate(text):
    keywords = ['''waitinglist command usage error, the following command formats are leagal:
1. `waitinglist add -p package_name1 -v >=1.0.0 -t pip`
Explanation: Add package_name1>=1.0.0 into waiting list(using pip), and version constraints string cannot contain spaces.
2. `waitinglist add -p package_name1 -t pip`
Explanation: Add package_name1 into waiting list, no `-v` means download the latest version by default.
3. `waitinglist addfile /path/to/file`
Explanation: Add all the items in the /path/to/file into waiting list. Note that you must make sure each line's item meet the formats like [package_name][version_constraints].
4. `waitinglist clear`
Explanation: Clear all the items in the waiting list.''', 
        'If you have multiple elements to add to the waitinglist, you can use && to connect multiple `waitinglist add` statements and surround them with ```bash and ```. Please make sure to write the complete statements; we will only recognize complete statements. Do not use ellipses or other incomplete forms.',
        '''conflictlist command usage error, the following command formats are legal:
1. `conflictlist solve`
Explanation: The standalone `conflictlist solve` command means not to impose any version constraints, i.e., to default to downloading the latest version of the third-party library. This will update the version constraint in the waiting list to be unrestricted.
2. `conflictlist solve -v "==2.0"`
Explanation: Adding -v followed by a version constraint enclosed in double quotes updates the version constraint in the waiting list to that specific range, such as "==2.0", meaning to take version 2.0.
3. `conflictlist solve -v ">3.0"`
Explanation: Similar to the command 2, this constraint specifies a version number greater than 3.0.
4. `conflictlist solve -u`
Explanation: Adding -u indicates giving up all the constraints in the conflict list while still retaining the constraints in the waiting list, i.e., not updating the constraints for that library in the waiting list.
5. `conflictlist clear`
Explanation: Clear all the items in the conflict list.''',
        'If you have multiple elements to remove from the conflict list, you can use && to connect multiple `conflictlist solve` statements and surround them with ```bash and ```. Please make sure to write the complete statements; we will only recognize complete statements. Do not use ellipses or other incomplete forms.'
        ]
    all_positions = {}
    for keyword in keywords:
        positions = [i for i in range(len(text)) if text.startswith(keyword, i)]
        if len(positions) > 1:
            all_positions[keyword] = positions

    if not all_positions:
        return text

    new_text = text
    keywords_to_remove = sorted(all_positions.items(), key=lambda item: item[1][-1], reverse=True)

    for keyword, positions in keywords_to_remove:
        last_position = positions[-1]
        before_last_position = new_text[:last_position].replace(keyword, "", len(positions) - 1)
        after_last_position = new_text[last_position:]
        new_text = before_last_position + after_last_position

    return new_text

class Configuration(Agent):
    def __init__(self, sandbox, image_name, full_name, root_dir, llm="gpt-4o-2024-05-13", max_turn=70, rocm_mode=False, plan=None, no_scale_down=False):
        self.model = llm
        self.root_dir = root_dir
        self.max_turn = max_turn
        self.rocm_mode = rocm_mode
        self.no_scale_down = no_scale_down
        self.sandbox = sandbox
        self.sandbox_session = self.sandbox.get_session()
        self.full_name = full_name
        self.plan = plan
        self.tool_lib = [
            Tools.waiting_list_add,
            Tools.waiting_list_add_file,
            Tools.waiting_list_clear,
            Tools.conflict_solve_constraints,
            Tools.conflict_solve_u,
            Tools.conflict_clear,
            Tools.conflict_list_show,
            Tools.waiting_list_show,
            Tools.download,
            Tools.runtest,
            Tools.poetryruntest,
            Tools.runpipreqs,
            Tools.change_python_version,
            Tools.change_base_image,
            Tools.clear_configuration,
        ]
        self.image_name = image_name
        self.outer_commands = list()
        tools_list = ""
        for tool in self.tool_lib:
            tools_list += f"{tool.value['command']} # {tool.value['description']}\n"

        self.system_prompt = self._build_system_prompt(tools_list)

    def _build_system_prompt(self, tools_list):
        """Build the system prompt, choosing plan-aware or full version."""
        has_plan = bool(self.plan)

        if has_plan:
            core_prompt = self._build_plan_aware_prompt(tools_list)
        else:
            core_prompt = self._build_full_prompt(tools_list)

        if self.rocm_mode:
            if has_plan:
                core_prompt += generate_rocm_prompt_section_with_plan(no_scale_down=self.no_scale_down)
            else:
                core_prompt += generate_rocm_prompt_section(no_scale_down=self.no_scale_down)

        if has_plan:
            core_prompt += "\n\n" + "=" * 60 + "\n"
            core_prompt += "STRATEGIC PLAN (generated from deep upfront repo analysis)\n"
            core_prompt += "=" * 60 + "\n"
            core_prompt += self.plan + "\n"
            core_prompt += "=" * 60 + "\n"
            core_prompt += (
                "Execute this plan immediately. The repo has already been analyzed — "
                "do NOT re-read the README, directory listing, or config files. "
                "Start from the first actionable step. Adapt if runtime observations "
                "reveal something unexpected.\n"
            )

        return core_prompt

    def _build_plan_aware_prompt(self, tools_list):
        """Condensed prompt when an upfront plan exists. Skips reconnaissance steps."""
        return f"""\
You are an expert environment configuration agent. A strategic plan has already analyzed
this repository's README, directory structure, config files, imports, and compatibility issues.
The plan is included below. Execute it immediately — do NOT re-read files that were already analyzed.

You are in the Docker environment of {self.image_name}. The base image has already been selected.

RESPONSE FORMAT:
Each response must contain a Thought and exactly ONE Action (one {BASH_FENCE[0]}...{BASH_FENCE[1]} block
or one {DIFF_FENCE[0]}...{DIFF_FENCE[1]} block). Write all commands on a single line using && to chain them.

AVAILABLE TOOLS:
{tools_list}
DEPENDENCY MANAGEMENT:
Use `waitinglist add/addfile` and `download` for batch installs. Use `conflictlist` to resolve version conflicts.
Use `pip install -q` for individual packages. Use `-q` mode where available to reduce output.

ERROR HANDLING:
Use `pipdeptree -p <pkg>` to inspect dependencies. Use `pip index versions <pkg>` to find available versions.
For import errors, check if the module exists in /repo before pip installing externally.
Do not use `git clone` or `wget` to download large files into /repo.

{INIT_PROMPT}
{EDIT_PROMPT}

RULES:
* Do not modify test files. Do not open new shells (hatch shell, etc.).
* Prefer minimal changes to original repository files.
* Write all commands on ONE line with &&. Do not use backslash line continuation.
* If the environment is too broken, use `clear_configuration` to reset.
"""

    def _build_full_prompt(self, tools_list):
        """Full prompt with 12-step work process (used when no plan exists)."""
        return f"""\
You are an expert skilled in environment configuration. You can refer to various files and structures in the repository such as `requirements.txt`, `setup.py`, etc., and use dependency prediction tools like pipreqs to install and download the corresponding third-party libraries in a given Docker image. This ensures that the repository can be successfully configured and able to correctly execute the specified tests.
* Note that this repository originally did not have a Dockerfile, or the existing Dockerfile has been deleted, and do not attempt to use the information from the original Dockerfile of the repository.*

* We have already configured poetry, pipdeptree, and pytest for you; no additional configuration is needed. However, you cannot directly invoke pytest; you need to run tests using `runtest` or `poetryruntest`.

WORK PROCESS:
1. **Read Directory Structure**: Check the folder structure in the root directory, focusing on the configuration files related to setting up the environment.
2. **Determine Python Version**: Decide if you need to switch the Python version in the Docker container. The current version is {self.image_name}. If you want to switch the Python version, please run the `change_python_version python_version` command, where python_version is the Python version number (for example, `change_python_version 3.9`), and you do not need to add quotation marks. If you do not need to make any changes, please ignore this step. You can also run these commands at any point later during the environment configuration to switch the Python version.
    *Note*: You can only switch the Python version within the container; switching to other images is not allowed.
3. **Check the configuration files in the root directory**: Read configuration files related to setting up the environment, such as: Information in the `.github` folder, `setup.py`, `setup.cfg`, `Pipfile` and `Pipfile.lock`, `environment.yml`, `poetry.lock` and `pyproject.toml`, etc.
3.5 **Try testing (optional)**: Using `runtest` command to check if it is possible to pass the tests directly without any additional configuration.
4. **Review Additional Files**: Consider other potential files and structures for environment configuration.
5. **Automatically install according to the installation script**: Based on the observed structure in the root directory, determine the necessary installation commands:
    a. Poetry Detected: If a poetry.lock file is present in the root directory, Install Poetry using the relevant method for your system. Execute the command `poetry install` to install the dependencies specified in the lock file.
    b. Setup.py Detected: If a setup.py file exists in the root directory, run the command `pip install -e .` to install the package in editable mode along with its dependencies.
    c. Other Descriptor Files: For other specific files that indicate dependency management, assess and determine the appropriate method to install the required dependencies.
    *Note*: We only consider automatically installation script in the repository. Do not consider `requirements.txt` directly in this step!
6. **Collecting Third-Party Library Download List**: In this step, you need to locate all files in the root directory that list dependencies line by line, such as `requirements.txt`, `requirements_dev.txt`, etc. Use a command like `queue_file /repo/requirements.txt` to submit them to the download list. I will handle the unified downloading later.
    If you have determined the path to the requirements file, please enter `waitinglist addfile` followed by the path to the requirements file. For example, `waitinglist addfile /repo/requirements.txt`.
    *Note*: The files you collect must follow the standard requirements.txt format. Do not collect files in any other formats. For instance, if you are unsure about the format of `/repo/requirements_test.txt`, you can use the command `cat /repo/requirements_test.txt` to read the file contents and ensure the file fully meets the requirements before submitting it. If no such dependency-listing files are found, you may skip this step.
    *Note*: In a standard requirements.txt file, each valid entry on a line consists of package_name followed by version_constraints (if there are no version_constraints, the latest version is implied). For example: "numpy==2.1", "numpy>2.0,<3.0", "numpy" (implies the latest version).
    *Note*: We will not collect items that are improperly formatted.
7. **Using pipreqs to Obtain Additional Lists**: In this step, you should use `runpipreqs` to analyze the third-party libraries that need to be installed based on the findings of the previous step. Simply use the command `get pipreqs`, and it will generate a `requirements_pipreqs.txt` file in the project root directory (/repo) and show you the warnings and errors.
    *Note*: If you have already collected some requirements.txt files in Step 5, you do not need to execute `runpipreqs` in this step. Avoid collecting too many dependencies repeatedly. You can directly execute `download` after handling conflicts and formatting errors. If errors occur in subsequent tests, you can then run `runpipreqs`.
8. **Handling pipreqs Warnings**: For warnings that appear in pipreqs, such as not being able to find a module on PyPI, it may be due to a discrepancy between the download name and the import name of a third-party library. For example, `import cv2` requires downloading not `cv2` but `opencv-python`. For each warning, you need to address the discrepancy by ensuring the correct package names are used for the downloads.
    You should review "pipreqs_output.txt" (used to store output during the pipreqs dependency generation process) and "requirements_pipreqs.txt" (the final generated dependency results) to check for any inconsistencies. Use ```diff and ``` to make adjustments to "requirements_pipreqs.txt" as needed.
    *Note*: If you did not execute runpipreqs, then you do not need to perform this step.
9. **Update lists**: Add the "requirements_pipreqs.txt" file generated by pipreqs and corrected by you (or maybe not) to the waiting list using the command `waitinglist addfile /repo/requirements_pipreqs.txt`.
    *Note*: If you did not execute runpipreqs, then you do not need to perform this step.
10. **Resolve version_constraint conflicts**: Process the elements in conflict_list. Based on the information in conflict_list, resolve any existing version_constraints conflicts. Only after these have been resolved can you proceed with the download.
11. **Unified download execution**: After analyzing the original requirements.txt of the repository and the "requirements.txt" generated by pipreqs, and resolving any conflict issues, you need to enter download to execute the unified `download`. This will download each element currently in the waiting_list one by one, and eventually, the download status will be returned.
12. **Error Handling**: After the download is complete, you need to handle the error messages based on the information provided. We will return the list of third-party libraries and their dependencies in your current environment. When resolving these errors, you need to ensure that these dependencies are properly managed. For example, you cannot directly uninstall a package that is required by another package, nor can you install a version that does not meet the constraints.
    For instance, if package A depends on package B with a requirement of "B>=1.0", you cannot simply run pip uninstall B as this would cause package A to lack its dependency. Similarly, you cannot run `pip install B==0.5` because this would not satisfy the requirement of "B>=1.0".
    You can make use of the following tools:
    a.(Strongly recommend) `pipdeptree`: Use pipdeptree to see the structure of the python third-party library downloaded.
    b.(Strongly recommend) `pipdeptree -p <package_name>`: Use pipdeptree -p followed by package_name can display the dependency information of a specific third-party library.
    c.(Strongly recommend) `pip index versions <package_name> --python-version <python_version>`: This command is used to query the versions of a specific package for a particular Python version, including pre-release versions. For example, `pip index versions requests --python-version 3.10` can be used to find the versions of requests that are available for Python 3.10.
    d. `pip install -q`: Use this command to install a specific version of a package with minimal output. This greatly reduces the verbosity, showing only crucial information and final status. It is recommended to specify the version with == to avoid conflicts with the existing environment. For example, use pip install -q requests==2.25.1 to ensure a quiet and specific version installation.
    e. `pip install -e`: Use this command to install a package in "editable" mode. This is particularly useful during development when you want to make changes to the source code and have them immediately reflected in the installed package without needing to reinstall it. For example, pip install -e . will install the package located in the current directory in editable mode. Another common use case is to install a package from a local path or a VCS repository while keeping it editable. For example, pip install -e git+https://github.com/username/repo.git#egg=package_name.
    f. `pip uninstall`: Use this command to uninstall a third-party library. This should be used cautiously as it may cause dependency issues. If you need to change the version of a package, it is better to use `pip install [package_name]==[version]` instead.
    g. `apt-get update -qq && apt-get install [package]=[version] -y -qq`: Use this command to install system packages if needed, remember to use `-y`. Use `-qq` to minimize the output if there is nothing wrong.
    h. `export <variable>=<value>`: Use this command to set system environment variables.
    i. You can use the `--help` parameter to view detailed usage instructions for various tools, such as `pipdeptree --help` and `pip install --help`, etc.
    j. You may also use other commands that are not listed here, including built-in Bash commands and other system commands.
    *Note*: Always consider the potential impact of each command on the system. Aim to achieve the best results with minimal changes.
    *Note*: For modules not found in the error message, first check if the corresponding module is available in the project folder before proceeding with external downloads. For example, if you encounter an error stating ModuleNotFoundError: No module named 'my_module', check if there is a file named my_module.py in your project directory. If it is not present, then you can look for the module externally and download it if necessary.
    *Note*: Do not use external download tools like `git clone` or `wget` to download a large number of files directly in the /repo folder (or its subdirectories) to avoid causing significant changes to the original repository.
    *Note*: Flexibility: You do not need to complete all configurations in one go. If you are unsure whether the configuration is approximately complete, you can use the `runtest` or `poetryruntest`(When you configured in poetry environment) command. I will check the configured environment and return any error messages. Based on the error messages, you can make further adjustments.
    *Note*: In special cases, if you feel that the Docker environment has become too messy to achieve your goal, you can use `clear_configuration` command to restore it to the initial Python 3.10 environment or `change_python_version` and start over.
**Most Important!** You can execute `runtest` or `poetryruntest` anywhere when you decide to test the environment. You do not need to complete all the previous steps; you can directly run `runtest` or `poetryruntest` to check if the configuration is complete and get feedback from the error messages. Be flexible. Our goal is to pass the runtest or poetryruntest checks.
If you encounter import errors such as ModuleNotFoundError or ImportError, you can consider two solutions. One solution is to use external tools like pip or apt-get to download these dependencies. The other solution is to check for local dependencies in the repository; if local dependencies are available, you can use `export PYTHONPATH=` statements to set environment variables (preferably), or modify the __init__.py file to resolve the issue.
Please note that when manually using pip, apt-get, poetry, or other tools to download third-party libraries, try to use the `-q` (quiet) mode if available to reduce intermediate progress bar outputs. Additionally, we will help remove more obvious progress bar information to minimize interference with the analysis.
If you need to install packages using pip, please consider adding them to the waiting list first, and then use the `download` command to proceed with the installation.
In each round of the conversation, we will inform you of the commands that have been correctly executed and have changed the state of the current Docker container. Please reflect on each round's Observation in relation to the current state of the Docker container and decide the subsequent Action.
If you need to run two or more commands, please strictly follow the order by enclosing each command in ONE {BASH_FENCE[0]} and {BASH_FENCE[1]} blocks connected by "&&" with ONE line! It is not recommended to use backslashes (\\) for line continuation. If you need to execute commands that span multiple lines, it is advisable to write them into a .sh file and then run the executable file. For example, if you want to enter the /repo directory and execute `poetry install`, you should input:
{BASH_FENCE[0]}
cd /repo && poetry install
{BASH_FENCE[1]}

It is not recommended to directly input:
{BASH_FENCE[0]}
cd /repo
{BASH_FENCE[1]}
{BASH_FENCE[0]}
poetry install
{BASH_FENCE[1]}

Nor is it recommended to input:
{BASH_FENCE[0]}
cd /repo \\
poetry install
{BASH_FENCE[1]}

We also strongly request that you try to write the instructions on the same line as much as possible, and do not break them into multiple lines, as this may cause parsing errors. Even if the line of instructions contains a lot of && connections, do not arbitrarily break it into multiple lines.

We will automatically maintain two lists in the background to facilitate the installation and download of third-party libraries. These are:
1. waiting list: Used to store third-party libraries waiting to be downloaded, including both pip and apt libraries. You can use `waitinglist show` to show all items in it.
2. conflict list: Used to store elements with the same name as those in the waiting list but with inconsistent version constraints. You can use `conflictlist show` to show all items in it.
*Note*: you only need to follow the prompts to complete operations on these lists during the running process and can only manipulate them using the provided commands.
*Note*: Before operating waiting list, conflict list, or download commands, you can use waitinglist show or conflictlist show to observe their structure each time.

{INIT_PROMPT}
You are now in the Docker environment of {self.image_name}. Please perform all operations within this environment.
CLI TOOLS: You can call CLI tools in  {BASH_FENCE[0]} ... {BASH_FENCE[1]} block as Action with a Thought. like:
### Thought: I need to understand the structure of the root directory.
### Action:
{BASH_FENCE[0]}
ls /repo
{BASH_FENCE[1]}

For another example:
### Thought: I need to read the README.md file.
### Action:
{BASH_FENCE[0]}
cat README.md
{BASH_FENCE[1]}

{EDIT_PROMPT}
*Note*: Do not make extensive changes to the existing files in the /repo folder. You may only make appropriate and necessary changes to the original repository files (e.g., when there are actual errors or tests that cannot be run).
*Very Important Note*: Passing tests by modifying testing functions is not allowed, and you should figure out how to make the current test functions run successfully!!!
In addition to typical bash commands, we also provide the following commands that can be used, you can use them flexibly if needed:
{tools_list}

VERY IMPORTANT TIPS: 
    * Your task is to configure the environment. Follow the steps and use various commands. After testing, ensure the environment passes.
    * You can directly run runtest or poetryruntest to check if the configuration is complete. Be flexible. Our goal is to pass the checks.
    * Passing tests by modifying testing functions is not allowed!
    * Try to write all commands on a single line with "&&" connections. Do not break them into multiple lines!
    * Avoid modifying or deleting original files, especially test files!
    * Do not use commands like `hatch shell` that open a new shell!
"""

    def show_init_prompt(self):
        print(self.system_prompt)
    
    def get_max_turn(self):
        return self.max_turn

    def _save_final_artifacts(self, waiting_list, conflict_list):
        """Collect and save pipdeptree, pip list, and diff artifacts."""
        results = {}
        for cmd_label, cmd in [
            ("pipdeptree_json", "pipdeptree --json-tree"),
            ("pipdeptree", "pipdeptree"),
            ("diff", "generate_diff"),
            ("pip_list", "$pip list --format json$"),
        ]:
            try:
                out, rc = self.sandbox_session.execute(cmd, waiting_list, conflict_list)
                results[cmd_label] = (out, rc)
            except Exception:
                results[cmd_label] = ("", -1)

        out_dir = f'{self.root_dir}/output/{self.full_name}'
        diff_out, diff_rc = results["diff"]
        if diff_out.strip() and diff_rc == 0:
            os.makedirs(f'{out_dir}/patch', exist_ok=True)
            with open(f'{out_dir}/patch/final_patch.diff', 'w') as f:
                f.write(diff_out)
        pj_out, pj_rc = results["pipdeptree_json"]
        if pj_rc == 0:
            with open(f'{out_dir}/pipdeptree.json', 'w') as f:
                f.write(pj_out)
        pn_out, pn_rc = results["pipdeptree"]
        if pn_rc == 0:
            with open(f'{out_dir}/pipdeptree.txt', 'w') as f:
                f.write(pn_out)
        pl_out, pl_rc = results["pip_list"]
        if pl_rc == 0:
            try:
                with open(f'{out_dir}/pip_list.json', 'w') as f:
                    f.write(json.dumps(json.loads(pl_out), indent=4))
            except Exception:
                pass
        return results

    def run(self, project_path, trajectory, waiting_list, conflict_list):
        log_header("CONFIGURATION AGENT", f"Model: {self.model} | Repo: {self.full_name} | ROCm: {self.rocm_mode}")
        log_phase("SYSTEM PROMPT", f"{len(self.system_prompt)} chars (passed via API system field)")
        log_info(f"Prompt preview (first 300): {self.system_prompt[:300]}...")

        start_time0 = time.time()
        self.messages = []
        user_message = {"role": "user", "content": "[Project root Path]: /repo"}
        self.messages.append(user_message)

        turn = 0
        cost_tokens = 0
        diff_no = 1
        rocm_runtest_blocked_count = 0

        def manage_token_usage(messages, max_tokens=150000):
            total_tokens = sum(len(str(message)) for message in messages)
            if total_tokens <= max_tokens:
                return messages
            new_messages = messages[:]
            while sum(len(str(message)) for message in new_messages) > max_tokens:
                new_messages = new_messages[:2] + new_messages[4:]
            return new_messages
        
        def extract_cmds(inner_commands):
            res_cmd = list()
            for inner_command in inner_commands:
                command = inner_command['command']
                dir = inner_command['dir'] if 'dir' in inner_command else '/'
                returncode = inner_command['returncode']
                action_name = command.split(' ')[0].strip()
                if str(returncode).strip() != '0':
                    continue
                if action_name in ['pipdeptree']:
                    continue
                if action_name in safe_cmd and '>' not in command:
                    continue
                if command == 'python /home/tools/runtest.py' or command == 'python /home/tools/poetryruntest.py' or command == 'python /home/tools/runpipreqs.py' or command == 'python /home/tools/generate_diff.py' or command == '$pwd$' or command == '$pip list --format json$':
                    continue
                if action_name == 'change_python_version':
                    res_cmd = list()
                    continue
                if action_name == 'change_base_image':
                    res_cmd = list()
                    continue
                if action_name == 'clear_configuration':
                    res_cmd = list()
                    continue
                if dir != '/':
                    res_cmd.append(f'cd {dir} && {command}')
                else:
                    res_cmd.append(command)
            return res_cmd

        while(turn < self.max_turn):
            turn += 1
            finish = False

            # ── LLM Call ──
            log_turn(turn, self.max_turn, self.model)
            GPT_start_time = time.time()
            current_messages = manage_token_usage(self.messages)
            log_prompt_sent(current_messages)

            configuration_agent_list, usage = get_llm_response(
                self.model, current_messages, system_prompt=self.system_prompt)
            GPT_end_time = time.time()
            GPT_elasped_time = GPT_end_time - GPT_start_time
            self.outer_commands.append({"GPT_time": GPT_elasped_time})
            configuration_agent = configuration_agent_list[0]
            cost_tokens += usage["total_tokens"]

            assistant_message = {"role": "assistant", "content": configuration_agent}
            self.messages.append(assistant_message)

            # ── Parse response ──
            init_commands = extract_commands(configuration_agent)
            total_bash_blocks = len(init_commands)

            log_llm_response(configuration_agent, GPT_elasped_time, usage, total_bash_blocks)

            # Enforce single-action-per-turn
            multi_action_warning = ""
            if total_bash_blocks > 1:
                log_multi_action_warning(total_bash_blocks, init_commands[0])
                init_commands = init_commands[:1]
                multi_action_warning = (
                    f"\n** WARNING: You provided {total_bash_blocks} bash blocks but I only executed the FIRST one. "
                    f"Provide exactly ONE ```bash``` block per response. "
                    f"Wait for the observation before deciding your next action. **\n"
                )

            commands = list()
            for ic in init_commands:
                commands.extend(split_cmd_statements(ic))
            diffs = extract_diffs(configuration_agent)

            system_res = '### Observation:\n'

            if len(diffs) != 0 and len(commands) != 0:
                system_res = f"ERROR! Your reply contains both bash block and diff block, which is not accepted. Each round of your reply can only contain one {BASH_FENCE[0]} {BASH_FENCE[1]} block or one {DIFF_FENCE[0]} {DIFF_FENCE[1]} block. Each round of your answers contain only *ONE* action!"
                log_error("Response contains both bash and diff blocks")

            elif len(commands) != 0:
                for i in range(len(commands)):
                    log_action(commands[i], i, len(commands))
                    self.outer_commands.append({"command": commands[i], "returncode": -2, "time": -1})
                    start_time = time.time()
                    
                    # ── change_python_version ──
                    if commands[i].strip().startswith('change_python_version'):
                        python_version = commands[i].strip().split('change_python_version')[1].strip()
                        try:
                            sandbox = self.sandbox_session.sandbox.change_python_version(python_version)
                            if isinstance(sandbox, str):
                                log_error(f"change_python_version failed: {sandbox}")
                                system_res += sandbox
                            else:
                                self.sandbox = sandbox
                                self.sandbox_session = self.sandbox.get_session()
                                res = f"You have successfully switched the docker container's Python version to {python_version}. If you want to revert to the previous environment, you can enter `change_python_version` followed by the previous python version."
                                self.sandbox.commands.append({"command": f'change_python_version {python_version}', "returncode": 0, "time": -1})
                                self.image_name = 'python:' + python_version
                                log_success(res)
                                system_res += res
                        except Exception as e:
                            res = f"Error to change the docker container's Python version to {python_version}, the error messages are: {e}"
                            log_error(res)
                            self.outer_commands[-1]["returncode"] = 1
                            system_res += res
                        end_time = time.time()
                        elasped_time = end_time - start_time
                        self.outer_commands[-1]["time"] = elasped_time
                        self.outer_commands[-1]["returncode"] = 0
                        if self.sandbox.commands[-1]['command'] == f'change_python_version {python_version}':
                            self.sandbox.commands[-1]["time"] = elasped_time
                        continue
                    
                    # ── clear_configuration ──
                    if commands[i].strip() == 'clear_configuration':
                        try:
                            sandbox = self.sandbox_session.sandbox.change_python_version('3.10')
                            self.sandbox = sandbox
                            self.sandbox_session = self.sandbox.get_session()
                            res = f"You have successfully switched the docker container's Python version to 3.10. If you want to revert to the previous environment, you can enter `change_python_version` followed by the previous python version."
                            self.sandbox.commands.append({"command": f'clear_configuration', "returncode": 0, "time": -1})
                            self.image_name = 'python:3.10'
                            log_success(res)
                            system_res += res
                        except Exception as e:
                            res = f"Error to change the docker container's Python version to 3.10, the error messages are: {e}"
                            log_error(res)
                            self.outer_commands[-1]["returncode"] = 1
                            system_res += res
                        end_time = time.time()
                        elasped_time = end_time - start_time
                        self.outer_commands[-1]["time"] = elasped_time
                        self.outer_commands[-1]["returncode"] = 0
                        if self.sandbox.commands[-1]['command'] == f'clear_configuration':
                            self.sandbox.commands[-1]["time"] = elasped_time
                        continue

                    # ── change_base_image ──
                    if commands[i].strip().startswith('change_base_image'):
                        base_image = commands[i].strip().split('change_base_image')[1].strip().lower()
                        log_info(f"Switching base image to: {base_image}")
                        try:
                            sandbox = self.sandbox_session.sandbox.change_base_image(base_image)
                            if not isinstance(sandbox, str):
                                self.sandbox = sandbox
                                self.sandbox_session = self.sandbox.get_session()
                                res = f"You have successfully switched the docker container's base image to {base_image}. If you want to revert to the previous environment, you can enter `change_python_version` followed by the previous python version or `change_base_image` followed by the previous base image name."
                                self.sandbox.commands.append({"command": f'change_base_image {base_image}', "returncode": 0, "time": -1})
                                self.image_name = base_image
                                log_success(res)
                                system_res += res
                            else:
                                log_error(f"change_base_image returned error: {sandbox}")
                                end_time = time.time()
                                elasped_time = end_time - start_time
                                self.outer_commands[-1]["time"] = elasped_time
                                self.outer_commands[-1]["returncode"] = 1
                                if self.sandbox.commands[-1]['command'] == f'change_base_image {base_image}':
                                    self.sandbox.commands[-1]["time"] = elasped_time
                                continue
                        except Exception as e:
                            res = f"Error to change the docker container's base image to {base_image}, the error messages are: {e}"
                            log_error(res)
                            self.outer_commands[-1]["returncode"] = 1
                            system_res += res
                        end_time = time.time()
                        elasped_time = end_time - start_time
                        self.outer_commands[-1]["time"] = elasped_time
                        self.outer_commands[-1]["returncode"] = 0
                        if self.sandbox.commands[-1]['command'] == f'change_base_image {base_image}':
                            self.sandbox.commands[-1]["time"] = elasped_time
                        continue

                    # ── ROCm mode: intercept runtest ──
                    if self.rocm_mode and commands[i].strip().lower() in ('runtest', 'poetryruntest'):
                        rocm_runtest_blocked_count += 1
                        log_rocm_no_test_block()
                        system_res += (
                            "\n** BLOCKED: `runtest` is disabled in ROCm mode. **\n"
                            "There are no unit tests in this repository.\n"
                            "Instead, verify the environment by:\n"
                            "1. Reading the README: `cat /repo/README.md`\n"
                            "2. Verifying imports work: `python -c 'import <main_package>'`\n"
                            "3. Create mock/dummy input data and ACTUALLY RUN the main script:\n"
                            "   e.g., `cd /repo && python example_mm.py --data_path /tmp/test_data`\n"
                            "   Note: `--help` alone is NOT sufficient. You must run with real/mock data.\n"
                            "Once the script produces actual output, declare success by outputting:\n"
                            "```bash\necho ROCM_ENV_VERIFIED\n```\n"
                        )
                        end_time = time.time()
                        self.outer_commands[-1]["time"] = end_time - start_time
                        self.outer_commands[-1]["returncode"] = 0
                        continue

                    # ── ROCm mode: detect success signal ──
                    # Flexible match: accept any command that contains ROCM_ENV_VERIFIED
                    # e.g. echo ROCM_ENV_VERIFIED, echo "ROCM_ENV_VERIFIED",
                    #      ls -la && echo ROCM_ENV_VERIFIED, etc.
                    if self.rocm_mode and 'ROCM_ENV_VERIFIED' in commands[i]:
                        log_rocm_success()
                        sandbox_res = "ROCM_ENV_VERIFIED\nCongratulations, you have successfully configured the environment!"
                        system_res += sandbox_res
                        self.outer_commands[-1]["returncode"] = 0
                        self.outer_commands[-1]["time"] = time.time() - start_time
                        self._save_final_artifacts(waiting_list, conflict_list)
                        log_success("ROCm environment verified. Agent finished.")
                        with open(f'{self.root_dir}/output/{self.full_name}/test.txt', 'w') as w3:
                            w3.write('ROCM_ENV_VERIFIED\n')
                        finish = True
                        break

                    # ── Execute normal command ──
                    sandbox_res, return_code = self.sandbox_session.execute(commands[i], waiting_list, conflict_list)
                    sandbox_res = res_truncate(sandbox_res)
                    system_res += sandbox_res
                    if return_code != 'unknown':
                        system_res += f'\n`{commands[i]}` executes with returncode: {return_code}\n'

                    log_observation(sandbox_res, return_code)

                    end_time = time.time()
                    elasped_time = end_time - start_time
                    self.outer_commands[-1]["time"] = elasped_time
                    self.outer_commands[-1]["returncode"] = 0

                    if TIME_OUT_LABEL in sandbox_res:
                        self.sandbox_session = self.sandbox.get_session()
                        self.outer_commands[-1]["returncode"] = 1

                    # ── ROCm mode: detect success signal in command output ──
                    # Catches cases where the LLM produced ROCM_ENV_VERIFIED via
                    # any means (echo with quotes, printf, a script that prints it, etc.)
                    if self.rocm_mode and 'ROCM_ENV_VERIFIED' in sandbox_res and return_code == 0:
                        log_rocm_success()
                        self._save_final_artifacts(waiting_list, conflict_list)
                        log_success("ROCm environment verified (detected in output). Agent finished.")
                        with open(f'{self.root_dir}/output/{self.full_name}/test.txt', 'w') as w3:
                            w3.write('ROCM_ENV_VERIFIED\n')
                        finish = True
                        break

                    # ── Standard success check (non-ROCm) ──
                    if 'Congratulations, you have successfully configured the environment!' in sandbox_res and '# This is $runtest.py$' not in sandbox_res:
                        if self.rocm_mode and 'No unit tests were detected' in sandbox_res:
                            log_rocm_no_test_block()
                            system_res += (
                                "\n\n** WARNING: No unit tests were detected. "
                                "In ROCm mode, this does NOT mean success. **\n"
                                "You MUST create mock/dummy data and ACTUALLY RUN the project's main script.\n"
                                "`--help` alone is NOT sufficient for verification.\n"
                                "Once the script produces actual output, output: ```bash\necho ROCM_ENV_VERIFIED\n```\n"
                            )
                            continue

                        self._save_final_artifacts(waiting_list, conflict_list)
                        log_success("Environment configured successfully!")
                        with open(f'{self.root_dir}/output/{self.full_name}/test.txt', 'w') as w3:
                            w3.write('\n'.join(sandbox_res.splitlines()[1:]))
                        finish = True
                        break

                if finish:
                    break

            elif len(diffs) != 0:
                if diffs.split('<<<<<<< SEARCH')[0].split('/')[-1].strip().startswith('test_') or diffs.split('<<<<<<< SEARCH')[0].split('/')[-1].strip().endswith('_test.py'):
                    self.outer_commands.append({"diff": diffs, "returncode": -2, "time": -1})
                    system_res += 'Running Edit...\n' + f"You are trying to modify file {diffs.split('<<<<<<< SEARCH')[0].split('/')[-1].strip()}, but we require that you should not modify the testing files. Please consider alternative solutions." + '\n'
                else:
                    self.outer_commands.append({"diff": diffs, "returncode": -2, "time": -1})
                    start_time = time.time()
                    tmp_name = save_diff_description(diffs)
                    sandbox_res, return_code =  self.sandbox_session.edit(tmp_name, project_path)
                    end_time = time.time()
                    elasped_time = end_time - start_time
                    self.outer_commands[-1]["returncode"] = 0
                    self.outer_commands[-1]["time"] = elasped_time
                    if return_code == 0:
                        try:
                            generate_diff, generate_diff_return_code = self.sandbox_session.execute('generate_diff', waiting_list, conflict_list)
                        except Exception as e:
                            log_error(f'Generate diff wrong: {e}!')
                        if not os.path.exists(f'{self.root_dir}/output/{self.full_name}/patch'):
                            os.system(f'mkdir {self.root_dir}/output/{self.full_name}/patch')
                        with open(f'{self.root_dir}/output/{self.full_name}/patch/patch_{diff_no}.diff', 'w') as w0:
                            w0.write(generate_diff + '\n')
                        diff_no += 1
                    system_res += sandbox_res
                    log_observation(sandbox_res, return_code)
                    if TIME_OUT_LABEL in sandbox_res:
                        self.sandbox_session = self.sandbox.get_session()
                    if HEAD not in diffs or DIVIDER not in diffs or UPDATED not in diffs:
                        self.outer_commands[-1]["returncode"] = 1
                        system_res += f"""#### Your patch is incomplete with {HEAD} or {DIVIDER} or {UPDATED} missing! ####            
The edit format is as follows: 

{DIFF_FENCE[0]}
/absolute/path/of/target.py
{HEAD}
    exact copy of old line(s) you would like to change
{DIVIDER}
    new line(s) to replace
{UPDATED}
"""
            else:
                self.outer_commands[-1]["returncode"] = 2
                system_res += "ERROR! Your reply does not contain valid block or final answer."
                log_error("No valid bash or diff block in LLM response")

            # ── Append multi-action warning if needed ──
            if multi_action_warning:
                system_res += multi_action_warning
            
            # ── Build context for next turn ──
            current_directory, return_code = self.sandbox_session.execute('$pwd$', waiting_list, conflict_list)
            current_directory_str = '\n[Current directory]:\n' + current_directory + '\n'
            system_res += current_directory_str
            system_res += f'You are currently in a [{self.image_name}] container.\n'
            reminder = f"\nENVIRONMENT REMINDER: You have {self.max_turn - turn} turns left to complete the task."
            system_res += reminder
            success_cmds = extract_cmds(self.sandbox.commands)

            if len(success_cmds) > 0:
                if self.rocm_mode:
                    appendix = '\nThe container has successfully executed the following commands in order. Please refer to the execution history, reflect, and decide the subsequent actions. You MUST actually run the project\'s main script with real or mock data (not just --help). Only after the script produces actual output, output `echo ROCM_ENV_VERIFIED` to finish.\n' + \
                        '\n'.join(success_cmds)
                else:
                    appendix = '\nThe container has successfully executed the following commands in order. Please refer to the execution history, reflect, and decide the subsequent actions. Remember, your ultimate goal is to pass the tests by executing `runtest` or `poetryruntest`.\n' + \
                        '\n'.join(success_cmds)
            else:
                appendix = '\nThe container remains in its original state.'
            pattern = r'python\s+/home/tools/pip_download.py\s+-p\s+(\S+)\s+-v\s+""([^""]+)""'
            replacement = r'pip install \1\2'
            appendix = re.sub(pattern, replacement, appendix)
            pattern1 = r'python\s+/home/tools/pip_download.py\s+-p\s+(\S+)'
            replacement1 = r'pip install \1'
            appendix = re.sub(pattern1, replacement1, appendix)
            
            system_res += appendix

            # ── Log context summary ──
            log_context_summary(current_directory.strip(), self.image_name, self.max_turn - turn, success_cmds)

            if "gpt" in self.model:
                system_message = {"role": "system", "content": system_res}
            else:
                system_message = {"role": "user", "content": system_res}
            self.messages.append(system_message)

            # ── Save intermediate outputs ──
            with open(f'{self.root_dir}/output/{self.full_name}/outer_commands.json', 'w') as w1:
                w1.write(json.dumps(self.outer_commands, indent=4))
            with open(f'{self.root_dir}/output/{self.full_name}/inner_commands.json', 'w') as w1:
                w1.write(json.dumps(self.sandbox.commands, indent=4))

            # NOTE: Artifacts are saved only on success (inside the success branches above).
            # Previously _save_final_artifacts was called every turn, which ran generate_diff
            # each iteration and caused unnecessary container commits/reverts.
        
        total_time = time.time() - start_time0
        log_finish_summary(turn, total_time, cost_tokens, finish)

        append_trajectory(trajectory, self.messages, 'configuration')
        trajectory.append({'agent': "configuration", 'cost_time': total_time, 'cost_tokens': cost_tokens}) 
        self.sandbox_session.close()
        return trajectory, self.outer_commands
