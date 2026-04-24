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


from enum import Enum
class Tools(Enum):
    # apt_download = {
    #     "command": "apt_download -p package_name [-v package_version]",
    #     "description": "Use apt-get to download a third-party system library, ensuring that this library is available in the system's package repositories."
    # }
    # pip_download = {
    #     "command": "pip_download -p package_name [-v package_version]",
    #     "description": "Use pip to download third-party libraries in the current Python environment."
    # }
    waiting_list_add = {
        "command": "waitinglist add -p package_name [-v version_constraints] -t tool",
        "description": "Add item into waiting list. If no 'version_constraints' are specified, the latest version will be downloaded by default."
    }
    waiting_list_add_file = {
        "command": "waitinglist addfile file_path",
        "description": "Add all entries from a file similar to requirements.txt format to the waiting list."
    }
    waiting_list_clear = {
        "command": "waitinglist clear",
        "description": "Used to clear all the items in the waiting list."
    }
    waiting_list_show = {
        "command": "waitinglist show",
        "description": "Used to show all the items in the waiting list."
    }
    conflict_solve_constraints = {
        "command": 'conflictlist solve -v "[version_cosntraints]"',
        "description": "Resolve the conflict for the first element in the conflict list, and update the version constraints for the corresponding package_name and tool to version_constraints. If no 'version_constraints' are specified, the latest version will be downloaded by default."
    }
    conflict_solve_u = {
        "command": "conflictlist solve -u",
        "description": "Keep the original version constraint that exists in the waiting list, and discard the other version constraints with the same name and tool in the conflict list."
    }
    conflict_clear = {
        "command": "conflictlist clear",
        "description": "Used to clear all the items in the conflict list."
    }
    conflict_list_show = {
        "command": "conflictlist show",
        "description": "Used to show all the items in the conflict list."
    }
    # errorformat_solve = {
    #     "command": 'errorformatlist solve ["package_name[version_constraints]" ...]',
    #     "description": "Used to extract the first element from the errorformat list that can be added to the waiting list. The entries must be enclosed in double quotes and can list multiple entries. If you run `errorformatlist solve` alone, it indicates that no third-party libraries need to be extracted for download from this format error entry."
    # }
    # errorformat_clear = {
    #     "command": 'errorformatlist clear',
    #     "description": "Used to clear all the items in the errorformat list."
    # }
    download = {
        "command": 'download',
        "description": "Download all pending elements in the waiting list at once."
    }
    runtest = {
        "command": 'runtest',
        "description": "Check if the configured environment is correct."
    }
    poetryruntest = {
        "command": 'poetryruntest',
        "description": "Check if the configured environment is correct in poetry environment! If you want to run tests in poetry environment, run it."
    }
    runpipreqs = {
        "command": 'runpipreqs',
        "description": "Generate 'requirements_pipreqs.txt' and 'pipreqs_output.txt' and 'pipreqs_error.txt'."
    }
    # rollback = {
    #     "command": 'rollback',
    #     "description": "Manually revert to the previous state, which means discarding the last successfully executed command. Note that you can only revert once and cannot continuously go back."
    # }
    change_python_version = {
        "command": 'change_python_version python_version',
        "description": "Switching the Python version in the Docker container will forgo any installations made prior to the switch. The Python version number should be represented directly with numbers and dots, without any quotation marks."
    }
    change_base_image = {
        "command": 'change_base_image base_image',
        "description": "Switching the base image in the Docker container will forgo any installations made prior to the switch. The base image does not necessarily have to follow the format 'python:<Python version>'. Preferably, specify it in the form of 'base_image_name:tag', such as 'pytorch/pytorch:1.10.0-cuda11.1-cudnn8-runtime'. If no tag is provided, it defaults to 'latest'. No any quotation marks are needed."
    }
    clear_configuration = {
        "command": 'clear_configuration',
        "description": "Reset all the configuration to the initial setting of python:3.10."
    }
    # ── Stage 5b: in-loop retrieval tools (graphify code + mempalace memory) ──
    mem_recall = {
        "command": 'mem_recall "<question>" [--rooms r1,r2,...] [--budget N]',
        "description": (
            "Query this run's memory (mempalace) for the slice most relevant to "
            "<question> instead of guessing. Default rooms: "
            "commands_success,commands_failed,fixes,decisions,patches,plan,"
            "experiment_state,context_refs. --budget is a token budget (default 1500). Use "
            "this BEFORE retrying a failed install or rerunning the same command."
        ),
    }
    paper_recall = {
        "command": 'paper_recall "<question>" [--budget N]',
        "description": (
            "Query paper-related context using graphify's static paper index PLUS "
            "run-state references from memory (`paper_experiments`, "
            "`experiment_state`, `context_refs`, `plan`, `decisions`). Use this "
            "BEFORE reading `/repo/paper.pdf` directly. Best for: 'what metric "
            "should I target?', 'what hyperparameters did the paper use?', "
            "'which experiment did the planner choose?', 'what tolerance applies?'."
        ),
    }
    graphify_query = {
        "command": 'graphify_query "<question>" [--scope paper|code|both] [--budget N]',
        "description": (
            "Ask the per-repo static graphify corpus for snippets relevant to "
            "<question>. `--scope code` (default) hits the tree-sitter code "
            "graph and returns ranked nodes (files, classes, functions) with "
            "absolute paths and line numbers — use it instead of `find -name`/"
            "`grep -r` to locate entry points, config loaders, model factories, "
            "etc. `--scope paper` queries the indexed paper PDF (sections, "
            "tables, hyperparameters); use it BEFORE reading `/repo/paper.pdf` "
            "directly. `--scope both` returns code AND paper results in a "
            "single call. --budget is a token budget (default 1500). "
            "`paper_recall` is now an alias for `--scope paper` plus run-state."
        ),
    }
    verify_paper_result = {
        "command": 'verify_paper_result --log <path> [--metric NAME=VALUE]... [--tolerance RULE]',
        "description": (
            "DETERMINISTIC verifier you MUST run BEFORE echoing "
            "PAPER_RESULT_REPRODUCED / PAPER_RESULT_NOT_REPRODUCED. Reads "
            "`<path>` (typically `/repo/paper_experiment.log`), extracts the "
            "named metrics by regex/JSON, compares each one to its expected "
            "value using the provided tolerance rule (or the chosen "
            "experiment's `tolerance_rule` from the plan), and prints a "
            "structured JSON verdict per metric. If you omit `--metric`, the "
            "verifier uses the chosen experiment's `expected_metric_name` and "
            "any `primary_metrics` list from the planner. The JSON it returns "
            "is the ONLY trusted source of truth for the marker line — do not "
            "fabricate numbers."
        ),
    }
    # ── PR-A: deterministic external lookups (no API key, no LLM) ──
    pypi_versions = {
        "command": 'pypi_versions <package_name> [--limit N]',
        "description": (
            "Look up recent PyPI versions of a package + release dates. Use "
            "this BEFORE pinning a CUDA-only wheel (e.g. flash-attn, "
            "bitsandbytes, xformers) to find a version that matches your ROCm "
            "torch. Cached for 7 days in the global KB."
        ),
    }
    dockerhub_tags = {
        "command": 'dockerhub_tags <image> [--limit N]',
        "description": (
            "Look up the most recently published tags on a Docker Hub repo "
            "(e.g. rocm/pytorch, rocm/vllm, rocm/sgl-dev). Use this BEFORE "
            "calling `change_base_image` to pick an actual current tag (the "
            "static catalog can go stale). Cached for 7 days in the global KB."
        ),
    }
    # ── PR-B: web search + URL fetcher (cached, soft-fail) ──
    web_search = {
        "command": 'web_search "<query>" [--max-results N]',
        "description": (
            "DuckDuckGo web search (no API key). Use this when in-repo / paper "
            "/ KB context can't answer (e.g. 'transformers SDPA tensor shape "
            "mismatch ROCm', 'flash-attn build error gfx942', 'undefined "
            "symbol libamdhip64'). Returns title+URL+snippet for top N hits "
            "(default 5). Cached 7 days. Pair with `visit_url` to read the "
            "best hit. Cheap; prefer over guessing-and-retrying."
        ),
    }
    visit_url = {
        "command": 'visit_url <url> [--max-chars N]',
        "description": (
            "Fetch a URL and return readable text (HTML stripped to markdown). "
            "Use this AFTER `web_search` to read a specific GitHub issue / "
            "ROCm doc / blog post that promises an answer. Default cap "
            "8000 chars; raise if you need more context. Cached 7 days."
        ),
    }
    # ── PR-C: deep research sub-agent (iterative web research loop) ──
    deep_research = {
        "command": 'deep_research "<question>" [--max-turns N] [--budget-s S]',
        "description": (
            "Run a BOUNDED sub-agent that iteratively searches the web, reads "
            "the most relevant pages, and returns ONE distilled answer with "
            "citations and (where applicable) verified install commands. Use "
            "this for niche errors that single-shot `web_search` can't crack "
            "in one round (e.g. SDPA tensor shape mismatches, undefined HIP "
            "symbols, multi-version transformers/torch/flash-attn dances). "
            "Defaults: max-turns=6, budget-s=90. Cached 14 days. Costs 4-6 "
            "small LLM calls; saves the parent ~10-25 turns of trial-error."
        ),
    }