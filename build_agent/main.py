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


import argparse
import json
import multiprocessing
import threading
import time
import os
import sys
from datetime import datetime, timedelta
from utils.sandbox import Sandbox
from agents.configuration import Configuration
from utils.llm import set_api_key
import subprocess
from utils.waiting_list import WaitingList
from utils.conflict_list import ConflictList
from utils.integrate_dockerfile import integrate_dockerfile
from utils.rich_logger import log_header, log_phase, log_success, log_error, log_info, log_container_op, console, init_file_log, close_file_log
from agents.planner import generate_plan, print_plan
import ast
import shutil

def move_files_to_repo(source_folder):
    # Temporary staging directory to flatten the repo's top-level folder
    target_folder = os.path.join(source_folder, 'repo_inner_directory_long_long_name_to_avoid_duplicate')
    
    # Create the staging directory if it doesn't exist
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    
    # Move all items into the staging directory
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        # Skip the staging directory itself
        if item == 'repo_inner_directory_long_long_name_to_avoid_duplicate':
            continue
        
        # Move file/directory into the staging directory
        shutil.move(item_path, os.path.join(target_folder, item))

    os.rename(target_folder, os.path.join(source_folder, 'repo'))

# Clone the repo into utils/repo/, flatten the top-level directory, and remove any existing Dockerfile.
def download_repo(root_path, full_name, sha):
    if len(full_name.split('/')) != 2:
        raise Exception("full_name Wrong!!!")
    author_name = full_name.split('/')[0]
    repo_name = full_name.split('/')[1]
    if not os.path.exists(f'{root_path}/utils/repo/{author_name}/{repo_name}'):
        os.system(f'mkdir -p {root_path}/utils/repo/{author_name}/{repo_name}')
    download_cmd = f"git clone https://github.com/{full_name}.git"
    subprocess.run(download_cmd, cwd=f'{root_path}/utils/repo/{author_name}', check=True, shell=True)
    move_files_to_repo(f'{root_path}/utils/repo/{author_name}/{repo_name}')
    if os.path.exists(f"{root_path}/utils/repo/{author_name}/{repo_name}/repo/Dockerfile") and not os.path.isdir(f"{root_path}/utils/repo/{author_name}/{repo_name}/repo/Dockerfile"):
        rm_dockerfile_cmd = f"rm -rf {root_path}/utils/repo/{author_name}/{repo_name}/repo/Dockerfile"
        subprocess.run(rm_dockerfile_cmd, check=True, shell=True)
    pipreqs_cmd = "pipreqs --savepath=.pipreqs/requirements_pipreqs.txt --force"
    os.system(f'mkdir {root_path}/utils/repo/{author_name}/{repo_name}/repo/.pipreqs')
    try:
        pipreqs_warnings = subprocess.run(pipreqs_cmd, cwd=f"{root_path}/utils/repo/{author_name}/{repo_name}/repo", check=True, shell=True, capture_output=True)
        with open(f'{root_path}/utils/repo/{author_name}/{repo_name}/repo/.pipreqs/pipreqs_output.txt', 'w') as w1:
            w1.write(pipreqs_warnings.stdout.decode('utf-8'))
        with open(f'{root_path}/utils/repo/{author_name}/{repo_name}/repo/.pipreqs/pipreqs_error.txt', 'w') as w2:
            w2.write(pipreqs_warnings.stderr.decode('utf-8'))
    except:
        pass

    checkout_cmd = f"git checkout {sha}"
    subprocess.run(checkout_cmd, cwd=f'{root_path}/utils/repo/{author_name}/{repo_name}/repo', capture_output=True, shell=True)

    # x = subprocess.run('git log -1 --format="%H"', cwd=f'{root_path}/utils/repo/{author_name}/{repo_name}/repo', capture_output=True, shell=True)
    with open(f'{root_path}/output/{author_name}/{repo_name}/sha.txt', 'w') as w1:
        w1.write(sha)

def main():
    # subprocess.run('docker rm -f $(docker ps -aq)', shell=True)
    parser = argparse.ArgumentParser(description='Run script with repository full name as an argument.')
    parser.add_argument('--full_name', type=str, help='The full name of the repository (e.g., user/repo).')
    parser.add_argument('--sha', type=str, help='sha')
    parser.add_argument('--root_path', type=str, help='root path')
    parser.add_argument('--llm', type=str, default='gpt-4o-2024-05-13', help='base LLM name')
    parser.add_argument('--rocm', action='store_true', default=False, help='Enable AMD ROCm mode for GPU workloads')
    parser.add_argument('--rocm-base-image', type=str, default=None, help='Override the default ROCm base image (e.g., rocm/pytorch:latest)')
    parser.add_argument('--api-key', type=str, default=None, help='AMD LLM API Gateway key (or set AMD_LLM_API_KEY env var)')
    parser.add_argument('--verbose', action='store_true', default=False, help='Show full LLM messages, responses, and context tables in terminal')
    parser.add_argument('--no-scale-down', action='store_true', default=False,
                        help='Run the repo exactly as the README describes — no scale-down of training params, no mock data. Uses real data/commands from the README as-is.')

    args = parser.parse_args()

    if args.verbose:
        from utils.rich_logger import set_verbose
        set_verbose(True)

    if args.api_key:
        set_api_key(args.api_key)
    elif os.environ.get("AMD_LLM_API_KEY"):
        set_api_key(os.environ["AMD_LLM_API_KEY"])

    waiting_list = WaitingList()
    conflict_list = ConflictList()

    root_path = args.root_path

    if not os.path.isabs(root_path):
        root_path = os.path.abspath(root_path)

    full_name = args.full_name
    sha = args.sha
    llm = args.llm
    rocm_mode = args.rocm
    rocm_base_image = args.rocm_base_image
    no_scale_down = args.no_scale_down

    log_header("REPO2RUN", f"Automated Environment Configuration Agent")
    log_phase("CONFIGURATION")
    log_info(f"Repository: {full_name}")
    log_info(f"SHA: {sha}")
    log_info(f"LLM: {llm}")
    log_info(f"ROCm Mode: {rocm_mode}")
    if no_scale_down:
        log_info(f"No-Scale-Down: ON (will follow README as-is, no mock data)")
    log_info(f"Root Path: {root_path}")
    print(full_name)
    print(sha)
    if os.path.exists(f'{root_path}/output/{full_name}/patch'):
        rm_cmd = f"rm -rf {root_path}/output/{full_name}/patch"
        subprocess.run(rm_cmd, shell=True, check=True)
    output_dir = f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}'
    if not os.path.exists(output_dir):
        subprocess.run(f'mkdir -p {output_dir}', shell=True)

    # Start file logging into the output directory
    init_file_log(output_dir)

    if os.path.exists(f'{root_path}/utils/repo/{full_name}'):
        init_cmd = f"rm -rf {root_path}/utils/repo/{full_name} && mkdir -p {root_path}/utils/repo/{full_name}"
    else:
        init_cmd = f"mkdir -p {root_path}/utils/repo/{full_name}"
    subprocess.run(init_cmd, check=True, shell=True)
    
    def timer():
        time.sleep(3600*2)  # 2-hour timeout
        print("Timeout for 2 hour!")
        os._exit(1)  # force-kill the process

    # Start the watchdog timer thread
    timer_thread = threading.Thread(target=timer)
    timer_thread.daemon = True
    timer_thread.start()

    log_phase("CLONING REPOSITORY", f"{full_name} @ {sha[:12]}")
    download_repo(root_path, full_name, sha)
    log_success(f"Repository cloned: {full_name}")

    # ── Upfront Planning Phase ──
    repo_path = f"{root_path}/utils/repo/{full_name}/repo"
    plan, recommended_image = generate_plan(
        repo_path=repo_path,
        full_name=full_name,
        rocm_mode=rocm_mode,
        llm=llm,
        no_scale_down=no_scale_down,
    )
    print_plan(plan)

    with open(f'{root_path}/output/{full_name}/plan.txt', 'w') as pf:
        pf.write(plan)
    log_success(f"Plan saved to output/{full_name}/plan.txt")

    trajectory = []

    if rocm_mode:
        if rocm_base_image:
            base_image = rocm_base_image
            log_info(f"ROCm mode: using CLI-specified image: {base_image}")
        elif recommended_image:
            base_image = recommended_image
            log_info(f"ROCm mode: using planner-recommended image: {base_image}")
        else:
            base_image = 'rocm/pytorch:latest'
            log_info(f"ROCm mode: using default fallback image: {base_image}")
    else:
        base_image = 'python:3.10'
        log_info(f"Using base image: {base_image}")

    log_phase("STARTING DOCKER CONTAINER")
    configuration_sandbox = Sandbox(base_image, full_name, root_path, rocm_mode=rocm_mode)
    configuration_sandbox.start_container()
    log_success("Container started and repo copied")

    log_phase("RUNNING CONFIGURATION AGENT")
    configuration_agent = Configuration(
        configuration_sandbox, base_image, full_name, root_path, llm, 100,
        rocm_mode=rocm_mode, plan=plan, no_scale_down=no_scale_down,
    )
    msg, outer_commands = configuration_agent.run('/tmp', trajectory, waiting_list, conflict_list)
    with open(f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}/track.json', 'w') as w1:
        w1.write(json.dumps(msg, indent=4))
    commands = configuration_sandbox.stop_container()
    with open(f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}/inner_commands.json', 'w') as w2:
        w2.write(json.dumps(commands, indent=4))
    with open(f'{root_path}/output/{full_name.split("/")[0]}/{full_name.split("/")[1]}/outer_commands.json', 'w') as w3:
        w3.write(json.dumps(outer_commands, indent=4))

    run_success = False
    try:
        integrate_dockerfile(f'{root_path}/output/{full_name}')
        msg_str = f'Generate success!'
        log_success(msg_str)
        with open(f'{root_path}/output/{full_name}/track.txt', 'a') as a1:
            a1.write(msg_str + '\n')
        run_success = True
    except Exception as e:
        msg_str = f'integrate_docker failed, reason:\n {e}'
        log_error(msg_str)
        with open(f'{root_path}/output/{full_name}/track.txt', 'a') as a1:
            a1.write(msg_str + '\n')

    close_file_log()
    log_info(f"Debug log saved to: {output_dir}/agent_debug_log.txt")

if __name__ == '__main__':
    try:
        subprocess.run('docker rmi $(docker images --filter "dangling=true" -q) > /dev/null 2>&1', shell=True)
    except:
        pass
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    log_phase("COMPLETE", f"Total time: {elapsed_time:.1f}s")
    console.print(f"\n[bold]Total elapsed time: {elapsed_time:.1f}s[/bold]\n")