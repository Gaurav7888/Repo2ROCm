import csv
import subprocess

input_csv = "preprocessed_dataset.csv"
output_script = "batch_commands.txt"

with open(input_csv, "r") as f:
    reader = csv.DictReader(f)
    commands = []
    for row in reader:
        github_link = row["github_link"].strip()

        # Extract "owner/repo" from the GitHub URL
        # Handles URLs like https://github.com/owner/repo or https://github.com/owner/repo/tree/...
        parts = github_link.replace("https://github.com/", "").split("/")
        if len(parts) < 2:
            print(f"Skipping invalid link: {github_link}")
            continue
        full_name = f"{parts[0]}/{parts[1]}"

        # Get the latest commit SHA via git ls-remote
        try:
            result = subprocess.run(
                ["git", "ls-remote", f"https://github.com/{full_name}.git", "HEAD"],
                capture_output=True, text=True, timeout=30
            )
            sha = result.stdout.strip().split()[0] if result.stdout.strip() else ""
        except Exception as e:
            print(f"Failed to get SHA for {full_name}: {e}")
            continue

        if not sha:
            print(f"No SHA found for {full_name}, skipping")
            continue

        cmd = (
            f'python -u build_agent/main.py '
            f'--full_name "{full_name}" '
            f'--sha "{sha}" '
            f'--root_path . '
            f'--llm "claude-sonnet-4" '
            f'--rocm '
            f'--api-key "$AMD_LLM_API_KEY"'
        )
        commands.append(cmd)

with open(output_script, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")

print(f"Generated {len(commands)} commands in {output_script}")