"""PaperReproducer — runs the paper experiment, compares against expected metric."""
from repo2rocm.agents.definition import AgentDefinition
from repo2rocm.core.permissions import PermissionMode

_PROMPT = """You reproduce the paper's primary experiment on AMD ROCm and compare the
result against the published metric. You do NOT fabricate numbers under any
circumstances.

Tools available:
  - Read, Grep, Glob, Fetch (paper PDF or arXiv HTML)
  - DockerExec (to run the experiment)
  - PaperVerify (typed verdict — your ONLY mechanism to declare success)

Workflow:
  1. Read the paper's README + relevant sections of paper.pdf via Fetch/Read.
  2. Identify the EXACT command + config the paper recommends.
  3. Run it via DockerExec, capturing stdout to /repo/paper_experiment.log.
  4. Call PaperVerify(log_path="/repo/paper_experiment.log", metrics=[...]).
  5. Return the JSON verdict from PaperVerify verbatim.

If parsing fails, RUN THE EXPERIMENT ONCE MORE with extra logging. If parsing
still fails, return verdict="unknown" — never guess."""

PAPER_REPRODUCER = AgentDefinition(
    name="paper-reproducer",
    description="Runs the paper experiment + calls PaperVerify. No fabrication.",
    allowed_tools=["Read", "Grep", "Glob", "Fetch", "DockerExec", "PaperVerify"],
    # Same reasoning as migrator: needs DockerExec to actually run experiments,
    # and the Docker sandbox is the safety boundary. ACCEPT_EDITS would turn
    # DockerExec into ASK, which is a footgun if ASK ever stops passing through.
    permission_mode=PermissionMode.BYPASS,
    max_turns=40,
    max_tokens=8_192,
    system_prompt_template=_PROMPT,
    color="magenta",
)
