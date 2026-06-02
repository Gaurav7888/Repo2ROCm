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
from utils.parser.parse_command import (
    match_mem_recall, match_paper_recall, match_graphify_query,
    match_pypi_versions, match_dockerhub_tags,
    match_web_search, match_visit_url,
    match_deep_research, match_verify_paper_result,
)
from utils.tools_config import Tools
from utils.split_cmd import split_cmd_statements
from knowledge.rocm_knowledge import generate_rocm_prompt_section, generate_rocm_prompt_section_with_plan
from utils.rich_logger import (
    log_header, log_phase, log_turn, log_prompt_sent, log_llm_response,
    log_multi_action_warning, log_action, log_observation, log_rocm_no_test_block,
    log_rocm_success, log_revert, log_context_summary, log_finish_summary,
    log_success, log_error, log_info, log_observer_note, console,
)
import re
import time

# ── Stage 1 (observation compactor) + Stage 2 (per-run mempalace) ──
try:
    from learning.observation_compactor import compact as _compact_obs
    _COMPACTOR_AVAILABLE = True
except Exception:
    _COMPACTOR_AVAILABLE = False
    def _compact_obs(text, **kw):  # type: ignore[no-redef]
        class _C:
            short = text
            full = text
            error_class = None
            metrics = []
            truncated = False
            orig_chars = len(text or "")
            compact_chars = len(text or "")
        return _C()

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
    def __init__(self, sandbox, image_name, full_name, root_dir, llm="gpt-4o-2024-05-13", max_turn=70, rocm_mode=False, plan=None, no_scale_down=False,
                 error_classifier=None, rule_engine=None, memory_provider=None,
                 trajectory_store=None, build_attempt=None, kb_context="",
                 optimize_kernels=False, use_claude_code=False,
                 reproduce_results=False, paper_pdf_path=None,
                 paper_experiments=None, paper_title="",
                 run_memory=None, graphify_provider=None,
                 observer_client=None,
                 run_mode: str = "env"):
        self.model = llm
        self.root_dir = root_dir
        self.max_turn = max_turn
        self.rocm_mode = rocm_mode
        self.no_scale_down = no_scale_down
        self.sandbox = sandbox
        self.sandbox_session = self.sandbox.get_session()
        self.full_name = full_name
        self.plan = plan
        self.error_classifier = error_classifier
        self.rule_engine = rule_engine
        self.memory_provider = memory_provider
        self.trajectory_store = trajectory_store
        self.build_attempt = build_attempt
        self.kb_context = kb_context
        self.optimize_kernels = optimize_kernels
        self.use_claude_code = use_claude_code
        self.run_memory = run_memory  # mempalace per-run store (Stage 2); may be None
        self.graphify_provider = graphify_provider  # per-repo code graph (Stage 4); may be None
        self.observer_client = observer_client
        self.run_mode = run_mode  # "env" | "reproduce" | "full"
        # Track Stage-1 compaction savings for diagnostics.
        self._compaction_orig_chars = 0
        self._compaction_short_chars = 0
        # Stage 5b + PR-A + PR-B + PR-C: track in-loop retrieval-tool usage.
        self._mem_recall_calls = 0
        self._paper_recall_calls = 0
        self._graphify_query_calls = 0
        self._pypi_versions_calls = 0
        self._dockerhub_tags_calls = 0
        self._web_search_calls = 0
        self._visit_url_calls = 0
        self._deep_research_calls = 0
        self._verify_paper_calls = 0
        # Stage 3: track latest activity to use as retrieval query for the appendix.
        self._last_action_text = ""
        self._last_obs_short = ""
        self._last_error_class = ""
        # Stage 3 diagnostics
        self._appendix_old_chars = 0
        self._appendix_new_chars = 0
        # Stage 2 paper-discipline guard: require retrieval before raw PDF reads.
        self._paper_retrieval_used = False
        self._graphify_code_lookup_used = False
        # When an AMD/ROCm-specific failure happens, require at least one live
        # internet-backed lookup before the next high-impact action.
        self._needs_live_amd_research = False
        self._amd_live_research_hint = ""
        # ── Hard-guard state (PR: tool-calling is a constraint, not advice) ──
        # `_verifier_records` keeps the most-recent structured JSON the
        # `verify_paper_result` tool produced, keyed by log path. The paper
        # marker handler refuses to emit a verdict that has not been backed by
        # a verifier record from THIS run.
        self._verifier_records: dict = {}
        # `_dockerhub_tags_seen`: set of image repos for which we have already
        # observed a successful `dockerhub_tags` lookup. The hard guard
        # blocks `change_base_image <repo>:<tag>` until we've checked.
        self._dockerhub_tags_seen: set = set()
        # `_pypi_versions_seen`: same idea for `pip install` of CUDA-only wheels.
        self._pypi_versions_seen: set = set()
        # `_gpu_check_seen`: True after the agent ran a successful
        # `torch.cuda.is_available()` / `rocm-smi` check. Required before
        # echoing `ROCM_ENV_VERIFIED`.
        self._gpu_check_seen: bool = False

        # ── STAGE 2 (paper reproduction) state ───────────────────────────────
        self.reproduce_results = bool(reproduce_results)
        self.paper_pdf_path = paper_pdf_path or ""
        self.paper_experiments = list(paper_experiments) if paper_experiments else []
        self.paper_title = paper_title or ""
        self._stage2_active = False
        self._stage2_announced = False
        self._paper_result_line = ""
        self._paper_verdict = ""  # "reproduced" | "not_reproduced" | ""
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
        # Stage 5b: only advertise retrieval tools when their backends are wired.
        if self.run_memory is not None and getattr(self.run_memory, "enabled", False):
            self.tool_lib.append(Tools.mem_recall)
            self.tool_lib.append(Tools.paper_recall)
        if self.graphify_provider is not None and getattr(self.graphify_provider, "enabled", False):
            self.tool_lib.append(Tools.graphify_query)
        # PR-A: external lookups always available (pure-stdlib HTTP, soft-fail).
        self.tool_lib.append(Tools.pypi_versions)
        self.tool_lib.append(Tools.dockerhub_tags)
        # Stage-2 deterministic verifier (only meaningful when reproducing a paper).
        if self.reproduce_results:
            self.tool_lib.append(Tools.verify_paper_result)
        self.image_name = image_name
        # The planner already validated the initial image's tags; treat its
        # repo as "seen" so the agent isn't asked to re-look it up just to
        # change to a different tag of the same repo.
        try:
            init_repo = (image_name or "").split(":", 1)[0].strip().lower()
            if init_repo:
                self._dockerhub_tags_seen.add(init_repo)
        except Exception:
            pass
        self.outer_commands = list()
        tools_list = ""
        for tool in self.tool_lib:
            tools_list += f"{tool.value['command']} # {tool.value['description']}\n"

        self.system_prompt = self._build_system_prompt(tools_list)

    # ── Stage 5b: in-loop retrieval tools (mempalace + graphify) ────────────

    def _tool_usage_snapshot(self) -> dict:
        return {
            "mem_recall": self._mem_recall_calls,
            "paper_recall": self._paper_recall_calls,
            "graphify_query": self._graphify_query_calls,
            "pypi_versions": self._pypi_versions_calls,
            "dockerhub_tags": self._dockerhub_tags_calls,
            "web_search": self._web_search_calls,
            "visit_url": self._visit_url_calls,
            "deep_research": self._deep_research_calls,
            "verify_paper_result": self._verify_paper_calls,
        }

    def _emit_observer_event(self, event_type: str, payload: dict) -> None:
        if self.observer_client is None:
            return
        try:
            self.observer_client.emit_event(event_type, payload)
        except Exception as e:
            log_info(f"[observer] emit failed: {e}")

    def _format_observer_advice(self, advice_rows: list) -> str:
        lines = [
            "### External Observer Notes",
            "An asynchronous observer reviewed the recent turn history and is "
            "preparing the next turn. Treat these as advisory readiness packs "
            "rather than executable commands.",
        ]
        for row in advice_rows:
            profile = str(row.get("profile_used") or "observerCritic")
            kind = str(row.get("kind") or "reactive")
            priority = str(row.get("priority") or "normal")
            diagnosis = str(row.get("diagnosis") or "").strip()
            strategy = str(row.get("recommended_strategy") or "").strip()
            predicted = str(row.get("predicted_failure") or "").strip()
            applies_before = str(row.get("applies_before") or "").strip()
            confidence = float(row.get("confidence", 0.0) or 0.0)
            expires = row.get("expires_after_turn")
            lines.append("")
            label_bits = [profile, kind, f"priority={priority}"]
            if applies_before:
                label_bits.append(f"applies_before={applies_before}")
            if isinstance(expires, (int, float)) and int(expires) >= 0:
                label_bits.append(f"expires_after_turn={int(expires)}")
            lines.append(
                "- Observer pack: "
                + " | ".join(label_bits)
                + f" (confidence={confidence:.2f}, turn_seen={row.get('turn_seen', '?')})"
            )
            if predicted:
                lines.append(f"  Predicted failure: {predicted}")
            if diagnosis and diagnosis != predicted:
                lines.append(f"  Diagnosis: {diagnosis}")
            if strategy:
                lines.append(f"  Strategy: {strategy}")
            for item in (row.get("suggested_questions_or_tools") or [])[:4]:
                lines.append(f"  Next: {str(item)[:240]}")
            for item in (row.get("evidence") or [])[:3]:
                lines.append(f"  Evidence: {str(item)[:240]}")
        lines.append("")
        lines.append(
            "Use these readiness packs to choose the next action. "
            "Do not blindly obey them and do not execute the observer's "
            "suggestions verbatim if the current local state contradicts them."
        )
        return "\n".join(lines)

    def _consume_observer_advice(self, turn: int) -> None:
        if self.observer_client is None:
            return
        try:
            fresh = self.observer_client.consume_new_advice(current_turn=turn)
        except Exception as e:
            log_info(f"[observer] advice read failed: {e}")
            return
        if not fresh:
            return
        note_text = self._format_observer_advice(fresh)
        role = "system" if "gpt" in self.model else "user"
        self.messages.append({"role": role, "content": note_text})
        self.outer_commands.append({"observer_advice": fresh, "turn": turn})
        log_observer_note(note_text)

    def _emit_turn_snapshot(self, turn: int, assistant_response: str,
                            commands: list, diffs: list, system_res: str,
                            turn_elapsed_s: float, outer_entries: list,
                            tool_counts_before: dict) -> None:
        if self.observer_client is None:
            return
        tool_counts_after = self._tool_usage_snapshot()
        tool_deltas = {
            key: tool_counts_after.get(key, 0) - tool_counts_before.get(key, 0)
            for key in tool_counts_after
        }
        return_codes = [
            entry.get("returncode")
            for entry in outer_entries
            if isinstance(entry, dict) and "returncode" in entry
        ]
        recent_commands = []
        for item in self.sandbox.commands[-8:]:
            if not isinstance(item, dict):
                continue
            recent_commands.append({
                "command": str(item.get("command") or "")[:480],
                "returncode": item.get("returncode"),
                "time": item.get("time"),
                "dir": item.get("dir"),
            })
        # Observation/response caps are intentionally generous: the observer
        # sidecar reasons over the raw text the way a human reads CI logs, so
        # cutting off mid-error-list (as the prior 2500-char cap did) hides
        # the cascading-error pattern that signals a stuck loop. The on-disk
        # event bus is cheap and the LLM can still rebudget per turn.
        payload = {
            "turn": turn,
            "stage": "stage2" if self._stage2_active else "stage1",
            "rocm_mode": self.rocm_mode,
            "paper_title": self.paper_title,
            "assistant_response": str(assistant_response or "")[:6000],
            "action_type": "bash" if commands else "diff" if diffs else "none",
            "commands": [str(cmd)[:480] for cmd in commands[:5]],
            "diff_present": bool(diffs),
            "return_codes": return_codes,
            "duration_s": round(float(turn_elapsed_s), 2),
            "error_class": self._last_error_class,
            "observation_excerpt": str(system_res or "")[:8000],
            "tool_deltas": tool_deltas,
            "recent_commands": recent_commands,
            "paper_retrieval_used": self._paper_retrieval_used,
            "graphify_code_lookup_used": self._graphify_code_lookup_used,
            "needs_live_amd_research": self._needs_live_amd_research,
        }
        self._emit_observer_event("turn_snapshot", payload)

    def _maybe_run_retrieval_tool(self, command: str):
        """
        If `command` is a retrieval tool (mem_recall / graphify_query), execute
        it locally and return (observation_text, return_code). Otherwise return
        None so the caller falls through to sandbox execution.

        Observation text is intentionally short: the goal is to give the agent a
        focused snippet of prior context, not another wall of text.
        """
        if not command or not isinstance(command, str):
            return None

        # mem_recall
        mr = match_mem_recall(command)
        if mr != -1:
            self._mem_recall_calls += 1
            if self.run_memory is None or not getattr(self.run_memory, "enabled", False):
                return (
                    "mem_recall is unavailable (run memory not enabled). "
                    "Install `mempalace` and re-run, or fall back to grep/find.\n"
                ), 1
            try:
                rooms = mr.get("rooms") or (
                    "commands_success", "commands_failed", "fixes",
                    "decisions", "patches", "plan", "experiment_state", "context_refs",
                )
                if isinstance(rooms, list):
                    rooms = tuple(rooms)
                pack = self.run_memory.recall_pack(
                    mr["question"], rooms=rooms, n_per_room=4,
                    token_budget=int(mr.get("budget", 1500)),
                ) or ""
                if mr.get("use_global"):
                    pack += (
                        "\n[note] cross-run natural-language lesson recall is "
                        "disabled. Use the structured KB guidance already in the "
                        "system prompt, or verify against the current repo and "
                        "tool output.\n"
                    )
                if not pack.strip():
                    pack = (
                        "mem_recall: no relevant prior context found for this "
                        "question. Either rephrase, broaden --rooms, or proceed "
                        "without memory hints.\n"
                    )
                if self._stage2_active:
                    self._paper_retrieval_used = True
                msg = f"Running `{command}`...\n" + pack
                return msg, 0
            except Exception as e:
                return f"mem_recall failed: {e}\n", 1

        # paper_recall
        pr = match_paper_recall(command)
        if pr != -1:
            self._paper_recall_calls += 1
            try:
                budget = int(pr.get("budget", 1500))
                # 1) Static paper corpus from graphify
                paper_pack = ""
                if self.graphify_provider is not None and getattr(self.graphify_provider, "enabled", False):
                    paper_pack = self.graphify_provider.query_paper(
                        pr["question"],
                        token_budget=max(800, (budget * 2) // 3),
                        max_chunks=6,
                        per_chunk_max_chars=1500,
                    ) or ""
                # 2) Dynamic run-state / references from mempalace
                state_pack = ""
                if self.run_memory is not None and getattr(self.run_memory, "enabled", False):
                    state_pack = self.run_memory.recall_pack(
                        pr["question"],
                        rooms=("paper_experiments", "experiment_state", "context_refs", "plan", "decisions"),
                        n_per_room=4,
                        token_budget=max(400, budget // 2),
                        header="PAPER RUN STATE (choices, refs, decisions)",
                    ) or ""
                pack = (paper_pack or "") + (state_pack or "")
                if pr.get("use_global"):
                    pack += (
                        "\n[note] cross-run natural-language lesson recall is "
                        "disabled. Prefer the current paper evidence, current "
                        "run-state, and deterministic verifier outputs.\n"
                    )
                if not pack.strip():
                    pack = (
                        "paper_recall: no relevant paper context found from graphify or run-state references. "
                        "Try a more specific paper question (metric, experiment name, hyperparameter), "
                        "or wait for an observer note with external evidence.\n"
                    )
                self._paper_retrieval_used = True
                msg = f"Running `{command}`...\n" + pack
                return msg, 0
            except Exception as e:
                return f"paper_recall failed: {e}\n", 1

        # graphify_query (now scope-aware: code | paper | both)
        gq = match_graphify_query(command)
        if gq != -1:
            self._graphify_query_calls += 1
            if self.graphify_provider is None or not getattr(self.graphify_provider, "enabled", False):
                return (
                    "graphify_query is unavailable (code graph not built). "
                    "Install `graphifyy[pdf]` and re-run, or fall back to "
                    "grep/find on /repo.\n"
                ), 1
            try:
                budget = int(gq.get("budget", 1500))
                scope = (gq.get("scope") or "code").lower()
                question = gq["question"]
                blocks: list = []
                if scope in ("code", "both"):
                    code_snip = self.graphify_provider.query(
                        question, token_budget=(budget if scope == "code" else max(800, (budget * 2) // 3)),
                    ) or ""
                    if code_snip.strip():
                        blocks.append(code_snip)
                if scope in ("paper", "both"):
                    paper_snip = self.graphify_provider.query_paper(
                        question,
                        token_budget=(budget if scope == "paper" else max(800, budget // 2)),
                        max_chunks=6,
                        per_chunk_max_chars=1500,
                    ) or ""
                    if paper_snip.strip():
                        blocks.append(paper_snip)
                snippet = "\n".join(blocks).strip()
                if not snippet:
                    snippet = (
                        f"graphify_query (--scope {scope}): no nodes matched. "
                        "Try different terms, change scope, or fall back to grep/find on /repo.\n"
                    )
                # When the paper corpus was queried we satisfy the Stage 2
                # paper-discipline guard. Pure-code queries don't.
                if scope in ("paper", "both") and self._stage2_active:
                    self._paper_retrieval_used = True
                if scope in ("code", "both"):
                    self._graphify_code_lookup_used = True
                msg = f"Running `{command}`...\n" + snippet
                return msg, 0
            except Exception as e:
                return f"graphify_query failed: {e}\n", 1

        # PR-A: pypi_versions
        pv = match_pypi_versions(command)
        if pv != -1:
            self._pypi_versions_calls += 1
            try:
                from tools.external_lookups import pypi_versions as _pv
                body, rc = _pv(pv["package"], limit=int(pv.get("limit", 12)))
                if rc == 0:
                    self._needs_live_amd_research = False
                    self._amd_live_research_hint = ""
                msg = f"Running `{command}`...\n" + body
                return msg, rc
            except Exception as e:
                return f"pypi_versions failed: {e}\n", 1

        # PR-A: dockerhub_tags
        dt = match_dockerhub_tags(command)
        if dt != -1:
            self._dockerhub_tags_calls += 1
            try:
                from tools.external_lookups import dockerhub_tags as _dt
                body, rc = _dt(dt["image"], limit=int(dt.get("limit", 12)))
                if rc == 0:
                    self._needs_live_amd_research = False
                    self._amd_live_research_hint = ""
                msg = f"Running `{command}`...\n" + body
                return msg, rc
            except Exception as e:
                return f"dockerhub_tags failed: {e}\n", 1

        # PR-B: web_search
        ws = match_web_search(command)
        if ws != -1:
            self._web_search_calls += 1
            try:
                from tools.web_search import web_search as _ws
                body, rc = _ws(ws["query"], max_results=int(ws.get("max_results", 5)))
                if self._stage2_active and rc == 0:
                    self._paper_retrieval_used = True
                if rc == 0:
                    self._needs_live_amd_research = False
                    self._amd_live_research_hint = ""
                msg = f"Running `{command}`...\n" + body
                return msg, rc
            except Exception as e:
                return f"web_search failed: {e}\n", 1

        # PR-B: visit_url
        vu = match_visit_url(command)
        if vu != -1:
            self._visit_url_calls += 1
            try:
                from tools.web_search import visit_url as _vu
                body, rc = _vu(vu["url"], max_chars=int(vu.get("max_chars", 8000)))
                if self._stage2_active and rc == 0:
                    self._paper_retrieval_used = True
                if rc == 0:
                    self._needs_live_amd_research = False
                    self._amd_live_research_hint = ""
                msg = f"Running `{command}`...\n" + body
                return msg, rc
            except Exception as e:
                return f"visit_url failed: {e}\n", 1

        # Stage-2 deterministic verifier (called BEFORE the marker line).
        vp = match_verify_paper_result(command)
        if vp != -1:
            self._verify_paper_calls += 1
            try:
                from tools.verify_paper_result import verify_paper_result as _vpr
                # If the LLM did not pass --metric, fill from the chosen
                # experiment's primary metrics (preferred) or the legacy single
                # `expected_metric_name` / `expected_metric_value` fields.
                metrics = list(vp.get("metrics") or [])
                if not metrics and self.paper_experiments:
                    chosen = self.paper_experiments[0]
                    if not isinstance(chosen, dict):
                        try:
                            chosen = chosen.to_dict()
                        except Exception:
                            chosen = {}
                    primary = chosen.get("primary_metrics") or []
                    if primary:
                        for pm in primary:
                            if isinstance(pm, dict) and pm.get("name"):
                                metrics.append({
                                    "name": pm["name"],
                                    "expected_value": pm.get("expected_value")
                                        if pm.get("expected_value") is not None
                                        else pm.get("value"),
                                    "tolerance": pm.get("tolerance") or "",
                                    "direction": pm.get("direction") or "",
                                })
                    elif chosen.get("expected_metric_name"):
                        metrics.append({
                            "name": chosen["expected_metric_name"],
                            "expected_value": chosen.get("expected_metric_value"),
                            "tolerance": chosen.get("tolerance_rule") or "",
                            "direction": "",
                        })

                # Resolve /repo/<x> log paths to a host path so we can read
                # them without a docker exec round-trip.
                log_path = vp.get("log_path") or ""
                if log_path.startswith("/repo/"):
                    host_repo = (
                        f"{self.root_dir}/utils/repo/{self.full_name}/repo"
                    )
                    os.environ["REPO2ROCM_HOST_REPO_PATH"] = host_repo
                    # Try to copy the log out of the container first; the
                    # sandbox is already running, so a docker cp is the
                    # safest way to see it from the host.
                    try:
                        cont_id = getattr(self.sandbox, "container", None)
                        cont_id = getattr(cont_id, "id", None) or ""
                        host_target = os.path.join(
                            host_repo, log_path[len("/repo/"):]
                        )
                        if cont_id:
                            import subprocess as _sp
                            _sp.run(
                                ["docker", "cp",
                                 f"{cont_id}:{log_path}",
                                 host_target],
                                check=False, capture_output=True, timeout=30,
                            )
                    except Exception:
                        pass

                body, rc, record = _vpr(
                    log_path=log_path,
                    metrics=metrics,
                    tolerance=vp.get("tolerance") or "",
                    direction=vp.get("direction") or "",
                )
                if log_path:
                    self._verifier_records[log_path] = record
                if self._stage2_active:
                    self._paper_retrieval_used = True
                msg = f"Running `{command}`...\n" + body
                return msg, rc
            except Exception as e:
                return f"verify_paper_result failed: {e}\n", 1

        # PR-C: deep_research (sub-agent)
        dr = match_deep_research(command)
        if dr != -1:
            self._deep_research_calls += 1
            try:
                from agents.researcher import research, format_for_observation
                note = research(
                    dr["question"],
                    llm=self.model,
                    max_turns=int(dr.get("max_turns", 6)),
                    budget_s=float(dr.get("budget_s", 90.0)),
                    use_cache=bool(dr.get("use_cache", True)),
                )
                if self._stage2_active:
                    self._paper_retrieval_used = True
                self._needs_live_amd_research = False
                self._amd_live_research_hint = ""
                msg = f"Running `{command}`...\n" + format_for_observation(note)
                rc = 0 if (note.get("confidence", 0) > 0 or note.get("_cache_hit")) else 1
                return msg, rc
            except Exception as e:
                return f"deep_research failed: {e}\n", 1

        return None

    # ── Hard guards: turn tool-calling from advice into a constraint ─────────
    #
    # Each guard returns (observation_text, return_code) just like
    # `_maybe_run_retrieval_tool`. Returning None means "no guard fired,
    # continue to sandbox dispatch".

    # CUDA-only wheels we never want pip-installed without first checking
    # PyPI. Maintained intentionally short — anything not on this list still
    # passes through.
    _CUDA_ONLY_WHEELS = (
        "flash-attn", "flash_attn", "bitsandbytes", "xformers",
        "nvidia-pyindex", "nvidia-cublas-cu11", "nvidia-cudnn-cu11",
        "deepspeed", "apex", "cupy",
    )

    @staticmethod
    def _command_first_token(command: str) -> str:
        if not command:
            return ""
        return command.strip().split(None, 1)[0].lower()

    @staticmethod
    def _extract_pip_install_packages(command: str):
        """Return the package names mentioned by a `pip install ...` command.

        Best-effort: strips flags, version specifiers, URLs, and -r/-c file
        forms. Returns lowercase short names.
        """
        if not command or "pip" not in command.lower():
            return []
        # only react to actual install verbs
        m = re.search(r"\bpip\d?\s+install\b(.*)$", command, re.IGNORECASE)
        if not m:
            return []
        rest = m.group(1)
        out: list = []
        for tok in re.split(r"\s+", rest.strip()):
            if not tok or tok.startswith("-"):
                continue
            if tok.startswith(("http://", "https://", "git+", "file://", ".", "/")):
                continue
            # split off the version specifier
            name = re.split(r"[<>=!~\[]", tok, maxsplit=1)[0].strip()
            if name:
                out.append(name.lower())
        return out

    @staticmethod
    def _looks_like_repo_discovery_command(command: str) -> bool:
        """Detect broad repo-search commands that graphify should replace."""
        if not command:
            return False
        c = command.strip().lower()
        return any(marker in c for marker in (
            "find /repo",
            "find . -name",
            "find . -path",
            "grep -r /repo",
            "grep -r .",
            "grep -r \"",
            "grep -r '",
            "grep -r ",
            "grep -R /repo",
            "ls -r /repo",
            "ls -R /repo",
            "tree /repo",
        ))

    @staticmethod
    def _looks_like_stage2_execution(command: str) -> bool:
        """Detect substantive paper-stage execution before evidence gathering."""
        if not command:
            return False
        c = command.strip().lower()
        if "paper_experiment.log" in c:
            return True
        return bool(re.search(r"\bpython\d?\s+\S+\.py\b", c))

    @staticmethod
    def _looks_like_amd_specific_issue(*texts: str) -> bool:
        """Best-effort detector for failures that need live AMD/ROCm research."""
        hay = " ".join((t or "") for t in texts).lower()
        markers = (
            "rocm", "hip", "amd", "gfx", "miopen", "rocblas", "rccl",
            "libamdhip64", "hiperror", "amdgpu", "hsa", "composable_kernel",
            "flash-attn", "flash_attn", "bitsandbytes", "xformers",
            "deepspeed", "triton", "sdpa", "torch.version.hip",
            "mi250", "mi300", "undefined symbol", "ck_tile",
        )
        return any(m in hay for m in markers)

    def _maybe_run_hard_guard(self, command: str):
        """Run the policy guards. Returns (obs, rc) or None to fall through."""
        if not command:
            return None
        c = command.strip()
        first = self._command_first_token(c)

        # Guard A: change_base_image must be preceded by a successful
        # `dockerhub_tags <repo>` lookup so the LLM can't invent a stale tag.
        if first == "change_base_image":
            target = c[len("change_base_image"):].strip().lower()
            repo = target.split(":", 1)[0]
            if repo and repo not in self._dockerhub_tags_seen:
                return (
                    "Hard guard: `change_base_image` requires a recent "
                    f"`dockerhub_tags {repo}` lookup BEFORE switching, so the "
                    "tag you pick is one Docker Hub actually serves. Run:\n"
                    f"    dockerhub_tags {repo} --limit 8\n"
                    "Then re-issue the change_base_image command with a tag "
                    "from that list. (If the repo is one of the canonical "
                    "rocm/* images, prefer `rocm/pytorch:latest` as a default.)"
                ), 1

        # Guard B: pip install <CUDA-only wheel> must be preceded by a
        # `pypi_versions <pkg>` lookup so the agent picks an installable
        # version rather than the latest CUDA-only build.
        for pkg in self._extract_pip_install_packages(c):
            for risky in self._CUDA_ONLY_WHEELS:
                if pkg == risky.lower() and risky.lower() not in self._pypi_versions_seen:
                    return (
                        f"Hard guard: `pip install {pkg}` is high-risk on "
                        "ROCm because PyPI ships CUDA-only wheels for it. "
                        "Run `pypi_versions " + pkg + " --limit 8` first "
                        "(zero LLM cost; cached) and pick a version that "
                        "matches your ROCm torch, OR follow the CUDA-to-ROCm "
                        "mapping in the system prompt (e.g. flash-attn "
                        "Triton-AMD install)."
                    ), 1

        # Guard C: ROCM_ENV_VERIFIED requires that we have observed at least
        # one successful GPU-availability check in this run. The check itself
        # is detected opportunistically by `_maybe_record_gpu_check` after
        # every sandbox execution.
        if (self.rocm_mode and not self._stage2_active
                and "ROCM_ENV_VERIFIED" in c
                and not self._gpu_check_seen):
            return (
                "Hard guard: do NOT echo `ROCM_ENV_VERIFIED` until you have "
                "verified GPU access in this run. Either run:\n"
                "    rocm-smi\n"
                "or:\n"
                "    python -c \"import torch; assert torch.cuda.is_available(); "
                "print('GPU OK', torch.cuda.get_device_name(0))\"\n"
                "and confirm the output mentions a real device. Then re-issue "
                "the ROCM_ENV_VERIFIED echo."
            ), 1

        # Guard D: PAPER_RESULT_* markers require a successful verify_paper_result
        # in THIS run that covered the same log file the marker references.
        if self._stage2_active and (
            "PAPER_RESULT_REPRODUCED" in c or "PAPER_RESULT_NOT_REPRODUCED" in c
        ):
            if not self._verifier_records:
                return (
                    "Hard guard: do NOT echo a PAPER_RESULT_* marker until you "
                    "have first run the deterministic verifier in this turn or "
                    "a previous turn. Run:\n"
                    "    verify_paper_result --log /repo/paper_experiment.log\n"
                    "(metric/tolerance default to the chosen experiment's "
                    "primary metrics). The verifier prints the JSON you must "
                    "echo. Do NOT invent numbers."
                ), 1

        # Guard E: stage 2 must consult at least one evidence tool before
        # launching the paper experiment itself.
        if (self._stage2_active
                and not self._paper_retrieval_used
                and self._looks_like_stage2_execution(c)):
            return (
                "Hard guard: before running the Stage 2 paper experiment, you "
                "must gather paper evidence in this run. Use one of:\n"
                "    paper_recall \"what metric / hyperparameters / command matter most?\"\n"
                "    graphify_query \"entrypoint / config / metric logging\" --scope both\n"
                                "    wait for an observer note with paper-specific external evidence\n"
                "Then run the experiment with those concrete details."
            ), 1

        # Guard F: broad repo discovery should use graphify, not shell search.
        if (self.graphify_provider is not None
                and getattr(self.graphify_provider, "enabled", False)
                and not self._graphify_code_lookup_used
                and self._looks_like_repo_discovery_command(c)):
            return (
                "Hard guard: broad repo discovery should use `graphify_query` "
                "before shell-wide search. Run something like:\n"
                "    graphify_query \"entrypoints, config loaders, metric logging\" --scope code\n"
                "Use `find`/`grep -r` only after graphify has narrowed the search."
            ), 1

        # Guard G: after an AMD/ROCm-specific failure, require at least one
        # live internet-backed lookup before the next high-impact action.
        if self._needs_live_amd_research and first not in safe_cmd:
            hint = self._amd_live_research_hint or "the last AMD/ROCm-specific failure"
            return (
                "Hard guard: static knowledge is not enough for this AMD/ROCm-specific issue. "
                f"Before another high-impact action, use live evidence for {hint!r}.\n"
                "Prefer one of:\n"
                                "    wait for the observer to attach internet-backed diagnosis before retrying\n"
                "Use `pypi_versions` / `dockerhub_tags` too if the issue is package- or image-specific."
            ), 1

        return None

    def _maybe_record_gpu_check(self, command: str, observation: str,
                                 return_code: int) -> None:
        """Detect a successful GPU check and remember it for the env guard."""
        if self._gpu_check_seen or return_code != 0:
            return
        c = (command or "").lower()
        o = observation or ""
        triggers = (
            "rocm-smi" in c,
            "torch.cuda.is_available" in c,
            "torch.cuda.get_device_name" in c,
        )
        if not any(triggers):
            return
        positive = (
            "True" in o
            or re.search(r"GPU\s*\[\s*\d+\s*\]", o)
            or "Device " in o
            or "rocm-smi" in o.lower() and "no devices" not in o.lower()
        )
        if positive:
            self._gpu_check_seen = True

    def _provide_causal_memory_per_turn(
        self,
        command_text: str,
        sandbox_res: str,
        return_code: int,
        classified_error,
        turn: int,
    ) -> str:
        """Per-turn causal-memory retrieval.

        Builds a `MemoryRequest` from the richer per-turn state (image,
        gpu_arch, error class, return code, sandbox observation) and asks
        the memory provider for the top causal transitions.  Returns the
        formatted `[CAUSAL]` lines (or an empty string when no transition
        applies).

        This runs in every mode, including `--mode env`, so the
        configuration agent always sees the relevant state→action→outcome
        priors.
        """
        if self.memory_provider is None:
            return ""

        from storage.models import MemoryRequest, MemoryPhase

        gpu_arch = ""
        image = ""
        if self.build_attempt is not None:
            image = getattr(self.build_attempt, "docker_image", "") or ""
            gpu_arch = getattr(self.build_attempt, "gpu_arch", "") or ""
        if not image:
            image = getattr(self, "image_name", "") or ""

        error_class = ""
        if classified_error is not None and not classified_error.is_novel:
            error_class = classified_error.error_class or ""

        ctx = {
            "rocm_mode": self.rocm_mode,
            "image": image,
            "gpu_arch": gpu_arch,
            "return_code": return_code,
            "error_class": error_class,
            "degradation_policy": "strict" if self.no_scale_down else "permissive",
        }

        request = MemoryRequest(
            query=command_text or "",
            context=ctx,
            phase=MemoryPhase.IN.value,
            fingerprint=getattr(self.build_attempt, "fingerprint", None)
                if self.build_attempt is not None else None,
            current_error=(sandbox_res or "")[:2000] if return_code != 0 else None,
            turn_number=turn,
        )

        try:
            # Gate by relevance (min_similarity=0.3): on repos that match no
            # stored/seeded transition this returns nothing instead of
            # injecting irrelevant seed advisories on every turn (the
            # context-length blow-up this fix targets).
            causal_items = self.memory_provider.provide_causal_memory(
                request, top_k=3, min_similarity=0.3,
            )
        except Exception:
            return ""

        if not causal_items:
            return ""
        # `MemoryItem.content` already carries the `[CAUSAL] state{...} →
        # action{...} → outcome{...}` line plus any `[counterfactual: ...]`
        # advisory lines.
        return "\n" + "\n".join(item.content for item in causal_items) + "\n"

    def _maybe_record_external_lookup(self, command: str, return_code: int) -> None:
        """Track `dockerhub_tags` / `pypi_versions` calls so the guard relaxes."""
        if return_code != 0 or not command:
            return
        c = command.strip()
        if c.lower().startswith("dockerhub_tags"):
            m = re.match(r"^\s*dockerhub_tags\s+([A-Za-z0-9._\-/]+)", c, re.IGNORECASE)
            if m:
                self._dockerhub_tags_seen.add(m.group(1).strip().lower())
        elif c.lower().startswith("pypi_versions"):
            m = re.match(r"^\s*pypi_versions\s+([A-Za-z0-9._\-]+)", c, re.IGNORECASE)
            if m:
                self._pypi_versions_seen.add(m.group(1).strip().lower())

    def _is_raw_paper_read_command(self, command: str) -> bool:
        """Detect direct shell reads of `/repo/paper.pdf`.

        Stage 2 should prefer `paper_recall` and `graphify_query` first, then
        absorb any observer note that brings in external internet evidence.
        Direct PDF reads remain allowed
        after at least one retrieval attempt, because sometimes the agent truly
        needs a verbatim passage.
        """
        if not command:
            return False
        c = command.strip().lower()
        if "/repo/paper.pdf" not in c:
            return False
        # Common raw-read patterns seen in logs
        raw_read_markers = (
            "pdftotext",          # pdftotext -layout /repo/paper.pdf - | head -c ...
            "fitz.open(",         # python -c "import fitz; fitz.open('/repo/paper.pdf')..."
            "open('/repo/paper.pdf'",
            'open("/repo/paper.pdf"',
            "python -c",          # broader, but gated by /repo/paper.pdf above
            "python3 -c",
        )
        return any(m in c for m in raw_read_markers)

    @staticmethod
    def _extract_paper_marker(text: str):
        """Extract a REAL `echo PAPER_RESULT_*` marker line from `text`.

        Returns ``(verdict, line)`` where ``verdict`` is ``"reproduced"`` or
        ``"not_reproduced"`` and ``line`` is the stripped echo invocation.
        Returns ``None`` if no concrete marker is present.

        We deliberately ignore lines that are clearly system-prompt or plan
        templates (e.g. ``9. `echo PAPER_RESULT_REPRODUCED metric=<name>
        actual=<v> ...```), shell help text (``echo PAPER_RESULT_REPRODUCED
        metric=<name>``), or any line whose payload still contains
        unsubstituted placeholders. This prevents a Stage-2 verdict from
        firing on the LLM merely echoing the workflow template back to us.
        """
        if not text:
            return None
        # Match any `<...>` placeholder, including ones with spaces or
        # hyphens (e.g. `<one-line reason>`, `<name>`, `<v>`). We also
        # treat angle-bracketed `[...]` placeholders as templates.
        placeholder_re = re.compile(r"<[^<>\n]{1,40}>|\[[A-Z][A-Z0-9_ -]{1,30}\]")
        # We require an `echo` (or `printf`) at the start of a real shell
        # command line. Strip common shell punctuation / list-item bullets.
        bullet_re = re.compile(r"^\s*(?:[-*+>]|\d+[.)]|`{1,3})?\s*")
        # Only accept lines that START with `echo` (or printf) followed by
        # the marker, to filter out narrative / template references.
        echo_re = re.compile(
            r"""^\s*(?:echo|printf)\s+
                (?:["']?)
                (PAPER_RESULT_(?:REPRODUCED|NOT_REPRODUCED))
                \b""",
            re.IGNORECASE | re.VERBOSE,
        )
        for raw in text.splitlines():
            # Peel off list-item bullets like "9.", "- ", "* ", "> ", etc.
            stripped = bullet_re.sub("", raw, count=1)
            # Some agents wrap the echo in backticks: `echo PAPER_RESULT_...`
            stripped = stripped.lstrip("`").rstrip("`").strip()
            m = echo_re.match(stripped)
            if not m:
                continue
            payload = stripped[m.end():]
            # Reject template echoes that still carry placeholders like
            # `<name>`, `<v>`, `<x>`, `<reason>`.
            if placeholder_re.search(payload):
                continue
            verdict = (
                "reproduced"
                if m.group(1).upper() == "PAPER_RESULT_REPRODUCED"
                else "not_reproduced"
            )
            return verdict, stripped
        return None

    def _mode_rule_block(self) -> str:
        """Return a short, mode-specific rule block for the system prompt."""
        mode = getattr(self, "run_mode", "env")
        if mode == "env":
            return (
                "**RUN MODE: 1 — ROCm Env Only**\n"
                "Your sole goal is ROCM_ENV_VERIFIED. Once the repo runs on the AMD GPU,\n"
                "echo ROCM_ENV_VERIFIED and stop. Do NOT attempt paper experiments.\n"
                "You MAY scale down training params (fewer epochs/steps) for a quick smoke test."
            )
        elif mode == "reproduce":
            return (
                "**RUN MODE: 2 — Paper Reproduce**\n"
                "Your primary goal is PAPER_RESULT_REPRODUCED or PAPER_RESULT_NOT_REPRODUCED.\n"
                "Complete env setup as fast as possible, then focus on:\n"
                "  1. Downloading the required datasets and checkpoints (see EXTERNAL ASSETS section).\n"
                "  2. Running the paper experiment with the EXACT paper config — no scale-down.\n"
                "  3. Echoing the result marker with metric details.\n"
                "You do NOT need to echo ROCM_ENV_VERIFIED — GPU confirmation is enough before moving on.\n"
                "Do NOT synthesise fake data. If real data is unavailable, echo PAPER_RESULT_NOT_REPRODUCED."
            )
        else:  # "full"
            return (
                "**RUN MODE: 3 — Full (Env → Paper Reproduce)**\n"
                "Two explicit stages:\n"
                "  Stage 1: Configure env, verify GPU with a smoke test, then echo ROCM_ENV_VERIFIED.\n"
                "  Stage 2: Download required assets, run paper experiment (exact config, no scale-down),\n"
                "           invoke paper-reproducer sub-agent, echo PAPER_RESULT_REPRODUCED or\n"
                "           PAPER_RESULT_NOT_REPRODUCED.\n"
                "Do NOT skip Stage 1 — the ROCM_ENV_VERIFIED marker is required before Stage 2 begins.\n"
                "Do NOT synthesise fake data."
            )

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

        if self.kb_context:
            core_prompt += "\n" + self.kb_context + "\n"

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

WHEN TO USE RETRIEVAL TOOLS (in escalating order of cost; all cached):
1. **mem_recall** "<failing command + error class>" — this-run memory only; ~free. Always FIRST after a failure.
2. **graphify_query** "<symbol or behavior you need>" — local code graph; ~free. Use INSTEAD of `find -name` / `grep -r`. Returns ranked file:line locations.
3. **paper_recall** "<question>" — graphify paper index + run-state references (`paper_experiments`, `experiment_state`, `context_refs`, `plan`, `decisions`). Use this BEFORE reading `/repo/paper.pdf` directly.
4. **mem_recall "<topic>" --rooms plan,experiment_state,context_refs,decisions** — ground hyperparameters / env vars in the planner output and run-state references. ~free.
5. **pypi_versions** <pkg> — local network ~1s. BEFORE pinning a CUDA-only wheel (flash-attn, bitsandbytes, xformers, triton). Returns currently-installable versions + dates.
6. **dockerhub_tags** <image> — local network ~1s. BEFORE `change_base_image`. Returns the actually-published tags for `rocm/pytorch`, `rocm/vllm`, etc.
7. **Observer-side internet research** — external web search and page reads are
handled asynchronously by the observer sidecar. When the run is stuck on an
AMD/ROCm/runtime or paper-fidelity issue, the observer may inject a note with
internet-backed evidence and a suggested strategy. Treat that note as advisory
guidance before your next action.

**AMD/ROCm-specific rule:** if the issue mentions ROCm/HIP/gfx/miopen/rocBLAS/libamdhip64,
or a fast-moving package like flash-attn/xformers/bitsandbytes/triton/deepspeed,
do NOT trust static knowledge alone. Use deterministic lookups first
(`pypi_versions`, `dockerhub_tags`) and then incorporate observer-provided
internet research when it arrives.

**AMD ROCm Ecosystem Reference:** A comprehensive NVIDIA→AMD mapping covering every ROCm
library (rocBLAS, MIOpen, MIGraphX, amd_gsplat, flash-attention, bitsandbytes, xFormers,
TunableOp, HIPIFY, etc.) with exact install commands and caveats is available at:
  `/Repo2ROCm/build_agent/knowledge/amd_rocm_ecosystem.md`
The plan's "AMD ROCm Ecosystem" section already lists the libraries relevant to THIS repo.
Always consult it before manually searching for an AMD alternative to a CUDA package.

**EXTERNAL DATA & MODEL ASSETS — read this before any training/inference step:**
GitHub enforces a 25 MB per-file limit. Therefore:
  - Large datasets are NEVER inside /repo. They live on HuggingFace, Google Drive,
    Baidu Yun, or a direct download URL documented in the README.
  - Pretrained checkpoints (.pth, .pt, .ckpt, .bin, .safetensors) are NEVER inside
    /repo. They must be downloaded from HuggingFace Hub, Google Drive, or a URL.
  - Pseudo-masks, annotation archives, and COLMAP sparse reconstructions that the
    paper trained its model with are NEVER inside /repo.

Before running any script that loads a dataset or checkpoint:
  1. Check the plan's "EXTERNAL ASSETS REQUIRED" section for the exact HF id,
     URL, or download script.
  2. `ls <expected_path>` to see if it already exists on disk.
  3. If missing: download it using the command in the plan.
  4. For HuggingFace: `pip install -q huggingface_hub && huggingface-cli download <id> --repo-type dataset --local-dir /data/<name>`
  5. For Google Drive: `pip install -q gdown && gdown <url> -O /data/<name>`
  6. For download scripts in the repo: `bash /repo/<script>`
  7. If the source is unclear: run `web_search "<repo_name> <asset_name> download HuggingFace"`
     to find the canonical source BEFORE attempting to create synthetic or mock data.

**DO NOT synthesise fake datasets or mock scenes** as a substitute for missing
external data during paper-reproduction runs. A synthetic scene will never
reproduce the paper's mIoU / PSNR / accuracy numbers. If the real dataset is
unavailable, report `PAPER_RESULT_NOT_REPRODUCED missing_data: <reason>`.

**Repo-discovery rule:** use `graphify_query` before broad `find`/`grep -r`
across `/repo`. Shell-wide search is a fallback, not the first move.

**Always prefer retrieval over guessing.** Local retrieval (`mem_recall`,
`graphify_query`, `paper_recall`) comes first. External web evidence should come
through the observer so the execution loop stays focused on grounded actions.

{self._mode_rule_block()}
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

    @staticmethod
    def _build_agentic_system_prompt_static(
        image_name: str,
        rocm_mode: bool = False,
        no_scale_down: bool = False,
        plan: str = "",
        kb_context: str = "",
        reproduce_results: bool = False,
    ) -> str:
        """Build a system prompt for Claude Code's full agentic mode.

        Used when --claude-code-agentic is passed: Claude Code drives the
        entire configuration process with its own built-in tools.
        """
        stage_line = (
            "5. Signal STAGE 1 success with: echo ROCM_ENV_VERIFIED"
            if reproduce_results else
            "5. Signal success with: echo ROCM_ENV_VERIFIED"
        )
        prompt = f"""\
You are an expert environment configuration agent running inside a Docker container
based on {image_name}. Your goal is to configure the repository at /repo so it
runs correctly on AMD ROCm GPUs.

WORKFLOW:
1. Inspect the repository structure and configuration files
2. Install dependencies (respecting CUDA-to-ROCm package mappings)
3. Apply code patches for ROCm compatibility
4. Run the main script to verify the environment works
{stage_line}

RULES:
- Do NOT modify test files
- Use pip install -q for quiet installs
- Prefer minimal changes to original source files
- If installing packages, check pipdeptree for conflicts afterward
- For import errors, check if the module exists locally before pip installing
"""

        if rocm_mode:
            prompt += """
ROCm-SPECIFIC RULES:
- Replace nvidia-* packages with ROCm equivalents
- Use torch from ROCm wheel index
- Verify GPU with: python -c "import torch; print(torch.cuda.is_available())"
- Guard cudnn flags: if not getattr(torch.version, 'hip', None)
- Replace nvidia-smi with rocm-smi
- Set WANDB_MODE=offline if wandb is used
"""

        prompt += """
REAL EXECUTION MODE:
- Do NOT create mock models, mock data, or stub scripts.
- Download and use the ACTUAL model specified in the README.
- Run the EXACT commands from the README with REAL arguments.
- If a HuggingFace model is specified, download and load that EXACT model.
- If OOM occurs, reduce batch_size or gen_length but keep the REAL model.
"""
        if no_scale_down:
            prompt += """
NO-SCALE-DOWN MODE:
Do NOT reduce epochs, iterations, batch sizes, or any training parameters.
Run the exact README commands with original args.
"""

        if reproduce_results:
            prompt += """
STAGE 2 - PAPER RESULT REPRODUCTION (only after ROCM_ENV_VERIFIED):
- The strategic plan below contains a PAPER REPRODUCTION TARGET section
  that names the Chosen experiment, its paper-reported metric, the suggested
  command, a tolerance rule, and fallback experiments. Follow it verbatim.
- The paper PDF is available at /repo/paper.pdf (Claude's Read tool handles PDFs).
- Run the chosen experiment with the EXACT paper/README config (obey --no-scale-down
  if set; do NOT reduce steps, batch size, seq length, or any hyperparameter).
- Tee the experiment's stdout+stderr to /repo/paper_experiment.log, e.g.:
    bash -lc "<cmd> 2>&1 | tee /repo/paper_experiment.log"
- Delegate judgement to the `paper-reproducer` sub-agent. Pass it the exact
  Chosen-experiment block from the plan and the path /repo/paper_experiment.log.
  It will read /repo/paper.pdf, compute a numeric delta (falling back to
  LLM-judge when units/numbers are not directly comparable), and return a
  strict JSON verdict.
- Based on the sub-agent's JSON verdict, echo EXACTLY one of:
    echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>
    echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>
- Do NOT fabricate metrics. If the run/parse fails, re-run the experiment ONCE;
  if it still fails, echo PAPER_RESULT_NOT_REPRODUCED with a brief parsing note.
- Only after echoing one of those markers is STAGE 2 complete.
"""

        if kb_context:
            prompt += "\n" + kb_context + "\n"

        if plan:
            prompt += "\n" + "=" * 60 + "\n"
            prompt += "STRATEGIC PLAN\n"
            prompt += "=" * 60 + "\n"
            prompt += plan + "\n"
            prompt += "=" * 60 + "\n"
            prompt += (
                "Execute this plan. The repo has already been analyzed. "
                "Start from the first actionable step.\n"
            )

        return prompt

    def _build_stage2_prompt_block(self) -> str:
        """Return the STAGE 2 instruction block injected when ROCM_ENV_VERIFIED fires
        and --reproduce-results is on. Self-contained so the LLM has everything it
        needs (chosen experiment, expected metric + portability class, tolerance,
        marker format, and sandbox hygiene hints)."""
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("STAGE 2 — PAPER RESULT REPRODUCTION")
        lines.append("=" * 70)
        if self.paper_title:
            lines.append(f"Paper: {self.paper_title}")
        if self.paper_pdf_path:
            lines.append(f"Paper PDF: {self.paper_pdf_path} (also at /repo/paper.pdf inside the container)")
        lines.append("")
        lines.append("ROCm Stage 1 is complete. Do NOT echo ROCM_ENV_VERIFIED again.")
        lines.append("IMPORTANT TOOL DISCIPLINE FOR STAGE 2:")
        lines.append("  Do NOT read /repo/paper.pdf directly as your first move.")
        lines.append("  Static paper content is indexed in graphify; dynamic experiment choices /")
        lines.append("  references live in memory. `paper_recall` merges both.")
        lines.append("  FIRST use retrieval tools in this order:")
        lines.append("    1. paper_recall \"what metric / experiment / hyperparameters do I need?\"")
        lines.append("    2. graphify_query \"where in the code/config is this experiment wired?\"")
        lines.append("    3. wait for observer-provided internet evidence when local retrieval is insufficient")
        lines.append("  Only after at least one retrieval attempt may you read /repo/paper.pdf directly.")
        lines.append("")
        lines.append("You MUST now run ONE paper experiment and verify its results match the")
        lines.append("paper's reported value. Then echo exactly ONE of:")
        lines.append("  echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>")
        lines.append("  echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>")
        lines.append("")

        # ── Cross-hardware guidance (generic, not paper-specific) ────────────
        lines.append("CROSS-HARDWARE NOTE:")
        lines.append("  We are running on AMD ROCm GPUs; the paper most likely used NVIDIA")
        lines.append("  hardware. ABSOLUTE throughput (tokens/s, samples/s, QPS) and latency")
        lines.append("  (ms) are NOT directly comparable across GPUs. For reproduction, prefer")
        lines.append("  hardware-portable metrics in this order:")
        lines.append("    1. ratios / speedups / percentages (e.g. '2.5x speedup' vs a baseline)")
        lines.append("    2. accuracy / F1 / EM / Top-k / pass@k / win-rate (portable)")
        lines.append("    3. perplexity / loss / NLL / reward (mostly portable)")
        lines.append("    4. absolute throughput / latency (GPU-dependent — last resort)")
        lines.append("  If the paper's headline claim is a SPEEDUP ratio (e.g. method X is 2.5x")
        lines.append("  faster than baseline Y), reproduce BOTH runs (baseline AND method) and")
        lines.append("  compute the RATIO locally; that ratio is what should match the paper,")
        lines.append("  not the absolute tokens/s of either run.")
        lines.append("")

        if not self.paper_experiments:
            lines.append("No experiments were shortlisted from the paper automatically.")
            lines.append("You have an unmodified paper.pdf at /repo/paper.pdf. Use `pdftotext -layout")
            lines.append("/repo/paper.pdf - | head -c 60000` to read it, pick the shortest experiment")
            lines.append("that exercises the paper's PROPOSED METHOD (not a baseline) and whose")
            lines.append("metric is hardware-portable (prefer speedups/percentages/accuracy over")
            lines.append("absolute tokens/s), run it with the EXACT paper/README config, parse the")
            lines.append("numeric metric(s) from its stdout, compare to the paper's reported value")
            lines.append("(tolerance: 15% relative for ratios, 3 absolute points for accuracy, 5%")
            lines.append("relative for PPL/loss, 25% relative for absolute throughput), and echo one")
            lines.append("of the two markers above.")
            self._append_stage2_sandbox_hygiene(lines)
            return "\n".join(lines)

        chosen = self.paper_experiments[0]
        def _g(d, k, default=""):
            return d.get(k, default) if isinstance(d, dict) else getattr(d, k, default)

        lines.append(f"Chosen experiment: {_g(chosen, 'name')}")
        if _g(chosen, "section"):
            lines.append(f"  Source: {_g(chosen, 'section')}")
        runtime = _g(chosen, "est_runtime_minutes", 0) or 0
        try:
            runtime_f = float(runtime)
        except (TypeError, ValueError):
            runtime_f = 0.0
        if runtime_f > 0:
            lines.append(f"  Estimated runtime: ~{runtime_f:.0f} min.")
        reason_bits = []
        if _g(chosen, "code_available"):
            reason_bits.append("code AVAILABLE in the repo")
        else:
            reason_bits.append("code NOT directly matched in the repo")
        if _g(chosen, "is_baseline"):
            reason_bits.append("(this is a BASELINE row — only picked because nothing better was found)")
        else:
            reason_bits.append("exercises the paper's proposed METHOD (not a baseline)")
        metric_class = _g(chosen, "metric_class")
        if metric_class:
            portability_label = {
                "ratio_speedup": "hardware-portable (ratio/speedup)",
                "accuracy": "hardware-portable (accuracy-style)",
                "quality": "mostly hardware-portable (quality metric)",
                "absolute_perf": "HARDWARE-DEPENDENT (absolute throughput/latency)",
                "other": "portability unknown",
            }.get(metric_class, "portability unknown")
            reason_bits.append(portability_label)
        lines.append("  Reason: " + "; ".join(reason_bits) + ".")
        metric_name = _g(chosen, "expected_metric_name")
        metric_value = _g(chosen, "expected_metric_value")
        metric_units = _g(chosen, "expected_metric_units")
        if metric_name:
            lines.append(
                f"  Paper-reported metric: {metric_name} = {metric_value} {metric_units}".rstrip()
            )
        if _g(chosen, "hardware"):
            lines.append(f"  Paper hardware: {_g(chosen, 'hardware')}")
        if _g(chosen, "suggested_command"):
            lines.append(f"  Suggested command (EXACT — includes all non-default flags):")
            lines.append(f"    {_g(chosen, 'suggested_command')}")
        # Paper config: render compactly as key=value pairs so the agent can
        # see every hyperparameter it must set.
        paper_config = _g(chosen, "paper_config") or {}
        if paper_config:
            lines.append(f"  Paper-exact hyperparameters (use ALL of these; do NOT rely on script defaults):")
            for k, v in list(paper_config.items())[:30]:
                if v not in (None, ""):
                    lines.append(f"    - {k} = {v}")
        if _g(chosen, "config_source"):
            lines.append(f"  Config source (paper + codebase): {_g(chosen, 'config_source')}")
        codebase_cfgs = _g(chosen, "codebase_config_files") or []
        if codebase_cfgs:
            lines.append("  Codebase config files governing this experiment (READ + OVERRIDE these,")
            lines.append("  do NOT guess values when the paper is ambiguous):")
            for cf in codebase_cfgs[:10]:
                lines.append(f"    - /repo/{cf}")
        missing_flags = _g(chosen, "missing_flags") or []
        if missing_flags:
            lines.append(f"  !!! Paper flags NOT exposed by the entry script (you must work around):")
            for mf in missing_flags[:10]:
                lines.append(f"        - {mf}  (patch the script or skip, then note in the verdict)")
        matched_files = _g(chosen, "matched_files") or []
        if matched_files:
            lines.append(f"  Matched files in repo: {', '.join(matched_files[:5])}")
        tolerance = _g(chosen, "tolerance_rule") or (
            "<=15% for ratios/speedups, <=3 abs pts for accuracy, <=5% for PPL/loss, <=25% for absolute throughput"
        )
        lines.append(f"  Tolerance: {tolerance}")
        caveats = _g(chosen, "caveats") or []
        if caveats:
            lines.append(f"  Caveats (disclaimers from the paper/README — READ CAREFULLY before running):")
            for cv in caveats[:6]:
                lines.append(f"    * {cv}")
            lines.append(
                "  If a caveat says the claimed metric requires a config we CANNOT use here"
            )
            lines.append(
                "  (e.g. a dataset not shipped in the repo, a batch size the script can't accept),"
            )
            lines.append(
                "  you SHOULD note that in the marker and echo PAPER_RESULT_NOT_REPRODUCED with"
            )
            lines.append(
                "  a reason that references the specific caveat, instead of guessing a number."
            )
        if _g(chosen, "notes"):
            lines.append(f"  Notes: {_g(chosen, 'notes')}")

        # If the chosen metric is absolute_perf, remind the agent to compute a
        # ratio locally by also running a baseline from the fallback list.
        if metric_class == "absolute_perf":
            lines.append("")
            lines.append("  !!! The chosen metric is an ABSOLUTE throughput/latency value. Because we")
            lines.append("      are on AMD hardware and the paper used a different GPU, the absolute")
            lines.append("      number will differ substantially. You SHOULD additionally run a")
            lines.append("      paired baseline from the 'Fallback experiments' list below and")
            lines.append("      compute the RATIO method/baseline locally; the ratio is what should")
            lines.append("      match the paper's speedup claim (if any). Report that ratio as the")
            lines.append("      reproduced metric (metric=speedup_ratio) instead of absolute tok/s.")

        if len(self.paper_experiments) > 1:
            lines.append("")
            lines.append("Fallback experiments (use only if the chosen one fails to run, or as")
            lines.append("baselines to compute a ratio when the chosen metric is absolute):")
            for fb in self.paper_experiments[1:4]:
                name = _g(fb, "name")
                cmd = _g(fb, "suggested_command")
                rt = _g(fb, "est_runtime_minutes", 0) or 0
                try:
                    rt_f = float(rt)
                except (TypeError, ValueError):
                    rt_f = 0.0
                rt_str = f"~{rt_f:.0f} min" if rt_f > 0 else "unknown"
                flags = []
                if _g(fb, "code_available"):
                    flags.append("code available")
                else:
                    flags.append("no direct code match")
                if _g(fb, "is_baseline"):
                    flags.append("BASELINE")
                fb_metric_class = _g(fb, "metric_class")
                if fb_metric_class:
                    flags.append(fb_metric_class)
                lines.append(f"  - {name} ({rt_str}, {', '.join(flags)})")
                if cmd:
                    lines.append(f"      cmd: {cmd}")
                fb_caveats = _g(fb, "caveats") or []
                for cv in fb_caveats[:2]:
                    lines.append(f"      caveat: {cv}")

        lines.append("")
        lines.append("Protocol (follow in order):")
        lines.append("  1. Use the EXACT paper/README config shown above (all hyperparameters). If")
        lines.append("     the entry script's CLI doesn't expose a flag the paper used, patch the")
        lines.append("     script in-place (via python -c / sed / the code_edit.py tool) to hardcode")
        lines.append("     that value. Do NOT silently rely on the script's default — that would")
        lines.append("     change the experiment and invalidate the comparison.")
        lines.append("  2. Run the chosen experiment (and, if computing a ratio, the matching")
        lines.append("     baseline) with that exact config. Tee stdout+stderr to a log:")
        lines.append("       ```bash")
        lines.append("       cd /repo && <suggested command> 2>&1 | tee /repo/paper_experiment.log")
        lines.append("       ```")
        lines.append("  3. Parse the relevant metric from the log (grep/awk/python). If the paper")
        lines.append("     reports a SPEEDUP and the script only prints raw tokens/s, compute the")
        lines.append("     ratio locally: method_tok/s / baseline_tok/s.")
        lines.append("  4. Compare to the paper-reported value using the tolerance above:")
        lines.append("       numeric first; if units differ or the paper has only a qualitative")
        lines.append("       claim, render an LLM-judge verdict based on the observed behavior.")
        lines.append("  5. In your FINAL bash block, echo exactly ONE of:")
        lines.append("       echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>")
        lines.append("       echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>  # cite the caveat if any")
        lines.append("  6. Do NOT fabricate numbers. If parsing fails, rerun the experiment ONCE;")
        lines.append("     if it still fails, echo PAPER_RESULT_NOT_REPRODUCED with a parsing note.")
        lines.append("  7. Do NOT echo ROCM_ENV_VERIFIED again — Stage 1 is done.")
        self._append_stage2_sandbox_hygiene(lines)
        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def _append_stage2_sandbox_hygiene(lines):
        """Append small hygiene rules to avoid known sandbox failure modes."""
        lines.append("")
        lines.append("Sandbox hygiene (to avoid wasted turns):")
        lines.append("  - AVOID multi-line heredoc file writes like `cat > file.py << 'EOF' ... EOF`;")
        lines.append("    the bash sandbox used here is interactive and often mangles heredocs.")
        lines.append("    To write a file, prefer ONE of:")
        lines.append("      (a) `python - <<'PY' ... PY`  (heredoc into python, single-arg)")
        lines.append("      (b) `python -c \"open('/repo/x.py','w').write('...content...')\"`")
        lines.append("      (c) echo '...single-line content...' > /repo/x.py")
        lines.append("      (d) sed -i / the project's code_edit.py tool for in-place edits")
        lines.append("  - Single-line bash blocks joined with `&&` work reliably.")
        lines.append("  - Prefer modifying an existing entry script over writing a new wrapper,")
        lines.append("    unless the entry script lacks timing/metric instrumentation.")

    def _handle_paper_marker(self, text: str, waiting_list, conflict_list) -> None:
        """Record the STAGE 2 verdict, persist artefacts, and log it.

        `text` is the source string containing one of the PAPER_RESULT_* tokens
        (either the LLM-produced command or the sandbox output).

        Trust hierarchy (strict):
          1. `verifier` (deterministic `verify_paper_result`) — always wins.
          2. LLM-issued marker — accepted as a DECLARATION, never as evidence.
             - LLM `not_reproduced` without a verifier ⇒ accepted (admitting
               failure does not need verification).
             - LLM `reproduced` without a verifier ⇒ REFUSED. Downgraded to
               `not_reproduced` with reason `verifier_never_called`. This is
               the guard that prevents a Stage-2 success from being claimed
               by simply echoing the workflow template.
        """
        marker = self._extract_paper_marker(text)
        if marker is not None:
            llm_verdict, marker_line = marker
        else:
            llm_verdict = "unknown"
            marker_line = ""

        reproduced_line = marker_line if llm_verdict == "reproduced" else ""
        not_reproduced_line = marker_line if llm_verdict == "not_reproduced" else ""

        # Pick the most recent verifier record (we don't try to be clever
        # about matching log paths because there is normally only one).
        verifier_record = None
        verifier_log_path = ""
        if self._verifier_records:
            verifier_log_path = list(self._verifier_records.keys())[-1]
            verifier_record = self._verifier_records[verifier_log_path]

        verifier_verdict = (verifier_record or {}).get("verdict", "")

        # Reconcile LLM marker with the verifier under the strict trust
        # hierarchy described in the docstring.
        verdict_source = "llm_marker"
        downgrade_reason = ""
        if verifier_verdict in ("reproduced", "not_reproduced"):
            if llm_verdict in ("reproduced", "not_reproduced") and llm_verdict != verifier_verdict:
                # Marker disagrees with deterministic verifier. Trust the
                # verifier — this is the EARTH-style "RMSE better but PCC much
                # worse" failure mode the verifier exists to catch.
                final_verdict = verifier_verdict
                verdict_source = "verifier_overrode_llm"
            else:
                final_verdict = verifier_verdict
                verdict_source = "verifier"
        elif llm_verdict == "reproduced":
            # SUCCESS without verifier evidence is not allowed. Downgrade.
            final_verdict = "not_reproduced"
            verdict_source = "downgraded_no_verifier"
            downgrade_reason = (
                "LLM emitted PAPER_RESULT_REPRODUCED but never called "
                "verify_paper_result; success requires deterministic "
                "verification of metric values against the paper."
            )
            # Surface the downgrade in the persisted not_reproduced_line so
            # downstream success-report explanations are explicit.
            not_reproduced_line = (
                f"DOWNGRADED from REPRODUCED (no verifier): {marker_line}"
                if marker_line else
                "DOWNGRADED from REPRODUCED (no verifier and no concrete marker)"
            )
            reproduced_line = ""
        elif llm_verdict == "not_reproduced":
            # Admitting failure without a verifier is fine.
            final_verdict = "not_reproduced"
            verdict_source = "llm_marker_unverified"
        else:
            final_verdict = "unknown"
            verdict_source = "no_marker_no_verifier"

        self._paper_verdict = final_verdict
        self._paper_result_line = marker_line

        out_dir = f'{self.root_dir}/output/{self.full_name}'
        os.makedirs(out_dir, exist_ok=True)

        test_lines = ["ROCM_ENV_VERIFIED"]
        if final_verdict == "reproduced":
            test_lines.append("PAPER_RESULT_REPRODUCED")
        elif final_verdict == "not_reproduced":
            test_lines.append("PAPER_RESULT_NOT_REPRODUCED")
        try:
            with open(f'{out_dir}/test.txt', 'w') as w:
                w.write('\n'.join(test_lines) + '\n')
        except Exception:
            pass

        chosen = self.paper_experiments[0] if self.paper_experiments else {}
        if not isinstance(chosen, dict):
            try:
                chosen = chosen.to_dict()
            except Exception:
                chosen = {}

        record = {
            "verdict": final_verdict,
            "verdict_source": verdict_source,
            "downgrade_reason": downgrade_reason,
            "llm_marker_verdict": llm_verdict,
            "verifier_verdict": verifier_verdict,
            "paper_title": self.paper_title,
            "paper_pdf_path": self.paper_pdf_path,
            "chosen_experiment": chosen,
            "fallback_experiments": [
                (c if isinstance(c, dict) else c.to_dict())
                for c in self.paper_experiments[1:4]
            ] if self.paper_experiments else [],
            "reproduced_line": reproduced_line,
            "not_reproduced_line": not_reproduced_line,
            "verifier": verifier_record or {},
            "verifier_log_path": verifier_log_path,
            "source": "configuration_agent_stage2",
        }

        # Build & embed the SuccessReport so downstream tooling has a single
        # numeric handle for every run.
        try:
            from storage.success_report import build_success_report
            sr = build_success_report(
                final_verdict=final_verdict,
                verifier_record=verifier_record,
                chosen_experiment=chosen,
                gpu_check_seen=self._gpu_check_seen,
                stage1_marker_emitted=True,  # we only get here AFTER stage 1
                turns_used=getattr(self, "_turns_used_so_far", 0),
                tool_calls={
                    "mem_recall": self._mem_recall_calls,
                    "paper_recall": self._paper_recall_calls,
                    "graphify_query": self._graphify_query_calls,
                    "pypi_versions": self._pypi_versions_calls,
                    "dockerhub_tags": self._dockerhub_tags_calls,
                    "web_search": self._web_search_calls,
                    "visit_url": self._visit_url_calls,
                    "deep_research": self._deep_research_calls,
                    "verify_paper_result": self._verify_paper_calls,
                },
                outer_commands=self.outer_commands,
            )
            record["success_report"] = sr
        except Exception as _sr_e:
            record["success_report_error"] = str(_sr_e)

        try:
            with open(f'{out_dir}/paper_reproduction.json', 'w') as w:
                w.write(json.dumps(record, indent=2, default=str))
        except Exception:
            pass

        try:
            self._save_final_artifacts(waiting_list, conflict_list)
        except Exception:
            pass

        if final_verdict == "reproduced":
            log_success(f"PAPER_RESULT_REPRODUCED ({verdict_source}): {marker_line}")
        elif final_verdict == "not_reproduced":
            display_line = not_reproduced_line or marker_line
            if verdict_source == "downgraded_no_verifier":
                log_error(
                    f"PAPER_RESULT_NOT_REPRODUCED ({verdict_source}): "
                    f"{downgrade_reason} | original_marker={marker_line!r}"
                )
            else:
                log_error(
                    f"PAPER_RESULT_NOT_REPRODUCED ({verdict_source}): {display_line}"
                )
        else:
            log_info(f"Paper reproduction verdict: unknown ({verdict_source})")

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
        chosen_experiment = ""
        try:
            if self.paper_experiments:
                first = self.paper_experiments[0]
                first_dict = first if isinstance(first, dict) else getattr(first, "to_dict", lambda: {})()
                chosen_experiment = str(first_dict.get("name") or first_dict.get("section") or "")[:240]
        except Exception:
            chosen_experiment = ""
        self._emit_observer_event("run_started", {
            "repo": self.full_name,
            "model": self.model,
            "rocm_mode": self.rocm_mode,
            "reproduce_results": self.reproduce_results,
            "paper_title": self.paper_title,
            "paper_pdf_path": self.paper_pdf_path,
            "chosen_experiment": chosen_experiment,
            "plan_excerpt": (self.plan or "")[:6000],
            "image_name": self.image_name,
            "kb_context_excerpt": (self.kb_context or "")[:2000],
        })

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
            self._turns_used_so_far = turn
            turn_started_at = time.time()
            turn_outer_start = len(self.outer_commands)
            tool_counts_before = self._tool_usage_snapshot()
            self._consume_observer_advice(turn)

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
                            "3. Run the EXACT commands from the README with REAL models and REAL data.\n"
                            "   Do NOT create mock/dummy models or data. Download and use the actual model.\n"
                            "   e.g., `cd /repo && python generate.py` (as specified in README)\n"
                            "   Note: `--help` alone is NOT sufficient. You must run with real data.\n"
                            "Once the script produces actual output, declare success by outputting:\n"
                            "```bash\necho ROCM_ENV_VERIFIED\n```\n"
                        )
                        end_time = time.time()
                        self.outer_commands[-1]["time"] = end_time - start_time
                        self.outer_commands[-1]["returncode"] = 0
                        continue

                    # ── STAGE 2 markers: check BEFORE Stage 1 marker so a combined ──
                    # bash block ending with PAPER_RESULT_* finishes cleanly.
                    # IMPORTANT: only fire on a REAL `echo PAPER_RESULT_*` line
                    # with concrete values (not the workflow template that
                    # appears verbatim inside the system prompt / plan / KB).
                    if self._stage2_active and self._extract_paper_marker(commands[i]):
                        self._handle_paper_marker(commands[i], waiting_list, conflict_list)
                        self.outer_commands[-1]["returncode"] = 0
                        self.outer_commands[-1]["time"] = time.time() - start_time
                        system_res += (
                            "\nPaper reproduction verdict recorded. Agent finished.\n"
                        )
                        finish = True
                        break

                    # ── ROCm mode: detect success signal ──
                    # Flexible match: accept any command that contains ROCM_ENV_VERIFIED
                    # e.g. echo ROCM_ENV_VERIFIED, echo "ROCM_ENV_VERIFIED",
                    #      ls -la && echo ROCM_ENV_VERIFIED, etc.
                    if self.rocm_mode and 'ROCM_ENV_VERIFIED' in commands[i] and not self._stage2_active:
                        log_rocm_success()
                        sandbox_res = "ROCM_ENV_VERIFIED\nCongratulations, you have successfully configured the environment!"
                        system_res += sandbox_res
                        self.outer_commands[-1]["returncode"] = 0
                        self.outer_commands[-1]["time"] = time.time() - start_time

                        if self.reproduce_results:
                            # ── Transition to STAGE 2 instead of finishing ──
                            self._stage2_active = True
                            with open(f'{self.root_dir}/output/{self.full_name}/test.txt', 'w') as w3:
                                w3.write('ROCM_ENV_VERIFIED\n')
                            log_success("ROCm environment verified. Transitioning to STAGE 2: paper reproduction.")
                            if not self._stage2_announced:
                                system_res += self._build_stage2_prompt_block()
                                self._stage2_announced = True
                            continue

                        self._save_final_artifacts(waiting_list, conflict_list)
                        log_success("ROCm environment verified. Agent finished.")
                        with open(f'{self.root_dir}/output/{self.full_name}/test.txt', 'w') as w3:
                            w3.write('ROCM_ENV_VERIFIED\n')
                        finish = True
                        break

                    # ── Rule matching before execution ──
                    matched_rule_ids = []
                    if self.rule_engine:
                        rule_context = {
                            "rocm_mode": self.rocm_mode,
                            "package_needed": commands[i].strip(),
                            "package_matches": commands[i].strip(),
                        }
                        rule_result = self.rule_engine.match(rule_context)
                        matched_rule_ids = rule_result.rule_ids
                        if rule_result.has_deterministic:
                            det_cmds = rule_result.all_deterministic_commands
                            if det_cmds:
                                kb_in_guidance_pre = self.rule_engine.format_for_prompt(rule_result)
                                system_res += f"\n{kb_in_guidance_pre}\n"

                    # ── Stage 5b: intercept retrieval tools BEFORE sandbox dispatch ──
                    intercepted = self._maybe_run_retrieval_tool(commands[i])
                    if intercepted is not None:
                        sandbox_res, return_code = intercepted
                        # External-lookup successes relax the hard guards.
                        self._maybe_record_external_lookup(commands[i], return_code)
                    else:
                        # Hard-guard layer: refuse risky actions that have not
                        # been informed by the cheap, cached tools first.
                        guard = self._maybe_run_hard_guard(commands[i])
                        if guard is not None:
                            sandbox_res, return_code = guard
                        elif (
                            self._stage2_active
                            and not self._paper_retrieval_used
                            and self._is_raw_paper_read_command(commands[i])
                        ):
                            # Stage 2 paper-discipline guard: do not raw-read
                            # the paper until at least one retrieval tool has
                            # been attempted.
                            sandbox_res = (
                                "Stage 2 paper-discipline guard: do NOT read "
                                "`/repo/paper.pdf` directly yet. First use one of:\n"
                                "  graphify_query \"...\" --scope paper\n"
                                "  paper_recall \"what metric / experiment / hyperparameter do I need?\"\n"
                                "  wait for the observer note if external evidence is needed\n"
                                "After at least one retrieval attempt, raw PDF reads are allowed.\n"
                            )
                            return_code = 1
                        else:
                            # ── Execute normal command ──
                            sandbox_res, return_code = self.sandbox_session.execute(commands[i], waiting_list, conflict_list)
                            sandbox_res = res_truncate(sandbox_res)
                            # Opportunistic GPU-check observation.
                            self._maybe_record_gpu_check(commands[i], sandbox_res, return_code)

                    # ── Error Classification (IN phase) ──
                    classified_error = None
                    kb_in_guidance = ""
                    rc_int = return_code if isinstance(return_code, int) else -1
                    if rc_int != 0 and self.rocm_mode and self._looks_like_amd_specific_issue(commands[i], sandbox_res):
                        self._needs_live_amd_research = True
                        hint_parts = []
                        if commands[i]:
                            hint_parts.append(commands[i][:120])
                        if sandbox_res:
                            hint_parts.append((sandbox_res or "").strip().splitlines()[-1][:160])
                        self._amd_live_research_hint = " | ".join(p for p in hint_parts if p)
                    if self.error_classifier and rc_int != 0 and sandbox_res:
                        classified_error, deterministic_cmds = self.error_classifier.classify_and_fix(
                            sandbox_res, rc_int
                        )
                        if deterministic_cmds and classified_error and not classified_error.is_novel:
                            kb_in_guidance += (
                                f"\n** KB MATCH: {classified_error.error_class} "
                                f"(confidence: {classified_error.confidence:.0%}) **\n"
                                f"Suggested fix: {' && '.join(deterministic_cmds)}\n"
                            )
                        if self.rule_engine and classified_error and classified_error.error_class:
                            err_rule_ctx = {
                                "rocm_mode": self.rocm_mode,
                                "error_class": classified_error.error_class,
                                "error_output": sandbox_res[:2000],
                                "error_pattern": sandbox_res[:2000],
                            }
                            err_rule_result = self.rule_engine.match(err_rule_ctx)
                            matched_rule_ids.extend(err_rule_result.rule_ids)
                            if err_rule_result.has_deterministic or err_rule_result.has_advisory:
                                kb_in_guidance += "\n" + self.rule_engine.format_for_prompt(err_rule_result)
                    elif self.memory_provider and rc_int != 0 and sandbox_res:
                        from storage.models import MemoryRequest, MemoryPhase
                        in_request = MemoryRequest(
                            query=commands[i],
                            context={"error_output": sandbox_res[:1000], "rocm_mode": self.rocm_mode},
                            phase=MemoryPhase.IN.value,
                            current_error=sandbox_res[:2000],
                            turn_number=turn,
                        )
                        in_memory = self.memory_provider.provide_memory(in_request)
                        kb_in_guidance = self.memory_provider.format_in_for_observation(in_memory)

                    # ── Causal migration memory (always-on per-turn) ───────
                    # Records richer per-turn state (image, gpu_arch, current
                    # error_class, return_code) and surfaces matching
                    # `state→action→outcome` priors as `[CAUSAL]` lines.
                    # Runs in every mode, including `--mode env`, and is a
                    # no-op when the KB has no transitions yet.
                    if self.memory_provider is not None:
                        try:
                            kb_in_guidance += self._provide_causal_memory_per_turn(
                                command_text=commands[i],
                                sandbox_res=sandbox_res or "",
                                return_code=rc_int,
                                classified_error=classified_error,
                                turn=turn,
                            )
                        except Exception:
                            pass

                    # ── Record rule outcomes ──
                    if self.rule_engine and matched_rule_ids:
                        for rid in set(matched_rule_ids):
                            self.rule_engine.record_outcome(rid, rc_int == 0)
                        if self.build_attempt:
                            for rid in matched_rule_ids:
                                if rid not in self.build_attempt.rules_applied:
                                    self.build_attempt.rules_applied.append(rid)

                    # ── Stage 1: compact the raw observation BEFORE injecting ──
                    compacted = _compact_obs(sandbox_res or "", action_content=commands[i])
                    self._compaction_orig_chars += compacted.orig_chars
                    self._compaction_short_chars += compacted.compact_chars

                    # ── Record trajectory ──
                    record = None
                    if self.trajectory_store and self.build_attempt:
                        from storage.models import TrajectoryRecord
                        record = TrajectoryRecord(
                            repo_id=self.full_name,
                            attempt_id=self.build_attempt.id,
                            agent="configuration",
                            action_type="bash",
                            action_content=commands[i],
                            observation_raw=sandbox_res[:5000] if sandbox_res else "",
                            outcome="success" if rc_int == 0 else "failure",
                            return_code=rc_int,
                            duration_seconds=time.time() - start_time,
                            turn_number=turn,
                            error_class=classified_error.error_class if classified_error else None,
                            novel_situation=classified_error.is_novel if classified_error else False,
                            kb_rules_applied=list(set(matched_rule_ids)),
                        )
                        self.trajectory_store.record_action(
                            record, self.build_attempt.trajectory_file
                        )

                    # ── Stage 2: write the turn to mempalace (write-only) ──
                    if self.run_memory is not None and getattr(self.run_memory, "enabled", False):
                        try:
                            self.run_memory.write_turn(
                                record if record is not None else {
                                    "turn_number": turn, "action_type": "bash",
                                    "action_content": commands[i],
                                    "outcome": "success" if rc_int == 0 else "failure",
                                    "return_code": rc_int,
                                    "duration_seconds": time.time() - start_time,
                                    "error_class": classified_error.error_class if classified_error else None,
                                },
                                full_observation=sandbox_res or "",
                                compact_obj=compacted,
                            )
                        except Exception as _mp_e:
                            log_info(f"[mempalace] write_turn failed: {_mp_e}")

                    # Inject the COMPACTED observation, not the raw output.
                    system_res += compacted.short
                    if kb_in_guidance:
                        system_res += kb_in_guidance
                    # Stage 3: stash for the per-turn recall query
                    self._last_action_text = commands[i] or ""
                    self._last_obs_short = compacted.short or ""
                    self._last_error_class = compacted.error_class or ""
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

                    # ── STAGE 2 markers in command output ──
                    # Only trigger on a REAL `echo PAPER_RESULT_*` invocation
                    # with concrete values. The workflow template appears
                    # verbatim inside `paper_recall` / `mem_recall` outputs
                    # because the plan room contains it; substring matching
                    # would otherwise treat that template as a verdict.
                    if (
                        self._stage2_active
                        and return_code == 0
                        and self._extract_paper_marker(sandbox_res)
                    ):
                        self._handle_paper_marker(sandbox_res, waiting_list, conflict_list)
                        finish = True
                        break

                    # ── ROCm mode: detect success signal in command output ──
                    # Catches cases where the LLM produced ROCM_ENV_VERIFIED via
                    # any means (echo with quotes, printf, a script that prints it, etc.)
                    if (
                        self.rocm_mode
                        and 'ROCM_ENV_VERIFIED' in sandbox_res
                        and return_code == 0
                        and not self._stage2_active
                    ):
                        log_rocm_success()
                        if self.reproduce_results:
                            # ── Transition to STAGE 2 instead of finishing ──
                            self._stage2_active = True
                            with open(f'{self.root_dir}/output/{self.full_name}/test.txt', 'w') as w3:
                                w3.write('ROCM_ENV_VERIFIED\n')
                            log_success("ROCm environment verified (detected in output). Transitioning to STAGE 2: paper reproduction.")
                            if not self._stage2_announced:
                                system_res += self._build_stage2_prompt_block()
                                self._stage2_announced = True
                            continue

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
                                "You MUST run the project's main script with REAL models and REAL data as described in the README.\n"
                                "Do NOT create mock/dummy models or data. Download and use the actual model.\n"
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
                        # Stage 2: persist the applied patch to mempalace
                        if self.run_memory is not None and getattr(self.run_memory, "enabled", False):
                            try:
                                self.run_memory.write_turn(
                                    {"turn_number": turn, "action_type": "diff",
                                     "action_content": diffs[:4000],
                                     "outcome": "success", "return_code": 0,
                                     "duration_seconds": elasped_time,
                                     "error_class": None},
                                    full_observation=sandbox_res or "",
                                    compact_obj=_compact_obs(sandbox_res or ""),
                                )
                            except Exception as _mp_e:
                                log_info(f"[mempalace] write_turn(diff) failed: {_mp_e}")
                    # Stage 1: compact diff observation before injection
                    _diff_compacted = _compact_obs(sandbox_res or "")
                    self._compaction_orig_chars += _diff_compacted.orig_chars
                    self._compaction_short_chars += _diff_compacted.compact_chars
                    system_res += _diff_compacted.short
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

            # Stage 3: build the appendix as
            #   (a) instruction prefix (unchanged, drives the agent's behavior)
            #   (b) tiny recency tail = last 5 success commands (orientation only)
            #   (c) per-run recall pack from mempalace (relevant prior context)
            #   (d) no second long-term lesson layer here; durable learning
            #       lives in the structured KB instead
            # Old behavior dumped ALL success_cmds verbatim → grew linearly each turn.

            if self._stage2_active:
                instr = (
                    "\nThe container has executed prior commands successfully. "
                    "STAGE 1 (ROCM_ENV_VERIFIED) is complete — do NOT echo ROCM_ENV_VERIFIED again. "
                    "Your ONLY remaining goal is to reproduce the CHOSEN paper experiment described "
                    "in the STAGE 2 block above: run it with the EXACT paper/README config, tee the "
                    "output to /repo/paper_experiment.log, compare the metric against the paper's "
                    "reported value, and echo EXACTLY ONE of:\n"
                    "  echo PAPER_RESULT_REPRODUCED metric=<name> actual=<v> expected=<v> delta_pct=<x>\n"
                    "  echo PAPER_RESULT_NOT_REPRODUCED <one-line reason>\n"
                    "Never fabricate numbers; if parsing fails, rerun once, then echo NOT_REPRODUCED.\n"
                )
            elif self.rocm_mode:
                instr = (
                    "\nThe container has executed prior commands successfully. "
                    "Reflect on the execution history below and decide the next action. "
                    "You MUST actually run the project's main script with REAL models and REAL data "
                    "as specified in the README (not just --help, not mock data). Only after the "
                    "script produces actual output, echo ROCM_ENV_VERIFIED to finish.\n"
                )
            else:
                instr = (
                    "\nThe container has executed prior commands successfully. "
                    "Reflect on the execution history below and decide the next action. "
                    "Remember, your ultimate goal is to pass the tests by executing "
                    "`runtest` or `poetryruntest`.\n"
                )

            if len(success_cmds) > 0:
                tail = success_cmds[-5:]
                recency = (
                    f"Total successful commands so far: {len(success_cmds)}. "
                    f"Most recent (last {len(tail)}):\n" + "\n".join(f"  - {c}" for c in tail)
                )
                # Old appendix size for diagnostics
                old_appendix_body = "\n".join(success_cmds)
                self._appendix_old_chars += len(old_appendix_body)
            else:
                recency = "The container remains in its original state.\n"

            # Per-run recall only. Long-term learning lives in the structured KB.
            recall_block = ""
            if self.run_memory is not None and getattr(self.run_memory, "enabled", False):
                # Build a focused query from the most recent activity.
                query_parts = []
                if self._last_action_text:
                    query_parts.append(self._last_action_text[:400])
                if self._last_error_class:
                    query_parts.append(f"error: {self._last_error_class}")
                if self._last_obs_short:
                    query_parts.append(self._last_obs_short[:400])
                query = "\n".join(query_parts).strip() or (success_cmds[-1] if success_cmds else "")
                if query:
                    try:
                        recall_block = self.run_memory.recall_pack(
                            query,
                            rooms=("commands_success", "commands_failed",
                                   "fixes", "decisions", "patches"),
                            n_per_room=4,
                            token_budget=1500,
                        ) or ""
                    except Exception as _e:
                        log_info(f"[mempalace] recall_pack failed: {_e}")

            appendix = instr + "\n" + recency + recall_block

            # Preserve the legacy normalization of pip_download wrapper paths.
            pattern = r'python\s+/home/tools/pip_download.py\s+-p\s+(\S+)\s+-v\s+""([^""]+)""'
            replacement = r'pip install \1\2'
            appendix = re.sub(pattern, replacement, appendix)
            pattern1 = r'python\s+/home/tools/pip_download.py\s+-p\s+(\S+)'
            replacement1 = r'pip install \1'
            appendix = re.sub(pattern1, replacement1, appendix)

            self._appendix_new_chars += len(appendix)
            system_res += appendix

            # ── Log context summary ──
            log_context_summary(current_directory.strip(), self.image_name, self.max_turn - turn, success_cmds)

            if "gpt" in self.model:
                system_message = {"role": "system", "content": system_res}
            else:
                system_message = {"role": "user", "content": system_res}
            self.messages.append(system_message)
            self._emit_turn_snapshot(
                turn=turn,
                assistant_response=configuration_agent,
                commands=commands,
                diffs=diffs,
                system_res=system_res,
                turn_elapsed_s=time.time() - turn_started_at,
                outer_entries=self.outer_commands[turn_outer_start:],
                tool_counts_before=tool_counts_before,
            )

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
        self._emit_observer_event("run_finished", {
            "total_turns": turn,
            "total_time": round(total_time, 2),
            "cost_tokens": cost_tokens,
            "finished": finish,
            "paper_verdict": self._paper_verdict,
        })

        append_trajectory(trajectory, self.messages, 'configuration')
        trajectory.append({'agent': "configuration", 'cost_time': total_time, 'cost_tokens': cost_tokens}) 
        self.sandbox_session.close()
        return trajectory, self.outer_commands
