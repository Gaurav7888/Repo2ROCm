"""
DAG Scheduler — executes an ExecutionDAG with parallel workers.

Independent branches run in parallel (thread pool), while respecting
dependency edges.  The scheduler supports:
- Dynamic replanning when critical-path nodes fail (up to max_replans)
- Fallback strategies (if node X fails, try node Y instead)
- Progress callbacks for the Rich UI
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional

from storage.models import ExecutionDAG, DAGNode, DAGNodeState


class DAGScheduler:
    """
    Parallel DAG executor with dependency-aware scheduling.

    Nodes whose dependencies are all satisfied are dispatched to the
    thread pool.  When a node completes, its dependents are checked
    and dispatched if ready.
    """

    def __init__(self, max_workers: int = 3,
                 on_node_start: Optional[Callable[[DAGNode], None]] = None,
                 on_node_complete: Optional[Callable[[DAGNode], None]] = None,
                 on_replan: Optional[Callable[[ExecutionDAG, DAGNode], Optional[ExecutionDAG]]] = None):
        self.max_workers = max_workers
        self.on_node_start = on_node_start
        self.on_node_complete = on_node_complete
        self.on_replan = on_replan
        self._lock = threading.Lock()
        self._node_executors: Dict[str, Callable] = {}

    def register_executor(self, agent_role: str,
                          executor: Callable[[DAGNode], Dict[str, Any]]):
        """
        Register an executor function for a specific agent role.

        The executor receives a DAGNode and returns a dict with:
          {"success": bool, "result": Any, "error": Optional[str]}
        """
        self._node_executors[agent_role] = executor

    def execute(self, dag: ExecutionDAG,
                node_executor: Optional[Callable[[DAGNode], Dict[str, Any]]] = None
                ) -> ExecutionDAG:
        """
        Execute the DAG, running independent branches in parallel.

        If node_executor is provided, it's used for all nodes.
        Otherwise, dispatches to role-specific executors registered
        via register_executor().

        Returns the DAG with all node states updated.
        """
        dag.get_critical_path()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: Dict[str, Future] = {}

            while not dag.is_complete():
                ready = dag.get_ready_nodes()

                for node in ready:
                    if node.id in futures:
                        continue

                    executor = node_executor or self._node_executors.get(node.agent_role)
                    if not executor:
                        with self._lock:
                            node.state = DAGNodeState.SKIPPED.value
                            node.error = f"No executor for role {node.agent_role}"
                        continue

                    with self._lock:
                        node.state = DAGNodeState.RUNNING.value
                        node.started_at = time.time()

                    if self.on_node_start:
                        self.on_node_start(node)

                    future = pool.submit(self._run_node, node, executor)
                    futures[node.id] = future

                completed_ids = []
                for node_id, future in futures.items():
                    if future.done():
                        completed_ids.append(node_id)
                        result = future.result()
                        node = dag.nodes[node_id]

                        with self._lock:
                            node.completed_at = time.time()
                            if result.get("success"):
                                node.state = DAGNodeState.SUCCESS.value
                                node.result = result.get("result")
                            else:
                                node.state = DAGNodeState.FAILED.value
                                node.error = result.get("error", "Unknown error")

                                if node.fallback_node_id and node.fallback_node_id in dag.nodes:
                                    fallback = dag.nodes[node.fallback_node_id]
                                    fallback.state = DAGNodeState.PENDING.value
                                    fallback.dependencies = [
                                        d for d in fallback.dependencies
                                        if d != node_id
                                    ]

                        if self.on_node_complete:
                            self.on_node_complete(node)

                        if (node.state == DAGNodeState.FAILED.value
                                and node.critical_path
                                and self.on_replan
                                and dag.replan_count < dag.max_replans):
                            new_dag = self.on_replan(dag, node)
                            if new_dag:
                                dag = new_dag
                                dag.replan_count += 1

                for cid in completed_ids:
                    del futures[cid]

                if not completed_ids and not ready:
                    if futures:
                        time.sleep(0.5)
                    else:
                        break

        return dag

    def _run_node(self, node: DAGNode,
                  executor: Callable[[DAGNode], Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single node, catching exceptions."""
        try:
            return executor(node)
        except Exception as e:
            return {"success": False, "error": str(e)}


def build_default_dag(fingerprint: Any = None, plan: str = "",
                      rocm_mode: bool = True) -> ExecutionDAG:
    """
    Build a default execution DAG for a ROCm build.

    The default DAG has this structure:

      [system_deps] ──┐
                       ├── [pip_deps] ── [special_deps] ── [verify]
      [gpu_check]  ───┘

    system_deps and gpu_check run in parallel.
    pip_deps depends on both system_deps and gpu_check.
    special_deps (flash-attn, etc.) depends on pip_deps.
    verify depends on special_deps.
    """
    dag = ExecutionDAG()

    system_deps = DAGNode(
        name="system_dependencies",
        agent_role="dependency",
        can_parallel=True,
        estimated_duration_minutes=2.0,
        failure_probability=0.1,
    )

    gpu_check = DAGNode(
        name="gpu_verification",
        agent_role="verification",
        can_parallel=True,
        estimated_duration_minutes=0.5,
        failure_probability=0.05,
        commands=[
            "python -c \"import torch; print('CUDA available:', torch.cuda.is_available()); "
            "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\""
        ],
    )

    sys_id = dag.add_node(system_deps)
    gpu_id = dag.add_node(gpu_check)

    pip_deps = DAGNode(
        name="pip_dependencies",
        agent_role="dependency",
        dependencies=[sys_id, gpu_id],
        can_parallel=False,
        estimated_duration_minutes=10.0,
        failure_probability=0.3,
    )
    pip_id = dag.add_node(pip_deps)

    special_deps = DAGNode(
        name="special_cuda_rocm_deps",
        agent_role="dependency",
        dependencies=[pip_id],
        can_parallel=False,
        estimated_duration_minutes=15.0,
        failure_probability=0.4,
    )
    special_id = dag.add_node(special_deps)

    verify = DAGNode(
        name="verification",
        agent_role="verification",
        dependencies=[special_id],
        can_parallel=False,
        estimated_duration_minutes=5.0,
        failure_probability=0.2,
        critical_path=True,
    )
    dag.add_node(verify)

    return dag
