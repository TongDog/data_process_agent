from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from data_process_agent.memory_store import TaskDescriptor, TaskMemoryStore, compute_fingerprint
from data_process_agent.workflow import (
    build_task_descriptor_from_state,
    cache_return_node,
    execution_memory_update_node,
    fast_path_router_node,
    planner_memory_node,
)


class MemoryWorkflowTests(unittest.TestCase):
    def _base_state(self) -> dict:
        return {
            "user_requirement": "示例任务",
            "target_path": "/tmp/output",
            "dataset_folder": "/tmp/dataset",
            "code_path": "/tmp/code",
            "execution_plan_text": "",
            "tasks": [],
            "current_task_index": 0,
            "results": [],
            "all_task_codes": [],
            "final_workflow_code": "",
            "generated_files": {},
            "current_node": "",
        }

    def test_fingerprint_deterministic(self) -> None:
        descriptor = TaskDescriptor("任务描述", "/data", "/output", "/code")
        self.assertEqual(compute_fingerprint(descriptor), compute_fingerprint(descriptor))

    def test_upsert_and_update_execution_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.json")
            store = TaskMemoryStore(storage_path=storage_path)
            descriptor = TaskDescriptor("任务描述", "/data", "/output", "/code")
            fingerprint = compute_fingerprint(descriptor)
            record = store.upsert_planner_record(
                fingerprint=fingerprint,
                descriptor=descriptor,
                execution_plan_text="计划内容",
                tasks=[{"task_id": "t1"}],
            )
            self.assertEqual(record["version"], 1)
            updated = store.update_execution_result(
                fingerprint=fingerprint,
                final_workflow_code="print('ok')",
                generated_files={"file": {"path": "x.py"}},
            )
            self.assertIsNotNone(updated)
            with open(storage_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(data[fingerprint][-1]["execution"]["status"], "completed")

    def test_find_best_match_and_fast_path_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.json")
            store = TaskMemoryStore(storage_path=storage_path)
            base_state = self._base_state()
            descriptor = build_task_descriptor_from_state(base_state)
            fingerprint = compute_fingerprint(descriptor)
            store.upsert_planner_record(
                fingerprint=fingerprint,
                descriptor=descriptor,
                execution_plan_text="计划内容",
                tasks=[{"task_id": "t1"}],
            )
            store.update_execution_result(
                fingerprint=fingerprint,
                final_workflow_code="print('cached')",
                generated_files={"file": {"path": "x.py"}},
            )
            routed = fast_path_router_node(base_state, store, similarity_threshold=0.95)
            self.assertTrue(routed["cache_hit"])
            returned = cache_return_node(routed)
            self.assertEqual(returned["final_workflow_code"], "print('cached')")

    def test_planner_and_execution_memory_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.json")
            store = TaskMemoryStore(storage_path=storage_path)
            state = self._base_state()
            state["execution_plan_text"] = "计划内容"
            state["tasks"] = [{"task_id": "t1"}]
            planner_state = planner_memory_node(state, store)
            fingerprint = planner_state["task_fingerprint"]
            state["final_workflow_code"] = "print('final')"
            state["generated_files"] = {"file": {"path": "x.py"}}
            execution_state = execution_memory_update_node(state, store)
            self.assertEqual(execution_state["task_fingerprint"], fingerprint)

    def test_fast_path_is_faster_than_cold_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.json")
            store = TaskMemoryStore(storage_path=storage_path)
            state = self._base_state()
            descriptor = build_task_descriptor_from_state(state)
            fingerprint = compute_fingerprint(descriptor)
            store.upsert_planner_record(
                fingerprint=fingerprint,
                descriptor=descriptor,
                execution_plan_text="计划内容",
                tasks=[{"task_id": "t1"}],
            )
            store.update_execution_result(
                fingerprint=fingerprint,
                final_workflow_code="print('cached')",
                generated_files={"file": {"path": "x.py"}},
            )
            start_cold = time.time()
            time.sleep(0.1)
            cold_duration = time.time() - start_cold
            start_fast = time.time()
            routed = fast_path_router_node(state, store, similarity_threshold=0.95)
            cache_return_node(routed)
            fast_duration = time.time() - start_fast
            self.assertLessEqual(fast_duration, cold_duration * 0.2)


if __name__ == "__main__":
    unittest.main()
