from __future__ import annotations

import os
import shutil
import uuid
from typing import Optional

from agent_state import AgentState


class CodeExecutor:
    def __init__(self, work_dir: str = "./runtime_workspace") -> None:
        self.work_dir = work_dir

    def setup_workspace(self, code_dir: str, data_source_path: str) -> str:
        execution_id = str(uuid.uuid4())
        task_dir = os.path.join(self.work_dir, execution_id)
        os.makedirs(task_dir, exist_ok=True)

        for name in os.listdir(code_dir):
            source = os.path.join(code_dir, name)
            target = os.path.join(task_dir, name)
            if os.path.isfile(source):
                shutil.copy2(source, target)

        if os.path.exists(data_source_path):
            data_name = os.path.basename(data_source_path.rstrip(os.sep))
            target_path = os.path.join(task_dir, data_name)
            if os.path.isdir(data_source_path):
                shutil.copytree(data_source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(data_source_path, target_path)
        return task_dir

    def run_code(
        self,
        task_dir: str,
        entry_file: Optional[str] = None,
        image: str = "python:3.10-slim",
        timeout: int = 3000,
    ) -> dict[str, str]:
        if entry_file is None:
            for candidate in ("main.py", "final_workflow.py"):
                if os.path.exists(os.path.join(task_dir, candidate)):
                    entry_file = candidate
                    break
        if entry_file is None:
            raise FileNotFoundError("No workflow entry file found.")

        try:
            import docker
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Docker SDK is not installed. Install with `pip install docker`.") from exc

        client = docker.from_env()
        container = None
        try:
            container = client.containers.run(
                image,
                ["python", entry_file],
                volumes={os.path.abspath(task_dir): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                detach=True,
            )
            result = container.wait(timeout=timeout)
            logs = container.logs().decode("utf-8", errors="ignore")
            status = "success" if result.get("StatusCode") == 0 else "error"
            return {"status": status, "logs": logs, "exit_code": str(result.get("StatusCode", ""))}
        except Exception as exc:  # pragma: no cover - runtime dependent
            return {"status": "error", "logs": f"Exception: {exc}", "exit_code": ""}
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def __call__(self, state: AgentState) -> AgentState:
        if not state.get("should_execute", False):
            return {
                "execution_result": "Execution skipped because should_execute is False.",
                "execution_status": "skipped",
                "results": state.get("results", []),
                "current_node": "code_executor",
            }

        code_dir = state.get("code_path", self.work_dir)
        dataset_folder = state.get("dataset_folder", "") or code_dir
        task_dir = self.setup_workspace(code_dir, dataset_folder)
        result = self.run_code(task_dir)
        logs = result.get("logs", "")
        return {
            "execution_result": logs,
            "execution_status": result.get("status", "error"),
            "results": state.get("results", []) + ([logs] if logs else []),
            "current_node": "code_executor",
        }
