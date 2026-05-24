from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from data_process_agent.models import FileInfo, Task


class AgentState(TypedDict, total=False):
    user_requirement: str
    target_path: str
    dataset_folder: str
    code_path: str
    folder_summary: str
    execution_plan: List[str]
    execution_plan_text: str
    generated_code: str
    execution_result: str
    execution_status: str
    tasks: Annotated[List[Task], operator.add]
    current_task_index: int
    current_task_code: str
    results: List[str]
    all_task_codes: Annotated[List[str], operator.add]
    final_workflow_code: str
    generated_files: Annotated[dict[str, FileInfo], operator.ior]
    current_node: str
    task_fingerprint: str
    cache_hit: bool
    cached_result: Dict[str, Any]
    should_execute: bool
    run_integrator: bool
