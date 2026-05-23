from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TaskDescriptor:
    user_requirement: str
    dataset_folder: str
    target_path: str
    code_path: str


def descriptor_to_dict(descriptor: TaskDescriptor) -> Dict[str, Any]:
    return {
        "user_requirement": descriptor.user_requirement,
        "dataset_folder": descriptor.dataset_folder,
        "target_path": descriptor.target_path,
        "code_path": descriptor.code_path,
    }


def compute_fingerprint(descriptor: TaskDescriptor) -> str:
    payload = json.dumps(descriptor_to_dict(descriptor), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return make_json_safe(value.dict())
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    return value


class TaskMemoryStore:
    def __init__(self, storage_path: Optional[str] = None) -> None:
        base_dir = os.path.dirname(__file__)
        self.storage_path = storage_path or os.path.join(base_dir, "task_memory.json")

    def _load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(self.storage_path):
            return {}
        with open(self.storage_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_all(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as handle:
            json.dump(make_json_safe(data), handle, ensure_ascii=False, indent=2)

    def _now(self) -> float:
        return time.time()

    def upsert_planner_record(
        self,
        *,
        fingerprint: str,
        descriptor: TaskDescriptor,
        execution_plan_text: str,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        all_data = self._load_all()
        existing_list = all_data.get(fingerprint, [])
        version = int(existing_list[-1].get("version", 0)) + 1 if existing_list else 1
        timestamp = self._now()
        record = {
            "fingerprint": fingerprint,
            "descriptor": descriptor_to_dict(descriptor),
            "planner": {
                "execution_plan_text": execution_plan_text,
                "tasks": make_json_safe(tasks),
            },
            "execution": {
                "final_workflow_code": None,
                "generated_files": {},
                "status": "planned",
            },
            "version": version,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        existing_list.append(record)
        all_data[fingerprint] = existing_list
        self._save_all(all_data)
        return record

    def update_execution_result(
        self,
        *,
        fingerprint: str,
        final_workflow_code: str,
        generated_files: Dict[str, Any],
        execution_status: str = "completed",
    ) -> Optional[Dict[str, Any]]:
        all_data = self._load_all()
        records = all_data.get(fingerprint)
        if not records:
            return None
        record = records[-1]
        record["execution"] = {
            "final_workflow_code": final_workflow_code,
            "generated_files": make_json_safe(generated_files),
            "status": execution_status,
        }
        record["updated_at"] = self._now()
        self._save_all(all_data)
        return record

    def find_best_match(
        self,
        descriptor: TaskDescriptor,
        threshold: float = 0.95,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        threshold = threshold if 0 <= threshold <= 1 else 0.95
        all_data = self._load_all()
        if not all_data:
            return None, 0.0

        target = json.dumps(descriptor_to_dict(descriptor), sort_keys=True, ensure_ascii=False)
        best_record: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for records in all_data.values():
            if not records:
                continue
            candidate = json.dumps(records[-1].get("descriptor", {}), sort_keys=True, ensure_ascii=False)
            score = SequenceMatcher(None, target, candidate).ratio()
            if score > best_score:
                best_record = records[-1]
                best_score = score
        return (best_record, best_score) if best_score >= threshold else (None, best_score)
