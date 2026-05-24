from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    required_libraries: List[str] = Field(..., description="Required libraries for this step")
    step_by_step_instruction: str = Field(..., description="Detailed implementation instruction")
    input_source: str
    output_target: str
    if_execute: bool = False
    preserve_directory_structure: bool = True
    target_extension: Optional[str] = None


class FunctionDetail(BaseModel):
    function_name: str = Field(description="Python function name")
    function_summary: str = Field(description="One-line purpose of the function")
    input_summary: str = Field(default="", description="Function inputs")
    output_summary: str = Field(default="", description="Function outputs")


class CoderOutput(BaseModel):
    file_name: str = Field(description="Python filename ending with .py")
    file_summary: str = Field(description="Summary of the generated file")
    code: str = Field(description="Complete executable Python code")
    functions: List[FunctionDetail] = Field(description="Functions defined in the file")


class FileInfo(BaseModel):
    file_summary: str
    file_pos: str
    functions: List[FunctionDetail]


class TasksList(BaseModel):
    tasks: List[Task]
