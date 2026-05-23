from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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


class TasksList(BaseModel):
    tasks: List[Task]


def task_planning_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    parser = PydanticOutputParser(pydantic_object=TasksList)
    prompt = ChatPromptTemplate.from_template(
        """
        你是一名技术项目经理。

        基于架构师提供的技术方案和用户需求，将其拆解成一组清晰、可执行的任务。

        用户需求:
        {user_requirement}

        技术方案:
        {execution_plan_text}

        请输出 JSON：
        {format_instructions}
        """
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    result = chain.invoke(
        {
            "user_requirement": state.get("user_requirement", ""),
            "execution_plan_text": state.get("execution_plan_text", ""),
        }
    )
    return {
        "tasks": result.tasks,
        "current_task_index": 0,
        "current_node": "planner",
    }
