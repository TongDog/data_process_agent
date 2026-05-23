from __future__ import annotations

from functools import partial
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .agents import FileStructureAgent
from .code_executor import CodeExecutor
from .code_integrator import CodeIntegratorAgent
from .config import load_llm_config
from .coder import CoderAgent
from .memory_store import TaskDescriptor, TaskMemoryStore, compute_fingerprint, make_json_safe
from .state import AgentState, task_planning_node


def build_task_descriptor_from_state(state: AgentState) -> TaskDescriptor:
    return TaskDescriptor(
        user_requirement=state.get("user_requirement", ""),
        dataset_folder=state.get("dataset_folder", ""),
        target_path=state.get("target_path", ""),
        code_path=state.get("code_path", ""),
    )


def file_structure_node_wrapper(state: AgentState) -> AgentState:
    llm_config = load_llm_config()
    agent = FileStructureAgent(
        model=llm_config.model,
        openai_api_key=llm_config.api_key,
        openai_api_base=llm_config.api_base,
    )
    response = agent.invoke(state["dataset_folder"], state["user_requirement"])
    return {
        "execution_plan_text": response,
        "folder_summary": state.get("folder_summary", ""),
        "current_node": "architect",
    }


def fast_path_router_node(
    state: AgentState,
    store: TaskMemoryStore,
    similarity_threshold: float = 0.95,
) -> AgentState:
    descriptor = build_task_descriptor_from_state(state)
    fingerprint = compute_fingerprint(descriptor)
    record, score = store.find_best_match(descriptor, threshold=similarity_threshold)
    if record and record.get("execution", {}).get("status") == "completed":
        return {
            "task_fingerprint": fingerprint,
            "cache_hit": True,
            "cached_result": {
                "final_workflow_code": record["execution"].get("final_workflow_code", ""),
                "generated_files": record["execution"].get("generated_files", {}),
                "version": record.get("version"),
                "similarity": score,
            },
            "current_node": "fast_path_router",
        }
    return {
        "task_fingerprint": fingerprint,
        "cache_hit": False,
        "current_node": "fast_path_router",
    }


def fast_path_route_decider(state: AgentState) -> str:
    return "cache_hit" if state.get("cache_hit") else "cache_miss"


def cache_return_node(state: AgentState) -> AgentState:
    cached = state.get("cached_result") or {}
    return {
        "final_workflow_code": cached.get("final_workflow_code", ""),
        "generated_files": cached.get("generated_files", {}),
        "current_node": "cache_return",
    }


def planner_memory_node(state: AgentState, store: TaskMemoryStore) -> AgentState:
    descriptor = build_task_descriptor_from_state(state)
    fingerprint = compute_fingerprint(descriptor)
    serialized_tasks = [make_json_safe(task) for task in state.get("tasks", [])]
    store.upsert_planner_record(
        fingerprint=fingerprint,
        descriptor=descriptor,
        execution_plan_text=state.get("execution_plan_text", ""),
        tasks=serialized_tasks,
    )
    return {"task_fingerprint": fingerprint, "current_node": "planner_memory"}


def execution_memory_update_node(state: AgentState, store: TaskMemoryStore) -> AgentState:
    descriptor = build_task_descriptor_from_state(state)
    fingerprint = compute_fingerprint(descriptor)
    store.update_execution_result(
        fingerprint=fingerprint,
        final_workflow_code=state.get("final_workflow_code", ""),
        generated_files=state.get("generated_files", {}),
        execution_status=state.get("execution_status", "completed") or "completed",
    )
    return {"task_fingerprint": fingerprint, "current_node": "memory_update"}


def check_if_finished(state: AgentState) -> str:
    return "continue" if state.get("current_task_index", 0) < len(state.get("tasks", [])) else "end"


def executor_route_decider(state: AgentState) -> str:
    return "run_integrator" if state.get("run_integrator") else "finish"


def build_workflow(*, enable_checkpoint: bool = False, redis_url: str = "redis://localhost:6379"):
    workflow = StateGraph(AgentState)
    llm_config = load_llm_config()
    memory_store = TaskMemoryStore()

    planner_llm = ChatOpenAI(
        model=llm_config.model,
        openai_api_key=llm_config.api_key,
        openai_api_base=llm_config.api_base,
        temperature=0,
    )
    coder_agent = CoderAgent(
        model=llm_config.model,
        openai_api_key=llm_config.api_key,
        openai_api_base=llm_config.api_base,
    )
    integrator_agent = CodeIntegratorAgent(
        model=llm_config.model,
        openai_api_key=llm_config.api_key,
        openai_api_base=llm_config.api_base,
    )
    executor_agent = CodeExecutor()

    workflow.add_node("fast_path_router", partial(fast_path_router_node, store=memory_store))
    workflow.add_node("cache_return", cache_return_node)
    workflow.add_node("architect", file_structure_node_wrapper)
    workflow.add_node("planner", partial(task_planning_node, llm=planner_llm))
    workflow.add_node("planner_memory", partial(planner_memory_node, store=memory_store))
    workflow.add_node("coder", coder_agent)
    workflow.add_node("integrator", integrator_agent)
    workflow.add_node("code_executor", executor_agent)
    workflow.add_node("memory_update", partial(execution_memory_update_node, store=memory_store))

    workflow.set_entry_point("fast_path_router")
    workflow.add_conditional_edges(
        "fast_path_router",
        fast_path_route_decider,
        {"cache_hit": "cache_return", "cache_miss": "architect"},
    )
    workflow.add_edge("cache_return", END)
    workflow.add_edge("architect", "planner")
    workflow.add_edge("planner", "planner_memory")
    workflow.add_edge("planner_memory", "coder")
    workflow.add_conditional_edges("coder", check_if_finished, {"continue": "coder", "end": "integrator"})
    workflow.add_edge("integrator", "code_executor")
    workflow.add_conditional_edges(
        "code_executor",
        executor_route_decider,
        {"run_integrator": "integrator", "finish": "memory_update"},
    )
    workflow.add_edge("memory_update", END)

    if not enable_checkpoint:
        return workflow.compile()

    from langgraph.checkpoint.redis import RedisSaver

    checkpointer = RedisSaver(redis_url)
    checkpointer.setup()
    return workflow.compile(checkpointer=checkpointer)


def to_serializable(obj: Any) -> Any:
    return make_json_safe(obj)
