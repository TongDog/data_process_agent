from __future__ import annotations

from agent_state import AgentState

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import TasksList


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
