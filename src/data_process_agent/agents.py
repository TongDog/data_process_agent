from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .data_reader import get_directory_structure_json
from .tools import read_json_structure


class FileStructureAgent(RunnablePassthrough):
    """Architect agent that turns folder structure into an execution plan."""

    llm: Optional[ChatOpenAI] = None
    structure_output_chain: Any = None
    prompt_template: ChatPromptTemplate | None = None
    system_prompt: str = """
    你是一名数据架构师。请根据用户需求和目录结构，为后续编码阶段输出精简、可执行的技术方案。

    输出要求：
    1. 文件分析：2-4 句话概括主要文件类型、重要目录和结构特征。
    2. 处理流程：用 3-6 个编号步骤描述“输入 -> 处理 -> 输出”。
    3. 重要约束：列出路径约束、性能风险、覆盖风险等。

    如果需要进一步了解某个 JSON 文件的字段结构，请调用 `read_json_structure`。

    用户需求：
    {context}

    文件夹结构快照：
    {folder_summary}
    """

    def __init__(self, *, model: str, openai_api_key: str | None, openai_api_base: str | None = None):
        super().__init__()
        self.prompt_template = ChatPromptTemplate.from_template(self.system_prompt)
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            temperature=0,
        ).bind_tools([read_json_structure])
        self.structure_output_chain = self.prompt_template | self.llm

    def invoke(self, datasets_folder: str, context: str) -> str:
        folder_summary = get_directory_structure_json(datasets_folder)
        prompt_value = self.prompt_template.invoke(
            {
                "context": context,
                "folder_summary": folder_summary,
            }
        )
        ai_msg = self.structure_output_chain.invoke(
            {
                "context": context,
                "folder_summary": folder_summary,
            }
        )
        if not ai_msg.tool_calls:
            return ai_msg.content

        messages = prompt_value.to_messages() + [ai_msg]
        for tool_call in ai_msg.tool_calls:
            if tool_call["name"] != "read_json_structure":
                continue
            tool_output = read_json_structure.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        response = self.llm.invoke(messages)
        return response.content
