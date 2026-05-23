from __future__ import annotations

import ast
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .state import AgentState


class CodeIntegratorAgent:
    def __init__(
        self,
        *,
        model: str,
        openai_api_key: str | None,
        openai_api_base: str | None = None,
        work_dir: str = "./generated_code",
    ) -> None:
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            temperature=0,
        )
        self.work_dir = work_dir
        self.prompt = ChatPromptTemplate.from_template(
            """
            你是一名 Python 架构师。请将多个步骤脚本整合为一个 `main.py`。

            用户需求：
            {user_requirement}

            技术方案：
            {execution_plan}

            已生成文件及函数信息：
            {functions}

            输入路径：
            - dataset_folder: {dataset_folder}
            - target_path: {target_path}

            要求：
            1. 通过 import 调用各步骤脚本，不重写其内部逻辑。
            2. 严格按照已有函数签名传参。
            3. 用一个 CONFIG 字典集中管理路径。
            4. 为每个步骤加轻量日志和 try/except。

            只输出完整 Python 代码，不要 Markdown。
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _save_code_to_file(self, code: str, filename: str) -> tuple[str, str]:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        file_path = os.path.join(self.work_dir, filename)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(clean_code)
        return file_path, clean_code

    def __call__(self, state: AgentState) -> AgentState:
        generated_files = state.get("generated_files", {})
        self.work_dir = state.get("code_path", self.work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

        chunks: list[str] = []
        for file_name, info in generated_files.items():
            chunks.append(f"文件名: {file_name}")
            chunks.append(f"文件路径: {info.file_pos}")
            chunks.append(f"功能摘要: {info.file_summary}")
            signatures: dict[str, str] = {}
            try:
                with open(info.file_pos, "r", encoding="utf-8") as handle:
                    module_ast = ast.parse(handle.read())
                for node in module_ast.body:
                    if isinstance(node, ast.FunctionDef):
                        arg_names = [arg.arg for arg in node.args.args]
                        signatures[node.name] = f"{node.name}({', '.join(arg_names)})"
            except Exception:
                signatures = {}

            chunks.append("包含函数:")
            for func in info.functions:
                sig_text = signatures.get(func.function_name, "")
                if sig_text:
                    chunks.append(f"- {sig_text}")
                else:
                    chunks.append(f"- {func.function_name}")
                if func.input_summary:
                    chunks.append(f"  输入: {func.input_summary}")
                if func.output_summary:
                    chunks.append(f"  输出: {func.output_summary}")
            chunks.append("--------------------")

        final_code = self.chain.invoke(
            {
                "user_requirement": state.get("user_requirement", ""),
                "execution_plan": state.get("execution_plan_text", ""),
                "functions": "\n".join(chunks),
                "dataset_folder": state.get("dataset_folder", ""),
                "target_path": state.get("target_path", ""),
            }
        )
        _, clean_code = self._save_code_to_file(final_code, "main.py")
        return {
            "final_workflow_code": clean_code,
            "current_node": "integrator",
        }
