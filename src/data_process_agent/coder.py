from __future__ import annotations

import json
import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent_state import AgentState

from .models import CoderOutput, FileInfo


class CoderAgent:
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
        self.parser = PydanticOutputParser(pydantic_object=CoderOutput)
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Senior Python Developer. Write exactly one self-contained Python module for one data-processing step.

            Project Structure Summary:
            {folder_summary}

            Task Goal:
            {task_description}

            Input Source:
            {input_source}

            Output Destination:
            {output_source}

            Required Libraries:
            {libraries}

            Requirements:
            1. Generate complete executable Python code for this step only.
            2. Use explicit function parameters instead of hard-coded absolute paths.
            3. Keep file-system side effects inside functions.
            4. Include light logging and basic error handling.
            5. Include at least one entry function.

            Output JSON only:
            {format_instructions}
            """
        ).partial(format_instructions=self.parser.get_format_instructions())
        self.chain = prompt | self.llm | self.parser

    def _save_code_to_file(self, code: str, filename: str) -> tuple[str, str]:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        file_path = os.path.join(self.work_dir, filename)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(clean_code)
        return file_path, clean_code

    def _save_metadata_to_json(self, generated_files: dict[str, FileInfo]) -> str:
        metadata_path = os.path.join(self.work_dir, "generated_files_metadata.json")
        payload = {key: info.model_dump() for key, info in generated_files.items()}
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return metadata_path

    def __call__(self, state: AgentState) -> AgentState:
        current_index = state.get("current_task_index", 0)
        tasks = state.get("tasks", [])
        self.work_dir = state.get("code_path", self.work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

        if current_index >= len(tasks):
            return {"current_task_index": current_index}

        current_task = tasks[current_index]
        response = self.chain.invoke(
            {
                "folder_summary": state.get("folder_summary", ""),
                "task_description": current_task.step_by_step_instruction,
                "libraries": current_task.required_libraries,
                "input_source": current_task.input_source,
                "output_source": current_task.output_target,
            }
        )

        base_name = response.file_name if response.file_name.endswith(".py") else f"{response.file_name}.py"
        file_name = f"task_{current_index}_{base_name}"
        file_path, clean_code = self._save_code_to_file(response.code, file_name)

        new_file_info = FileInfo(
            file_summary=response.file_summary,
            file_pos=file_path,
            functions=response.functions,
        )
        all_generated_files = state.get("generated_files", {}).copy()
        all_generated_files[file_name] = new_file_info
        self._save_metadata_to_json(all_generated_files)

        return {
            "all_task_codes": [clean_code],
            "current_task_code": clean_code,
            "current_task_index": current_index + 1,
            "generated_files": {file_name: new_file_info},
            "current_node": f"coder_task_{current_index}",
        }
