import json
import os
import time
from typing import Any, Dict, List, Optional
from common import client, model
from function_calling import (
    get_celsius_temperature,
    get_currency,
    retrieve_long_term_memory,
    tools as assistant_tools_schema,
)
from memory_manager import MemoryManager, set_memory_manager_instance
from gemmin_agent import GemminAgent
import threading


class AssistantsManager:
    """Lightweight helper to manage Assistants API resources and local context."""

    def __init__(
        self,
        model: str,
        system_role: str,
        instruction: str,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        assistant_tools: Optional[List[Dict[str, Any]]] = None,
        memory_manager: Optional[MemoryManager] = None,
        user: Optional[str] = None,
    ):
        self.client = client
        self.model = model
        self.system_role = system_role
        self.instruction = instruction
        self.assistant_tools = assistant_tools or []
        self.memory_manager = memory_manager
        self.user = user
        self.assistant_id = self._ensure_assistant_id(assistant_id)
        self.thread_id = self._ensure_thread_id(thread_id)
        self._context: List[Dict[str, Any]] = []
        # 유지보수를 위해 시스템 메시지는 로컬 컨텍스트에만 기록
        self.add_message('developer', system_role, saved=True, persist_to_thread=False)

    @property
    def context(self) -> List[Dict[str, Any]]:
        return self._context

    def _instruction_payload(self) -> str:
        parts = []
        if self.system_role:
            parts.append(self.system_role.strip())
        if self.instruction:
            parts.append(self.instruction.strip())
        return '\n'.join(part for part in parts if part)

    def _ensure_assistant_id(self, override_id: Optional[str]) -> str:
        candidate_id = override_id or os.getenv('ASSISTANT_ID')
        if candidate_id:
            try:
                self.client.beta.assistants.retrieve(candidate_id)
                return candidate_id
            except Exception as exc:
                print(f'기존 Assistant(ID: {candidate_id}) 조회 실패: {exc}')

        try:
            assistant = self.client.beta.assistants.create(
                model=self.model,
                instructions=self._instruction_payload(),
                tools=self.assistant_tools,
            )
            print(f'생성된 ID {assistant.id}를 .env 파일에 ASSISTANT_ID로 추가하세요')
            return assistant.id
        except Exception as exc:
            raise RuntimeError('Failed to create Assistant for Chatbot') from exc

    def _ensure_thread_id(self, override_id: Optional[str]) -> str:
        if override_id:
            return override_id

        stored_thread_id: Optional[str] = None
        if self.memory_manager and self.user:
            stored_thread_id = self.memory_manager.get_thread_id(self.user)

        if stored_thread_id:
            try:
                self.client.beta.threads.retrieve(stored_thread_id)
                return stored_thread_id
            except Exception as exc:
                print(f'기존 Thread(ID: {stored_thread_id}) 조회 실패: {exc}')

        try:
            thread = self.client.beta.threads.create()
            if self.memory_manager and self.user:
                self.memory_manager.save_thread_id(self.user, thread.id)
            return thread.id
        except Exception as exc:
            raise RuntimeError('Failed to create Thread for Chatbot') from exc

    def add_message(
        self,
        role: str,
        content: str,
        saved: bool = False,
        persist_to_thread: bool = False,
    ) -> Dict[str, Any]:
        message = {'role': role, 'content': content, 'saved': saved}
        self._context.append(message)
        if persist_to_thread and role in ('user', 'assistant'):
            try:
                self.client.beta.threads.messages.create(
                    thread_id=self.thread_id,
                    role=role,
                    content=content,
                )
            except Exception as exc:
                print(f'AssistantsManager message sync failed: {exc}')
        return message

    def add_user_message(self, content: str) -> Dict[str, Any]:
        return self.add_message('user', content, saved=False, persist_to_thread=True)

    def add_assistant_message(self, content: str) -> Dict[str, Any]:
        # 어시스턴트 응답은 Run 결과에 의해 Thread에 자동 기록될 예정이므로
        # 로컬 컨텍스트에만 기록한다.
        return self.add_message('assistant', content, saved=False, persist_to_thread=False)

    def create_run(self):
        return self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
        )

    def get_last_message(self):
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread_id,
                order='desc',
                limit=1,
            )
        except Exception as exc:
            print(f'AssistantsManager.get_last_message error: {exc}')
            return None

        if not messages.data:
            return None

        latest = messages.data[0]
        text = ''
        for block in getattr(latest, 'content', []):
            text_block = getattr(block, 'text', None)
            if text_block and getattr(text_block, 'value', None):
                text = text_block.value
                break

        if latest.role in ('assistant', 'user') and text:
            if not self._context or self._context[-1].get('content') != text:
                self.add_message(latest.role, text, saved=False, persist_to_thread=False)
        return text


TOOL_FUNCTIONS = {
    'get_celsius_temperature': get_celsius_temperature,
    'get_currency': get_currency,
    'retrieve_long_term_memory': retrieve_long_term_memory,
}


class Chatbot:

    def __init__(self, model, system_role, instruction, **kwargs):
        self.model = model
        self.instruction = instruction
        self.kwargs = kwargs
        self.user = kwargs['user']
        self.assistant = kwargs['assistant']
        self.gemminAgent = self._create_gemmin_agent()
        self.memoryManager = MemoryManager(**kwargs)
        set_memory_manager_instance(self.memoryManager)
        self.assistantsManager = AssistantsManager(
            model=self.model,
            system_role=system_role,
            instruction=instruction,
            assistant_id=kwargs.get('assistant_id'),
            thread_id=kwargs.get('thread_id'),
            assistant_tools=assistant_tools_schema,
            memory_manager=self.memoryManager,
            user=self.user,
        )
        self.thread_id = self.assistantsManager.thread_id
        self.assistant_id = self.assistantsManager.assistant_id
        # 데몬 구동
        bg_thread = threading.Thread(target=self.background_task)
        bg_thread.daemon = True
        bg_thread.start()

    def background_task(self):
        while True:
            self.save_chat()
            self.memoryManager.build_memory()
            # time.sleep(3600)     # 1시간마다 반복
            time.sleep(120)  # 테스트 용도

    def add_user_message(self, message):
        self.assistantsManager.add_user_message(message)

    def get_last_response(self):
        last_message = self.assistantsManager.get_last_message()
        return last_message or ''

    def save_chat(self):
        self.memoryManager.save_chat(self.context)

    def _create_gemmin_agent(self):
        return GemminAgent(
                    model=model.advanced,
                    user=self.user,
                    assistant=self.assistant,
               )

    @property
    def context(self):
        return self.assistantsManager.context

    def get_final_response(self, poll_interval: int = 1) -> str:
        """Create a Run, handle tool calls, and return the final response."""
        try:
            run = self.assistantsManager.create_run()
        except Exception as exc:
            print(f'Run creation failed: {exc}')
            return '[대화를 시작할 수 없습니다. 잠시 후 다시 시도해주세요.]'

        while True:
            try:
                run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run.id,
                )
            except Exception as exc:
                print(f'Run retrieve failed: {exc}')
                return '[대화 상태를 확인할 수 없습니다. 잠시 후 다시 시도해주세요.]'

            status = getattr(run, 'status', None)
            if status == 'completed':
                message = self.get_last_response()
                return message or '[응답을 가져오지 못했습니다.]'
            if status == 'requires_action':
                self._handle_tool_calls(run)
                continue
            if status in ('failed', 'cancelled', 'expired'):
                reason = getattr(run, 'last_error', None)
                error_message = (
                    reason.get('message', '')
                    if isinstance(reason, dict)
                    else str(reason or '')
                )
                print(f'Run terminated with status {status}: {error_message}')
                return '[대화를 완료하지 못했습니다. 잠시 후 다시 시도해주세요.]'
            time.sleep(poll_interval)

    def _handle_tool_calls(self, run) -> None:
        submit_action = getattr(run, 'required_action', None)
        submit_payload = (
            getattr(submit_action, 'submit_tool_outputs', None) if submit_action else None
        )
        tool_calls = getattr(submit_payload, 'tool_calls', []) if submit_payload else []
        if not tool_calls:
            return

        outputs = []
        for call in tool_calls:
            tool_name = getattr(getattr(call, 'function', None), 'name', '')
            func = TOOL_FUNCTIONS.get(tool_name)
            raw_args = getattr(getattr(call, 'function', None), 'arguments', '{}')
            args = self._parse_tool_arguments(raw_args)

            if not func:
                result = f'[지원하지 않는 도구: {tool_name}]'
            else:
                try:
                    result = func(**args)
                except Exception as exc:
                    result = f'[도구 실행 실패: {exc}]'

            outputs.append(
                {
                    'tool_call_id': call.id,
                    'output': json.dumps(result, ensure_ascii=False)
                    if isinstance(result, (dict, list))
                    else str(result),
                }
            )

        if outputs:
            try:
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread_id,
                    run_id=run.id,
                    tool_outputs=outputs,
                )
            except Exception as exc:
                print(f'submit_tool_outputs failed: {exc}')

    @staticmethod
    def _parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if not isinstance(raw_args, str):
            return {}
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
