from typing import Any, Dict, List, Optional
from common import client
from function_calling import tools as assistant_tools_schema
from memory_manager import MemoryManager, set_memory_manager_instance
from gemmin_agent import GemminAgent
import threading
import time


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
    ):
        self.client = client
        self.model = model
        self.system_role = system_role
        self.instruction = instruction
        self.assistant_tools = assistant_tools or []
        self.assistant_id = assistant_id or self._create_assistant()
        self.thread_id = thread_id or self._create_thread()
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

    def _create_assistant(self) -> str:
        try:
            assistant = self.client.beta.assistants.create(
                model=self.model,
                instructions=self._instruction_payload(),
                tools=self.assistant_tools,
            )
            return assistant.id
        except Exception as exc:
            raise RuntimeError('Failed to create Assistant for Chatbot') from exc

    def _create_thread(self) -> str:
        try:
            thread = self.client.beta.threads.create()
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
                    model=self.model,
                    user=self.user,
                    assistant=self.assistant,
               )

    @property
    def context(self):
        return self.assistantsManager.context
