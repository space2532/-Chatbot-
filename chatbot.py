import math
from common import client, makeup_response, gpt_num_tokens
from memory_manager import MemoryManager
from gemmin_agent import GemminAgent
import threading
import time


class Chatbot:

    def __init__(self, model, system_role, instruction, **kwargs):
        self.context = [{'role': 'developer', 'content': system_role}]
        self.model = model
        self.max_token_size = 16 * 1024
        self.instruction = instruction
        self.kwargs = kwargs
        self.user = kwargs['user']
        self.assistant = kwargs['assistant']
        self.gemminAgent = self._create_gemmin_agent()
        self.memoryManager = MemoryManager(**kwargs)
        self.context.extend(self.memoryManager.restore_chat())
        # 데몬 구동
        bg_thread = threading.Thread(target=self.background_task)
        bg_thread.daemon = True
        bg_thread.start()

    def background_task(self):
        while True:
            self.save_chat()
            self.context = [ {'role': v['role'], 'content': v['content'], 'saved': True} for v in self.context ]
            self.memoryManager.build_memory()
            # time.sleep(3600)     # 1시간마다 반복
            time.sleep(120)  # 테스트 용도

    def add_user_message(self, message):
        self.context.append({'role': 'user', 'content': message, 'saved': False})

    def _send_request(self):
        try:
            if gpt_num_tokens(self.context) > self.max_token_size:
                self.context.pop()
                return makeup_response('메시지를 조금 짧게 보내줄래?')
            else:
                context = self.to_openai_context()
                response = client.responses.create(
                model=self.model,
                instructions = self.instruction,
                input=context
                )
        except Exception as e:
            print(f'> Exception 오류({type(e)}) 발생:{e} ')
            return makeup_response('[내 찐친 챗봇에 문제가 발생했습니다. 잠시 뒤 이용해주세요]')
        return response

    def send_request(self):
        print(1)
        memory_instruction = self.retrieve_memory()
        self.context[-1]['content'] += memory_instruction if memory_instruction is not None else ''
        if self.gemminAgent.monitor_user(self.to_openai_context()):
            return makeup_response(self.gemminAgent.retort_user()) 
        else:
            return self._send_request()
        
    def retrieve_memory(self):
        user_message = self.context[-1]['content']
        if not self.memoryManager.needs_memory(user_message):
            return

        memory = self.memoryManager.retrieve_memory(user_message)  
        if memory is not None:
            whisper = (f'[귓속말]\n{self.assistant}야! 기억 속 대화 내용이야. 앞으로 이 내용을 참조하면서 답해줘. '
                       f'알마 전에 나누었던 대화라는 점을 자연스럽게 말해줘:\n{memory}')
            self.add_user_message(whisper)
            return None
        else:
            return '[기억이 안난다고 답할 것!]'

    def add_response(self, response):
        self.context.append({
            'role': response.output[-1].role,
            'content': response.output_text,
            'saved': False
        })
        

    def get_last_response(self):
        return self.context[-1]['content']

    def to_openai_context(self):
        return [{'role': v['role'], 'content': v['content']} for v in self.context]

    def save_chat(self):
        self.memoryManager.save_chat(self.context)

    # def clean_context(self):
    #     for idx in reversed(range(len(self.context))):
    #         if self.context[idx]['role'] == 'user':
    #             content = self.context[idx].get('content', '')
    #             if content and isinstance(content, str):
    #                 self.context[idx]['content'] = content.split('instruction:\n')[0].strip()
    #             break

    def handle_token_limit(self, response):
        # 누적 토큰 수가 임계점을 넘지 않도록 제어한다.
        try:
            if response['usage']['total_tokens'] > self.max_token_size:
                remove_size = math.ceil(len(self.context) / 10)
                self.context = [self.context[0]] + self.context[remove_size+1:]
        except Exception as e:
            print(f'> handle_token_limit exception:{e}')

    def _create_gemmin_agent(self):
        return GemminAgent(
                    model=self.model,
                    user=self.user,
                    assistant=self.assistant,
               )
