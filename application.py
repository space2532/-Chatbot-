import atexit
import json
import sys
import time
from flask import Flask, render_template, request
from common import client, model
from chatbot import Chatbot
from characters import system_role, instruction
from function_calling import (
    get_celsius_temperature,
    get_currency,
    retrieve_long_term_memory,
)

# 실행 방법: python application.py 8080
# 웹페이지 주소 http://localhost:8080/chat-app

# jjinchin 인스턴스 생성
jjinchin = Chatbot(
    model=model.basic, # 모델 선택택
    system_role=system_role,
    instruction=instruction,
    user = '유진',
    assistant = '나양이'
)

application = Flask(__name__)

TOOL_FUNCTIONS = {
    'get_celsius_temperature': get_celsius_temperature,
    'get_currency': get_currency,
    'retrieve_long_term_memory': retrieve_long_term_memory,
}


@application.route('/')
def hello():
    return 'Hello goorm!'


@application.route('/chat-app')
def chat_app():
    return render_template('chat.html')


@application.route('/chat-api', methods=['POST'])
def chat_api():
    request_message = request.json['request_message']
    print('> request_message:', request_message)
    jjinchin.add_user_message(request_message)
    if jjinchin.gemminAgent.monitor_user(jjinchin.context):
        retort = jjinchin.gemminAgent.retort_user()
        jjinchin.assistantsManager.add_assistant_message(retort)
        print('> response_message (retort):', retort)
        return {'response_message': retort}

    run = jjinchin.assistantsManager.create_run()
    response_message = _wait_for_run_completion(run.id)
    print('> response_message:', response_message)
    return {'response_message': response_message}


def _wait_for_run_completion(run_id, poll_interval=1):
    """Poll the Assistants API Run until completion, handling any tool calls."""
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=jjinchin.thread_id,
            run_id=run_id,
        )
        status = getattr(run, 'status', None)
        if status == 'completed':
            message = jjinchin.get_last_response()
            return message or '[응답을 가져오지 못했습니다.]'
        if status == 'requires_action':
            _handle_tool_calls(run)
        elif status in ('failed', 'cancelled', 'expired'):
            reason = getattr(run, 'last_error', None)
            error_message = (
                reason.get('message', '') if isinstance(reason, dict) else str(reason or '')
            )
            print(f'Run terminated with status {status}: {error_message}')
            return '[대화를 완료하지 못했습니다. 잠시 후 다시 시도해주세요.]'
        time.sleep(poll_interval)


def _handle_tool_calls(run):
    tool_calls = getattr(run.required_action.submit_tool_outputs, 'tool_calls', [])
    outputs = []
    for call in tool_calls:
        tool_name = getattr(call.function, 'name', '')
        func = TOOL_FUNCTIONS.get(tool_name)
        raw_args = getattr(call.function, 'arguments', '{}')
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
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
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=jjinchin.thread_id,
            run_id=run.id,
            tool_outputs=outputs,
        )
    

@atexit.register
def shutdown():
    print('flask shutting down...')
    jjinchin.save_chat()

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=int(sys.argv[1]))
