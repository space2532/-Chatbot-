import atexit
from flask import Flask, render_template, request
import sys
from common import model
from chatbot import Chatbot
from characters import system_role, instruction
from function_calling import FunctionCalling, tools    # 단일 함수 호출

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

func_calling = FunctionCalling(model=model.basic)


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

    # GPT에게 함수사양을 토대로 사용자 메시지에 호응하는 함수 정보를 분석해달라고 요청
    response, response_type = func_calling.analyze(request_message, tools)
    if response_type == 'function_call':    # GPT가 함수 호출이 필요하다고 분석했는지 여부 체크
        # GPT가 분석해준 대로 함수 호출
        response = func_calling.run(response, jjinchin.context[:]) 
        jjinchin.add_response(response) 
    else:
        response = jjinchin.send_request()
        print(2)
        jjinchin.add_response(response)
    
    response_message = jjinchin.get_last_response()
#    jjinchin.clean_context()
    print('> response_message:', response_message)
    return {'response_message': response_message}
    

@atexit.register
def shutdown():
    print('flask shutting down...')
    jjinchin.save_chat()

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=int(sys.argv[1]))
