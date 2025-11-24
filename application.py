import atexit
import sys
from flask import Flask, render_template, request
from common import model
from chatbot import Chatbot
from characters import system_role, instruction

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

    response_message = jjinchin.get_final_response()
    print('> response_message:', response_message)
    return {'response_message': response_message}
    

@atexit.register
def shutdown():
    print('flask shutting down...')
    jjinchin.save_chat()

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=int(sys.argv[1]))
