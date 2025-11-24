from openai import OpenAI
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv() #.env의 API Key 값을 미리 로드

import pytz
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Model:
    basic: str = 'gpt-5-mini'
    advanced: str = 'gpt-5'


model = Model()

client = OpenAI(timeout=30, max_retries=1)


import tiktoken

def gpt_num_tokens(messages, model='gpt-4o'):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3    # 모든 메시지는 다음 형식을 따른다: <|start|>{role/name}\n{content}<|end|>\n
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for _, value in message.items():
            # value가 문자열이 아닌 경우 문자열로 변환
            if not isinstance(value, str):
                value = str(value)
            num_tokens += len(encoding.encode(value))
    num_tokens += 3    # 모든 메시지는 다음 형식으로 assistant의 답변을 준비한다: <|start|>assistant<|message|>
    return num_tokens


from types import SimpleNamespace

def dict_to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [dict_to_namespace(i) for i in data]
    else:
        return data

def makeup_response(message):
    data = {'output': [ { 'content': [ { 'text': message } ],
                          'role': 'assistant' } ],
            'usage': {'total_tokens': 0},
            'output_text': message
        }
    return dict_to_namespace(data)

def today():
    korea = pytz.timezone('Asia/Seoul')  # 한국 시간대를 얻습니다.
    now = datetime.now(korea)            # 현재 시각을 얻습니다.
    return(now.strftime('%Y%m%d'))       # 시각을 원하는 형식의 문자열로 변환합니다.

def yesterday():    
    korea = pytz.timezone('Asia/Seoul')    # 한국 시간대를 얻습니다.
    now = datetime.now(korea)              # 현재 시각을 얻습니다.
    one_day = timedelta(days=1)            # 하루 (1일)를 나타내는 timedelta 객체를 생성합니다.
    yesterday = now - one_day              # 현재 날짜에서 하루를 빼서 어제의 날짜를 구합니다.
    return yesterday.strftime('%Y%m%d')    # 어제의 날짜를 yyyymmdd 형식으로 변환합니다.

def currTime():
    korea = pytz.timezone('Asia/Seoul')                 # 한국 시간대를 얻습니다.
    now = datetime.now(korea)                           # 현재 시각을 얻습니다.
    formatted_now = now.strftime('%Y.%m.%d %H:%M:%S')   # 시각을 원하는 형식의 문자열로 변환합니다.
    return(formatted_now)
