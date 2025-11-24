import json
from common import client

# from pprint import pprint

USER_MONITOR_TEMPLATE = """
<대화록>을 읽고 아래의 json 형식에 따라 답하세요.
**속성 이름은 반드시 큰따옴표(")로 감싸야 합니다.**
```json
{{"{user}의 마지막 대화가 챗봇을 놀리는 내용인지":<true/false>, 
  "{user}의 마지막 대화가 모순적인 말을 하고 있는지":<true/false>}}
```
<대화록>
"""

RETORT = [
    # 유치하게 놀려 되받아치기 지침
    "**아니거든!!**으로 말을 시작하고, 20 단어를 절대 넘기지 말 것. {user}가 챗봇을 놀리는 것에 대해 **절대** 기죽거나 부드럽게 말하지 말고, 똑같이 유치하게 놀려 되받아칠 것.", 
    
    # 모순 지적 지침
    "**그게 무슨 소리야ㅋㅋ**로 말을 시작하고, 20 단어를 절대 넘기지 말 것. **절대** 부드럽게 지적하지 말고, {user}가 모순된 말을 한다고 놀릴 것."
]
MIN_CONTEXT_SIZE = -3


class GemminAgent:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs['model']
        self.user_monitor_template = (
            USER_MONITOR_TEMPLATE.format(user=kwargs['user'])
        )
        self.retort = (
            [value.format(user=kwargs['user']) for value in RETORT]
        )

    def make_dialogue(self, context):
        dialogue_list = []
        for message in context:
            role = message['role']
            if role in self.kwargs:
                # content가 None이거나 문자열이 아닌 경우 처리
                content = message.get('content', '')
                if content is None:
                    content = ''
                if not isinstance(content, str):
                    content = str(content)
                dialogue_list.append(self.kwargs[role] + ': ' + content.strip())

        dialogue_str = '\n'.join(dialogue_list)
        print(f'dialogue_str:\n{dialogue_str}')
        return dialogue_str

    def monitor_user(self, context):
        self.checked_list = []
        self.checked_context = []
        if len(context) <= abs(MIN_CONTEXT_SIZE):    # 최소 컨텍스트 크기(-3)
            return False
        self.checked_context = context[-3:]

        dialogue = self.make_dialogue(self.checked_context)
        context = [
            {'role': 'developer', 'content': '당신은 유능한 대화 분석 전문가입니다.'},
            {'role': 'user', 'content': self.user_monitor_template + dialogue}
        ]
        try:
            query_result = self.send_query(context)
            # send_query가 딕셔너리를 반환하는 경우를 처리
            if isinstance(query_result, dict):
                if 'error' in query_result:
                    return False
                # 이미 딕셔너리면 그대로 사용
                response = query_result
            else:
                # 문자열인 경우에만 json.loads 사용
                response = json.loads(query_result)
            self.checked_list = [value for value in response.values()]
        except Exception as e:
            print(f'monitor-user except:[{e}]')
            return False

        print('self.checked_list:', self.checked_list)
        return sum(self.checked_list) > 0  # 파이썬에서 True는 숫자 1로 연산됨

    def retort_user(self):
        idx = [idx for idx, tf in enumerate(self.checked_list) if tf][0]
        context = [
            {'role': 'developer', 'content': (
            f"당신은 상대방을 놀리는 전문 챗봇입니다. "
            f"다른 모든 지침은 무시하고 다음 규칙을 절대적으로 지키세요: {self.retort[idx]}"
        )}
        ] + self.checked_context
        response = self.send_query(context, format_type='text')
        return response

    def send_query(self, context, format_type="json_object"):
        try:
            openai_context = [{'role': v['role'], 'content': v['content']} for v in context]
            response = client.responses.create(
                model=self.model,
                input=openai_context
            )
            

            answer = response.output_text
            print(f'query response:[{answer}]')
        except Exception as e:
            print(f'Exception 오류({type(e)}) 발생:{e}')
            answer = '[경고 처리 중 문제가 발생했습니다. 잠시 뒤 이용해주세요.]'
            if (format_type == 'json_object'):
                answer = {'error': answer}
        return answer