import os
from pymongo import MongoClient
import json
from pinecone.grpc import PineconeGRPC as Pinecone

from common import client, model, today, yesterday, currTime
from pinecone.grpc import PineconeGRPC as Pinecone

pinecone = Pinecone()
pinecone_index = pinecone.Index('jjinchin-memory')

mongo_cluster = MongoClient(os.getenv('MONGO_URI'))
mongo_chats_collection = mongo_cluster['jjinchin']['chats']
mongo_memory_collection = mongo_cluster['jjinchin']['memory']

embedding_model = 'text-embedding-ada-002'

# 아래 사용자 질의가 오늘 이전의 기억에 대해 묻는 것인지 참/거짓으로만 응답하세요.
NEEDS_MEMORY_TEMPLATE = """
Answer only true/false if the user query below asks about memories before today.
```
{message}
"""

# statement1은 기억에 대한 질문입니다.
# statement2는 민지와 고비가 공유하는 기억입니다.
# statment2는 statement1에 대한 가억으로 적절한지 아래 json 포맷으로 답하세요
# {'0과 1 사이의 확률': <확률값>}
MEASURING_SIMILARITY_SYSTEM_ROLE = """
statement1 is a question about memory.
statement2 is a memory shared by '민지' and '고비'.
Answer whether statement2 is appropriate as a memory for statement1 in the following JSON format
{"probability": <between 0 and 1>}
"""

SUMMARIZING_TEMPLATE = """
당신은 사용자의 메시지를 아래의 JSON 형식으로 대화 내용을 주제별로 요약하는 기계입니다.
1. 주제는 구체적이며 의미가 있는 것이어야 합니다.
2. 요약 내용에는 '민지는...', '고비는...'처럼 대화자의 이름이 들어가야 합니다.
3. 원문을 최대한 유지하며 요약해야 합니다. 
4. 주제의 갯수는 무조건 5개를 넘지 말아야 하며 비슷한 내용은 하나로 묶어야 합니다. 
5. "```json"과 같은 부가 정보를 포함하지 않습니다.
```
{
    "data":
            [
                {"주제":<주제>, "요약":<요약>},
                {"주제":<주제>, "요약":<요약>},
            ]
}
"""


class MemoryManager:
    
    def __init__(self, **kwargs):
        self.user = kwargs['user']
        self.assistant = kwargs['assistant']
    
    def search_mongo_db(self, _id):
        search_result = mongo_memory_collection.find_one({'_id': int(_id)})
        print('search_result', search_result)
        return search_result['summary']

    def search_vector_db(self, message):
        query_vector = ( 
            client.embeddings.create(input=message, model=embedding_model).data[0].embedding
        )
        results = pinecone_index.query(top_k=1, vector=query_vector, include_metadata=True)
        id, score = results['matches'][0]['id'], results['matches'][0]['score']
        print('> id', id, 'score', score)
        return id if score > 0.7 else None

    def filter(self, message, memory, threshhold=0.6):
        try:
            response = client.responses.create(
                model=model.advanced,
                input=[
                    {'role': 'developer', 'content': MEASURING_SIMILARITY_SYSTEM_ROLE},
                    {'role': 'user', 'content': f'{{"statement1": {message}, "statement2": {memory}}}'},
                ],
            )
            prob = json.loads(response.output_text)['probability']
            print('> filter prob:', prob)
        except Exception as e:
            print('> filter error:', e)
            prob = 0
        return prob >= threshhold

    def retrieve_memory(self, message):
        vector_id = self.search_vector_db(message)
        if not vector_id:
            return None

        memory = self.search_mongo_db(vector_id)
        if self.filter(message, memory):
            return memory
        else:
            return None

    def needs_memory(self, message):
        try:
            response = client.responses.create(
                model=model.advanced, 
                input=NEEDS_MEMORY_TEMPLATE.format(message=message),
            )

            print('> needs_memory:', response.output_text)
            return (True if response.output_text.upper() == 'TRUE' else False)

        except Exception:
            return False

    def save_chat(self, context):        
        messages = []
        unsaved_refs = []
        for message in context:
            if message.get('saved', True): 
                continue
            messages.append({'date': today(), 'role': message['role'], 'content': message['content']})
            unsaved_refs.append(message)
                        
        if len(messages) > 0:           
            mongo_chats_collection.insert_many(messages)
            for message in unsaved_refs:
                message['saved'] = True

    def restore_chat(self, date=None):
        search_date = date if date is not None else today()        
        search_results = mongo_chats_collection.find({'date': search_date})
        restored_chat = [ {'role': v['role'], 'content': v['content'], 'saved': True} for v in search_results ]
        return restored_chat
    
    def summarize(self, messages):
        altered_messages = [ 
            {
                f"{self.user if message['role'] == 'user' else self.assistant}": message['content'] 
            } for message in messages
        ]
        
        try:
            context = [ {'role': 'developer', 'content': SUMMARIZING_TEMPLATE},
                        {'role': 'user', 'content': json.dumps(altered_messages, ensure_ascii=False)} ]
            response = client.responses.create(
                model=model.basic,
                input=context,
            )
            print('> summarize:', response.output_text)
            return json.loads(response.output_text)['data']
        except Exception as e:
            print('> Exception:', e)
            return []

    def delete_by_date(self, date):
        search_results = mongo_memory_collection.find({'date': date})
        ids = [str(v['_id']) for v in search_results]
        if len(ids) == 0:
            return

        pinecone_index.delete(ids=ids)
        mongo_memory_collection.delete_many({'date': date})

    def save_to_memory(self, summaries, date):
        next_id = self.next_memory_id()

        for summary in summaries:
            vector = client.embeddings.create(
                input=summary['요약'], 
                model=embedding_model
            ).data[0].embedding
            metadata = {'date': date, 'keyword': summary['주제']}
            pinecone_index.upsert([(str(next_id), vector, metadata)])

            query = {'_id': next_id}  # 조회조건
            newvalues = {'$set': {'date': date, 'keyword': summary['주제'], 'summary': summary['요약']} }
            mongo_memory_collection.update_one(query, newvalues, upsert=True)
            next_id += 1

    def next_memory_id(self):
        result = mongo_memory_collection.find_one(sort=[('_id', -1)])
        return 1 if result is None else result['_id'] + 1

    def build_memory(self):
        print(f'{currTime()}: build_memory started...')
        # date = yesterday()
        date = today()    # 테스트 용도

        memory_results = mongo_memory_collection.find({'date': date})
        if len(list(memory_results)) > 0:
            return

        chats_results = self.restore_chat(date)
        if len(list(chats_results)) == 0:
            return

        summaries = self.summarize(chats_results)    # 주제별 요약하기

        self.delete_by_date(date)                    # 날짜별 삭제하기

        self.save_to_memory(summaries, date)         # Database에 저장하기


_memory_manager_singleton = None


def set_memory_manager_instance(instance):
    global _memory_manager_singleton
    _memory_manager_singleton = instance


def get_memory_manager_instance():
    return _memory_manager_singleton