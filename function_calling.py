import requests
from memory_manager import get_memory_manager_instance

# 위도 경도
global_lat_lon = {
           '서울':[37.57,126.98],'강원도':[37.86,128.31],'경기도':[37.44,127.55],
           '경상남도':[35.44,128.24],'경상북도':[36.63,128.96],'광주':[35.16,126.85],
           '대구':[35.87,128.60],'대전':[36.35,127.38],'부산':[35.18,129.08],
           '세종시':[36.48,127.29],'울산':[35.54,129.31],'전라남도':[34.90,126.96],
           '전라북도':[35.69,127.24],'제주도':[33.43,126.58],'충청남도':[36.62,126.85],
           '충청북도':[36.79,127.66],'인천':[37.46,126.71],
           'Boston':[42.36, -71.05], '도쿄':[35.68, 139.69]
          }

# 화폐 코드
global_currency_code = {'달러': 'USD', '엔화': 'JPY', '유로화': 'EUR', '위안화': 'CNY', '파운드': 'GBP'}


def get_celsius_temperature(**kwargs):
    location = kwargs['location']
    lat_lon = global_lat_lon.get(location, None)
    if lat_lon is None:
        return None
    lat = lat_lon[0]
    lon = lat_lon[1]

    # API endpoint
    url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true'

    # API를 호출하여 데이터 가져오기
    response = requests.get(url)
    # 응답을 JSON 형태로 변환
    data = response.json()
    # 현재 온도 가져오기 (섭씨)
    temperature = data['current_weather']['temperature']

    print('> temperature:', temperature)
    return temperature


def get_currency(**kwargs):
    currency_name = kwargs['currency_name']
    currency_name = currency_name.replace('환율', '')
    currency_code = global_currency_code.get(currency_name, 'USD')

    if currency_code is None:
        return None

    response = requests.get(f'https://api.exchangerate-api.com/v4/latest/{currency_code}')
    data = response.json()
    krw = data['rates']['KRW']

    print('> 환율:', krw)
    return krw


def retrieve_long_term_memory(user_message: str):
    """Lookup long-term memory via the active MemoryManager instance."""
    manager = get_memory_manager_instance()
    if manager is None:
        return '[기억을 불러올 수 없습니다. 잠시 후 다시 시도해주세요.]'

    memory = manager.retrieve_memory(user_message)
    if memory is None:
        return '[기억이 안난다고 답할 것!]'
    return memory


tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'get_celsius_temperature',
                    'description': '지정된 위치의 현재 섭씨 날씨 확인',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': '광역시도, e.g. 서울, 경기',
                            }
                        },
                        'required': ['location'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_currency',
                    'description': '지정된 통화의 원(KRW) 기준의 환율 확인.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'currency_name': {
                                'type': 'string',
                                'description': '통화명, e.g. 달러환율, 엔화환율',
                            }
                        },
                        'required': ['currency_name'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'retrieve_long_term_memory',
                    'description': '사용자의 질문과 관련된 장기 기억(대화 요약)을 검색',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'user_message': {
                                'type': 'string',
                                'description': '사용자의 최근 질문 혹은 발화',
                            }
                        },
                        'required': ['user_message'],
                    },
                },
            },
        ]
