# RAG-AI_agent-Study

## 환경 활성화
1. uv venv
2. source .venv/bin/activate
3. uv pip install -r requirements.txt

### 추가 라이브러리 설치 이후
uv pip freeze > requirements.txt

## chromaDB
chromaDB2 는 장르 원핫인코딩도 포함된 버전




response_object_ott = [<AllowedOTTs.NETFLIX: 'Netflix'>, ...] 이런 형식에서

for ott_enum in response_object_ott:
        # 1. .value 속성을 사용하여 순수한 문자열 'Netflix'를 추출
        ott_value = ott_enum.value

이런식으로 사용가능

