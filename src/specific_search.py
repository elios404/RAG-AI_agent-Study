# %%
from state import AgentState
from service import llm, retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List


# %%
from typing import List

# 노드
def format_state_to_string(state: AgentState) -> AgentState:
    """
    AgentState 딕셔너리를 LLM 프롬프트에 넣기 좋은
    "Key: Value" 문자열 형식으로 변환합니다.
    None 값이거나 빈 리스트/빈 문자열인 항목은 제외합니다.
    """
    formatted_lines = []

    keys_to_include = [
        'query', 'status', 'title', 'year', 
        'casts', 'director', 'genre', 'ott', 'info'
    ]

    for key in keys_to_include:
        value = state.get(key)
        
        # 값이 비어있는지 확인 (None, "", [])
        if not value:
            continue
            
        # 값이 리스트인 경우 (예: genre, ott)
        if isinstance(value, List):
            # Enum 객체일 경우 .value를, 일반 문자열일 경우 그대로 사용
            str_values = [
                str(v.value if hasattr(v, 'value') else v) for v in value
            ]
            value_str = ", ".join(str_values)
        else:
            value_str = str(value)
            
        # "Key: Value" 형식으로 추가
        formatted_lines.append(f"- {key.capitalize()}: {value_str}")

    if not formatted_lines:
        return {"rag_context": " - (No specific query details provided) -"}
        
    return {"rag_context" : "\n".join(formatted_lines)}
    # return "\n".join(formatted_lines)

# %%
rag_specialized_prompt_template = """
당신의 역할은 state 정보를 보고 RAG 유사도 검색에 알맞은 쿼리를 생성하는 것입니다.
RAG 안에 Document content는 다음과 같은 형태로 되어있습니다.

=== RAG content 예시(few Shot) ===
"[제목] 승부\n[영문 제목] The Match\n
[줄거리] 세계 최고 바둑 대회에서 국내 최초 우승자..(중략)…\n
[장르] 드라마\n
[키워드] based on true story, go\n
[주요 출연진] 이병헌, 유아인, ..(중략)\n
[감독] 김형주"
=== 예시 끝 ===

**state 내용**
{rag_context}
"""

rag_query_prompt = ChatPromptTemplate.from_template(rag_specialized_prompt_template)

# %%
rag_query_generation_chain = (rag_query_prompt | llm | StrOutputParser())

# 노드
def generate_rag_query(state: AgentState) -> AgentState:
    """
    Rag_context 정보를 바탕으로 RAG 쿼리를 생성합니다.
    """
    print("--- RAG 쿼리 생성 중 ---")
    
    # state 전체를 체인에 전달
    rag_query = rag_query_generation_chain.invoke({"rag_context": state['rag_context']})
    
    print(f"생성된 RAG 쿼리: {rag_query}")
    return {"rag_query": rag_query}

# %%
# 노드
def retrieve(state: AgentState) -> AgentState:
    """ 
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state
    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.        
    """

    rag_query = state.get('rag_query')
    if not rag_query:
        print("경고: RAG 쿼리가 비어있어 원본 쿼리를 사용합니다.")
        rag_query = state['query']
    docs = retriever.invoke(rag_query)
    return {'context': docs}

# %%
generate_prompt_str = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

def format_docs_to_string(docs: List[Document]) -> str:
    """검색된 Document 리스트를 LLM이 읽기 좋은 단일 문자열로 합칩니다."""
    formatted_docs = []
    for doc in docs:
        # Document의 page_content를 가져와서 추가
        formatted_docs.append(doc.page_content)
    
    # 각 문서를 명확하게 분리
    return "\n\n---\n\n".join(formatted_docs)

rag_chain = (
    ChatPromptTemplate.from_template(generate_prompt_str)
    | llm
    | StrOutputParser()
)

# 노드
def generate_answer(state: AgentState) -> AgentState:
    """ 
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.
    """
    print("--- 3. 검색된 컨텍스트로 답변 생성 ---")
    
    query = state['query']
    context_docs = state['context'] # List[Document]

    formatted_context = format_docs_to_string(context_docs)

    response = rag_chain.invoke({
        'question': query, 
        'context': formatted_context
    })

    print(f"생성된 답변: {response}")

    return {'answer': response}


