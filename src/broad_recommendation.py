# broad_recommendation.py

# %%
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from state import AgentState
# retriever가 아닌 vector_store를 직접 임포트하여 filter 기능 사용
from service import llm, vector_store 

# %%
# --- Node 2: 광범위 추천 RAG 쿼리 생성 ---

broad_rag_prompt_template = """
당신은 RAG 검색을 위한 추천 쿼리 생성기입니다.
제공된 '검색 조건'을 바탕으로, 이 조건에 부합하는 작품(영화, 드라마)을 찾기 위한 RAG 검색 쿼리를 생성해주세요.

쿼리는 사용자가 찾고자 하는 작품의 핵심 특징(장르, 분위기, 키워드 등)을 요약해야 합니다.

[검색 조건]
{rag_context}

[조건 요약 RAG 검색 쿼리]
"""

broad_rag_prompt = ChatPromptTemplate.from_template(broad_rag_prompt_template)
broad_rag_chain = broad_rag_prompt | llm | StrOutputParser()

def generate_broad_rag_query(state: AgentState) -> AgentState:
    """
    (플로우 2단계)
    'rag_context' (검색 조건)를 기반으로
    광범위한 추천을 위한 RAG 쿼리를 생성하여 'rag_query'에 덮어씌웁니다.
    """
    print("--- RAG: 광범위 추천 쿼리 생성 중 ---")
    rag_context = state.get('rag_context')

    if not rag_context or rag_context == " - (No specific query details provided) -":
        print("경고: rag_context가 비어있어 원본 쿼리로 쿼리 생성 시도.")
        rag_context = state.get('query')

    new_rag_query = broad_rag_chain.invoke({"rag_context": rag_context})
    
    print(f"--- 생성된 광범위 추천 쿼리 (rag_query로 업데이트): {new_rag_query} ---")
    return {"rag_query": new_rag_query}

# %%
# --- Node 3: 메타데이터 필터링을 통한 검색 ---

def _build_metadata_filter(state: AgentState) -> Optional[Dict[str, Any]]:
    """
    AgentState를 기반으로 ChromaDB에서 사용할 메타데이터 필터를 생성합니다.
    """
    
    # $and 조건을 기본으로 하는 필터 리스트
    filter_list = []
    
    # --- List 기반 필드 처리 (genre, ott, casts, director) ---
    # 사용자가 ["Action", "SF"]를 요청하면, 
    # "Action" 이나 "SF"가 포함된 문서를 찾도록 $or 로직을 구성합니다.
    
    list_fields = ['genre', 'ott', 'casts', 'director']
    
    for field in list_fields:
        values = state.get(field)
        if values:
            # Enum/str-to-str 변환
            str_values = [v.value if hasattr(v, 'value') else str(v) for v in values]
            
            if str_values:
                if len(str_values) == 1:
                    # 값이 하나면 $eq 사용 (예: {"genre": {"$eq": "Action"}})
                    filter_list.append({field: {"$eq": str_values[0]}})
                else:
                    # 값이 여러 개면 $or 사용 (예: {"$or": [{"genre": ...}, {"genre": ...}]})
                    or_clause = [{field: {"$eq": v}} for v in str_values]
                    filter_list.append({"$or": or_clause})

    # --- 단일 값 필드 처리 (year) ---
    if state.get('year'):
        # (예: {"year": {"$eq": 2020}})
        filter_list.append({"year": {"$eq": state['year']}})
    
    # --- 최종 필터 조합 ---
    if not filter_list:
        return None # 적용할 필터 없음
    
    if len(filter_list) == 1:
        return filter_list[0] # 단일 조건
    
    # 여러 조건이 있으면 $and 로 묶음
    return {"$and": filter_list}


def retrieve_with_filter(state: AgentState) -> AgentState:
    """
    (플로우 3단계)
    'rag_query' (semantic)와 state의 세부 정보 (metadata filter)를
    모두 사용하여 RAG 문서를 검색합니다.
    """
    print("--- RAG: 메타데이터 필터링으로 검색 중 ---")
    
    rag_query = state.get('rag_query')
    if not rag_query:
        print("경고: RAG 쿼리가 없어 원본 쿼리로 시맨틱 검색 시도.")
        rag_query = state['query']

    # 1. 메타데이터 필터 생성
    metadata_filter = _build_metadata_filter(state)
    
    search_kwargs = {'k': 3} # service.py의 기본값
    
    if metadata_filter:
        print(f"--- 적용된 메타데이터 필터: {metadata_filter} ---")
        search_kwargs['filter'] = metadata_filter
        
        # 필터가 있으면 vector_store.similarity_search 사용
        docs = vector_store.similarity_search(
            query=rag_query,
            **search_kwargs
        )
    else:
        print("--- 메타데이터 필터 없음. 시맨틱 검색만 수행 ---")
        docs = vector_store.similarity_search(
            query=rag_query,
            **search_kwargs
        )
    
    return {'context': docs}