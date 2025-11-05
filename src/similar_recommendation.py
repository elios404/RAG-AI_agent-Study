# similar_recommendation.py

# %%
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from state import AgentState
from service import llm, retriever

# %%
# --- Node 3: 특정 작품 검색 및 rag_context 덮어쓰기 ---

def retrieve_and_update_rag_context(state: AgentState) -> AgentState:
    """
    (플로우 3단계)
    state의 'rag_query'를 사용해 특정 작품 정보를 검색하고,
    검색된 첫 번째 문서의 내용을 'rag_context'에 덮어씌웁니다.
    """
    print("--- RAG: 특정 작품 정보 검색 중 ---")
    rag_query = state.get('rag_query')
    if not rag_query:
        print("경고: RAG 쿼리가 없어 원본 쿼리 사용")
        rag_query = state['query']

    docs: List[Document] = retriever.invoke(rag_query)

    if not docs:
        print("경고: 특정 작품 정보를 찾지 못했습니다. 원본 쿼리로 추천을 시도합니다.")
        # Fallback: 원본 쿼리 자체를 rag_context로 사용하여 다음 단계 진행
        return {"rag_context": state['query']}

    # 가장 관련성 높은 (첫 번째) 문서의 내용을 rag_context로 설정
    specific_item_context = docs[0].page_content
    print(f"--- 검색된 작품 정보 (rag_context로 업데이트): {specific_item_context[:150]}... ---")
    
    return {"rag_context": specific_item_context}

# %%
# --- Node 4: 추천 검색 쿼리 생성 ---

recommend_query_prompt_template = """
당신은 RAG 검색을 위한 추천 쿼리 생성기입니다.
제공된 작품의 상세 정보(rag_context)를 바탕으로, 이와 '유사한' 다른 작품(영화, 드라마)을 찾기 위한 RAG 검색 쿼리를 생성해주세요.

쿼리는 작품의 핵심 특징(장르, 줄거리, 분위기, 배우 등)을 반영해야 합니다.
기존 작품의 제목은 제외하고, 특징에 집중해주세요.

[작품 상세 정보]
{rag_context}

[유사 작품 추천을 위한 검색 쿼리]
"""

recommend_query_prompt = ChatPromptTemplate.from_template(recommend_query_prompt_template)
recommend_query_chain = recommend_query_prompt | llm | StrOutputParser()

def generate_recommendation_query(state: AgentState) -> AgentState:
    """
    (플로우 4단계)
    특정 작품의 정보('rag_context')를 기반으로
    유사한 작품을 찾기 위한 새로운 RAG 쿼리를 생성하여 'rag_query'에 덮어씌웁니다.
    """
    print("--- RAG: 유사 작품 추천 쿼리 생성 중 ---")
    rag_context = state.get('rag_context')

    if not rag_context:
        print("경고: rag_context가 비어있어 추천 쿼리 생성 실패.")
        # Fallback: 원본 쿼리를 기반으로 생성 시도
        rag_context = state.get('query')

    new_rag_query = recommend_query_chain.invoke({"rag_context": rag_context})
    
    print(f"--- 생성된 추천 쿼리 (rag_query로 업데이트): {new_rag_query} ---")
    return {"rag_query": new_rag_query}