# %%
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from schemas import *
from service import llm

# %%
queryDetail_prompt_template = """
당신의 역할은 사용자의 쿼리를 분석하여 Pydantic 스키마 형식에 맞게 핵심 요소들을 추출하는 것입니다.

**사용자 쿼리:**
{query}

---
**추출 가이드라인:**
- 쿼리를 분석하여 Pydantic 스키마의 각 필드에 알맞은 값을 추출합니다.
- 'genre'와 'ott' 필드는 **반드시 스키마에 정의된 허용된 Enum 값 중에서만** 선택해야 합니다.
- 쿼리에 "공상과학"이 언급되면 "SF" Enum 값을 선택해야 합니다.
- 쿼리에 "넷플"이 언급되면 "Netflix" Enum 값을 선택해야 합니다.
- 쿼리에 정보가 없다면 해당 필드는 비워둡니다 (default=None).
"""

# %%
queryDetail_generate_prompt = ChatPromptTemplate.from_template(queryDetail_prompt_template)
structed_llm = llm.with_structured_output(QueryDetails)
query_analysis_chain = queryDetail_generate_prompt | structed_llm

# %%
# --- 테스트 ---
# test_query = "이병헌이랑 유아인이 나오는 2020년 이후 공상과학 액션 영화 찾아줘. 넷플릭스에 있으면 좋겠어."
# response_object = query_analysis_chain.invoke({"query": test_query})

# print(response_object)

# %%
def generate_query_analysis(state: AgentState) -> AgentState:
    """
    쿼리에서 영화와 관련된 기본 요소를 분리해서 세부정보를 반환합니다.
    Args:
        state (AgentState): 기본 state
        
    Returns:
        state (AgnetState) : title, year, casts 등을 추출해서 담고있는 state
    """

    query = state['query']
    response = query_analysis_chain.invoke({"query": query})

    return response.model_dump()

# %%
def route_query_type(state: AgentState) -> Literal['specific_search', 'similar_recommendation', 'broad_recommendation']:
    """
    쿼리 분석 결과를 기반으로 LangGraph의 다음 단계를 결정합니다.
    """

    query = state.get('query', "")
    status = state.get('status',"")
    title = state.get('title', "")

    if status == "search":
        return "specific_search"
    else:
        if title is not None and title != "":
            return "similar_recommendation"
        else:
            return "broad_recommendation"


