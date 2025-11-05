import streamlit as st
from main_graph import build_graph

# --- 1. 그래프 로드 (캐시 사용) ---
# @st.cache_resource: 앱이 실행될 때 그래프를 한 번만 빌드하고 캐시에 저장
@st.cache_resource
def get_rag_app():
    """
    LangGraph 앱을 빌드하고 반환합니다.
    """
    # service.py, .env, ChromaDB 등이 모두 준비되어 있어야 함
    try:
        app = build_graph()
        return app
    except Exception as e:
        st.error(f"그래프 빌드 중 오류 발생: {e}")
        return None

# 그래프 빌드 시도
rag_app = get_rag_app()

# --- 2. Streamlit UI 설정 ---
st.title("🎬 OTT RAG 챗봇")
st.caption("LangGraph와 Streamlit으로 만든 영화/드라마 추천 봇입니다.")

# --- 3. 채팅 기록 세션 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. 채팅 기록 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. 사용자 입력 및 챗봇 응답 ---
if prompt := st.chat_input("영화 '승부'에 대해 알려줘"):
    
    # 그래프가 정상적으로 로드되었는지 확인
    if rag_app is None:
        st.error("챗봇 애플리케이션을 로드하지 못했습니다. 관리자에게 문의하세요.")
    else:
        # 1. 사용자 메시지를 기록하고 UI에 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 봇 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다... 🤖"):
                try:
                    # LangGraph 앱 호출
                    inputs = {"query": prompt}
                    
                    # .invoke()를 사용해 최종 상태(답변)를 받음
                    final_state = rag_app.invoke(inputs)
                    
                    # 최종 답변 추출
                    response = final_state.get('answer', '죄송합니다. 답변을 생성하지 못했습니다.')
                    
                    st.markdown(response)
                    
                    # 3. 봇 응답을 기록
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    response = f"답변 생성 중 오류가 발생했습니다: {e}"
                    st.error(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})