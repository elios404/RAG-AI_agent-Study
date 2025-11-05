# services.py

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()

# --- 1. LLM 및 Embedding 초기화 ---
llm = ChatOpenAI(model="gpt-5")
embedding = OpenAIEmbeddings(model='text-embedding-3-large')


# --- 2. VectorStore 및 Retriever 초기화 ---
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='./db/chromaDB2',
    collection_name='movie_rag_collection'
)
retriever = vector_store.as_retriever(search_kwargs={'k':3})
