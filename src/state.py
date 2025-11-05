from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from typing import Optional, List
from typing import Literal

class AgentState(TypedDict):
    query : str
    rag_query : str
    recommend_query : str
    rag_context : str
    context : List[Document]
    answer : str

    # 세부사항
    status: Literal['search', 'recommend']
    title: Optional[str]
    year: Optional[int]
    casts: Optional[List[str]]         # 'actor: str' (X) -> 'casts: Optional[List[str]]' (O)
    director: Optional[List[str]]      # 'director: str' (X) -> 'director: Optional[List[str]]' (O)
    genre: Optional[List[str]]         # 'genre: str' (X) -> 'genre: Optional[List[str]]' (O)
    ott: Optional[List[str]]           # 'ott: str' (X) -> 'ott: Optional[List[str]]' (O)
    info: Optional[str]