from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

# 1. 허용되는 장르 목록을 Enum으로 정의
# (메타데이터 키 'genre_SF' -> Enum 값 'SF')
class AllowedGenres(str, Enum):
    ACTION_ADVENTURE = "Action & Adventure"
    REALITY = "Reality"
    SF = "SF"
    SCI_FI_FANTASY = "Sci-Fi & Fantasy"
    가족 = "가족"
    공포 = "공포"
    다큐멘터리 = "다큐멘터리"
    드라마 = "드라마"
    로맨스 = "로맨스"
    모험 = "모험"
    미스터리 = "미스터리"
    범죄 = "범죄"
    스릴러 = "스릴러"
    애니메이션 = "애니메이션"
    액션 = "액션"
    역사 = "역사"
    음악 = "음악"
    전쟁 = "전쟁"
    코미디 = "코미디"
    판타지 = "판타지"

# 2. 허용되는 OTT 목록을 Enum으로 정의
class AllowedOTTs(str, Enum):
    DISNEY_PLUS = "Disney Plus"
    FILMBOX_PLUS = "FilmBox+"
    NETFLIX = "Netflix"
    NETFLIX_STANDARD_ADS = "Netflix Standard with Ads"
    TVING = "TVING"
    WATCHA = "Watcha"
    WAVVE = "wavve"

# %%
class QueryDetails(BaseModel):
    """
    사용자의 쿼리에서 추출한 RAG 필터링용 핵심 정보
    """
    status: Literal['search', 'recommend'] = Field(description="쿼리의 내용이 정보 검색(search), 다른 영화 드라마 추천인지(recommend)")
    title: Optional[str] = Field(None, description="쿼리에서 언급된 영화나 드라마의 제목")
    year: Optional[int] = Field(None, description="쿼리에서 언급된 특정 연도")
    casts: Optional[List[str]] = Field(None, description="쿼리에서 언급된 배우 이름 목록")
    director: Optional[List[str]] = Field(None, description="쿼리에서 언급된 감독 이름 목록")
    
    # 3. List[str] 대신 List[AllowedGenres]와 List[AllowedOTTs]를 사용
    genre: Optional[List[AllowedGenres]] = Field(
        None, 
        description="쿼리에서 언급된 장르 목록. 반드시 스키마에 정의된 허용된 값 중에서만 선택해야 함."
    )
    ott: Optional[List[AllowedOTTs]] = Field(
        None, 
        description="쿼리에서 언급된 OTT 플랫폼 목록. 반드시 스키마에 정의된 허용된 값 중에서만 선택해야 함."
    )
    info: Optional[str] = Field(None, description="기타 줄거리 관련 키워드")