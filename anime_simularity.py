# pip install sentence-transformers
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterable, Optional
from sentence_transformers import SentenceTransformer

# ---------- 0) 노이즈에 강한 정규화 ----------
# - 한글/히라가나/가타카나/한자/영문/숫자만 남김
# - 괄호/부제/기호/이상문자 제거 후 공백 정리
NOISE_KEEP = re.compile(r"[^0-9A-Za-z\uAC00-\uD7A3\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\s]+")


def _normalize_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)              # 괄호 내 부제 제거
    s = NOISE_KEEP.sub(" ", s)                   # 허용 외 문자 제거
    s = re.sub(r"\s+", " ", s).strip()         # 공백 정리
    return s


class AnimeTitleSearcher:
    """제목 임베딩 기반 애니 검색기 (클래스 버전)

    - SentenceTransformer 임베딩으로 인덱스를 구성하고, 단건/배치 검색을 제공
    - 기존 함수형 구현(search_title/batch_search)을 객체 메서드로 통합
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        normalize_embeddings: bool = True,
    ) -> None:
        # 모델은 한 번만 로드
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model: Optional[SentenceTransformer] = SentenceTransformer(model_name)

        # 인덱스 관련 버퍼
        self.emb: Optional[np.ndarray] = None  # (N, d)
        self.docs: List[str] = []              # 정규화된 제목 문자열
        self.id_map: List[int] = []            # anime_id 리스트
        self.titles_df: Optional[pd.DataFrame] = None

    # ---------- 인덱스 구축 ----------
    def fit(self, df: pd.DataFrame, title_col: str = "title", id_col: str = "anime_id") -> "AnimeTitleSearcher":
        """주어진 DataFrame으로 인덱스를 재구성합니다."""
        assert title_col in df.columns and id_col in df.columns, f"DataFrame must contain '{title_col}' and '{id_col}'"
        self.titles_df = df[[id_col, title_col]].copy()
        self.docs = [_normalize_title(t) for t in df[title_col].tolist()]
        self.id_map = df[id_col].tolist()
        self.emb = self.model.encode(
            self.docs,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")
        return self

    @classmethod
    def from_titles(
        cls,
        titles: Iterable[dict],
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        normalize_embeddings: bool = True,
    ) -> "AnimeTitleSearcher":
        """[{"anime_id": int, "title": str}, ...] 리스트로 바로 생성"""
        df = pd.DataFrame(titles)
        return cls(model_name=model_name, normalize_embeddings=normalize_embeddings).fit(df)

    # ---------- 단건 검색 ----------
    def search(self, query: str, k: int = 5, cutoff: float = 0.55) -> List[Tuple[int, float, str]]:
        """질의 문자열로 상위 k개 결과 반환: [(anime_id, score, matched_norm_title), ...]"""
        assert self.emb is not None and len(self.id_map) > 0, "Index is empty. Call fit() first."
        qn = _normalize_title(query)
        qv = self.model.encode([qn], normalize_embeddings=self.normalize_embeddings).astype("float32")[0]
        sims = self.emb @ qv  # 코사인 유사도 (normalize_embeddings=True 전제)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        hits = [(self.id_map[i], float(sims[i]), self.docs[i]) for i in idx if sims[i] >= cutoff]
        return hits

    # ---------- 배치 검색 ----------
    def batch_search(self, queries: List[str], k: int = 5, cutoff: float = 0.55) -> List[List[Tuple[int, float, str]]]:
        """여러 질의를 한 번에 검색"""
        assert self.emb is not None and len(self.id_map) > 0, "Index is empty. Call fit() first."
        qn = [_normalize_title(q) for q in queries]
        qv = self.model.encode(qn, normalize_embeddings=self.normalize_embeddings).astype("float32")  # (B, d)
        sims = qv @ self.emb.T  # (B, N)
        results: List[List[Tuple[int, float, str]]] = []
        for r in range(sims.shape[0]):
            row = sims[r]
            kk = min(k, len(row))
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx])]
            hits = [(self.id_map[i], float(row[i]), self.docs[i]) for i in idx if row[i] >= cutoff]
            results.append(hits)
        return results

    # ---------- 인덱스 확장(옵션) ----------
    def add_titles(self, new_titles: Iterable[dict], title_col: str = "title", id_col: str = "anime_id") -> None:
        """새로운 타이틀들을 인덱스에 추가(증분 인코딩)."""
        df_new = pd.DataFrame(new_titles)
        assert title_col in df_new.columns and id_col in df_new.columns, f"DataFrame must contain '{title_col}' and '{id_col}'"
        docs_new = [_normalize_title(t) for t in df_new[title_col].tolist()]
        ids_new = df_new[id_col].tolist()
        emb_new = self.model.encode(docs_new, normalize_embeddings=self.normalize_embeddings).astype("float32")
        # concat
        if self.emb is None:
            self.emb = emb_new
            self.docs = docs_new
            self.id_map = ids_new
        else:
            self.emb = np.vstack([self.emb, emb_new])
            self.docs.extend(docs_new)
            self.id_map.extend(ids_new)


# ---------- 데모 ----------
if __name__ == "__main__":
    titles = [
        # Naruto-verse
        {"anime_id": 1,  "title": "Naruto"},
        {"anime_id": 2,  "title": "Naruto Shippuden"},
        {"anime_id": 3,  "title": "Boruto: Naruto Next Generations"},
        # Big shonen
        {"anime_id": 4,  "title": "One Piece"},
        {"anime_id": 5,  "title": "Bleach"},
        {"anime_id": 6,  "title": "Bleach: Thousand-Year Blood War"},
        {"anime_id": 7,  "title": "Dragon Ball Z"},
        {"anime_id": 8,  "title": "JoJo's Bizarre Adventure"},
        {"anime_id": 9,  "title": "Slam Dunk"},
        {"anime_id": 10, "title": "Detective Conan"},
        # Modern hits
        {"anime_id": 11, "title": "Attack on Titan"},
        {"anime_id": 12, "title": "Demon Slayer: Kimetsu no Yaiba"},
        {"anime_id": 13, "title": "Jujutsu Kaisen"},
        {"anime_id": 14, "title": "My Hero Academia"},
        {"anime_id": 15, "title": "Chainsaw Man"},
        {"anime_id": 16, "title": "SPY×FAMILY"},
        {"anime_id": 17, "title": "Haikyu!!"},
        {"anime_id": 18, "title": "Blue Lock"},
        {"anime_id": 19, "title": "Oshi no Ko"},
        {"anime_id": 20, "title": "Frieren: Beyond Journey's End"},
        # Classics
        {"anime_id": 21, "title": "Fullmetal Alchemist: Brotherhood"},
        {"anime_id": 22, "title": "Death Note"},
        {"anime_id": 23, "title": "Neon Genesis Evangelion"},
        {"anime_id": 24, "title": "Steins;Gate"},
        {"anime_id": 25, "title": "Made in Abyss"},
        # SAO / Re:Zero / Mushoku
        {"anime_id": 26, "title": "Sword Art Online"},
        {"anime_id": 27, "title": "Re:Zero − Starting Life in Another World"},
        {"anime_id": 28, "title": "Mushoku Tensei: Jobless Reincarnation"},
        # Sports / Misc
        {"anime_id": 29, "title": "Kaguya-sama: Love Is War"},
        {"anime_id": 30, "title": "Vinland Saga"},
        {"anime_id": 31, "title": "Dr. Stone"},
        {"anime_id": 32, "title": "Mob Psycho 100"},
        # Your examples
        {"anime_id": 33, "title": "Dandadan"},
    ]

    # 인스턴스 생성 및 인덱스 구축
    searcher = AnimeTitleSearcher()
    searcher = searcher.from_titles(titles)

    # 단건 검색
    print(searcher.search("귀멸의 칼날", k=5, cutoff=0.50))

    # 배치 검색
    queries = [
        "나루토 질풍전ㅣㄴㅁ;ㅣㅇ;",   # → Naruto Shippuden
        "단다단1-₩129812ㅑ",          # → Dandadan
        "진격의거인!!",                # → Attack on Titan
    ]
    # 가장 높은 순위 3개의 (id, 유사도, 애니 이름)
    for q, hits in zip(queries, searcher.batch_search(queries, k=5, cutoff=0.50)):
        print(q, "->", hits[:3])
    
    # 인덱스만
    # for q, hits in zip(queries, searcher.batch_search(queries, k=5, cutoff=0.50)):
        # print(q, "->", hits[0][0])