# -*- coding: utf-8 -*-
"""
rag_utils.py
--------------
PDF 문서 기반 RAG (정보 질문 답변 / 추천 이유 설명 보조).

가벼운 구현:
  1) pypdf 로 PDF 텍스트 추출
  2) 문단 단위로 분할
  3) TF-IDF(문자 n-gram, 한국어 토크나이저 불필요) 로 관련 문단 검색
  4) LLM 이 있으면 검색 문단을 근거로 답변 정리, 없으면 검색 문단을 그대로 정리해 반환

중요: RAG 는 원두 추천 순위를 결정하지 않는다. 정보 질문 답변/설명 보조에만 사용한다.
"""

import os
import re

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False


def _read_pdf_text(path):
    if PdfReader is None or not os.path.exists(path):
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _split_paragraphs(text, min_len=40):
    # 빈 줄 기준 분할 후 짧은 조각 병합
    raw = re.split(r"\n\s*\n", text)
    chunks, buf = [], ""
    for r in raw:
        r = re.sub(r"\s+", " ", r).strip()
        if not r:
            continue
        if len(buf) < min_len:
            buf = (buf + " " + r).strip()
        else:
            chunks.append(buf)
            buf = r
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= 10]


class RagIndex:
    """여러 PDF를 묶어 TF-IDF 검색 인덱스를 만든다."""

    def __init__(self, pdf_paths):
        self.paragraphs = []   # [{text, source}]
        self.ready = False
        self.error = None
        self._vectorizer = None
        self._matrix = None
        self._build(pdf_paths)

    def _build(self, pdf_paths):
        for path in pdf_paths:
            text = _read_pdf_text(path)
            if not text.strip():
                continue
            src = os.path.basename(path)
            for para in _split_paragraphs(text):
                self.paragraphs.append({"text": para, "source": src})

        if not self.paragraphs:
            self.error = "PDF 문서를 읽지 못했습니다. data/ 폴더에 PDF가 있는지 확인하세요."
            return
        if not _SKLEARN:
            self.error = "scikit-learn 미설치로 TF-IDF 검색을 사용할 수 없습니다."
            self.ready = True  # 키워드 매칭 폴백은 가능
            return

        corpus = [p["text"] for p in self.paragraphs]
        # 한국어 토크나이저 없이도 동작하도록 문자 n-gram 사용
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._matrix = self._vectorizer.fit_transform(corpus)
        self.ready = True

    def search(self, query, top_k=4):
        """질의와 관련된 문단 top_k 반환: [{text, source, score}]"""
        if not self.paragraphs:
            return []
        if self._vectorizer is not None:
            qv = self._vectorizer.transform([query])
            sims = cosine_similarity(qv, self._matrix)[0]
            order = sims.argsort()[::-1][:top_k]
            return [
                {**self.paragraphs[i], "score": float(sims[i])}
                for i in order if sims[i] > 0
            ]
        # 폴백: 단순 키워드 포함 점수
        scored = []
        terms = [t for t in re.split(r"\s+", query) if len(t) >= 2]
        for p in self.paragraphs:
            score = sum(p["text"].count(t) for t in terms)
            if score:
                scored.append({**p, "score": float(score)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def answer_question(rag_index, question, llm_fn=None):
    """정보 질문에 대해 문서 기반 답변을 생성한다.
    llm_fn(system, user) -> str|None 이 주어지면 LLM 으로 정리, 아니면 검색 문단 요약.
    반환: (answer_text, sources: list[str])
    """
    if rag_index is None or not rag_index.paragraphs:
        return ("죄송합니다. 참고할 PDF 문서를 불러오지 못했습니다. "
                "data/ 폴더에 '커피 주요 산지.pdf', '커피재배, 수확, 가공.pdf'를 두었는지 확인해 주세요.", [])

    hits = rag_index.search(question, top_k=4)
    if not hits:
        return ("문서에서 관련 내용을 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요.", [])

    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    sources = sorted({h["source"] for h in hits})

    if llm_fn is not None:
        system = (
            "너는 커피 정보 안내 도우미다. 아래 '문서 발췌'에 근거해서만 한국어로 간결하게 답하라. "
            "문서에 없는 내용은 추측하지 말고 모른다고 말하라. 답변은 3~5문장 이내."
        )
        user = f"[문서 발췌]\n{context}\n\n[질문]\n{question}"
        out = llm_fn(system, user)
        if out:
            return out.strip(), sources

    # 폴백: 검색 문단을 정리해서 제시 (가장 관련도 높은 1~2개 문단을 자른다)
    snippet = hits[0]["text"]
    if len(snippet) > 600:
        snippet = snippet[:600] + " ..."
    answer = (
        "문서에서 찾은 관련 내용입니다:\n\n"
        + snippet
        + "\n\n(API 키를 설정하면 더 자연스러운 요약 답변을 제공할 수 있습니다.)"
    )
    return answer, sources
