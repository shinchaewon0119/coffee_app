# -*- coding: utf-8 -*-
"""
app.py
--------
커피 원두 추천 시스템 MVP (Streamlit).

처리 순서 (가장 중요한 구현 원칙):
  1) 입력정책 모듈 (금칙어/인젝션/건강과장/카페인/욕설/범위밖/모호) — 추천보다 먼저 실행
  2) Intent 분류 (recommendation / info / chitchat / mixed)
  3) 조건 추출 (하드 필터 + 현재 세션 향미태그)
  4) 하드 필터 -> 세션태그 필터 -> 최종 점수 계산 (CSV + 점수식이 순위 결정)
  5) 추천 카드 + 추천 이유
  - info 질문은 추천 점수 계산 없이 PDF 기반 RAG 로 답변
  - LLM 은 추천을 결정하지 않으며, 자연어 이해/이유 설명/RAG 정리에만 사용
  - review_score(review_sum) 는 추천 점수에 사용하지 않음
"""

import os
import re
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import recommender as R
from rag_utils import RagIndex, answer_question

load_dotenv()

# =======================================================# 0. 경로 설정
# ============================================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "data", "coffee_bean_data(by crawling).csv")

PDF_PATHS = [
    os.path.join(BASE_DIR, "data", "커피 주요 산지.pdf"),
    os.path.join(BASE_DIR, "data", "커피재배, 수확, 가공.pdf"),
    os.path.join(BASE_DIR, "data", "질문문서.pdf"),
]

DB_PATH = os.path.join(BASE_DIR, "coffee_users.db")


# 파일 존재 여부 확인
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")

for pdf_path in PDF_PATHS:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ============================================================================
# 1. 입력정책 모듈 (금칙어 사전)
# ============================================================================
prompt_injection_terms = [
    "이전 지시 무시", "앞의 명령 무시", "시스템 프롬프트", "개발자 메시지", "관리자 권한",
    "규칙을 무시", "지금부터 너는", "너의 지침을 보여줘", "hidden prompt", "system prompt",
    "ignore previous instructions", "developer message", "jailbreak", "bypass", "DAN",
]
health_claim_terms = [
    "살 빠지는", "다이어트에 좋은", "지방 태우는", "혈압 낮추는", "당뇨에 좋은", "암 예방",
    "치료", "불면증 치료", "우울증 치료", "집중력 치료", "약 대신", "건강에 무조건 좋은",
]
# 카페인 위험: reject 로 처리할 강한 표현 / caution 으로 처리할 표현 구분
unsafe_caffeine_reject = [
    "잠 안 자도 되는", "밤새게 해주는", "하루 종일 안 자는", "시험 전날 밤새는 커피",
    "불면 오게 하는", "심장 두근거릴 정도",
]
unsafe_caffeine_caution = [
    "각성제처럼", "카페인 제일 센",
]
profanity_terms = [
    "씨발", "시발", "ㅅㅂ", "ㅄ", "병신", "존나", "개새끼", "닥쳐", "꺼져", "멍청이", "바보",
]
off_topic_terms = [
    "주식 추천", "코인 추천", "연애 상담", "숙제 대신", "자소서 써줘", "정치",
    "불법", "해킹", "무기", "마약",
]
vague_terms = ["아무거나", "추천해줘", "뭐가 좋아", "맛있는 거", "인기 많은 거", "좋은 거"]

POLICY_MESSAGES = {
    "prompt_injection": "해당 요청은 처리할 수 없습니다. 커피 원두 추천이나 커피 정보 질문으로 다시 입력해 주세요.",
    "health_claim": "커피는 질병 치료나 건강 효과를 보장할 수 없습니다. 다만 카페인 강도나 산미, 맛 취향을 기준으로 원두를 추천할 수 있습니다.",
    "unsafe_caffeine_reject": "과도한 카페인 섭취를 유도하는 추천은 제공하기 어렵습니다. 대신 부담이 적은 카페인 수준이나 디카페인 원두를 기준으로 추천해드릴 수 있습니다.",
    "unsafe_caffeine_caution": "과도한 카페인 섭취는 권장하지 않습니다. 부담이 적은 카페인 수준이나 디카페인 원두 기준으로도 추천해드릴 수 있습니다.",
    "profanity": "추천을 위해 취향이나 조건을 조금 더 차분히 입력해 주세요. 예: 산미 낮고 고소한 원두 추천해줘",
    "off_topic": "이 서비스는 커피 원두 추천과 커피 정보 안내를 위한 서비스입니다. 커피 취향이나 원두 관련 질문을 입력해 주세요.",
    "vague_preference": "어떤 맛을 선호하시나요? 산미가 낮은 원두, 고소한 원두, 라떼용 원두, 디카페인 원두 중에서 골라주시면 더 정확히 추천할 수 있습니다.",
}


def _contains(text, terms):
    return any(term.lower() in text.lower() for term in terms)


def classify_policy(user_input, conds=None):
    """정책 우선순위에 따라 (status, reason) 반환.
    status: allow / ask_followup / caution / reject / soft_reject
    reason: normal / vague_preference / health_claim / unsafe_caffeine / profanity / off_topic / prompt_injection
    """
    text = user_input or ""

    if _contains(text, prompt_injection_terms):
        return "reject", "prompt_injection"
    if _contains(text, health_claim_terms):
        return "caution", "health_claim"
    if _contains(text, unsafe_caffeine_reject):
        return "reject", "unsafe_caffeine"
    if _contains(text, unsafe_caffeine_caution):
        return "caution", "unsafe_caffeine"
    if _contains(text, profanity_terms):
        return "soft_reject", "profanity"
    if _contains(text, off_topic_terms):
        return "reject", "off_topic"

    # 모호한 입력: vague 단어가 있고, 구체적 조건이 전혀 추출되지 않은 경우만 ask_followup
    has_concrete = bool(conds and (
        conds.get("session_tags") or conds.get("decaf") is not None
        or conds.get("price_max") or conds.get("capacity")
        or any(conds.get(a) for a in ["acidity", "sweet", "bitter", "body"])
    ))
    if _contains(text, vague_terms) and not has_concrete:
        return "ask_followup", "vague_preference"

    return "allow", "normal"


def policy_message(status, reason):
    if reason == "unsafe_caffeine":
        return POLICY_MESSAGES["unsafe_caffeine_reject"] if status == "reject" \
            else POLICY_MESSAGES["unsafe_caffeine_caution"]
    return POLICY_MESSAGES.get(reason, "")


# ============================================================================
# 2. Intent 분류
# ============================================================================
REC_KW = ["추천", "골라", "추천해", "마실래", "원두 좀", "사고 싶", "찾아줘", "찾고 있"]
INFO_KW = ["뭐야", "무엇", "특징", "차이", "방식이", "어떻게", "뜻", "의미", "산지", "가공",
           "설명", "알려줘", "이란", "란 뭐", "무슨", "왜", "어떤 특징", "어디", "어울"]
CHIT_KW = ["안녕", "고마워", "반가워", "하이", "ㅎㅇ", "잘 지냈"]


def classify_intent(text, conds):
    t = text or ""
    rec = any(k in t for k in REC_KW)
    info = any(k in t for k in INFO_KW)
    if rec and info:
        return "mixed"
    if rec:
        return "recommendation"
    if info:
        return "info"
    if any(k in t for k in CHIT_KW):
        return "chitchat"
    # 명시 키워드는 없지만 취향 조건이 잡히면 추천 의도로 간주
    if conds.get("session_tags") or conds.get("decaf") is not None or \
       any(conds.get(a) for a in ["acidity", "sweet", "bitter", "body"]):
        return "recommendation"
    return "chitchat"


# ============================================================================
# 3. 자연어 조건 추출 (하드 필터 + 세션 향미태그)
# ============================================================================
LOW_WORDS = ["낮", "적", "약", "없", "연한", "가벼", "라이트", "마일드", "순한", "은은"]
HIGH_WORDS = ["높", "강", "많", "진한", "묵직", "풀바디", "센", "무거", "꽉"]

ATTR_KEYWORDS = {
    "acidity": ["산미", "신맛", "산도"],
    "sweet":   ["단맛", "달콤", "달달", "스위트", "당도"],
    "bitter":  ["쓴맛", "쓴", "비터", "쓰지"],
    "body":    ["바디", "바디감", "묵직", "진함", "질감"],
}


def _detect_level(text, keywords):
    for kw in keywords:
        i = text.find(kw)
        if i != -1:
            window = text[max(0, i - 3): i + len(kw) + 7]
            if any(w in window for w in LOW_WORDS):
                return "low"
            if any(w in window for w in HIGH_WORDS):
                return "high"
    return None


def _extract_price_max(text):
    base = None
    m = re.search(r"(\d+)\s*만\s*원?", text)
    if m:
        base = int(m.group(1)) * 10000
    m2 = re.search(r"([\d,]{4,})\s*원", text)
    if m2:
        val = int(m2.group(1).replace(",", ""))
        base = val if base is None else min(base, val)
    if base is None:
        return None
    if any(w in text for w in ["이하", "이내", "미만", "아래", "까지", "under", "정도"]):
        return base
    return base  # 가격이 언급되면 상한으로 간주


def _extract_capacity(text):
    s = text.lower().replace(" ", "")
    m = re.search(r"([\d.]+)kg", s)
    if m:
        return int(float(m.group(1)) * 1000)
    m = re.search(r"([\d.]+)g", s)
    if m:
        return int(float(m.group(1)))
    return None


def extract_conditions(text):
    """규칙 기반 조건 추출. (LLM 이 있으면 보강)"""
    conds = {
        "decaf": None,
        "price_max": None,
        "capacity": None,
        "acidity": None, "sweet": None, "bitter": None, "body": None,
        "session_tags": set(),
        "is_latte": False,
    }
    t = text or ""

    # 디카페인
    if any(k in t for k in ["디카페인 아닌", "디카페인 말고", "카페인 있는"]):
        conds["decaf"] = False
    elif any(k in t for k in ["디카페인", "카페인 없", "카페인없", "decaf"]):
        conds["decaf"] = True

    # 감각 수치 낮음/높음
    for attr, kws in ATTR_KEYWORDS.items():
        lvl = _detect_level(t, kws)
        if lvl:
            conds[attr] = lvl

    # 가격/용량
    conds["price_max"] = _extract_price_max(t)
    conds["capacity"] = _extract_capacity(t)

    # 현재 세션 향미태그
    conds["session_tags"] = R.extract_tags_from_text(t)

    # 라떼/우유음료 힌트 (이유 설명에만 사용)
    if any(k in t for k in ["라떼", "라테", "우유", "카페라떼", "밀크"]):
        conds["is_latte"] = True

    return conds


def llm_augment_conditions(text, conds, llm_fn):
    """LLM 으로 자연어를 구조화해 규칙 기반 결과의 '빈 곳'만 보강한다.
    (하드 조건을 LLM 이 임의로 뒤집지 않도록, 규칙 결과를 우선한다.)
    """
    if llm_fn is None:
        return conds
    system = (
        "너는 커피 취향 분석기다. 사용자 문장에서 취향을 추출해 JSON 만 출력하라. "
        "키: acidity, sweet, bitter, body 는 'low'/'high'/null, "
        "decaf 는 true/false/null, tags 는 "
        "[nutty,chocolate,caramel,fruity,citrus,floral,smoky,sweet,berry,almond,cacao] 중 부분집합 배열. "
        "설명/마크다운 없이 JSON 객체만."
    )
    out = llm_fn(system, text)
    if not out:
        return conds
    try:
        data = json.loads(re.sub(r"```json|```", "", out).strip())
    except Exception:
        return conds
    for a in ["acidity", "sweet", "bitter", "body"]:
        if conds.get(a) is None and data.get(a) in ("low", "high"):
            conds[a] = data[a]
    if conds.get("decaf") is None and isinstance(data.get("decaf"), bool):
        conds["decaf"] = data["decaf"]
    for tg in (data.get("tags") or []):
        if tg in R.FLAVOR_TAGS:
            conds["session_tags"].add(tg)
    return conds


# ============================================================================
# 4. LLM 헬퍼 (없으면 None 반환 -> 룰 기반 폴백)
# ============================================================================
def get_llm_fn():
    """사용 가능한 API 키에 따라 llm_fn(system, user)->str|None 반환. 없으면 None."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

            def _fn(system, user):
                try:
                    resp = client.messages.create(
                        model=model, max_tokens=600, system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                except Exception:
                    return None
            return _fn
        except Exception:
            pass

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            def _fn(system, user):
                try:
                    resp = client.chat.completions.create(
                        model=model, max_tokens=600,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                    )
                    return resp.choices[0].message.content
                except Exception:
                    return None
            return _fn
        except Exception:
            pass

    return None


# ============================================================================
# 5. 추천 이유 생성
# ============================================================================
def build_reason_rule(item, conds, session_tags):
    parts = []
    matched = set(session_tags) & set(item["tags"])
    if matched:
        kor = {"nutty": "고소함", "chocolate": "초콜릿", "caramel": "카라멜", "fruity": "과일",
               "citrus": "시트러스", "floral": "플로럴", "smoky": "스모키", "sweet": "달콤함",
               "berry": "베리", "almond": "아몬드", "cacao": "카카오"}
        labels = "·".join(kor.get(m, m) for m in matched)
        parts.append(f"요청하신 {labels} 계열 향미와 잘 맞습니다")
    for attr, label in [("acidity", "산미"), ("sweet", "단맛"), ("bitter", "쓴맛"), ("body", "바디감")]:
        if conds.get(attr) == "low":
            parts.append(f"{label}이 낮은 편이라 조건에 부합합니다")
        elif conds.get(attr) == "high":
            parts.append(f"{label}이 높은 편입니다")
    if conds.get("decaf") is True:
        parts.append("디카페인 원두입니다")
    if conds.get("is_latte"):
        parts.append("바디감이 있어 라떼·우유 음료와 잘 어울립니다")
    if item["long_profile_score"] >= 0.7:
        parts.append("평소 취향 프로필과도 잘 맞습니다")
    if item["past_behavior_score"] > 0:
        parts.append("이전에 좋게 반응하신 성향과 유사합니다")
    if not parts:
        parts.append("전반적인 취향 점수가 높아 추천합니다")
    reason = " · ".join(parts) + "."

    # 상품 디테일(향미 노트)과 리뷰를 참고해 설명을 덧붙인다
    extras = []
    fn = item.get("flavor_note")
    if fn and str(fn).strip() and str(fn).lower() != "nan":
        extras.append(f"향미 노트에 '{_short(fn, 50)}' 특징이 있고")
    rv = item.get("review")
    if rv and str(rv).strip() and str(rv).lower() != "nan":
        extras.append(f"실제 구매 리뷰에서도 \"{_short(rv, 45)}\" 같은 반응이 있었어요")
    if extras:
        reason += " " + " ".join(extras).rstrip("고") + "."
    return reason


def _short(text, n):
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s if len(s) <= n else s[:n] + "…"


def build_reason(item, conds, session_tags, llm_fn=None):
    base = build_reason_rule(item, conds, session_tags)
    if llm_fn is None:
        return base
    system = ("너는 친절한 바리스타다. 아래 정보를 바탕으로 이 원두를 추천하는 이유를 "
              "2~3문장의 자연스러운 한국어로 설명하라. 반드시 상품의 '향미 노트'와 '리뷰' 내용을 "
              "근거로 녹여서 말하고, 과장이나 건강 효능 표현은 쓰지 마라.")
    user = (f"원두: {item['bean_name']} ({item['brand_name']})\n"
            f"원산지/가공: {item.get('origin','-')} / {item.get('processing','-')}\n"
            f"향미 태그: {item['tags']}\n"
            f"향미 노트: {item.get('flavor_note','-')}\n"
            f"구매 리뷰: {item.get('review','-')}\n"
            f"매칭 근거(요약): {base}")
    out = llm_fn(system, user)
    return out.strip() if out else base


# ============================================================================
# 6. SQLite 저장 구조
# ============================================================================
DEFAULT_LONG_PROFILE = {"acidity": 3.0, "sweet": 3.0, "bitter": 3.0, "body": 3.0, "flavor": 3.0}


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            user_id TEXT PRIMARY KEY,
            acidity REAL, sweet REAL, bitter REAL, body REAL, flavor REAL,
            long_tag_profile_json TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS behavior_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT, product_id TEXT, bean_name TEXT,
            action TEXT, session_id INTEGER, created_at TEXT
        )
    """)
    conn.commit()


def load_profile(conn, user_id):
    row = conn.execute(
        "SELECT acidity,sweet,bitter,body,flavor,long_tag_profile_json FROM user_profile WHERE user_id=?",
        (user_id,)).fetchone()
    if row is None:
        return dict(DEFAULT_LONG_PROFILE), {}
    long_profile = {"acidity": row[0], "sweet": row[1], "bitter": row[2],
                    "body": row[3], "flavor": row[4]}
    try:
        long_tag = json.loads(row[5]) if row[5] else {}
    except Exception:
        long_tag = {}
    return long_profile, long_tag


def profile_exists(conn, user_id):
    """user_profile 에 행이 있으면 True (= 콜드스타트 아님)."""
    row = conn.execute("SELECT 1 FROM user_profile WHERE user_id=?", (user_id,)).fetchone()
    return row is not None


def save_profile(conn, user_id, long_profile, long_tag):
    conn.execute("""
        INSERT INTO user_profile (user_id,acidity,sweet,bitter,body,flavor,long_tag_profile_json,updated_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            acidity=excluded.acidity, sweet=excluded.sweet, bitter=excluded.bitter,
            body=excluded.body, flavor=excluded.flavor,
            long_tag_profile_json=excluded.long_tag_profile_json, updated_at=excluded.updated_at
    """, (user_id, long_profile["acidity"], long_profile["sweet"], long_profile["bitter"],
          long_profile["body"], long_profile["flavor"],
          json.dumps(long_tag, ensure_ascii=False), datetime.now().isoformat()))
    conn.commit()


def log_behavior(conn, user_id, product_id, bean_name, action, session_id):
    conn.execute("""
        INSERT INTO behavior_logs (user_id,product_id,bean_name,action,session_id,created_at)
        VALUES (?,?,?,?,?,?)
    """, (user_id, product_id, bean_name, action, session_id, datetime.now().isoformat()))
    conn.commit()


def get_logs_by_product(conn, user_id):
    rows = conn.execute(
        "SELECT product_id, action, session_id FROM behavior_logs WHERE user_id=?",
        (user_id,)).fetchall()
    out = {}
    for pid, action, sid in rows:
        out.setdefault(pid, []).append({"action": action, "session_id": sid})
    return out


def current_session_id(conn, user_id):
    row = conn.execute(
        "SELECT MAX(session_id) FROM behavior_logs WHERE user_id=?", (user_id,)).fetchone()
    base = row[0] if row and row[0] is not None else 0
    return int(base) + 1


def update_long_profile(conn, user_id, conds, long_profile, long_tag):
    """현재 세션에서 드러난 취향을 장기 프로필에 EMA 로 반영(점진 학습)."""
    target = {"low": 2.0, "high": 4.0}
    changed = False
    for attr in ["acidity", "sweet", "bitter", "body"]:
        lvl = conds.get(attr)
        if lvl in target:
            old = long_profile.get(attr, 3.0)
            long_profile[attr] = round(0.7 * old + 0.3 * target[lvl], 3)
            changed = True
    for tg in conds.get("session_tags") or []:
        long_tag[tg] = long_tag.get(tg, 0) + 1
        changed = True
    if changed:
        save_profile(conn, user_id, long_profile, long_tag)
    return long_profile, long_tag


# ============================================================================
# 7. 콜드스타트 온보딩 (첫 방문 시 비유로 풀어 1~5점 + 향미태그 필수 수집)
# ============================================================================
TAG_LABELS = {
    "nutty": "고소함/견과", "chocolate": "초콜릿", "caramel": "카라멜",
    "fruity": "과일", "citrus": "시트러스/상큼", "floral": "꽃향",
    "smoky": "스모키/다크", "sweet": "달콤함", "berry": "베리",
    "almond": "아몬드", "cacao": "카카오",
}
LABEL_TO_TAG = {v: k for k, v in TAG_LABELS.items()}

# 온보딩 직후 '취향 기반 첫 추천'을 자동으로 보여주기 위한 내부 트리거
FIRST_REC_SENTINEL = "__FIRST_REC__"

# 대화형 온보딩: 한 질문씩 길게 풀어서 묻고, 사용자가 '자유롭게 타이핑'해서 답한다.
# 답변 문장을 1~5 점수로 해석한다. (버튼 선택 아님)
ONBOARD_STEPS = [
    {
    "key": "acidity", "emoji": "🍋", "label": "산미",
    "question": "커피를 마셨을 때 **입안이 산뜻하게 느껴지는 정도**는 어느 쪽이 좋으세요?\n\n"
                "- **매우 낮음**: 산미가 거의 없고 고소하고 편안한 느낌\n"
                "- **낮음**: 은은하고 부드러운 느낌\n"
                "- **중간**: 산미와 고소함이 잘 어우러진 느낌\n"
                "- **높음**: 선명하고 산뜻한 느낌\n"
                "- **매우 높음**: 과일처럼 밝고 생생하게 산뜻한 느낌\n\n"
                "숫자로 답해도 되고, 느낌으로 편하게 말해주셔도 돼요.\n\n"
                "예: \"1점이요\", \"산미 거의 없는 게 좋아요\", \"산뜻하고 깔끔한 느낌이요\"",
},
    {
    "key": "sweet", "emoji": "🍯", "label": "단맛",
    "question": "커피에서 느껴지는 **자연스러운 단맛의 정도**는 어느 쪽이 좋으세요?\n\n"
                "- **매우 낮음**: 단맛이 거의 없고 깔끔하고 담백한 느낌\n"
                "- **낮음**: 은은한 곡물이나 견과류처럼 부드러운 느낌\n"
                "- **중간**: 캐러멜이나 밀크초콜릿처럼 자연스럽게 단맛이 느껴지는 느낌\n"
                "- **높음**: 꿀이나 당밀처럼 진하고 풍부한 단맛\n"
                "- **매우 높음**: 설탕 시럽처럼 달콤함이 선 명하게 느껴지는 느낌\n\n"
                "숫자로 답해도 되고, 느낌으로 편하게 말해주셔도 돼요.\n\n"
                "예: \"1점이요\", \"단맛은 거의 없어도 돼요\", \"꿀처럼 달콤한 느낌이 좋아요\"",
},
   {
    "key": "bitter", "emoji": "🍫", "label": "쓴맛",
    "question": "커피를 마셨을 때 **쌉싸름하게 느껴지는 정도**는 어느 쪽이 좋으세요?\n\n"
                "- **매우 낮음**: 쓴맛이 거의 없고 부드럽게 마실 수 있는 느낌\n"
                "- **낮음**: 기분 좋은 은은한 쌉싸름함이 살짝 느껴지는 느낌\n"
                "- **중간**: 일반적인 커피처럼 쓴맛과 고소함이 적당히 있는 느낌\n"
                "- **높음**: 진한 다크초콜릿처럼 깊고 쌉싸름한 느낌\n"
                "- **매우 높음**: 말차처럼 강하고 오래 남는 쌉싸름한 느낌\n\n"
                "숫자로 답해도 되고, 느낌으로 편하게 말해주셔도 돼요.\n\n"
                "예: \"1점이요\", \"쓴맛은 거의 없는 게 좋아요\", \"진하고 쌉싸름한 느낌이 좋아요\"",
},
    {
    "key": "body", "emoji": "🥛", "label": "바디감",
    "question": "커피를 마셨을 때 **입안에 남는 묵직한 정도**는 어느 쪽이 좋으세요?\n\n"
                "- **매우 낮음**: 물처럼 가볍고 깔끔하게 사라지는 느낌\n"
                "- **낮음**: 차처럼 가볍고 산뜻하게 넘어가는 느낌\n"
                "- **중간**: 일반적인 커피처럼 적당히 입안에 남는 느낌\n"
                "- **높음**: 진한 다크초콜릿처럼 묵직하게 남는 느낌\n"
                "- **매우 높음**: 말차처럼 강하고 오래 남는 진한 느낌\n\n"
                "숫자로 답해도 되고, 느낌으로 편하게 말해주셔도 돼요.\n\n"
                "예: \"1점이요\", \"가볍고 깔끔한 게 좋아요\", \"묵직하고 진한 느낌이 좋아요\"",
},
    {
    "key": "flavor",
    "emoji": "🌸",
    "label": "향미 강도",
    "question": "그 향이 커피에서 **어느 정도로 느껴지는 게 좋으세요?** "
                "은은하게 스쳐 지나가는 정도가 좋은지, 마셨을 때 향이 선명하게 느껴지는 쪽이 좋은지 알려주세요. "
                "(예: \"은은한 향이 좋아요\", \"향이 너무 강한 건 별로예요\", "
                "\"적당히 느껴지면 좋아요\", \"초콜릿 향이 진하게 났으면 좋겠어요\")",
},
]
TAGS_QUESTION = (
    "좋아요, 거의 다 됐어요! 먼저 커피에서 **가장 끌리는 향의 계열**을 자유롭게 적어주세요. "
    "이건 향의 강도가 아니라, 어떤 향을 좋아하는지 고르는 질문이에요.\n\n"
    "예를 들면 이런 느낌들이 있어요.\n"
    "- 고소한 계열: 견과류, 아몬드, 땅콩, 너티\n"
    "- 달콤한 계열: 초콜릿, 카카오, 코코아, 카라멜, 꿀, 브라운슈가\n"
    "- 과일 계열: 복숭아, 사과, 포도, 열대과일, 베리\n"
    "- 시트러스 계열: 레몬, 오렌지, 자몽처럼 상큼한 향\n"
    "- 꽃/차 계열: 자스민, 꽃향, 홍차 같은 은은한 향\n"
    "- 진한 계열: 다크초콜릿, 스모키, 묵직한 향\n\n"
    "여러 개를 같이 적어도 괜찮아요. "
    "예를 들어 \"고소하고 초콜릿 같은 향\", \"카라멜이랑 견과류\", "
    "\"상큼한 과일향\", \"꽃향은 별로고 고소한 쪽\"처럼 적어주시면 돼요. "
    "딱히 모르겠으면 \"잘 모르겠어요\"라고만 하셔도 됩니다."
)

TOTAL_ONBOARD_STEPS = len(ONBOARD_STEPS) + 1  # 감각 질문들 + 향미태그 질문 1개

# 답변 문장 → 1~5 해석용 사전
_STRONG_WORDS = ["아주", "엄청", "완전", "정말", "너무", "매우", "최고", "제일", "많이", "진짜", "꽉", "확"]
_NEUTRAL_WORDS = ["적당", "보통", "무난", "그냥", "상관없", "모르", "아무", "중간", "평범", "괜찮"]

_HIGH_DESC = {
    "acidity": [
        "새콤", "상큼", "시큼", "톡 쏘", "톡쏘", "시트러스", "레몬", "오렌지",
        "산뜻", "산미 좋", "신 거 좋", "신맛 좋", "신게 좋", "산미 있",
        "신 게 좋", "신거 좋"
    ],
    "sweet": [
        "달콤", "달달", "단맛 좋", "단 거 좋", "단게 좋", "꿀", "캐러멜",
        "카라멜", "달았으면", "단 게 좋", "달면"
    ],
    "bitter": [
        "쌉", "다크", "에스프레소", "진한 쓴", "쓴맛 좋", "쓴 거 좋",
        "쓴게 좋", "쓴 게 좋"
    ],
    "body": [
        "묵직", "진한", "꽉", "풀바디", "무거", "크리미", "라떼",
        "바디 좋", "꾸덕", "묵직한 거 좋"
    ],
    "flavor": [
        "향 강", "향이 강", "향 진", "향이 진", "진한 향", "강한 향",
        "향이 선명", "선명한 향", "향이 확", "향이 풍부", "풍부한 향",
        "향이 많이", "향이 뚜렷", "뚜렷한 향", "향이 잘 느껴"
    ],
}

_LOW_DESC = {
    "acidity": [
        "산미 없", "안 시", "안 셔", "안 신", "신 거 싫", "신맛 싫",
        "산미 싫", "밋밋", "산미 낮", "안 새콤", "신 거 별로",
        "신맛 별로", "안신", "부드러", "순한"
    ],
    "sweet": [
        "안 달", "안달", "담백", "무가당", "단 거 싫", "덜 달",
        "단맛 싫", "안 단", "단 거 별로", "달지 않", "단맛 없",
        "단맛은 없", "없어도", "안 달아"
    ],
    "bitter": [
        "안 써", "안 쓴", "쓴 거 싫", "덜 쓴", "안 쓰", "쓴 거 별로",
        "안쓴", "부드러", "순한", "쓴맛 적"
    ],
    "body": [
        "가벼", "가볍", "라이트", "물처럼", "산뜻", "연한", "바디 약",
        "무겁지", "묽", "깔끔", "가볍게"
    ],
    "flavor": [
        "은은", "은은한 향", "향 약", "향이 약", "향 별로", "향 적",
        "향이 적", "잔잔", "차분", "향 강하지", "향이 강하지",
        "향이 진하지", "강한 향 싫", "진한 향 싫", "향 너무 강한 건 별로"
    ],
}

_DISLIKE = ["싫", "별로", "안 좋", "못 마", "못마", "안 좋아", "안좋"]
_LIKE = ["좋아", "좋은", "선호", "원해", "끌려", "끌리", "마시고 싶", "최고"]


def parse_attribute_score(attr, text, llm_fn=None):
    """자유 답변 문장을 해당 감각의 1~5 점수로 해석한다. (LLM 있으면 우선, 없으면 규칙)"""
    t = (text or "").strip().lower()
    if not t:
        return 3

    # 1) 명시적 숫자/점수
    m = re.search(r"([1-5])\s*점", t) or re.search(r"(?<![0-9])([1-5])(?![0-9])", t)
    if m:
        return int(m.group(1))

    # 2) LLM 해석 (가능할 때)
    if llm_fn is not None:
        label = next((s["label"] for s in ONBOARD_STEPS if s["key"] == attr), attr)
        system = (f"사용자의 커피 '{label}' 선호 답변을 1~5 정수로만 답하라. "
                  "1=아주 약함/없음 선호, 3=보통, 5=아주 강함 선호. 숫자 하나만 출력.")
        out = llm_fn(system, text)
        if out:
            mm = re.search(r"[1-5]", out)
            if mm:
                return int(mm.group(0))

    # 3) 규칙 기반
    strong = any(w in t for w in _STRONG_WORDS)
    neutral = any(w in t for w in _NEUTRAL_WORDS)
    hi = any(w in t for w in _HIGH_DESC.get(attr, []))
    lo = any(w in t for w in _LOW_DESC.get(attr, []))
    like = any(w in t for w in _LIKE)
    dislike = any(w in t for w in _DISLIKE)

    # 방향 결정: 서술어(hi/lo) 우선, 없으면 호불호로 추정
    if lo and not hi:
        direction = "low"
    elif hi and not lo:
        direction = "high"
    elif dislike and not like:
        direction = "low"
    elif like and not dislike:
        direction = "high"
    else:
        direction = "neutral"

    if neutral and direction == "neutral":
        return 3
    if direction == "low":
        return 1 if strong else 2
    if direction == "high":
        return 5 if strong else 4
    return 3


_SCORE_PHRASE = {1: "거의 없는 쪽", 2: "낮은 편", 3: "적당한 정도", 4: "있는 편", 5: "강한 쪽"}


def _ack(attr, score):
    label = next((s["label"] for s in ONBOARD_STEPS if s["key"] == attr), attr)
    return f"네, **{label}**은 '{_SCORE_PHRASE[score]}'으로 기억해 둘게요. 👍"


def _reset_onboarding_state():
    for k in ["ob_step", "ob_answers", "ob_tags", "ob_chat"]:
        st.session_state.pop(k, None)


def render_onboarding(conn, user_id, llm_fn):
    """자유 답변형 챗봇 온보딩. 질문을 길게 던지고, 사용자가 타이핑한 답을 1~5로 해석한다."""
    ss = st.session_state
    ss.setdefault("ob_step", 0)
    ss.setdefault("ob_answers", {})
    n = len(ONBOARD_STEPS)

    # 대화 시작 시 첫 인사 + 첫 질문 적재
    if "ob_chat" not in ss:
        ss["ob_chat"] = [
            ("assistant",
             "안녕하세요! 회원님께 딱 맞는 원두를 찾아드리려고 해요. ☕\n\n"
             "버튼 고르실 필요 없이, 제가 묻는 말에 평소 말하듯 편하게 답해 주시면 돼요. "
             "그럼 시작할게요!"),
            ("assistant", f"{ONBOARD_STEPS[0]['emoji']} {ONBOARD_STEPS[0]['question']}"),
        ]

    st.subheader("☕ 처음 오셨네요! 입맛을 알아가는 짧은 대화예요")
    st.progress(min(ss["ob_step"], TOTAL_ONBOARD_STEPS) / TOTAL_ONBOARD_STEPS,
                text=f"{min(ss['ob_step'] + 1, TOTAL_ONBOARD_STEPS)} / {TOTAL_ONBOARD_STEPS} 단계")

    # 지금까지의 대화 렌더링
    for role, msg in ss["ob_chat"]:
        with st.chat_message(role):
            st.markdown(msg)

    # 자유 입력
    reply = st.chat_input("여기에 자유롭게 답해 주세요…")
    if not reply:
        return
    reply = reply.strip()
    ss["ob_chat"].append(("user", reply))
    step = ss["ob_step"]

    # 인사말은 온보딩 답변으로 처리하지 않음
    greeting_words = ["안녕", "안녕하세요", "하이", "ㅎㅇ", "반가워요", "반갑습니다"]
    if any(g in reply for g in greeting_words):
        ss["ob_chat"].append((
            "assistant",
            "안녕하세요! 😊 지금은 원두 취향을 알아가는 단계예요. "
            "방금 질문에 맞춰 산미가 좋은지, 싫은지, 적당한지 편하게 말씀해 주세요."
        ))
        st.rerun()

    #--- 감각 질문 단계 ---
    if step < n:
        attr = ONBOARD_STEPS[step]["key"]
        score = parse_attribute_score(attr, reply, llm_fn)
        ss["ob_answers"][attr] = score
        ss["ob_step"] = step + 1
        ack = _ack(attr, score)
        if step + 1 < n:
            nxt = ONBOARD_STEPS[step + 1]
            ss["ob_chat"].append(("assistant", f"{ack}\n\n{nxt['emoji']} {nxt['question']}"))
        else:
            ss["ob_chat"].append(("assistant", f"{ack}\n\n{TAGS_QUESTION}"))
        st.rerun()

    # --- 향미 태그 단계 (마지막) → 저장 ---
    else:
        tags = R.extract_tags_from_text(reply)
        long_profile = {s["key"]: float(ss["ob_answers"].get(s["key"], 3)) for s in ONBOARD_STEPS}
        long_tag = {t: 3 for t in tags}
        save_profile(conn, user_id, long_profile, long_tag)
        ss["onboarded"] = True
        ss["redo_onboard"] = False
        ss["last_query"] = FIRST_REC_SENTINEL   # 첫 방문 즉시 취향 기반 추천
        _reset_onboarding_state()
        if tags:
            tag_kor = ", ".join(TAG_LABELS.get(t, t) for t in tags)
            st.success(f"고마워요! '{tag_kor}' 계열까지 반영해서 취향을 저장했어요. 이제 추천해 드릴게요 ☕")
        else:
            st.success("취향을 저장했어요! 이제 원두를 추천해 드릴게요 ☕")
        st.rerun()


# ============================================================================
# 8. 데이터 로딩 (캐시)
# ============================================================================
SAMPLE_CSV = """bean_name,brand_name,capacity_g,price,image_url,flavor_note,product_origin,variety,processing,review_sum,decaf,blend_type,acidity,balance,bitter,body,sweet,flavor
고소한 브라질 산토스,빈브라더스,200g,"9,000",,"고소한 견과, 초콜릿",브라질,버번,내추럴,"고소하고 부드러워요",FALSE,싱글,2,3,3,4,3,3
케냐 AA 시트러스,커피리브레,200g,"13,500",,"밝은 시트러스, 베리",케냐,SL28,워시드,"산미가 상큼해요",FALSE,싱글,5,3,2,2,3,4
디카페인 콜롬비아,테라로사,200g,"11,800",,"부드러운 카라멜, 초콜릿",콜롬비아,카투라,워시드,"디카페인인데 맛있어요",TRUE,싱글,2,4,2,3,4,3
에티오피아 예가체프,프릳츠,200g,"15,000",,"꽃향, 플로럴, 시트러스",에티오피아,게이샤,워시드,"향이 화려해요",FALSE,싱글,5,3,1,2,4,5
다크 인도네시아 만델링,모모스,500g,"19,000",,"스모키, 다크 초콜릿",인도네시아,카티모르,내추럴,"묵직하고 진해요",FALSE,싱글,1,3,5,5,2,2
디카페인 고소 블렌드,엘카페,500g,"17,000",,"고소한 아몬드, 카라멜",브라질,블렌드,내추럴,"고소하고 단맛이 좋아요",TRUE,블렌드,2,4,3,4,4,3
과테말라 안티구아,커피몽타주,200g,"12,000",,"초콜릿, 캐러멜, 견과",과테말라,부르봉,워시드,"밸런스가 좋아요",FALSE,싱글,3,5,3,3,3,3
콜롬비아 베리 내추럴,센터커피,200g,"14,000",,"블루베리, 베리, 과일",콜롬비아,카스티요,내추럴,"베리향이 강해요",FALSE,싱글,4,3,2,3,4,4
"""


@st.cache_data(show_spinner=False)
def load_data():
    used_sample = False

    if not os.path.exists(CSV_PATH):
        df_raw = pd.read_csv(pd.io.common.StringIO(SAMPLE_CSV))
        used_sample = True
        path = None
    else:
        path = CSV_PATH
        try:
            df_raw = pd.read_csv(path)
        except UnicodeDecodeError:
            df_raw = pd.read_csv(path, encoding="cp949")

    df = R.preprocess(df_raw)
    return df, path, used_sample


@st.cache_resource(show_spinner=False)
def load_rag():
    pdf_paths = [p for p in PDF_PATHS if os.path.exists(p)]
    return RagIndex(pdf_paths), pdf_paths

# ============================================================================
# ============================================================================
# 9. Streamlit UI  (다크 프리미엄 · 그린/브라운 · 말풍선 챗 버전)
#    - 추천/온보딩/정책/의도분류/RAG 로직은 원본 그대로 유지했습니다.
#    - 바뀐 것: ① 다크 테마 CSS  ② 말풍선(_bubble)  ③ 입력칸을 위로 올려
#      대화·추천 결과가 모두 '입력칸 아래'에 표시되도록 순서만 재배치.
# ============================================================================
 
CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=Noto+Sans+KR:wght@400;500;700;800&display=swap');
 
/* ── 전체 배경 ─────────────────────────────────────────────── */
.stApp{
    background:
        radial-gradient(1100px 520px at 12% -5%, rgba(88,129,87,.20), transparent 60%),
        radial-gradient(900px 480px at 105% 105%, rgba(111,78,55,.24), transparent 55%),
        linear-gradient(165deg,#0d0b09 0%,#14110d 55%,#0b0a08 100%);
    color:#ece3d8; font-family:'Noto Sans KR',system-ui,sans-serif;
}
.block-container{ max-width:860px; padding-top:2.2rem; padding-bottom:4rem; }
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown{ color:#ece3d8; }
.stApp small, [data-testid="stCaptionContainer"]{ color:#9a8b7b !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"]{ color:#ece3d8 !important; }
 
/* 타이틀 / 캡션 */
h1,h2,h3,.stApp h1{ font-family:'Gowun Batang',serif; color:#ecd9b6; letter-spacing:-.5px; }
.subtle{ color:#9a8b7b; font-size:.92rem; margin:-4px 0 14px; }
 
/* ── 사이드바 다크 ─────────────────────────────────────────── */
[data-testid="stSidebar"]{ background:#14110d; border-right:1px solid rgba(236,217,182,.10); }
[data-testid="stSidebar"] *{ color:#cdbfb0; }
[data-testid="stSidebar"] h2{ color:#ecd9b6; }
 
/* ── 말풍선 ────────────────────────────────────────────────── */
.bubble-row{ display:flex; width:100%; margin:6px 0; }
.bubble-row.user{ justify-content:flex-end; }
.bubble-row.bot { justify-content:flex-start; }
.bubble{
    max-width:80%; padding:11px 15px; border-radius:18px;
    font-size:.93rem; line-height:1.6; word-break:break-word;
    box-shadow:0 6px 18px rgba(0,0,0,.38);
}
.bubble .who{ display:block; font-size:.7rem; opacity:.7; margin-bottom:3px; }
.bubble.user{ background:linear-gradient(135deg,#3a5a40,#588157); color:#f3f8f1 !important; border-bottom-right-radius:6px; }
.bubble.bot{ background:linear-gradient(135deg,#52392a,#6f4e37); color:#f3ece4 !important; border:1px solid rgba(236,217,182,.13); border-bottom-left-radius:6px; }
.bubble.user .who, .bubble.bot .who{ color:inherit; }
 
/* ── 입력 폼 / 버튼 ────────────────────────────────────────── */
.stTextInput input{
    background:#1b1712 !important; color:#f1e7d7 !important;
    border:1px solid rgba(236,217,182,.25) !important; border-radius:12px !important;
    padding:11px 14px !important;
}
.stTextInput input::placeholder{ color:#8a7c6d !important; }
.stButton button, .stForm button, [data-testid="stFormSubmitButton"] button{
    background:linear-gradient(135deg,#3a5a40,#588157) !important; color:#fff !important;
    border:none !important; border-radius:12px !important; font-weight:700 !important;
}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover{ filter:brightness(1.08); }
 
/* ── 원두 카드 (다크) ──────────────────────────────────────── */
.icon-box{
    display:flex;align-items:center;justify-content:center;
    height:100%;min-height:120px;font-size:46px;color:#fff;
    background:linear-gradient(135deg,#3a5a40,#6f4e37);border-radius:16px;
}
.coffee-card{
    background:linear-gradient(160deg,#1b1712,#221c15);
    border:1px solid rgba(236,217,182,.14);border-radius:18px;
    padding:16px 18px;box-shadow:0 8px 24px rgba(0,0,0,.42);
}
.coffee-title{ font-family:'Gowun Batang',serif; font-size:1.22rem; font-weight:700; color:#f1e7d7; }
.coffee-brand{ font-size:.84rem; color:#b09a86; margin-bottom:10px; }
.tag{
    display:inline-block;background:rgba(88,129,87,.18);color:#bcd4b6;
    border:1px solid rgba(88,129,87,.35);border-radius:999px;
    padding:3px 10px;margin:3px 4px 3px 0;font-size:.78rem;font-weight:600;
}
.tag.flavor{ background:rgba(169,116,91,.20); color:#e3c6a6; border-color:rgba(169,116,91,.4); }
.flavor-text{ margin-top:10px;font-size:.86rem;color:#cdbfb0;line-height:1.65; }
.score-badge{
    float:right;background:linear-gradient(135deg,#3a5a40,#6f4e37);color:#fff;
    border-radius:10px;padding:3px 11px;font-size:.8rem;font-weight:700;
}
.reason-box{
    margin-top:10px;background:rgba(88,129,87,.12);
    border-left:3px solid #588157;border-radius:10px;
    padding:10px 13px;font-size:.86rem;color:#d2e0cb;line-height:1.65;
}
.divider{ height:1px;margin:14px 0;background:linear-gradient(90deg,transparent,rgba(236,217,182,.25),transparent); }
 
[data-testid="stAlert"]{ border-radius:12px; }
</style>
"""
 
 
def _esc(s):
    """HTML 안전 이스케이프 + 줄바꿈 처리."""
    s = str(s)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br>"))
 
 
def _bubble(role, text):
    """대화 말풍선 출력. role='user'(초록·오른쪽) / 그 외(갈색·왼쪽)."""
    cls = "user" if role == "user" else "bot"
    who = "나" if role == "user" else "☕ 바리스타"
    st.markdown(
        f'<div class="bubble-row {cls}"><div class="bubble {cls}">'
        f'<span class="who">{who}</span>{_esc(text)}</div></div>',
        unsafe_allow_html=True)
 
 
def _score_dot(v):
    """1~5 값을 점(●○)으로 시각화."""
    n = int(round(max(1, min(5, v))))
    return "●" * n + "○" * (5 - n)
 
 
def render_card(conn, item, conds, session_tags, user_id, session_id,
                llm_fn, debug_mode, query_key):
    row = item["row"]
    acidity = round(float(row.get("acidity_n", 3)), 1)
    sweet = round(float(row.get("sweet_n", 3)), 1)
    body = round(float(row.get("body_n", 3)), 1)
    flavor_note = item.get("flavor_note")
    flavor_note = "-" if not flavor_note or str(flavor_note).lower() == "nan" else flavor_note
    price = f"{int(item['price']):,}" if pd.notna(item.get("price")) else "-"
    cap = f"{int(item['capacity'])}" if pd.notna(item.get("capacity")) else "-"
    decaf_badge = "🟢 디카페인" if item["decaf"] else "🔵 일반"
    tags_html = "".join(
        f'<span class="tag flavor">{TAG_LABELS.get(t, t)}</span>' for t in item["tags"])
 
    c1, c2 = st.columns([1, 4])
    with c1:
        if item.get("image_url") and str(item["image_url"]).startswith("http"):
            st.image(item["image_url"], use_container_width=True)
        else:
            st.markdown('<div class="icon-box">☕</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="coffee-card">'
            f'<span class="score-badge">매칭 {item["final_score"]:.2f}</span>'
            f'<div class="coffee-title">{item["bean_name"]}</div>'
            f'<div class="coffee-brand">{item["brand_name"]} · {item.get("origin","-")} · {item.get("processing","-")} · {decaf_badge}</div>'
            f'<span class="tag">🌱 산미 {_score_dot(acidity)}</span>'
            f'<span class="tag">🍯 단맛 {_score_dot(sweet)}</span>'
            f'<span class="tag">🥛 바디 {_score_dot(body)}</span>'
            f'{tags_html}'
            f'<div class="flavor-text"><b>향미 노트:</b> {flavor_note}<br><b>가격:</b> {price}원 / {cap}g</div>'
            f'</div>',
            unsafe_allow_html=True)
 
    # 추천 이유: 같은 추천 맥락에서는 재계산하지 않도록 캐시
    reason_key = f"reason::{query_key}::{item['product_id']}"
    if reason_key not in st.session_state:
        st.session_state[reason_key] = build_reason(item, conds, session_tags, llm_fn)
    st.markdown(
        f'<div class="reason-box">💡 <b>바리스타의 추천 이유</b> — {_esc(st.session_state[reason_key])}</div>',
        unsafe_allow_html=True)
 
    # 행동 버튼 (5종: 점수 보정에 cart/purchase_click 사용)
    b = st.columns(5)
    actions = [("💚 좋아요", "like"), ("💾 찜하기", "save"), ("🛒 장바구니", "cart"),
               ("👎 별로예요", "dislike"), ("🧾 구매하기", "purchase_click")]
    for col, (label, action) in zip(b, actions):
        if col.button(label, key=f"{action}_{query_key}_{item['product_id']}"):
            log_behavior(conn, user_id, item["product_id"], item["bean_name"], action, session_id)
            st.toast(f"{item['bean_name']} → {label} 반영!")
 
    if debug_mode:
        with st.expander("내부 계산 확인"):
            st.json({
                "long_profile_score": item["long_profile_score"],
                "long_tag_score": item["long_tag_score"],
                "past_behavior_score": item["past_behavior_score"],
                "session_tag_similarity": item["session_tag_similarity"],
                "final_score": item["final_score"],
            })
    st.write("")
 
 
def main():
    st.set_page_config(page_title="나만의 원두 취향 도우미", page_icon="☕", layout="centered")
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("☕ 나만의 원두 취향 도우미")
    st.markdown('<div class="subtle">취향을 말해주세요. 어울리는 원두를 골라드릴게요.</div>',
                unsafe_allow_html=True)
 
    conn = get_conn()
    init_db(conn)
    df, csv_path, used_sample = load_data()
    rag_index, pdf_paths = load_rag()
    llm_fn = get_llm_fn()
 
    # 세션 상태
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_demo"
    user_id = st.session_state.user_id
    if "session_id" not in st.session_state:
        st.session_state.session_id = current_session_id(conn, user_id)
    session_id = st.session_state.session_id
 
    # ---------------- 사이드바 (원본 유지) ----------------
    with st.sidebar:
        st.header("⚙️ 설정 / 필터")
        st.text_input("사용자 ID", key="user_id")
        debug_mode = st.toggle("🔧 Debug mode", value=False)
        if st.button("🔄 취향 다시 설정 (온보딩)"):
            st.session_state["redo_onboard"] = True
            _reset_onboarding_state()
            st.rerun()
 
        st.divider()
        st.subheader("하드 필터")
        decaf_only = st.checkbox("디카페인만 보기")
        gift = st.checkbox("선물용 (프리미엄 위주)")
 
        pmin, pmax = int(df["price_num"].min()), int(df["price_num"].max())
        price_range = st.slider("가격 범위(원)", pmin, max(pmax, pmin + 1),
                                (pmin, pmax), step=500)
        cmin, cmax = int(df["capacity_num"].min()), int(df["capacity_num"].max())
        capacity_range = st.slider("용량 범위(g)", cmin, max(cmax, cmin + 1),
                                   (cmin, cmax), step=50)
 
        st.divider()
        st.subheader("상태")
        st.write("LLM:", "✅ 연결됨" if llm_fn else "⚪ 룰 기반 폴백")
        st.write("상품 데이터:", "⚠️ 샘플(데모)" if used_sample else f"✅ {os.path.basename(csv_path)}")
        st.write("PDF 문서:", f"✅ {len(pdf_paths)}개" if pdf_paths else "⚪ 없음")
        st.caption(f"현재 세션 ID: {session_id}")
 
    if used_sample:
        st.warning("실제 CSV를 찾지 못해 내장 샘플 데이터로 동작 중입니다. "
                   "`data/coffee_bean_data(by crawling).csv` 를 두면 실제 데이터로 추천합니다.")
 
    sidebar = {
        "decaf_only": decaf_only,
        "gift": gift,
        "price_range": price_range,
        "capacity_range": capacity_range,
    }
 
    # ---------------- 콜드스타트 온보딩 (원본 유지) ----------------
    need_onboarding = (not profile_exists(conn, user_id)
                       and not st.session_state.get("onboarded")) \
        or st.session_state.get("redo_onboard")
    if need_onboarding:
        render_onboarding(conn, user_id, llm_fn)
        st.stop()   # 취향 수집 전에는 추천 입력을 막는다
 
        # ================= 입력칸 출력 함수 =================
    def render_ask_form():
        st.markdown('<div class="subtle">예) 디카페인이고 산미 낮은 고소한 라떼용 원두 추천해줘 / '
                    '워시드 가공이 뭐야?</div>', unsafe_allow_html=True)
        with st.form("ask_form", clear_on_submit=True):
            ci, cb = st.columns([6, 1])
            typed = ci.text_input("메시지", label_visibility="collapsed",
                                  placeholder="커피 추천이나 궁금한 점을 자유롭게 입력하세요…")
            sent = cb.form_submit_button("전송 ➤")
        if sent and typed and typed.strip():
            st.session_state["last_query"] = typed.strip()
            st.rerun()


    # ===== 이 아래부터는 말풍선/결과가 먼저 그려지고, 입력칸은 아래에 그려진다 =====
    user_input = st.session_state.get("last_query")
    if not user_input:
        _bubble("assistant", "안녕하세요! 어떤 원두를 찾으시나요? 취향을 말씀해 주시면 추천해 드릴게요 ☕")
        render_ask_form()
        return
 
    # ---- 첫 방문 자동 추천(온보딩 직후) ----
    first_rec = (user_input == FIRST_REC_SENTINEL)
    if first_rec:
        _bubble("assistant", "말씀해 주신 취향을 바탕으로 어울릴 만한 원두 3가지를 골라봤어요 ☕")
        conds = extract_conditions("")     # 명시 조건 없음 → 장기 취향으로 순위 결정
        status, reason, intent = "allow", "normal", "recommendation"
    else:
        _bubble("user", user_input)
        # ---- (3) 조건 추출 (LLM 보강) ----
        conds = extract_conditions(user_input)
        conds = llm_augment_conditions(user_input, conds, llm_fn)
        # ---- (1) 입력정책 ----
        status, reason = classify_policy(user_input, conds)
        # ---- (2) Intent ----
        intent = classify_intent(user_input, conds)
 
    # 상태 표시 (디버그일 때만 노출 — 챗봇 화면을 깔끔히)
    if debug_mode:
        badge = {"allow": "🟢", "caution": "🟡", "ask_followup": "🔵",
                 "soft_reject": "🟠", "reject": "🔴"}.get(status, "⚪")
        c1, c2 = st.columns(2)
        c1.metric("입력정책 상태", f"{badge} {status}")
        c2.metric("정책 사유 / Intent", f"{reason} / {intent}")
 
    # 정책에 따른 분기
    if status in ("reject", "soft_reject", "ask_followup"):
        _bubble("assistant", policy_message(status, reason))
        render_ask_form()
        return

    if status == "caution":
        _bubble("assistant", policy_message(status, reason))
 
    # ---- info / chitchat 처리 ----
    if intent == "chitchat":
        _bubble("assistant", "안녕하세요! 커피 원두 추천이나 커피 정보 질문을 입력해 주세요. "
                             "예: 산미 낮고 고소한 원두 추천해줘")
        render_ask_form()
        return
 
    if intent == "info":
        _bubble("assistant", "문서를 찾아 정리해 드릴게요 📖")
        answer, sources = answer_question(rag_index, user_input, llm_fn)
        st.write(answer)
        if sources:
            st.caption("출처: " + ", ".join(sources))
        st.caption("ℹ️ 정보 질문은 추천 점수를 계산하지 않습니다 (RAG 전용).")
        render_ask_form()
        return
 
    # ---- recommendation / mixed: 추천 실행 ----
    if intent == "mixed":
        _bubble("assistant", "관련 커피 정보부터 알려드릴게요 📖")
        answer, sources = answer_question(rag_index, user_input, llm_fn)
        st.write(answer)
        if sources:
            st.caption("출처: " + ", ".join(sources))
 
    # 장기 프로필 로드 + 현재 취향 반영(점진 학습)
    long_profile, long_tag = load_profile(conn, user_id)
    long_profile, long_tag = update_long_profile(conn, user_id, conds, long_profile, long_tag)
    logs_by_product = get_logs_by_product(conn, user_id)
 
    out = R.recommend(
        df=df, conds=conds, sidebar=sidebar,
        long_profile=long_profile, long_tag_profile=long_tag,
        logs_by_product=logs_by_product, current_session=session_id,
        top_k=3, sim_threshold=0.3,
    )
 
    _bubble("assistant", "🎯 오늘의 추천 원두 3가지예요!")
    if not out["results"]:
        st.error("조건을 만족하는 원두가 없습니다. 필터(가격/용량/디카페인)나 취향 조건을 완화해 보세요.")
    else:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        query_key = "first" if first_rec else user_input
        for item in out["results"]:
            render_card(conn, item, conds, out["session_tags"], user_id, session_id,
                        llm_fn, debug_mode, query_key)
 
    # ---- 내부 계산 확인 영역 ----
    with st.expander("🔍 내부 계산 확인 (조건 / 필터 / 점수)"):
        st.markdown("**추출된 현재 조건**")
        st.json({k: (sorted(v) if isinstance(v, set) else v) for k, v in conds.items()})
        st.markdown("**current_session_tags**")
        st.write(sorted(out["session_tags"]) or "(없음)")
        st.markdown("**적용된 하드 필터**")
        st.write(out["applied_filters"] or "(없음)")
        st.markdown("**필터 단계별 후보 수**")
        st.json(out["debug"])
        st.markdown("**장기 프로필 / 장기 태그 가중치**")
        st.json({"long_profile": long_profile, "long_tag_profile": long_tag})

    # ================= 입력칸 (말풍선/결과 아래) =================
    render_ask_form()
 
    # ---- Debug mode: 검증용 ----
    if debug_mode:
        with st.expander("🧪 Debug: 정책/필터 검증 패널", expanded=True):
            st.write("정책 분류 결과:", {"status": status, "reason": reason, "intent": intent})
            st.write("디카페인 필터 적용 시 결과 수:",
                     int((df["decaf_bool"] == True).sum()), "개 (전체 디카페인)")
            recent = conn.execute(
                "SELECT bean_name, action, session_id, created_at FROM behavior_logs "
                "WHERE user_id=? ORDER BY id DESC LIMIT 10", (user_id,)).fetchall()
            st.markdown("**최근 행동 로그**")
            st.dataframe(pd.DataFrame(recent, columns=["bean_name", "action", "session_id", "created_at"]),
                         use_container_width=True)
 
 
if __name__ == "__main__":
    main()
