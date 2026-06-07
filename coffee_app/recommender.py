# -*- coding: utf-8 -*-
"""
recommender.py
----------------
상품 데이터 전처리, 하드 필터, 현재 세션 향미태그 필터(자카드),
장기 프로필 점수 / 장기 태그 점수 / 과거 행동 점수, 최종 점수식을 담당한다.

핵심 원칙
- 현재 사용자가 말한 조건은 '필터'로 먼저 보장한다 (점수 평균에 섞지 않는다).
- 장기 취향은 남은 후보의 '순위 계산'에 사용한다.
- 과거 행동은 추천 점수를 '보정'한다.
- review_score(review_sum)는 추천 점수 계산에 사용하지 않는다.
"""

import re
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. 향미 태그 정의 및 한국어 키워드 매핑
# ----------------------------------------------------------------------------
FLAVOR_TAGS = [
    "nutty", "chocolate", "caramel", "fruity", "citrus",
    "floral", "smoky", "sweet", "berry", "almond", "cacao",
]

TAG_KEYWORD_MAP = {
    "nutty":     ["고소", "견과", "아몬드", "너티", "땅콩", "nutty", "nut"],
    "chocolate": ["초콜릿", "초코", "카카오", "코코아", "chocolate", "cocoa"],
    "caramel":   ["카라멜", "캐러멜", "caramel"],
    "fruity":    ["과일", "복숭아", "열대과일", "사과", "포도", "fruit", "fruity", "tropical"],
    "citrus":    ["시트러스", "레몬", "오렌지", "자몽", "citrus", "lemon", "orange"],
    "floral":    ["꽃", "플로럴", "자스민", "floral", "jasmine"],
    "smoky":     ["스모키", "탄맛", "다크", "smoky", "dark"],
    "sweet":     ["달콤", "꿀", "브라운슈가", "설탕", "sweet", "honey", "sugar"],
    "berry":     ["베리", "블루베리", "딸기", "berry", "blueberry", "strawberry"],
    "almond":    ["아몬드", "almond"],
    "cacao":     ["카카오", "카카오닙스", "cacao", "cocoa nibs"],
}

# 점수 계산에 사용하는 수치형 감각 컬럼 (정규화 후 1~5 스케일)
SENSORY_COLS = ["acidity", "sweet", "bitter", "body", "flavor"]

# 과거 행동 가중치
BEHAVIOR_WEIGHTS = {
    "cart":           0.35,
    "save":           0.30,
    "like":           0.25,
    "dislike":       -0.50,
    "purchase_click": -0.80,   # 이미 구매로 이어진 상품은 반복 추천을 줄인다
}

# 하드 필터 임계값 (정규화된 1~5 스케일 기준)
LOW_THRESHOLD = 2.5
HIGH_THRESHOLD = 3.5


# ----------------------------------------------------------------------------
# 2. 전처리 유틸
# ----------------------------------------------------------------------------
def parse_price(v):
    """'9,000' / '9000원' / '11,800' -> 숫자만 추출하여 int."""
    if pd.isna(v):
        return np.nan
    digits = re.sub(r"[^0-9]", "", str(v))
    return int(digits) if digits else np.nan


def parse_capacity(v):
    """'200g' -> 200, '500g' -> 500, '1kg' -> 1000."""
    if pd.isna(v):
        return np.nan
    s = str(v).lower().replace(" ", "")
    m = re.search(r"([\d.]+)\s*kg", s)
    if m:
        return int(float(m.group(1)) * 1000)
    m = re.search(r"([\d.]+)\s*g", s)
    if m:
        return int(float(m.group(1)))
    digits = re.sub(r"[^0-9]", "", s)
    return int(digits) if digits else np.nan


def parse_decaf(v):
    """TRUE/true/1/yes -> True, 그 외 -> False."""
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"true", "t", "1", "1.0", "yes", "y", "decaf", "디카페인", "o"}


def _normalize_to_1_5(series):
    """임의 스케일의 수치 컬럼을 1~5 스케일로 정규화한다.
    원본 스케일을 알 수 없어도 점수식(abs diff / 4)과 임계값이 일관되게 동작하도록 한다.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([3.0] * len(s), index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([3.0] * len(s), index=s.index)
    norm = 1.0 + 4.0 * (s - mn) / (mx - mn)
    return norm.fillna(3.0)


def extract_tags_from_text(text):
    """텍스트에서 향미 태그 집합을 추출한다."""
    if not text:
        return set()
    low = str(text).lower()
    tags = set()
    for tag, kws in TAG_KEYWORD_MAP.items():
        for kw in kws:
            if kw.lower() in low:
                tags.add(tag)
                break
    return tags


def _extract_product_tags(row):
    """상품 1건의 향미 태그를 flavor_note, review_sum, product_origin, processing 등에서 추출."""
    src_cols = ["flavor_note", "review_sum", "product_origin", "processing", "flavor"]
    text = " ".join(str(row.get(c, "")) for c in src_cols)
    return extract_tags_from_text(text)


def preprocess(df):
    """원본 CSV DataFrame을 추천 가능한 형태로 전처리한다."""
    df = df.copy()

    # 컬럼 누락 방어
    expected = [
        "bean_name", "brand_name", "capacity_g", "price", "image_url",
        "flavor_note", "product_origin", "variety", "processing", "review_sum",
        "decaf", "blend_type", "acidity", "balance", "bitter", "body",
        "sweet", "flavor",
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    df = df.reset_index(drop=True)
    df["product_id"] = df.index.astype(str)

    df["price_num"] = df["price"].apply(parse_price)
    df["capacity_num"] = df["capacity_g"].apply(parse_capacity)
    df["decaf_bool"] = df["decaf"].apply(parse_decaf)

    # 감각 수치 정규화 (1~5)
    for col in SENSORY_COLS:
        df[col + "_n"] = _normalize_to_1_5(df[col])

    # 향미 태그
    df["tags"] = df.apply(_extract_product_tags, axis=1)

    # 가격 결측 처리(필터/정렬 안정성)
    df["price_num"] = df["price_num"].fillna(df["price_num"].median())
    df["capacity_num"] = df["capacity_num"].fillna(df["capacity_num"].median())

    return df


# ----------------------------------------------------------------------------
# 3. 하드 필터
# ----------------------------------------------------------------------------
def apply_hard_filters(df, conds, sidebar):
    """현재 세션에서 명확히 말한 조건 + 사이드바 조건을 필터로 먼저 보장한다.
    반환: (filtered_df, applied_filters: list[str])
    """
    work = df.copy()
    applied = []

    # --- 디카페인 ---
    decaf_req = conds.get("decaf")
    if sidebar.get("decaf_only"):
        decaf_req = True
    if decaf_req is True:
        work = work[work["decaf_bool"] == True]
        applied.append("decaf = True (디카페인만)")
    elif decaf_req is False:
        work = work[work["decaf_bool"] == False]
        applied.append("decaf = False (일반 원두만)")

    # --- 가격 ---
    pmax = conds.get("price_max")
    if pmax is not None:
        work = work[work["price_num"] <= pmax]
        applied.append(f"price <= {pmax:,}원")
    sb_pmin, sb_pmax = sidebar.get("price_range", (None, None))
    if sb_pmin is not None:
        work = work[work["price_num"] >= sb_pmin]
    if sb_pmax is not None:
        work = work[work["price_num"] <= sb_pmax]
    if sb_pmin is not None or sb_pmax is not None:
        applied.append(f"가격 범위 {sb_pmin:,} ~ {sb_pmax:,}원")

    # --- 용량 ---
    cap = conds.get("capacity")
    if cap is not None:
        work = work[work["capacity_num"] == cap]
        applied.append(f"capacity = {cap}g")
    sb_cmin, sb_cmax = sidebar.get("capacity_range", (None, None))
    if sb_cmin is not None:
        work = work[work["capacity_num"] >= sb_cmin]
    if sb_cmax is not None:
        work = work[work["capacity_num"] <= sb_cmax]
    if sb_cmin is not None or sb_cmax is not None:
        applied.append(f"용량 범위 {sb_cmin} ~ {sb_cmax}g")

    # --- 감각 수치 (낮음/높음) ---
    labels = {"acidity": "산미", "sweet": "단맛", "bitter": "쓴맛", "body": "바디감"}
    for attr, label in labels.items():
        lvl = conds.get(attr)
        if lvl == "low":
            work = work[work[attr + "_n"] <= LOW_THRESHOLD]
            applied.append(f"{label} 낮음 ({attr}_n <= {LOW_THRESHOLD})")
        elif lvl == "high":
            work = work[work[attr + "_n"] >= HIGH_THRESHOLD]
            applied.append(f"{label} 높음 ({attr}_n >= {HIGH_THRESHOLD})")

    # --- 선물용 (전용 컬럼이 없어 가격 중앙값 이상을 프리미엄으로 간주) ---
    if sidebar.get("gift"):
        med = df["price_num"].median()
        work = work[work["price_num"] >= med]
        applied.append(f"선물용(가격 >= 중앙값 {med:,.0f}원)")

    return work, applied


# ----------------------------------------------------------------------------
# 4. 현재 세션 향미태그 필터 (자카드 유사도)
# ----------------------------------------------------------------------------
def jaccard(a, b):
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def apply_session_tag_filter(df, session_tags, threshold=0.3):
    """현재 세션 태그와 상품 태그의 자카드 유사도가 threshold 이상인 후보만 유지.
    세션 태그가 없으면 필터를 건너뛴다.
    반환: (filtered_df, sims: dict[index -> sim])
    """
    sims = {}
    if not session_tags:
        return df, sims
    keep = []
    for idx, row in df.iterrows():
        sim = jaccard(session_tags, row["tags"])
        sims[idx] = sim
        if sim >= threshold:
            keep.append(idx)
    return df.loc[keep], sims


# ----------------------------------------------------------------------------
# 5. 점수 계산
# ----------------------------------------------------------------------------
def long_profile_score(row, long_profile):
    """수치형 장기 취향과 상품 감각값의 일치도. score_each = 1 - abs/4, 평균."""
    scores = []
    for k in SENSORY_COLS:
        uv = long_profile.get(k)
        pv = row.get(k + "_n")
        if uv is None or pd.isna(pv):
            continue
        scores.append(1.0 - abs(float(uv) - float(pv)) / 4.0)
    return float(np.mean(scores)) if scores else 0.0


def long_tag_score(product_tags, long_tag_profile):
    """장기 향미태그 누적 가중치 기반 점수."""
    total = sum(long_tag_profile.values())
    if total <= 0 or not product_tags:
        return 0.0
    return sum(long_tag_profile.get(t, 0) for t in product_tags) / total


def past_behavior_score(logs, current_session):
    """과거 행동 점수. session_decay = 0.5 ** (session_gap / 3)."""
    total = 0.0
    for log in logs:
        w = BEHAVIOR_WEIGHTS.get(log["action"], 0.0)
        gap = max(0, current_session - int(log.get("session_id", current_session)))
        decay = 0.5 ** (gap / 3.0)
        total += w * decay
    return total


# 최종 점수 기본 가중치
# 재방문자 기준: 장기 취향 45%, 장기 향미태그 35%, 과거 행동 20%
FINAL_WEIGHTS = {
    "long_profile": 0.45,
    "long_tag": 0.35,
    "past_behavior": 0.20,
}


def resolve_weights(has_behavior):
    """
    행동 로그 유무에 따라 최종 점수 가중치를 조정한다.

    1) 행동 로그가 있는 경우
       - 기존 구조 그대로 사용
       - long_profile 0.45
       - long_tag 0.35
       - past_behavior 0.20

    2) 행동 로그가 없는 경우, 즉 첫 방문/콜드스타트
       - past_behavior는 계산식에서 제외
       - 빠진 0.20 가중치를 long_profile과 long_tag에 기존 비율대로 재분배
       - long_profile 0.5625
       - long_tag 0.4375
       - past_behavior 0.0
    """
    weights = dict(FINAL_WEIGHTS)

    if not has_behavior:
        behavior_weight = weights["past_behavior"]
        base_weight = weights["long_profile"] + weights["long_tag"]

        if base_weight > 0:
            weights["long_profile"] += behavior_weight * (
                weights["long_profile"] / base_weight
            )
            weights["long_tag"] += behavior_weight * (
                weights["long_tag"] / base_weight
            )

        weights["past_behavior"] = 0.0

    return weights


def final_score(lp, lt, pb, weights):
    """
    최종 추천 점수를 계산한다.

    lp: 장기 취향 프로필 점수
    lt: 장기 향미태그 점수
    pb: 과거 행동 점수

    첫 방문자는 resolve_weights(False)를 통해
    past_behavior 가중치가 0으로 처리되므로,
    행동 점수가 최종 점수에 반영되지 않는다.
    """
    return (
        weights["long_profile"] * lp
        + weights["long_tag"] * lt
        + weights["past_behavior"] * pb
    )


# ----------------------------------------------------------------------------
# 6. 추천 파이프라인
# ----------------------------------------------------------------------------
def recommend(df, conds, sidebar, long_profile, long_tag_profile,
              logs_by_product, current_session, top_k=5, sim_threshold=0.3):
    """전체 추천 파이프라인 실행.
    반환: dict {results, applied_filters, debug}
    """
    debug = {"total": len(df)}

    # 5. 하드 필터
    work, applied = apply_hard_filters(df, conds, sidebar)
    debug["after_hard_filter"] = len(work)

    # 6. 현재 세션 향미태그 필터
    session_tags = conds.get("session_tags") or set()
    work, sims = apply_session_tag_filter(work, session_tags, threshold=sim_threshold)
    debug["after_session_tag_filter"] = len(work)
    debug["session_tags"] = sorted(session_tags)

    # 7. 최종 점수 계산
    # 행동 로그가 하나도 없으면 첫 방문자로 보고,
    # past_behavior 가중치를 빼서 long_profile / long_tag에 재분배한다.
    has_behavior = bool(logs_by_product) and any(bool(v) for v in logs_by_product.values())
    weights = resolve_weights(has_behavior)

    debug["used_behavior"] = has_behavior
    debug["weights"] = {
        "long_profile": round(weights["long_profile"], 4),
        "long_tag": round(weights["long_tag"], 4),
        "past_behavior": round(weights["past_behavior"], 4),
    }

    results = []

    for idx, row in work.iterrows():
        lp = long_profile_score(row, long_profile)
        lt = long_tag_score(row["tags"], long_tag_profile)

        # 상품별 과거 행동 로그
        product_logs = logs_by_product.get(row["product_id"], []) if logs_by_product else []
        pb = past_behavior_score(product_logs, current_session)

        # 첫 방문이면 weights["past_behavior"]가 0이므로 pb는 최종 점수에 반영되지 않음
        fs = final_score(lp, lt, pb, weights)

        results.append({
            "index": idx,
            "product_id": row["product_id"],
            "bean_name": row.get("bean_name"),
            "brand_name": row.get("brand_name"),
            "image_url": row.get("image_url"),
            "price": row.get("price_num"),
            "capacity": row.get("capacity_num"),
            "origin": row.get("product_origin"),
            "processing": row.get("processing"),
            "decaf": bool(row.get("decaf_bool")),
            "tags": sorted(row["tags"]),
            "flavor_note": row.get("flavor_note"),
            "review": row.get("review_sum"),
            "variety": row.get("variety"),

            "long_profile_score": round(lp, 4),
            "long_tag_score": round(lt, 4),
            "past_behavior_score": round(pb, 4),
            "session_tag_similarity": round(sims.get(idx, None), 4) if sims else None,
            "final_score": round(fs, 4),

            "row": row,
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)

    return {
        "results": results[:top_k],
        "applied_filters": applied,
        "debug": debug,
        "session_tags": session_tags,
    }