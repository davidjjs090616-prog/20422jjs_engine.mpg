"""Auto MPG 데이터 로딩 및 공통 데이터 정의."""

from __future__ import annotations

import pandas as pd
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/davidjjs090616-prog/"
    "jjs-data/refs/heads/main/auto-mpg.data"
)

RAW_COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]

KM_PER_LITER_FACTOR = 0.4251437

FEATURE_COLUMNS = ["weight", "displacement", "horsepower"]
TARGET_COLUMN = "연비(km/L)"

FEATURE_LABELS = {
    "weight": "자동차 무게",
    "displacement": "배기량",
    "horsepower": "마력",
}

COLUMN_LABELS = {
    "mpg": "연비(MPG)",
    "cylinders": "실린더 수",
    "displacement": "배기량",
    "horsepower": "마력",
    "weight": "자동차 무게",
    "acceleration": "가속 성능",
    "model_year": "모델 연도",
    "origin": "제조 지역",
    "car_name": "자동차 이름",
    TARGET_COLUMN: TARGET_COLUMN,
}

ORIGIN_LABELS = {
    1: "미국",
    2: "유럽",
    3: "일본",
}


class DataLoadError(RuntimeError):
    """데이터를 불러오거나 해석하지 못했을 때 발생하는 예외."""


@st.cache_data(show_spinner="자동차 데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    """지정한 GitHub Raw URL에서 Auto MPG 데이터를 읽어 반환한다.

    원본 파일은 공백 구분 형식이며, car_name은 큰따옴표로 묶인
    문자열이므로 CSV 파서의 quotechar 처리를 사용한다.
    """
    try:
        df = pd.read_csv(
            DATA_URL,
            sep=r"\s+",
            header=None,
            names=RAW_COLUMNS,
            na_values=["?"],
            quotechar='"',
            engine="c",
        )
    except Exception as exc:  # 네트워크 오류와 파싱 오류를 함께 안내
        raise DataLoadError(
            "GitHub 데이터 주소에서 자동차 데이터를 불러오지 못했습니다. "
            "인터넷 연결과 GitHub 저장소의 파일 경로를 확인해 주세요. "
            f"원인: {exc}"
        ) from exc

    if list(df.columns) != RAW_COLUMNS:
        raise DataLoadError("데이터의 열 구성이 예상한 Auto MPG 형식과 다릅니다.")

    if df.empty:
        raise DataLoadError("데이터 파일은 열렸지만 자동차 데이터가 비어 있습니다.")

    numeric_columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 원본의 70~82를 1970~1982년으로 바꾸어 표시한다.
    df["model_year"] = df["model_year"] + 1900

    # 실린더 수는 범주형 특성이므로 숫자 크기의 연속적 관계로 해석하지 않는다.
    # 데이터 표에는 원래 개수를 유지하고 그래프/상관관계에서 범주형으로 처리한다.
    df["origin"] = df["origin"].map(ORIGIN_LABELS).fillna("기타")
    df["car_name"] = df["car_name"].astype("string").str.strip()

    df[TARGET_COLUMN] = df["mpg"] * KM_PER_LITER_FACTOR
    return df


def get_display_data(df: pd.DataFrame) -> pd.DataFrame:
    """한국어 열 이름으로 바꾼 탐색용 데이터프레임을 반환한다."""
    display_df = df.copy()
    display_df = display_df.rename(columns=COLUMN_LABELS)
    return display_df
