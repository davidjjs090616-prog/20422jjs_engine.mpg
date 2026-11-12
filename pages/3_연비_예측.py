from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    KM_PER_LITER_FACTOR,
    DataLoadError,
    load_data,
)
from utils.ui import apply_app_styles

st.set_page_config(page_title="연비 예측 · 자동차 연비 예측 실습실", page_icon="🚘", layout="wide")
apply_app_styles()

st.title("🚘 3. 자동차 연비 예측하기")
st.write("새로운 자동차의 특성을 직접 입력하고 선형 회귀 모델이 예상하는 연비를 확인해 봅니다.")

try:
    df = load_data()
except DataLoadError as exc:
    st.error(str(exc))
    st.stop()

bundle = st.session_state.get("auto_mpg_model_bundle")
if bundle is None:
    st.warning("먼저 2. 선형 회귀 모델 만들기 페이지에서 모델을 학습해 주세요.")
    st.page_link("pages/2_선형_회귀.py", label="선형 회귀 모델 학습 페이지로 이동", icon="📈")
    st.stop()

st.subheader("① 새로운 자동차의 특성 입력")
st.caption("입력 범위는 원본 자동차 데이터의 최솟값과 최댓값 안에서 설정됩니다.")

slider_settings = {
    "weight": {"step": 1.0, "format": "%.0f"},
    "displacement": {"step": 1.0, "format": "%.1f"},
    "horsepower": {"step": 1.0, "format": "%.1f"},
}

with st.form("auto_mpg_prediction_form"):
    input_columns = st.columns(3, gap="large")
    input_values: dict[str, float] = {}
    for column_container, feature in zip(input_columns, FEATURE_COLUMNS):
        series = df[feature].dropna()
        minimum = float(series.min())
        maximum = float(series.max())
        default = float(series.median())
        settings = slider_settings[feature]
        with column_container:
            st.markdown(f"**{FEATURE_LABELS[feature]}**")
            input_values[feature] = st.slider(
                FEATURE_LABELS[feature],
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=settings["step"],
                format=settings["format"],
                key=f"prediction_{feature}",
                help=f"원본 데이터 범위: {minimum:g} ~ {maximum:g}",
            )
    submitted = st.form_submit_button("연비 예측하기", type="primary", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
        predicted_km_l = float(bundle["model"].predict(input_df)[0])
        predicted_mpg = predicted_km_l / KM_PER_LITER_FACTOR
        st.session_state["auto_mpg_prediction_result"] = {
            "inputs": input_values.copy(),
            "predicted_km_l": predicted_km_l,
            "predicted_mpg": predicted_mpg,
        }
    except Exception as exc:
        st.error(f"연비 예측 중 문제가 발생했습니다: {exc}")
        st.stop()

result = st.session_state.get("auto_mpg_prediction_result")
if result is not None:
    if result["inputs"] == input_values:
        st.divider()
        st.subheader("② 예측 결과")
        result_col1, result_col2 = st.columns(2)
        result_col1.metric("예상 자동차 연비", f"{result['predicted_km_l']:.2f} km/L")
        result_col2.metric("MPG로 환산한 값", f"{result['predicted_mpg']:.2f} MPG")
        st.success("2페이지에서 학습한 선형 회귀 모델을 사용해 예측했습니다.")
        st.markdown("**이번 예측에 사용한 입력값**")
        input_summary = pd.DataFrame(
            {
                "독립변수": [FEATURE_LABELS[feature] for feature in FEATURE_COLUMNS],
                "입력값": [result["inputs"][feature] for feature in FEATURE_COLUMNS],
            }
        )
        st.dataframe(input_summary, use_container_width=True, hide_index=True)
    else:
        st.info("입력값이 이전 예측과 달라졌습니다. 변경된 값으로 다시 예측하려면 '연비 예측하기' 버튼을 눌러 주세요.")
else:
    st.info("자동차의 특성을 입력한 뒤 '연비 예측하기' 버튼을 누르면 예상 연비가 표시됩니다.")

st.caption(
    "참고: 이 모델은 1970~1982년 자동차 데이터로 학습되었습니다. "
    "입력 범위 안의 값을 사용하더라도 실제 최신 자동차의 연비와 차이가 있을 수 있습니다."
)
