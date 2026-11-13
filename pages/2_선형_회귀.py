
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from utils.data_loader import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    DataLoadError,
    TARGET_COLUMN,
    load_data,
)
from utils.model_utils import train_linear_regression
from utils.ui import apply_app_styles

st.set_page_config(
    page_title="선형 회귀 · 자동차 연비 예측 실습실",
    page_icon="📈",
    layout="wide"
)
apply_app_styles()

st.title("📈 2. 선형 회귀 모델 만들기")
st.write(
    "자동차의 무게, 배기량, 마력을 이용하여 선형 회귀 모델을 학습하고 "
    "회귀식의 의미를 알아봅니다."
)

try:
    df = load_data()
except DataLoadError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("① 독립변수와 종속변수")

x_col, y_col = st.columns(2)

with x_col:
    st.markdown("**독립변수(X)**")
    st.write("- 자동차 무게 (`weight`)")
    st.write("- 배기량 (`displacement`)")
    st.write("- 마력 (`horsepower`)")

with y_col:
    st.markdown("**종속변수(Y)**")
    st.write("- 연비(km/L)")
    st.caption("이 실습에서는 X와 Y를 변경하지 않습니다.")

st.info(
    "자동차 무게·배기량·마력 세 변수를 사용합니다. "
    "결측값 대체와 선형 회귀는 하나의 Pipeline 안에서 처리합니다."
)

st.subheader("② 데이터 분할 및 모델 학습")
st.write(
    "학습 데이터 80%, 테스트 데이터 20%, "
    "random_state=42 조건을 사용합니다."
)

if st.button(
    "선형 회귀 모델 학습하기",
    type="primary",
    use_container_width=True
):
    try:
        st.session_state["auto_mpg_model_bundle"] = train_linear_regression(df)
        st.session_state.pop("auto_mpg_prediction_result", None)
        st.success(
            "선형 회귀 모델 학습을 완료했습니다. "
            "아래에서 회귀 결과를 확인하세요."
        )
    except Exception as exc:
        st.error(f"모델 학습 중 문제가 발생했습니다: {exc}")
        st.stop()

bundle = st.session_state.get("auto_mpg_model_bundle")

if bundle is None:
    st.warning(
        "아직 학습된 모델이 없습니다. "
        "위의 '선형 회귀 모델 학습하기' 버튼을 눌러 주세요."
    )
    st.stop()

model = bundle["model"]
regressor = model.named_steps["linear_regression"]
coefficients = regressor.coef_
intercept = float(regressor.intercept_)

st.divider()
st.subheader("③ 회귀 모델 결과")

info_cols = st.columns(3)
info_cols[0].metric("학습 데이터", f"{bundle['train_count']:,}대")
info_cols[1].metric("테스트 데이터", f"{bundle['test_count']:,}대")
info_cols[2].metric("사용한 독립변수", f"{len(FEATURE_COLUMNS)}개")

coefficient_df = pd.DataFrame(
    {
        "독립변수": [
            FEATURE_LABELS[column] for column in FEATURE_COLUMNS
        ],
        "원본 변수명": FEATURE_COLUMNS,
        "회귀 계수": [
            float(value) for value in coefficients
        ],
        "영향 방향": [
            "증가 방향" if value > 0
            else "감소 방향" if value < 0
            else "변화 없음"
            for value in coefficients
        ],
    }
)

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("**회귀 계수**")
    st.dataframe(
        coefficient_df,
        use_container_width=True,
        hide_index=True
    )
    st.metric("절편", f"{intercept:.6f}")

with right:
    st.markdown("**학습 데이터로 계산된 회귀식**")

    equation_terms = [f"{intercept:.6f}"]

    for feature, coefficient in zip(FEATURE_COLUMNS, coefficients):
        equation_terms.append(
            f"({float(coefficient):+.6f} × {FEATURE_LABELS[feature]})"
        )

    equation = "예측 연비(km/L) = " + " ".join(equation_terms)

    st.code(equation, language="text")

    st.markdown(
        "회귀 계수는 다른 독립변수의 값이 같을 때 해당 변수가 "
        "1단위 증가하면 예측 연비가 얼마나 변하는지를 나타냅니다. "
        "양수이면 증가 방향, 음수이면 감소 방향입니다."
    )

    horsepower_median = bundle["imputation_medians"]["horsepower"]

    st.caption(
        f"마력 결측값은 학습 데이터에서 계산한 중앙값 "
        f"{horsepower_median:.2f}으로 대체했습니다."
    )

st.divider()
st.subheader("④ 회귀 결과 시각화")

results = bundle["test_results"].copy()

scatter_col, residual_col = st.columns(2, gap="large")

with scatter_col:
    st.markdown("**실제 연비와 예측 연비 비교**")

    actual_pred_fig = px.scatter(
        results,
        x="실제 연비(km/L)",
        y="예측 연비(km/L)",
        hover_data=["자동차 이름", "잔차(km/L)"],
        labels={
            "실제 연비(km/L)": "실제 연비(km/L)",
            "예측 연비(km/L)": "예측 연비(km/L)"
        },
    )

    lower = float(
        min(
            results["실제 연비(km/L)"].min(),
            results["예측 연비(km/L)"].min()
        )
    )

    upper = float(
        max(
            results["실제 연비(km/L)"].max(),
            results["예측 연비(km/L)"].max()
        )
    )

    actual_pred_fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="실제값과 예측값이 같은 기준선",
            line={"dash": "dash"}
        )
    )

    st.plotly_chart(actual_pred_fig, use_container_width=True)

    st.caption(
        "점이 대각선 기준선에 가까울수록 실제 연비와 예측 연비가 비슷합니다."
    )

with residual_col:
    st.markdown("**잔차 그래프**")

    residual_fig = px.scatter(
        results,
        x="예측 연비(km/L)",
        y="잔차(km/L)",
        hover_data=["자동차 이름", "실제 연비(km/L)"],
        labels={
            "예측 연비(km/L)": "예측 연비(km/L)",
            "잔차(km/L)": "잔차(km/L)"
        },
    )

    residual_fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#263b53"
    )

    st.plotly_chart(residual_fig, use_container_width=True)

    st.caption(
        "잔차는 실제 연비에서 예측 연비를 뺀 값입니다. "
        "0을 기준으로 점들이 어떻게 흩어지는지 살펴보세요."
    )

st.markdown("**테스트 자동차별 실제값과 예측값 비교**")

line_results = results.reset_index().sort_values("원본 데이터 행")

comparison_fig = go.Figure()

comparison_fig.add_trace(
    go.Scatter(
        x=line_results["원본 데이터 행"],
        y=line_results["실제 연비(km/L)"],
        mode="lines+markers",
        name="실제 연비"
    )
)

comparison_fig.add_trace(
    go.Scatter(
        x=line_results["원본 데이터 행"],
        y=line_results["예측 연비(km/L)"],
        mode="lines+markers",
        name="예측 연비"
    )
)

comparison_fig.update_layout(
    xaxis_title="원본 데이터 행 번호",
    yaxis_title="연비(km/L)",
    legend_title="구분",
)

st.plotly_chart(comparison_fig, use_container_width=True)

st.markdown("**실제값과 예측값 비교 테이블**")
st.dataframe(results, use_container_width=True)

# ==========================================
# ⑤ 모델 성능 평가
# ==========================================

st.divider()
st.subheader("⑤ 모델 성능 평가")

# 테스트 데이터의 실제값과 예측값을 이용하여 계산
y_test = results["실제 연비(km/L)"]
y_pred = results["예측 연비(km/L)"]

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        label="MSE (평균제곱오차)",
        value=f"{mse:.4f}"
    )

with metric_col2:
    st.metric(
        label="MAE (평균절대오차)",
        value=f"{mae:.4f} km/L"
    )

with metric_col3:
    st.metric(
        label="R² (결정계수)",
        value=f"{r2:.4f}"
    )

st.markdown("**성능평가지표 이해하기**")

with st.expander("MSE - 평균제곱오차"):
    st.write(
        "실제 연비와 예측 연비의 차이를 제곱한 뒤 평균한 값입니다. "
        "큰 예측 오차에 더 큰 영향을 받으며, 일반적으로 작을수록 좋습니다."
    )
    st.latex(
        r"MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2"
    )

with st.expander("MAE - 평균절대오차"):
    st.write(
        "실제 연비와 예측 연비의 차이를 절댓값으로 바꾸어 평균한 값입니다. "
        "예측값이 실제값과 평균적으로 얼마나 차이 나는지 보여 줍니다. "
        "단위는 km/L이며, 작을수록 좋습니다."
    )
    st.latex(
        r"MAE = \\frac{1}{n}\\sum_{i=1}^{n}|y_i-\\hat{y}_i|"
    )

with st.expander("R² - 결정계수"):
    st.write(
        "모델이 실제 연비 데이터의 변동을 얼마나 설명하는지 나타내는 지표입니다. "
        "일반적으로 1에 가까울수록 좋습니다. "
        "음수가 나올 수도 있으며, 이 경우에는 평균값만으로 예측하는 것보다 "
        "모델의 예측이 좋지 않다는 뜻입니다."
    )
    st.latex(
        r"R^2 = 1-\\frac{\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2}"
        r"{\\sum_{i=1}^{n}(y_i-\\bar{y})^2}"
    )

st.caption(
    "성능평가지표는 학습에 사용하지 않은 테스트 데이터로 계산했습니다. "
    "따라서 새로운 자동차에 대한 예측 성능을 살펴보는 데 활용할 수 있습니다."
)

st.divider()
st.subheader("⑥ 3페이지와 연결")

st.success(
    "학습한 모델과 전처리 과정이 이 브라우저 세션에 저장되었습니다. "
    "이제 '3. 자동차 연비 예측하기' 페이지에서 같은 모델을 사용할 수 있습니다."
)

st.page_link(
    "pages/3_연비_예측.py",
    label="자동차 연비 예측 페이지로 이동",
    icon="🚘",
    use_container_width=True
)
