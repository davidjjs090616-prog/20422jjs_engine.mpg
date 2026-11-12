from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    COLUMN_LABELS,
    DataLoadError,
    TARGET_COLUMN,
    get_display_data,
    load_data,
)
from utils.ui import apply_app_styles

st.set_page_config(page_title="데이터 탐색 · 자동차 연비 예측 실습실", page_icon="🔍", layout="wide")
apply_app_styles()

st.title("🔍 1. 데이터 탐색")
st.write("자동차 데이터를 살펴보고, 자동차의 특성과 연비 사이에 어떤 관계가 있는지 탐구합니다.")

try:
    df = load_data()
except DataLoadError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("① 데이터 기본 정보")
summary_columns = st.columns(5)
summary_columns[0].metric("전체 자동차 수", f"{len(df):,}대")
summary_columns[1].metric("평균 연비", f"{df[TARGET_COLUMN].mean():.2f} km/L")
summary_columns[2].metric("최고 연비", f"{df[TARGET_COLUMN].max():.2f} km/L")
summary_columns[3].metric("최저 연비", f"{df[TARGET_COLUMN].min():.2f} km/L")
summary_columns[4].metric("전체 결측값", f"{int(df.isna().sum().sum()):,}개")

st.divider()
st.subheader("② 자동차 데이터 표")
display_df = get_display_data(df)
search_text = st.text_input("데이터 검색", placeholder="자동차 이름, 연도, 배기량 등을 입력하세요.")
if search_text.strip():
    searchable = display_df.astype("string").fillna("")
    matched = searchable.apply(
        lambda column: column.str.contains(search_text.strip(), case=False, regex=False),
        axis=0,
    ).any(axis=1)
    display_df = display_df.loc[matched]

preview_choice = st.selectbox("표시할 행 수", ["5", "10", "20", "50", "전체"], index=1)
if preview_choice == "전체":
    table_df = display_df
else:
    table_df = display_df.head(int(preview_choice))

st.caption(f"검색 결과 {len(display_df):,}대 중 {len(table_df):,}대를 표시합니다.")
st.dataframe(table_df, use_container_width=True, hide_index=True)
csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "CSV 다운로드",
    data=csv_bytes,
    file_name="auto_mpg_탐색_데이터.csv",
    mime="text/csv",
    use_container_width=False,
)

st.divider()
st.subheader("③ 변수별 분포")
distribution_labels = {
    TARGET_COLUMN: TARGET_COLUMN,
    "displacement": "배기량",
    "horsepower": "마력",
    "weight": "자동차 무게",
    "acceleration": "가속 성능",
    "model_year": "모델 연도",
    "cylinders": "실린더 수",
    "origin": "제조 지역",
}
selected_distribution = st.selectbox(
    "분포를 확인할 변수",
    options=list(distribution_labels.keys()),
    format_func=lambda key: distribution_labels[key],
)

categorical_columns = {"cylinders", "origin"}
if selected_distribution in categorical_columns:
    if selected_distribution == "cylinders":
        counts = df[selected_distribution].dropna().astype(int).astype(str).value_counts().sort_index()
        x_title = "실린더 수"
    else:
        counts = df[selected_distribution].dropna().value_counts().sort_values(ascending=False)
        x_title = "제조 지역"
    distribution_fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        labels={"x": x_title, "y": "자동차 수"},
        title=f"{x_title}별 자동차 수",
        text=counts.values,
    )
    distribution_fig.update_traces(textposition="outside")
    st.plotly_chart(distribution_fig, use_container_width=True)
    st.caption("범주별 자동차 수를 비교한 그래프입니다. 실린더 수와 제조 지역은 범주로 비교합니다.")
else:
    histogram_bins = st.slider("히스토그램 구간 수", min_value=8, max_value=40, value=20, step=1)
    distribution_fig = px.histogram(
        df,
        x=selected_distribution,
        nbins=histogram_bins,
        labels={selected_distribution: distribution_labels[selected_distribution]},
        title=f"{distribution_labels[selected_distribution]} 분포",
    )
    distribution_fig.update_layout(yaxis_title="자동차 수")
    st.plotly_chart(distribution_fig, use_container_width=True)
    st.caption("막대가 높을수록 해당 구간에 속하는 자동차가 많다는 뜻입니다.")

st.divider()
st.subheader("④ 연비와 변수 간 산점도")
scatter_options = {
    "weight": "자동차 무게",
    "displacement": "배기량",
    "horsepower": "마력",
    "acceleration": "가속 성능",
    "model_year": "모델 연도",
}
scatter_left, scatter_right = st.columns([1, 2])
with scatter_left:
    scatter_x = st.selectbox(
        "X축 변수",
        options=list(scatter_options.keys()),
        format_func=lambda key: scatter_options[key],
    )
    show_trend = st.checkbox("선형 추세선 표시", value=True)
with scatter_right:
    valid_scatter = df[[scatter_x, TARGET_COLUMN, "car_name", "origin"]].dropna()
    scatter_fig = px.scatter(
        valid_scatter,
        x=scatter_x,
        y=TARGET_COLUMN,
        color="origin",
        hover_name="car_name",
        hover_data={
            scatter_x: ":.2f",
            TARGET_COLUMN: ":.2f",
            "origin": True,
        },
        labels={scatter_x: scatter_options[scatter_x], TARGET_COLUMN: TARGET_COLUMN, "origin": "제조 지역"},
        title=f"{scatter_options[scatter_x]}과 연비의 관계",
    )
    if show_trend and len(valid_scatter) >= 2 and valid_scatter[scatter_x].nunique() >= 2:
        x_values = valid_scatter[scatter_x].to_numpy(dtype=float)
        y_values = valid_scatter[TARGET_COLUMN].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
        scatter_fig.add_trace(
            go.Scatter(
                x=line_x,
                y=slope * line_x + intercept,
                mode="lines",
                name="선형 추세선",
                line={"dash": "dash", "width": 2, "color": "#263b53"},
            )
        )
    st.plotly_chart(scatter_fig, use_container_width=True)
st.caption("각 점은 자동차 한 대를 나타냅니다. 추세선은 두 변수 사이의 전체적인 선형 경향을 살펴보기 위한 보조선입니다.")

st.divider()
st.subheader("⑤ 변수 간 상관관계")
correlation_columns = [
    TARGET_COLUMN,
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]
correlation_labels = {
    TARGET_COLUMN: TARGET_COLUMN,
    "displacement": "배기량",
    "horsepower": "마력",
    "weight": "자동차 무게",
    "acceleration": "가속 성능",
    "model_year": "모델 연도",
}
correlation_df = df[correlation_columns].corr(numeric_only=True)
correlation_df = correlation_df.rename(index=correlation_labels, columns=correlation_labels)
heatmap_fig = px.imshow(
    correlation_df,
    text_auto=".2f",
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="연비와 수치형 변수의 상관관계",
    labels={"color": "상관계수"},
)
heatmap_fig.update_layout(height=560)
st.plotly_chart(heatmap_fig, use_container_width=True)
st.caption(
    "상관계수는 두 변수의 선형적인 관계 방향과 정도를 나타냅니다. "
    "실린더 수와 제조 지역은 범주형 특성이므로 이 상관관계 계산에서 제외했습니다. "
    "상관관계가 높다고 해서 한 변수가 다른 변수의 원인이라는 뜻은 아닙니다."
)
