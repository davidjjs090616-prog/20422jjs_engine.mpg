from __future__ import annotations

import streamlit as st

from utils.data_loader import DataLoadError, TARGET_COLUMN, load_data
from utils.ui import apply_app_styles

st.set_page_config(
    page_title="자동차 연비 예측 실습실",
    page_icon="🚗",
    layout="wide",
)
apply_app_styles()

st.markdown('<div class="app-eyebrow">AUTO MPG · MACHINE LEARNING LAB</div>', unsafe_allow_html=True)
st.title("🚗 자동차 연비 예측 실습실")
st.markdown(
    """
    <section class="app-hero">
      <p>오늘의 탐구 질문</p>
      <h2>자동차의 여러 특성을 입력하면<br>1L의 연료로 몇 km를 주행할 수 있을까?</h2>
      <p>데이터를 탐색하고, 선형 회귀 모델을 학습한 뒤, 새로운 자동차의 연비를 예측해 봅니다.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    data = load_data()
except DataLoadError as exc:
    st.error(str(exc))
    st.info("데이터 주소가 브라우저에서 열리는지, GitHub 저장소의 파일이 공개되어 있는지 확인해 주세요.")
    st.stop()

metric_columns = st.columns(4)
metric_columns[0].metric("전체 자동차 데이터", f"{len(data):,}대")
metric_columns[1].metric("전체 컬럼 수", f"{len(data.columns) - 1:,}개")
metric_columns[2].metric("평균 연비", f"{data[TARGET_COLUMN].mean():.2f} km/L")
metric_columns[3].metric("마력 결측 데이터", f"{int(data['horsepower'].isna().sum()):,}개")

st.subheader("학습 단계")
st.caption("세 단계가 순서대로 연결됩니다. 먼저 데이터를 살펴보고, 회귀 모델을 학습한 다음, 새로운 자동차의 연비를 예측하세요.")

cards = st.columns(3, gap="large")
page_cards = [
    ("🔍 1. 데이터 탐색", "자동차 데이터의 구조와 분포를 확인하고, 여러 특성과 연비 사이의 관계를 그래프로 탐구합니다.", "pages/1_데이터_탐색.py"),
    ("📈 2. 선형 회귀 모델 만들기", "자동차 무게·배기량·마력을 독립변수로 사용해 다중 선형 회귀 모델을 학습하고 회귀식을 살펴봅니다.", "pages/2_선형_회귀.py"),
    ("🚘 3. 자동차 연비 예측하기", "새 자동차의 무게·배기량·마력을 입력하고 학습한 모델의 예상 연비를 확인합니다.", "pages/3_연비_예측.py"),
]
for column, (heading, description, page_path) in zip(cards, page_cards):
    with column:
        with st.container(border=True):
            st.subheader(heading)
            st.write(description)
            st.page_link(page_path, label="페이지 열기", use_container_width=True)

st.divider()
st.subheader("이번 실습에서 사용하는 변수")
left, right = st.columns(2)
with left:
    st.markdown("**독립변수(X)**")
    st.write("자동차 무게 · 배기량 · 마력")
with right:
    st.markdown("**종속변수(Y)**")
    st.write("연비(km/L)")
st.caption("모든 학습 페이지에서 독립변수와 종속변수는 동일하게 고정됩니다.")
