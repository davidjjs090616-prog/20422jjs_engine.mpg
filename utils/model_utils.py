"""고정된 독립변수로 선형 회귀 모델을 학습하는 공통 함수."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.data_loader import FEATURE_COLUMNS, TARGET_COLUMN

TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_linear_regression(df: pd.DataFrame) -> dict[str, Any]:
    """고정된 X와 Y로 데이터를 분할하고 선형 회귀 모델을 학습한다.

    결측값 대체 중앙값은 학습 데이터에서만 계산되며,
    Pipeline이 테스트 데이터와 새 입력값에도 같은 값을 적용한다.
    """
    missing_columns = [
        column
        for column in [*FEATURE_COLUMNS, TARGET_COLUMN, "car_name"]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "모델 학습에 필요한 데이터 열이 없습니다: " + ", ".join(missing_columns)
        )

    # 종속변수가 없는 행은 학습할 수 없으므로 제외한다.
    modeling_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    X = modeling_df[FEATURE_COLUMNS].copy()
    y = modeling_df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("linear_regression", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    predicted = pipeline.predict(X_test)
    test_results = pd.DataFrame(
        {
            "자동차 이름": modeling_df.loc[X_test.index, "car_name"].to_numpy(),
            "실제 연비(km/L)": y_test.to_numpy(),
            "예측 연비(km/L)": predicted,
        },
        index=X_test.index,
    )
    test_results.index.name = "원본 데이터 행"
    test_results["잔차(km/L)"] = (
        test_results["실제 연비(km/L)"] - test_results["예측 연비(km/L)"]
    )
    test_results = test_results.sort_index()

    imputer = pipeline.named_steps["imputer"]
    median_by_feature = {
        feature: float(value)
        for feature, value in zip(FEATURE_COLUMNS, imputer.statistics_)
    }

    return {
        "model": pipeline,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "target_column": TARGET_COLUMN,
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "imputation_medians": median_by_feature,
        "test_results": test_results,
        "train_indices": X_train.index.to_list(),
        "test_indices": X_test.index.to_list(),
    }
