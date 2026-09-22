import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# ------------------------------------------------------------
# 페이지 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 분류 모델 만들기")
st.write("나이, 혈당, 체질량지수 등의 정보로 뇌졸중을 예측하는 모델을 만들어봅시다.")

st.divider()

# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ------------------------------------------------------------
# 열 이름 <-> 우리말 이름 매칭 사전
# ------------------------------------------------------------
col_to_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_to_col = {v: k for k, v in col_to_kor.items()}

all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features_kor = [col_to_kor[c] for c in all_features if c != "bmi"]  # bmi 제외 기본값

# ------------------------------------------------------------
# 1. 입력 속성 선택 (우리말 이름으로 표시)
# ------------------------------------------------------------
st.header("1️⃣ 입력으로 사용할 속성 고르기")

selected_kor = st.multiselect(
    "모델의 입력으로 사용할 속성을 선택하세요 (최소 2개)",
    options=[col_to_kor[c] for c in all_features],
    default=default_features_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해야 모델을 만들 수 있어요. 속성을 더 선택해주세요.")
    st.stop()

selected_features = [kor_to_col[k] for k in selected_kor]
use_bmi = "bmi" in selected_features

st.divider()

# ==============================================================
# 2. 데이터 준비 (정렬 -> 10명씩 묶어 앞 3명 테스트, 뒤 7명 학습)
# ==============================================================
st.header("2️⃣ 데이터 나누기")

df_sorted = df.sort_values("id").reset_index(drop=True)

# 10명씩 묶었을 때 그룹 내 순번 (0~9) 계산
group_position = np.arange(len(df_sorted)) % 10

test_mask = group_position < 3   # 각 그룹의 앞 3명
train_mask = ~test_mask          # 나머지 7명

train_df = df_sorted[train_mask].copy()
test_df = df_sorted[test_mask].copy()

st.write(f"전체 {len(df_sorted):,}명 중 학습용 **{len(train_df):,}명**, 테스트용 **{len(test_df):,}명**으로 나누었습니다.")

# ------------------------------------------------------------
# bmi 결측치 처리: 선택된 경우에만, 훈련용 중앙값으로 채움
# ------------------------------------------------------------
if use_bmi:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.write(f"체질량지수(bmi)의 빈 값은 훈련용 중앙값인 **{bmi_median:.2f}**로 채웠습니다.")

# ------------------------------------------------------------
# 학습용 데이터에서 클래스 크기 맞추기 (언더샘플링)
# ------------------------------------------------------------
train_majority = train_df[train_df["stroke"] == 0]
train_minority = train_df[train_df["stroke"] == 1]

train_majority_downsampled = resample(
    train_majority,
    replace=False,
    n_samples=len(train_minority),
    random_state=42
)

train_balanced = pd.concat([train_majority_downsampled, train_minority]).reset_index(drop=True)

st.write(
    f"학습용 데이터의 크기를 맞춘 뒤(언더샘플링), "
    f"뇌졸중 있음 **{len(train_minority)}명**, 뇌졸중 없음 **{len(train_majority_downsampled)}명**으로 "
    f"총 **{len(train_balanced)}명**을 학습에 사용합니다."
)

X_train = train_balanced[selected_features]
y_train = train_balanced["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

st.divider()

# ==============================================================
# 3. 모델 학습
# ==============================================================
st.header("3️⃣ 모델 학습 결과")

# 로지스틱 회귀
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)

# 의사결정트리 (질문 최대 3번, 마지막 마디 5명 미만이면 그만)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# 더미 모델 (입력을 보지 않고 훈련용에서 많은 쪽으로 답)
dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)

def get_accuracies(model):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model)

# ------------------------------------------------------------
# 정확도 카드 세 개
# ------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{log_test_acc*100:.1f} %"
    )
    st.caption(f"훈련 정확도: {log_train_acc*100:.1f}% ・ 테스트 정확도: {log_test_acc*100:.1f}%")

with col2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{tree_test_acc*100:.1f} %"
    )
    st.caption(f"훈련 정확도: {tree_train_acc*100:.1f}% ・ 테스트 정확도: {tree_test_acc*100:.1f}%")

with col3:
    st.metric(
        label="입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        value=f"{dummy_test_acc*100:.1f} %"
    )
    st.caption(f"훈련 정확도: {dummy_train_acc*100:.1f}% ・ 테스트 정확도: {dummy_test_acc*100:.1f}%")

st.divider()

# ==============================================================
# 4. 산점도 + 로지스틱 회귀 경계선 + 의사결정트리 영역
# ==============================================================
st.header("4️⃣ 산점도로 살펴보기")

col_x_kor, col_y_kor = st.columns(2)

with col_x_kor:
    x_axis_kor = st.selectbox("가로축으로 사용할 속성", options=selected_kor, index=0)

with col_y_kor:
    remaining_kor = [k for k in selected_kor if k != x_axis_kor]
    y_axis_kor = st.selectbox("세로축으로 사용할 속성", options=remaining_kor, index=0)

x_axis = kor_to_col[x_axis_kor]
y_axis = kor_to_col[y_axis_kor]

# 두 축이 아닌 나머지 속성은 테스트 데이터의 중앙값으로 고정
other_features = [f for f in selected_features if f not in [x_axis, y_axis]]
fixed_values = {}
for f in other_features:
    fixed_values[f] = X_test[f].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_to_kor[k]} = {v:.2f}" for k, v in fixed_values.items()])
    st.write(f"📌 그래프에 나타나지 않는 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 두 개뿐이라 고정할 속성이 없습니다.")

# ------------------------------------------------------------
# 그리드 범위 설정 (테스트 데이터 기준)
# ------------------------------------------------------------
x_min, x_max = X_test[x_axis].min(), X_test[x_axis].max()
y_min, y_max = X_test[y_axis].min(), X_test[y_axis].max()

x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 1
y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 1

grid_size = 200
xx, yy = np.meshgrid(
    np.linspace(x_min - x_margin, x_max + x_margin, grid_size),
    np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
)

# 그리드에 대응하는 전체 입력 데이터프레임 만들기 (다른 속성은 고정값)
grid_df = pd.DataFrame({x_axis: xx.ravel(), y_axis: yy.ravel()})
for f, v in fixed_values.items():
    grid_df[f] = v
grid_df = grid_df[selected_features]  # 학습 시 순서와 동일하게 맞춤

# 의사결정트리 배경 영역 (옅은 색)
tree_pred_grid = tree_model.predict(grid_df).reshape(xx.shape)

fig = go.Figure()

# 배경 영역 (의사결정트리 분류 결과)
fig.add_trace(go.Contour(
    x=np.linspace(x_min - x_margin, x_max + x_margin, grid_size),
    y=np.linspace(y_min - y_margin, y_max + y_margin, grid_size),
    z=tree_pred_grid,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "blue"], [1, "red"]],
    contours=dict(start=0, end=1, size=1),
    line=dict(width=0),
    name="의사결정트리 영역"
))

# 실제 테스트 데이터 점 찍기 (뇌졸중 여부로 색 구분)
test_plot_df = test_df.copy()
test_plot_df["뇌졸중 여부"] = test_plot_df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

for label, color in [("뇌졸중 없음", "blue"), ("뇌졸중 있음", "red")]:
    subset = test_plot_df[test_plot_df["뇌졸중 여부"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_axis],
        y=subset[y_axis],
        mode="markers",
        name=label,
        marker=dict(color=color, size=6, opacity=0.6)
    ))

# ------------------------------------------------------------
# 로지스틱 회귀 0.5 결정 경계선 계산
# ------------------------------------------------------------
# 로지스틱 회귀는 선형 결합이 0일 때 확률 0.5가 됨
# w0*x0 + w1*x1 + ... + b = 0 을 이용해서
# x_axis, y_axis에 대한 선을 구함 (다른 항은 고정값으로 대입)

coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

feature_index = {f: i for i, f in enumerate(selected_features)}
x_idx = feature_index[x_axis]
y_idx = feature_index[y_axis]

# 고정된 다른 속성들의 기여도 합
fixed_contrib = intercept
for f, v in fixed_values.items():
    fixed_contrib += coef[feature_index[f]] * v

w_x = coef[x_idx]
w_y = coef[y_idx]

line_drawn = False
if abs(w_y) > 1e-10:
    # y = -(w_x * x + fixed_contrib) / w_y
    x_line = np.linspace(x_min - x_margin, x_max + x_margin, 200)
    y_line = -(w_x * x_line + fixed_contrib) / w_y

    # 그림 범위 안에 들어오는 부분만 표시
    in_range = (y_line >= y_min - y_margin) & (y_line <= y_max + y_margin)

    if in_range.any():
        fig.add_trace(go.Scatter(
            x=x_line[in_range],
            y=y_line[in_range],
            mode="lines",
            name="로지스틱 회귀 경계선(확률 0.5)",
            line=dict(color="black", width=2, dash="dash")
        ))
        line_drawn = True

fig.update_layout(
    title="테스트 데이터 산점도와 결정 경계",
    xaxis_title=x_axis_kor,
    yaxis_title=y_axis_kor,
    legend_title="구분"
)

st.plotly_chart(fig, use_container_width=True)

if not line_drawn:
    st.write("📌 로지스틱 회귀의 확률 0.5 경계선은 이 그림의 범위 밖에 있어서 표시되지 않았습니다.")

st.divider()

# ==============================================================
# 5. 의사결정트리 가지 그림 (graphviz DOT 문자열)
# ==============================================================
st.header("5️⃣ 의사결정트리 가지 그림")

tree_ = tree_model.tree_
feature_names = selected_features

def build_dot(tree_, feature_names):
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    def recurse(node_id):
        n_samples = tree_.n_node_samples[node_id]
        # value: [클래스0 개수, 클래스1 개수] (가중치 반영된 값일 수 있음 -> 반올림)
        value = tree_.value[node_id][0]
        n_stroke = value[1]
        ratio = n_stroke / (value[0] + value[1]) if (value[0] + value[1]) > 0 else 0

        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]

        if is_leaf:
            # 답을 내는 마디: 다수결로 예측 클래스 결정
            predicted_class = 1 if value[1] > value[0] else 0
            label_text = "뇌졸중 있음" if predicted_class == 1 else "뇌졸중 없음"
            color = "#ffcccc" if predicted_class == 1 else "#cce5ff"

            label = (
                f"{label_text}\\n"
                f"인원: {int(round(n_samples))}명\\n"
                f"뇌졸중: {int(round(n_stroke))}명\\n"
                f"비율: {ratio*100:.1f}%"
            )
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
        else:
            feature = feature_names[tree_.feature[node_id]]
            feature_kor = col_to_kor[feature]
            threshold = tree_.threshold[node_id]

            label = (
                f"{feature_kor} <= {threshold:.2f} ?\\n"
                f"인원: {int(round(n_samples))}명\\n"
                f"뇌졸중: {int(round(n_stroke))}명\\n"
                f"비율: {ratio*100:.1f}%"
            )
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="#f2f2f2"];')

            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]

            dot_lines.append(f'{node_id} -> {left_id} [label="예"];')
            dot_lines.append(f'{node_id} -> {right_id} [label="아니요"];')

            recurse(left_id)
            recurse(right_id)

    recurse(0)
    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_string = build_dot(tree_, feature_names)
st.graphviz_chart(dot_string)

st.divider()

# ------------------------------------------------------------
# 답을 내는 마디(리프 노드) 요약
# ------------------------------------------------------------
st.header("6️⃣ 의사결정트리 요약")

leaf_ids = [i for i in range(tree_.node_count) if tree_.children_left[i] == tree_.children_right[i]]
n_leaf = len(leaf_ids)

n_leaf_negative = 0
for leaf_id in leaf_ids:
    value = tree_.value[leaf_id][0]
    predicted_class = 1 if value[1] > value[0] else 0
    if predicted_class == 0:
        n_leaf_negative += 1

st.write(f"- 답을 내는 마디는 모두 **{n_leaf}칸**이고, 그중 **{n_leaf_negative}칸**이 '뇌졸중 없음'이라고 답합니다.")

used_features_idx = set(tree_.feature[tree_.feature >= 0])
used_features = [feature_names[i] for i in used_features_idx]
used_features_kor = [col_to_kor[f] for f in used_features]

if used_features_kor:
    st.write(f"- 이 나무가 실제로 물어본 속성은 **{', '.join(used_features_kor)}**입니다.")
else:
    st.write("- 이 나무는 어떤 속성도 사용하지 않고 바로 답을 냈습니다.")
