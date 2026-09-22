import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------------------------------------
# 페이지 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 데이터 탐색")
st.write("뇌졸중 데이터를 다양한 그래프와 표로 살펴봅시다.")

st.divider()

# ------------------------------------------------------------
# 데이터 불러오기 (첫 화면과 동일한 방식)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ==============================================================
# 1. 나이와 평균 혈당의 분포 - 히스토그램 두 개 나란히
# ==============================================================
st.header("1️⃣ 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x="age", nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# ==============================================================
# 2. 뇌졸중 여부에 따른 나이·평균 혈당 상자그림 비교
# ==============================================================
st.header("2️⃣ 뇌졸중 여부에 따른 나이·평균 혈당 비교")

# stroke 열을 문자로 바꿔서 그래프에 표시하기 쉽게 만듦
df_box = df.copy()
df_box["stroke_label"] = df_box["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

col3, col4 = st.columns(2)

with col3:
    fig_age_box = px.box(
        df_box, x="stroke_label", y="age",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col4:
    fig_glucose_box = px.box(
        df_box, x="stroke_label", y="avg_glucose_level",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 두 그룹의 평균값 표
mean_table = df_box.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().round(2)
mean_table.columns = ["평균 나이", "평균 혈당"]
mean_table.index.name = "그룹"

st.write("**두 그룹의 평균값**")
st.dataframe(mean_table, use_container_width=True)

st.divider()

# ==============================================================
# 3. 고혈압/심장병 유무에 따른 뇌졸중 비율 막대그래프
# ==============================================================
st.header("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hyp
