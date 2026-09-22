import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------------------------------------
# 페이지 기본 설정 (브라우저 탭 제목, 아이콘, 레이아웃)
# ------------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------------------------
# 앱 제목 (화면 맨 위)
# ------------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 데이터를 살펴보고, 예측 모델을 만들어보는 실습 공간입니다.")

st.divider()

# ------------------------------------------------------------
# 데이터 불러오기 함수
# ------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ------------------------------------------------------------
# 소개 화면 제목
# ------------------------------------------------------------
st.header("📌 데이터 소개")
st.write(
    "이 데이터는 환자들의 건강 정보와 뇌졸중(stroke) 발생 여부를 담고 있습니다. "
    "아래에서 데이터의 전체 모습을 먼저 확인해봅시다."
)

# ------------------------------------------------------------
# 큰 숫자 카드 네 개
# ------------------------------------------------------------
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = round(stroke_count / total_people * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중(stroke=1) 인원", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio} %")

st.divider()

# ------------------------------------------------------------
# 열 이름 / 우리말 뜻 / 값의 종류 / 빈 값 개수 표
# ------------------------------------------------------------
st.header("📋 열(컬럼) 정보 표")
st.write("아래 표의 **'우리말 뜻'** 칸은 비어 있습니다. 교재를 참고해서 직접 채워보세요!")

def get_value_summary(col):
    unique_vals = df[col].dropna().unique()
    if df[col].dtype in ["int64", "float64"] and len(unique_vals) > 10:
        return f"숫자형 (예: {round(df[col].min(), 1)} ~ {round(df[col].max(), 1)})"
    else:
        sorted_vals = sorted(unique_vals, key=lambda x: str(x))
        return ", ".join(str(v) for v in sorted_vals)

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": ["" for _ in df.columns],
    "값의 종류": [get_value_summary(col) for col in df.columns],
    "빈 값 개수": [df[col].isnull().sum() for col in df.columns]
})

st.dataframe(column_info, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------
# 데이터 처음 다섯 줄
# ------------------------------------------------------------
st.header("🔍 데이터 미리보기 (상위 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ------------------------------------------------------------
# 데이터 출처
# ------------------------------------------------------------
st.header("📚 데이터 출처")
st.info("여기에 교재를 참고하여 데이터 출처를 직접 작성해보세요.")

source_text = st.text_area(
    "데이터 출처를 아래에 작성하세요:",
    placeholder="예) OOO 논문, OOO 기관, OOO 사이트 등...",
    height=100
)

if source_text:
    st.success("작성한 출처 내용:")
    st.write(source_text)
