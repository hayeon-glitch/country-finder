import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import json

st.set_page_config(page_title="나의 가치관으로 찾는 나라 🌍", layout="centered")

# ── 데이터 로드 & 전처리 (캐시) ──
@st.cache_data
def load_data():
    df = pd.read_csv("world-happiness-report-2021.csv")
    features = [
        "Logged GDP per capita", "Social support",
        "Healthy life expectancy", "Freedom to make life choices",
        "Perceptions of corruption"
    ]
    df_c = df[["Country name"] + features + ["Ladder score"]].dropna().reset_index(drop=True)

    X = df_c[features].copy()
    X["Perceptions of corruption"] = 1 - X["Perceptions of corruption"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    FLAG_MAP = {
        "Finland":"🇫🇮","Denmark":"🇩🇰","Switzerland":"🇨🇭","Iceland":"🇮🇸",
        "Netherlands":"🇳🇱","Norway":"🇳🇴","Sweden":"🇸🇪","New Zealand":"🇳🇿",
        "Austria":"🇦🇹","Australia":"🇦🇺","Israel":"🇮🇱","Germany":"🇩🇪",
        "Canada":"🇨🇦","Ireland":"🇮🇪","United Kingdom":"🇬🇧","United States":"🇺🇸",
        "France":"🇫🇷","Spain":"🇪🇸","Italy":"🇮🇹","Japan":"🇯🇵",
        "South Korea":"🇰🇷","Poland":"🇵🇱","Czech Republic":"🇨🇿","Brazil":"🇧🇷",
        "Argentina":"🇦🇷","Mexico":"🇲🇽","Chile":"🇨🇱","China":"🇨🇳",
        "India":"🇮🇳","Indonesia":"🇮🇩","Thailand":"🇹🇭","Vietnam":"🇻🇳",
        "Philippines":"🇵🇭","Russia":"🇷🇺","Ukraine":"🇺🇦","Turkey":"🇹🇷",
        "Greece":"🇬🇷","Portugal":"🇵🇹","Nigeria":"🇳🇬","Kenya":"🇰🇪",
        "Ethiopia":"🇪🇹","South Africa":"🇿🇦","Afghanistan":"🇦🇫","Pakistan":"🇵🇰",
        "Taiwan Province of China":"🇹🇼","Costa Rica":"🇨🇷","Luxembourg":"🇱🇺",
        "United Arab Emirates":"🇦🇪","Saudi Arabia":"🇸🇦","Malta":"🇲🇹",
        "Romania":"🇷🇴","Latvia":"🇱🇻","Estonia":"🇪🇪","Lithuania":"🇱🇹",
        "Hungary":"🇭🇺","Croatia":"🇭🇷","Slovakia":"🇸🇰","Slovenia":"🇸🇮",
        "Peru":"🇵🇪","Colombia":"🇨🇴","Bangladesh":"🇧🇩","Zimbabwe":"🇿🇼",
    }

    CLUSTER_NAMES = {
        0:"🏙️ 성장 중인 동유럽·CIS형",
        1:"🌿 공동체 중심 개도국형",
        2:"🏆 북유럽 복지국가형",
        3:"🌄 소박하고 강인한 생존형",
        4:"⚡ 자유롭고 활기찬 선진국형",
        5:"🏜️ 저개발·생존형"
    }

    countries = []
    for i, row in df_c.iterrows():
        name = row["Country name"]
        countries.append({
            "country": name,
            "flag": FLAG_MAP.get(name, "🌐"),
            "cluster": int(labels[i]),
            "cluster_name": CLUSTER_NAMES[int(labels[i])],
            "happiness": float(row["Ladder score"]),
            "gdp": float(row["Logged GDP per capita"]),
            "health": float(row["Healthy life expectancy"]),
            "social": float(row["Social support"]),
            "freedom": float(row["Freedom to make life choices"]),
            "corruption": float(row["Perceptions of corruption"]),
        })

    return countries, X_scaled

def cosine_sim(a, b):
    dot = np.dot(a, b)
    return dot / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# ── UI ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🌍 나의 가치관으로 찾는 나라")
st.markdown("**5개 슬라이더**로 가치관을 설정하면 가장 잘 맞는 나라 TOP 5를 찾아드려요")
st.markdown("---")

countries, X_scaled = load_data()

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 💡 나의 가치관 설정")
    v0 = st.slider("💰 경제적 풍요",  1, 5, 3, help="1: 소박한 삶 / 5: 경제적 성공이 우선")
    v1 = st.slider("🤝 공동체 연대",  1, 5, 3, help="1: 개인주의 / 5: 가족·이웃이 중요")
    v2 = st.slider("🌿 건강·워라밸", 1, 5, 3, help="1: 빠른 성취 중시 / 5: 여유로운 삶")
    v3 = st.slider("🗽 자유와 선택", 1, 5, 3, help="1: 안정·규칙 선호 / 5: 내 삶은 내가 결정")
    v4 = st.slider("🏛️ 사회 청렴도", 1, 5, 3, help="1: 현실적 타협도 OK / 5: 투명·공정 필수")

st.markdown("---")

if st.button("🔍 나에게 맞는 나라 찾기", use_container_width=True):
    raw = np.array([
        (v0 - 1) / 4,
        (v1 - 1) / 4,
        (v2 - 1) / 4,
        (v3 - 1) / 4,
        1 - (v4 - 1) / 4,
    ])

    sims = [cosine_sim(raw, row) for row in X_scaled]
    top5_idx = np.argsort(sims)[::-1][:5]
    top5 = [(countries[i], sims[i]) for i in top5_idx]

    top1 = top5[0][0]

    # 결과 헤더
    st.markdown("---")
    st.markdown(f"## {top1['flag']} {top1['country']}형 삶이 어울려요!")
    st.markdown(f"**{top1['cluster_name']}**")
    st.markdown("---")

    # TOP5
    st.markdown("### 🏅 당신과 맞는 나라 TOP 5")
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    for i, (c, sim) in enumerate(top5):
        pct = round(sim * 100)
        col_a, col_b, col_c = st.columns([0.5, 3, 1])
        with col_a:
            st.markdown(f"### {medals[i]}")
        with col_b:
            st.markdown(f"**{c['flag']} {c['country']}**")
            st.caption(f"행복지수 {c['happiness']:.2f} · {c['cluster_name']}")
            st.progress(pct)
        with col_c:
            st.markdown(f"**{pct}%**")
        st.markdown("")

    # 레이더 차트
    st.markdown("### 📊 가치관 레이더 차트")
    labels_r = ["경제적 풍요","공동체 연대","건강·워라밸","자유와 선택","사회 청렴도"]
    user_vals = [(v0-1)/4, (v1-1)/4, (v2-1)/4, (v3-1)/4, (v4-1)/4]
    top1_row = X_scaled[top5_idx[0]]
    top1_vals = [top1_row[0], top1_row[1], top1_row[2], top1_row[3], 1-top1_row[4]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_vals + [user_vals[0]],
        theta=labels_r + [labels_r[0]],
        fill="toself", name="나의 가치관",
        line_color="#01696f", fillcolor="rgba(1,105,111,0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=top1_vals + [top1_vals[0]],
        theta=labels_r + [labels_r[0]],
        fill="toself", name=top1["country"],
        line_color="#4f98a3", fillcolor="rgba(79,152,163,0.10)",
        line_dash="dash"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, height=420,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상세 지표
    st.markdown(f"### 📈 {top1['country']} 주요 지표")
    c1, c2, c3 = st.columns(3)
    c1.metric("😊 행복지수",  f"{top1['happiness']:.2f}")
    c2.metric("💰 GDP(로그)", f"{top1['gdp']:.2f}")
    c3.metric("🌿 건강수명",  f"{top1['health']:.1f}세")
    c4, c5, c6 = st.columns(3)
    c4.metric("🤝 사회적 지지", f"{top1['social']:.3f}")
    c5.metric("🗽 자유도",      f"{top1['freedom']:.3f}")
    c6.metric("🏛️ 부패인식",   f"{top1['corruption']:.3f}")

st.markdown("---")
st.caption("데이터: World Happiness Report 2021 (Kaggle) · K-Means Clustering + Cosine Similarity · 149개국")
