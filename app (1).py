import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

st.set_page_config(page_title="나의 가치관으로 찾는 나라 🌍", layout="centered")

# ── 전체 CSS — 원본 HTML 디자인 재현 ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');

:root {
  --teal: #01696f;
  --teal-light: #4f98a3;
  --teal-bg: #cedcd8;
  --bg: #f7f6f2;
  --text: #28251d;
  --muted: #7a7974;
}

html, body, [class*="css"], .stApp {
  font-family: 'Noto Sans KR', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* 헤더 숨김 */
#MainMenu, header, footer { visibility: hidden; }

/* Hero 배너 */
.hero {
  background: linear-gradient(135deg, #01696f 0%, #4f98a3 60%, #7ec8cc 100%);
  color: white;
  text-align: center;
  padding: 3rem 1.5rem 3.5rem;
  border-radius: 0 0 50% 50% / 0 0 30px 30px;
  margin: -1rem -1rem 2rem -1rem;
}
.hero h1 { font-size: 2.2rem; font-weight: 900; margin-bottom: 0.5rem; }
.hero p { font-size: 1rem; opacity: 0.9; }
.hero-badge {
  display: inline-block;
  background: rgba(255,255,255,0.2);
  border: 1px solid rgba(255,255,255,0.4);
  padding: 0.3rem 1rem; border-radius: 9999px;
  font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem;
}

/* 카드 */
.custom-card {
  background: white;
  border: 1px solid #dcd9d5;
  border-radius: 1rem;
  padding: 1.8rem;
  box-shadow: 0 4px 20px rgba(1,105,111,0.10);
  margin-bottom: 1.4rem;
}
.card-title {
  font-size: 0.9rem; font-weight: 700; color: var(--teal);
  text-transform: uppercase; letter-spacing: 0.08em;
  margin-bottom: 1.2rem;
}

/* 슬라이더 커스텀 */
.stSlider > div > div > div > div {
  background: var(--teal) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--teal) !important;
  border: 3px solid white !important;
  box-shadow: 0 2px 8px rgba(1,105,111,0.35) !important;
}

/* 버튼 */
.stButton > button {
  width: 100%;
  background: linear-gradient(135deg, #01696f, #4f98a3) !important;
  color: white !important;
  font-family: 'Noto Sans KR', sans-serif !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 0.5rem !important;
  padding: 0.8rem !important;
  box-shadow: 0 4px 16px rgba(1,105,111,0.3) !important;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(1,105,111,0.4) !important;
}

/* 결과 헤더 */
.result-hero {
  background: linear-gradient(135deg, #01696f, #4f98a3);
  color: white; border-radius: 1rem;
  padding: 2rem; text-align: center; margin-bottom: 1.4rem;
}
.result-hero h2 { font-size: 1.6rem; font-weight: 900; margin-bottom: 0.3rem; }
.cluster-badge {
  display: inline-block;
  background: rgba(255,255,255,0.25);
  border: 1px solid rgba(255,255,255,0.5);
  padding: 0.3rem 1rem; border-radius: 9999px;
  font-size: 0.85rem; font-weight: 600; margin-top: 0.7rem;
}

/* TOP5 아이템 */
.top5-item {
  display: flex; align-items: center; gap: 1rem;
  padding: 0.9rem 1rem; border-radius: 0.5rem;
  border: 1px solid #dcd9d5; margin-bottom: 0.6rem;
  background: #fafaf8;
}
.top5-item.first { background: linear-gradient(135deg,#f0f9f8,#e8f5f4); border-color: #cedcd8; }
.rank { font-size: 1.4rem; font-weight: 900; color: var(--teal); width: 2.2rem; text-align: center; }
.country-flag { font-size: 1.6rem; }
.country-info { flex: 1; }
.country-name { font-size: 1rem; font-weight: 700; }
.country-meta { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }
.sim-pct { font-size: 0.75rem; font-weight: 700; color: var(--teal); text-align: right; margin-bottom: 3px; }
.sim-bar-bg { height: 4px; border-radius: 9999px; background: #e0e0e0; overflow: hidden; width: 80px; }
.sim-bar-fill { height: 100%; background: linear-gradient(to right, #01696f, #4f98a3); border-radius: 9999px; }

/* stat grid */
.stat-box {
  background: #f7f6f2; border-radius: 0.5rem;
  padding: 0.9rem; text-align: center;
}
.stat-val { font-size: 1.1rem; font-weight: 700; color: var(--teal); }
.stat-label { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }

/* 라벨 색 */
label { color: var(--text) !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 ──
@st.cache_data
def load_data():
    df = pd.read_csv("world-happiness-report-2021.csv")
    features = ["Logged GDP per capita","Social support",
                "Healthy life expectancy","Freedom to make life choices",
                "Perceptions of corruption"]
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
        "United Arab Emirates":"🇦🇪","Saudi Arabia":"🇸🇦","Romania":"🇷🇴",
        "Latvia":"🇱🇻","Estonia":"🇪🇪","Lithuania":"🇱🇹","Hungary":"🇭🇺",
        "Croatia":"🇭🇷","Slovakia":"🇸🇰","Slovenia":"🇸🇮","Peru":"🇵🇪",
        "Colombia":"🇨🇴","Bangladesh":"🇧🇩","Zimbabwe":"🇿🇼","Malta":"🇲🇹",
    }
    CLUSTER_NAMES = {
        0:"🏙️ 성장 중인 동유럽·CIS형", 1:"🌿 공동체 중심 개도국형",
        2:"🏆 북유럽 복지국가형",       3:"🌄 소박하고 강인한 생존형",
        4:"⚡ 자유롭고 활기찬 선진국형", 5:"🏜️ 저개발·생존형"
    }
    countries = []
    for i, row in df_c.iterrows():
        name = row["Country name"]
        countries.append({
            "country": name, "flag": FLAG_MAP.get(name,"🌐"),
            "cluster": int(labels[i]), "cluster_name": CLUSTER_NAMES[int(labels[i])],
            "happiness": float(row["Ladder score"]),
            "gdp": float(row["Logged GDP per capita"]),
            "health": float(row["Healthy life expectancy"]),
            "social": float(row["Social support"]),
            "freedom": float(row["Freedom to make life choices"]),
            "corruption": float(row["Perceptions of corruption"]),
        })
    return countries, X_scaled

def cosine_sim(a, b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-10)

countries, X_scaled = load_data()

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="hero-badge">🌍 WORLD HAPPINESS REPORT 2021</div>
  <h1>나의 가치관으로<br>찾는 나라</h1>
  <p>5개의 슬라이더로 삶의 가치관을 설정하면<br>가장 잘 맞는 나라 TOP 5를 찾아드려요</p>
</div>
""", unsafe_allow_html=True)

# ── 슬라이더 카드 ──
st.markdown('<div class="custom-card"><div class="card-title">💡 나의 가치관 설정</div>', unsafe_allow_html=True)

v0 = st.slider("💰 경제적 풍요",  1, 5, 3, help="1: 소박한 삶으로 충분해  |  5: 경제적 성공이 우선이야")
v1 = st.slider("🤝 공동체 연대",  1, 5, 3, help="1: 나 혼자도 괜찮아 (개인)  |  5: 가족·이웃이 중요해 (공동체)")
v2 = st.slider("🌿 건강·워라밸", 1, 5, 3, help="1: 빠른 성취가 중요해  |  5: 건강하고 여유로운 삶")
v3 = st.slider("🗽 자유와 선택", 1, 5, 3, help="1: 안정·규칙을 선호해  |  5: 내 삶은 내가 결정해")
v4 = st.slider("🏛️ 사회 청렴도", 1, 5, 3, help="1: 현실적 타협도 OK  |  5: 투명·공정한 사회 필수")

run = st.button("🔍 나에게 맞는 나라 찾기")
st.markdown('</div>', unsafe_allow_html=True)

# ── 분석 ──
if run:
    raw = np.array([
        (v0-1)/4, (v1-1)/4, (v2-1)/4, (v3-1)/4,
        1-(v4-1)/4
    ])
    sims = [cosine_sim(raw, row) for row in X_scaled]
    top5_idx = np.argsort(sims)[::-1][:5]
    top5 = [(countries[i], sims[i]) for i in top5_idx]
    top1 = top5[0][0]

    # 결과 Hero
    st.markdown(f"""
    <div class="result-hero">
      <div style="font-size:3.5rem;margin-bottom:0.5rem">{top1['flag']}</div>
      <h2>{top1['country']}형 삶이 어울려요!</h2>
      <p style="opacity:0.9">당신의 가치관과 가장 잘 맞는 나라</p>
      <div class="cluster-badge">{top1['cluster_name']}</div>
    </div>
    """, unsafe_allow_html=True)

    # TOP5
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    top5_html = '<div class="custom-card"><div class="card-title">🏅 당신과 맞는 나라 TOP 5</div>'
    for i, (c, sim) in enumerate(top5):
        pct = round(sim*100)
        cls = "top5-item first" if i==0 else "top5-item"
        top5_html += f"""
        <div class="{cls}">
          <div class="rank">{medals[i]}</div>
          <div class="country-flag">{c['flag']}</div>
          <div class="country-info">
            <div class="country-name">{c['country']}</div>
            <div class="country-meta">행복지수 {c['happiness']:.2f} &nbsp;·&nbsp; {c['cluster_name']}</div>
          </div>
          <div>
            <div class="sim-pct">{pct}%</div>
            <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{pct}%"></div></div>
          </div>
        </div>"""
    top5_html += '</div>'
    st.markdown(top5_html, unsafe_allow_html=True)

    # 레이더 차트
    st.markdown('<div class="custom-card"><div class="card-title">📊 가치관 레이더 차트</div>', unsafe_allow_html=True)
    labels_r = ["경제적 풍요","공동체 연대","건강·워라밸","자유와 선택","사회 청렴도"]
    user_v = [(v0-1)/4,(v1-1)/4,(v2-1)/4,(v3-1)/4,(v4-1)/4]
    t1r = X_scaled[top5_idx[0]]
    top1_v = [t1r[0],t1r[1],t1r[2],t1r[3],1-t1r[4]]
    fig = go.Figure()
    for vals, name, color, fill in [
        (user_v, "나의 가치관", "#01696f", "rgba(1,105,111,0.15)"),
        (top1_v, top1['country'], "#4f98a3", "rgba(79,152,163,0.10)")
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=labels_r+[labels_r[0]],
            fill="toself", name=name,
            line=dict(color=color, width=2.5),
            fillcolor=fill
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#f7f6f2",
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12, family="Noto Sans KR"))
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    font=dict(size=12, family="Noto Sans KR")),
        height=380, margin=dict(l=40,r=40,t=20,b=60),
        paper_bgcolor="white", plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 상세 지표
    st.markdown(f'<div class="custom-card"><div class="card-title">📈 {top1["country"]} 주요 지표</div>', unsafe_allow_html=True)
    stats = [
        ("😊 행복지수", f"{top1['happiness']:.2f}"),
        ("💰 GDP(로그)", f"{top1['gdp']:.2f}"),
        ("🌿 건강수명", f"{top1['health']:.1f}세"),
        ("🤝 사회적 지지", f"{top1['social']:.3f}"),
        ("🗽 자유도", f"{top1['freedom']:.3f}"),
        ("🏛️ 부패인식", f"{top1['corruption']:.3f}"),
    ]
    cols = st.columns(3)
    for i, (label, val) in enumerate(stats):
        with cols[i%3]:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p style="text-align:center;font-size:0.78rem;color:#bab9b4;margin-top:2rem;">데이터: World Happiness Report 2021 (Kaggle) · K-Means Clustering + Cosine Similarity · 149개국</p>', unsafe_allow_html=True)
