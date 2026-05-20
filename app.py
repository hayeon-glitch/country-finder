import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import json

# ── 1. CSV 로드 ──────────────────────────────────────────────────
df = pd.read_csv("world-happiness-report-2021.csv")

# ── 2. 피처 선택 & 전처리 ────────────────────────────────────────
features = [
    "Logged GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Perceptions of corruption"
]
df_clean = df.dropna(subset=features).copy().reset_index(drop=True)

X = df_clean[features].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. K-Means 클러스터링 (Unsupervised Learning) ─────────────────
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_clean["cluster"] = kmeans.fit_predict(X_scaled)

# ── 4. 국기 매핑 ─────────────────────────────────────────────────
FLAG_MAP = {
    "Finland":"🇫🇮","Denmark":"🇩🇰","Switzerland":"🇨🇭","Iceland":"🇮🇸",
    "Netherlands":"🇳🇱","Norway":"🇳🇴","Sweden":"🇸🇪","Luxembourg":"🇱🇺",
    "New Zealand":"🇳🇿","Austria":"🇦🇹","Australia":"🇦🇺","Israel":"🇮🇱",
    "Germany":"🇩🇪","Canada":"🇨🇦","Ireland":"🇮🇪","Costa Rica":"🇨🇷",
    "United Kingdom":"🇬🇧","Czech Republic":"🇨🇿","United States":"🇺🇸",
    "Belgium":"🇧🇪","France":"🇫🇷","Bahrain":"🇧🇭","Malta":"🇲🇹",
    "Taiwan Province of China":"🇹🇼","United Arab Emirates":"🇦🇪",
    "Saudi Arabia":"🇸🇦","Spain":"🇪🇸","Italy":"🇮🇹","Slovakia":"🇸🇰",
    "Lithuania":"🇱🇹","Slovenia":"🇸🇮","Romania":"🇷🇴","Estonia":"🇪🇪",
    "Poland":"🇵🇱","Cyprus":"🇨🇾","Latvia":"🇱🇻","Hungary":"🇭🇺",
    "Portugal":"🇵🇹","Singapore":"🇸🇬","Serbia":"🇷🇸","Japan":"🇯🇵",
    "South Korea":"🇰🇷","Croatia":"🇭🇷","Bolivia":"🇧🇴","Moldova":"🇲🇩",
    "Russia":"🇷🇺","Honduras":"🇭🇳","Kazakhstan":"🇰🇿","Belarus":"🇧🇾",
    "Mongolia":"🇲🇳","Ukraine":"🇺🇦","Bulgaria":"🇧🇬","Greece":"🇬🇷",
    "Argentina":"🇦🇷","Brazil":"🇧🇷","Colombia":"🇨🇴","Indonesia":"🇮🇩",
    "China":"🇨🇳","Ecuador":"🇪🇨","Panama":"🇵🇦","Uruguay":"🇺🇾",
    "Mexico":"🇲🇽","Jamaica":"🇯🇲","Peru":"🇵🇪","Albania":"🇦🇱",
    "Chile":"🇨🇱","Thailand":"🇹🇭","Malaysia":"🇲🇾","North Cyprus":"🏳️",
    "Nicaragua":"🇳🇮","El Salvador":"🇸🇻","Iraq":"🇮🇶","Uzbekistan":"🇺🇿",
    "Kyrgyzstan":"🇰🇬","Dominican Republic":"🇩🇴","Armenia":"🇦🇲",
    "Paraguay":"🇵🇾","Georgia":"🇬🇪","Vietnam":"🇻🇳","Philippines":"🇵🇭",
    "Nepal":"🇳🇵","Pakistan":"🇵🇰","Nigeria":"🇳🇬","Cameroon":"🇨🇲",
    "Kenya":"🇰🇪","Cambodia":"🇰🇭","Bangladesh":"🇧🇩","Ghana":"🇬🇭",
    "Myanmar":"🇲🇲","Lebanon":"🇱🇧","India":"🇮🇳","Sri Lanka":"🇱🇰",
    "Iran":"🇮🇷","Morocco":"🇲🇦","Egypt":"🇪🇬","Tunisia":"🇹🇳",
    "Palestine":"🇵🇸","Chad":"🇹🇩","Ethiopia":"🇪🇹","Sierra Leone":"🇸🇱",
    "Zimbabwe":"🇿🇼","Tanzania":"🇹🇿","Zambia":"🇿🇲","Lesotho":"🇱🇸",
    "Senegal":"🇸🇳","Madagascar":"🇲🇬","Rwanda":"🇷🇼","Togo":"🇹🇬",
    "Malawi":"🇲🇼","Uganda":"🇺🇬","Burkina Faso":"🇧🇫","Niger":"🇳🇪",
    "Benin":"🇧🇯","Congo (Kinshasa)":"🇨🇩","Mali":"🇲🇱","Mozambique":"🇲🇿",
    "Yemen":"🇾🇪","Afghanistan":"🇦🇫","South Africa":"🇿🇦","Guatemala":"🇬🇹",
    "Cuba":"🇨🇺","Trinidad and Tobago":"🇹🇹","Haiti":"🇭🇹","Jordan":"🇯🇴",
    "Libya":"🇱🇾","Algeria":"🇩🇿","Azerbaijan":"🇦🇿","Kosovo":"🏳️",
    "North Macedonia":"🇲🇰","Bosnia and Herzegovina":"🇧🇦","Montenegro":"🇲🇪",
    "Turkey":"🇹🇷","Hong Kong S.A.R. of China":"🇭🇰","Tajikistan":"🇹🇯",
    "Ivory Coast":"🇨🇮","Maldives":"🇲🇻","Turkmenistan":"🇹🇲","Gambia":"🇬🇲",
    "Namibia":"🇳🇦","Botswana":"🇧🇼","Angola":"🇦🇴","Venezuela":"🇻🇪",
    "Congo (Brazzaville)":"🇨🇬","Comoros":"🇰🇲","Mauritania":"🇲🇷",
    "Liberia":"🇱🇷","Guinea":"🇬🇳","Mauritius":"🇲🇺","Eswatini":"🇸🇿",
}

# ── 5. JSON 직렬화용 레코드 생성 ─────────────────────────────────
records = []
for _, row in df_clean.iterrows():
    records.append({
        "country":    row["Country name"],
        "flag":       FLAG_MAP.get(row["Country name"], "🌐"),
        "gdp":        round(float(row["Logged GDP per capita"]), 3),
        "social":     round(float(row["Social support"]), 3),
        "health":     round(float(row["Healthy life expectancy"]), 1),
        "freedom":    round(float(row["Freedom to make life choices"]), 3),
        "corruption": round(float(row["Perceptions of corruption"]), 3),
        "happiness":  round(float(row["Ladder score"]), 2),
        "cluster":    int(row["cluster"])
    })

# ── 6. HTML에 내장할 JS 변수 준비 ────────────────────────────────
countries_js = json.dumps(records, ensure_ascii=False)
x_scaled_js  = json.dumps(X_scaled.tolist())
scaler_js    = json.dumps({
    "data_min_":   scaler.data_min_.tolist(),
    "data_range_": scaler.data_range_.tolist()
})

# ── 7. HTML 생성 ──────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🌍 나의 가치관으로 찾는 나라</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --teal:#01696f;--teal-light:#4f98a3;--teal-bg:#cedcd8;
  --bg:#f7f6f2;--surface:#ffffff;--border:#dcd9d5;
  --text:#28251d;--muted:#7a7974;--faint:#bab9b4;
  --radius:1rem;--radius-sm:0.5rem;
  --shadow:0 4px 20px rgba(1,105,111,0.10);
}}
body{{font-family:'Noto Sans KR',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}}
.hero{{
  background:linear-gradient(135deg,#01696f 0%,#4f98a3 60%,#7ec8cc 100%);
  color:#fff;text-align:center;padding:3.5rem 1.5rem 4rem;position:relative;overflow:hidden;
}}
.hero::after{{
  content:'';position:absolute;bottom:-2px;left:0;right:0;height:60px;
  background:var(--bg);clip-path:ellipse(55% 100% at 50% 100%);
}}
.hero h1{{font-size:clamp(1.6rem,4vw,2.4rem);font-weight:900;letter-spacing:-0.02em;margin-bottom:0.6rem}}
.hero p{{font-size:1rem;opacity:0.9;line-height:1.7}}
.hero-badge{{
  display:inline-block;background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.4);
  padding:0.3rem 1rem;border-radius:9999px;font-size:0.8rem;font-weight:600;margin-bottom:1rem;letter-spacing:0.05em;
}}
.container{{max-width:680px;margin:0 auto;padding:0 1.2rem 4rem}}
.card{{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.8rem;box-shadow:var(--shadow);margin-bottom:1.4rem;
}}
.card-title{{
  font-size:0.9rem;font-weight:700;color:var(--teal);
  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1.4rem;
  display:flex;align-items:center;gap:0.5rem;
}}
.slider-wrap{{margin-bottom:1.4rem}}
.slider-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem}}
.slider-label-text{{font-size:0.95rem;font-weight:600}}
.slider-val{{
  background:var(--teal);color:#fff;padding:0.15rem 0.6rem;
  border-radius:9999px;font-size:0.78rem;font-weight:700;min-width:1.8rem;text-align:center;
}}
input[type=range]{{
  -webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:9999px;
  background:linear-gradient(to right,var(--teal) 0%,var(--teal) 50%,#e0e0e0 50%,#e0e0e0 100%);
  outline:none;cursor:pointer;margin-bottom:0.4rem;
}}
input[type=range]::-webkit-slider-thumb{{
  -webkit-appearance:none;appearance:none;width:22px;height:22px;border-radius:50%;
  background:var(--teal);border:3px solid #fff;box-shadow:0 2px 8px rgba(1,105,111,0.35);transition:transform 0.15s;
}}
input[type=range]::-webkit-slider-thumb:hover{{transform:scale(1.2)}}
.slider-ends{{display:flex;justify-content:space-between;font-size:0.72rem;color:var(--muted)}}
.btn{{
  width:100%;padding:1rem;border:none;
  background:linear-gradient(135deg,var(--teal),var(--teal-light));
  color:#fff;font-family:inherit;font-size:1rem;font-weight:700;
  border-radius:var(--radius-sm);cursor:pointer;
  box-shadow:0 4px 16px rgba(1,105,111,0.3);transition:all 0.2s;letter-spacing:0.02em;
}}
.btn:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(1,105,111,0.4)}}
.btn:active{{transform:translateY(0)}}
#result{{display:none}}
.result-hero{{
  background:linear-gradient(135deg,var(--teal),var(--teal-light));
  color:#fff;border-radius:var(--radius);padding:2rem;text-align:center;margin-bottom:1.4rem;
}}
.result-hero .top-flag{{font-size:3.5rem;margin-bottom:0.5rem}}
.result-hero h2{{font-size:1.6rem;font-weight:900;margin-bottom:0.3rem}}
.result-hero p{{opacity:0.9;font-size:0.95rem}}
.cluster-badge{{
  display:inline-block;background:rgba(255,255,255,0.25);border:1px solid rgba(255,255,255,0.5);
  padding:0.3rem 1rem;border-radius:9999px;font-size:0.82rem;font-weight:600;margin-top:0.7rem;
}}
.top5-item{{
  display:flex;align-items:center;gap:1rem;padding:0.9rem 1rem;
  border-radius:var(--radius-sm);border:1px solid var(--border);
  margin-bottom:0.6rem;background:#fafaf8;transition:box-shadow 0.15s;
}}
.top5-item:hover{{box-shadow:0 2px 12px rgba(1,105,111,0.12)}}
.top5-item:first-child{{background:linear-gradient(135deg,#f0f9f8,#e8f5f4);border-color:var(--teal-bg)}}
.rank{{font-size:1.4rem;font-weight:900;color:var(--teal);width:2.2rem;text-align:center;flex-shrink:0}}
.country-flag{{font-size:1.6rem;flex-shrink:0}}
.country-info{{flex:1}}
.country-name{{font-size:1rem;font-weight:700}}
.country-meta{{font-size:0.8rem;color:var(--muted);margin-top:0.2rem}}
.sim-bar-wrap{{width:80px;flex-shrink:0}}
.sim-pct{{font-size:0.75rem;font-weight:700;color:var(--teal);text-align:right;margin-bottom:3px}}
.sim-bar{{height:4px;border-radius:9999px;background:#e0e0e0;overflow:hidden}}
.sim-bar-fill{{height:100%;background:linear-gradient(to right,var(--teal),var(--teal-light));border-radius:9999px}}
.chart-wrap{{position:relative;height:320px;margin-top:0.5rem}}
.stats-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;margin-top:0.5rem}}
.stat-box{{background:var(--bg);border-radius:var(--radius-sm);padding:0.9rem;text-align:center}}
.stat-val{{font-size:1.1rem;font-weight:700;color:var(--teal)}}
.stat-label{{font-size:0.72rem;color:var(--muted);margin-top:0.2rem}}
footer{{text-align:center;padding:2rem;font-size:0.78rem;color:var(--faint)}}
@media(max-width:480px){{
  .stats-grid{{grid-template-columns:1fr 1fr}}
  .sim-bar-wrap{{display:none}}
}}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-badge">🌍 WORLD HAPPINESS REPORT 2021 · 149개국</div>
  <h1>나의 가치관으로<br>찾는 나라</h1>
  <p>5개의 슬라이더로 삶의 가치관을 설정하면<br>가장 잘 맞는 나라 TOP 5를 찾아드려요</p>
</div>

<div class="container">
  <div class="card">
    <div class="card-title">💡 나의 가치관 설정</div>

    <div class="slider-wrap">
      <div class="slider-header"><span class="slider-label-text">💰 경제적 풍요</span><span class="slider-val" id="v0">3</span></div>
      <input type="range" min="1" max="5" value="3" id="s0" oninput="updateSlider(0,this.value)">
      <div class="slider-ends"><span>소박한 삶으로 충분해</span><span>경제적 성공이 우선이야</span></div>
    </div>

    <div class="slider-wrap">
      <div class="slider-header"><span class="slider-label-text">🤝 공동체 연대</span><span class="slider-val" id="v1">3</span></div>
      <input type="range" min="1" max="5" value="3" id="s1" oninput="updateSlider(1,this.value)">
      <div class="slider-ends"><span>나 혼자도 괜찮아 (개인)</span><span>가족·이웃이 중요해 (공동체)</span></div>
    </div>

    <div class="slider-wrap">
      <div class="slider-header"><span class="slider-label-text">🌿 건강·워라밸</span><span class="slider-val" id="v2">3</span></div>
      <input type="range" min="1" max="5" value="3" id="s2" oninput="updateSlider(2,this.value)">
      <div class="slider-ends"><span>빠른 성취가 중요해</span><span>건강하고 여유로운 삶</span></div>
    </div>

    <div class="slider-wrap">
      <div class="slider-header"><span class="slider-label-text">🗽 자유와 선택</span><span class="slider-val" id="v3">3</span></div>
      <input type="range" min="1" max="5" value="3" id="s3" oninput="updateSlider(3,this.value)">
      <div class="slider-ends"><span>안정·규칙을 선호해</span><span>내 삶은 내가 결정해</span></div>
    </div>

    <div class="slider-wrap">
      <div class="slider-header"><span class="slider-label-text">🏛️ 사회 청렴도</span><span class="slider-val" id="v4">3</span></div>
      <input type="range" min="1" max="5" value="3" id="s4" oninput="updateSlider(4,this.value)">
      <div class="slider-ends"><span>현실적 타협도 OK</span><span>투명·공정한 사회 필수</span></div>
    </div>

    <button class="btn" onclick="analyze()">🔍 나에게 맞는 나라 찾기</button>
  </div>

  <div id="result">
    <div class="result-hero">
      <div class="top-flag" id="res-flag">🌐</div>
      <h2 id="res-country">—</h2>
      <p>당신의 가치관과 가장 잘 맞는 나라</p>
      <div class="cluster-badge" id="res-cluster">—</div>
    </div>

    <div class="card">
      <div class="card-title">🏅 당신과 맞는 나라 TOP 5</div>
      <div id="top5-list"></div>
    </div>

    <div class="card">
      <div class="card-title">📊 가치관 레이더 차트</div>
      <div class="chart-wrap"><canvas id="radarChart"></canvas></div>
    </div>

    <div class="card">
      <div class="card-title">📈 <span id="stats-country-name">—</span> 주요 지표</div>
      <div class="stats-grid" id="stats-grid"></div>
    </div>
  </div>
</div>

<footer>
  데이터: World Happiness Report 2021 (Kaggle) · 149개국 실제 데이터<br>
  K-Means Clustering (k=6) + Cosine Similarity 기반
</footer>

<script>
const COUNTRIES = {countries_js};
const X_SCALED  = {x_scaled_js};
const SCALER    = {scaler_js};

const CLUSTER_NAMES = {{
  0:"🏙️ 중간 소득·성장형",
  1:"⚡ 자유롭고 활기찬 선진국형",
  2:"🏆 북유럽 복지국가형",
  3:"🌿 공동체 중심 개도국형",
  4:"🌄 저개발·생존형",
  5:"🌍 중진국·전환기형"
}};

const MEDALS = ["🥇","🥈","🥉","4️⃣","5️⃣"];
let radarChart = null;

function updateSlider(i, val) {{
  document.getElementById('v'+i).textContent = val;
  const inp = document.getElementById('s'+i);
  const pct = ((val-1)/4)*100;
  inp.style.background = `linear-gradient(to right,var(--teal) 0%,var(--teal) ${{pct}}%,#e0e0e0 ${{pct}}%,#e0e0e0 100%)`;
}}
[0,1,2,3,4].forEach(i => updateSlider(i,3));

function cosineSim(a, b) {{
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){{dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}}
  return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-10);
}}

function analyze() {{
  const raw = [
    (parseInt(document.getElementById('s0').value)-1)/4,
    (parseInt(document.getElementById('s1').value)-1)/4,
    (parseInt(document.getElementById('s2').value)-1)/4,
    (parseInt(document.getElementById('s3').value)-1)/4,
    1-(parseInt(document.getElementById('s4').value)-1)/4
  ];

  const sims = X_SCALED.map(row => cosineSim(raw, row));
  const indexed = sims.map((s,i) => ({{idx:i,sim:s}}));
  indexed.sort((a,b) => b.sim-a.sim);
  const top5 = indexed.slice(0,5);
  const top1c = COUNTRIES[top5[0].idx];

  document.getElementById('result').style.display='block';
  document.getElementById('res-flag').textContent = top1c.flag||'🌐';
  document.getElementById('res-country').textContent = top1c.country+'형 삶이 어울려요!';
  document.getElementById('res-cluster').textContent = CLUSTER_NAMES[top1c.cluster]||'';

  const listEl = document.getElementById('top5-list');
  listEl.innerHTML = '';
  top5.forEach((item,rank) => {{
    const c = COUNTRIES[item.idx];
    const simPct = Math.round(item.sim*100);
    listEl.innerHTML += `
      <div class="top5-item">
        <div class="rank">${{MEDALS[rank]}}</div>
        <div class="country-flag">${{c.flag||'🌐'}}</div>
        <div class="country-info">
          <div class="country-name">${{c.country}}</div>
          <div class="country-meta">행복지수 ${{c.happiness.toFixed(2)}} &nbsp;·&nbsp; ${{CLUSTER_NAMES[c.cluster]||''}}</div>
        </div>
        <div class="sim-bar-wrap">
          <div class="sim-pct">${{simPct}}%</div>
          <div class="sim-bar"><div class="sim-bar-fill" style="width:${{simPct}}%"></div></div>
        </div>
      </div>`;
  }});

  const userDisplay = [
    (parseInt(document.getElementById('s0').value)-1)/4,
    (parseInt(document.getElementById('s1').value)-1)/4,
    (parseInt(document.getElementById('s2').value)-1)/4,
    (parseInt(document.getElementById('s3').value)-1)/4,
    (parseInt(document.getElementById('s4').value)-1)/4,
  ];
  const top1Row = X_SCALED[top5[0].idx];
  const top1Display = [top1Row[0],top1Row[1],top1Row[2],top1Row[3],1-top1Row[4]];
  const labels = ['경제적 풍요','공동체 연대','건강·워라밸','자유와 선택','사회 청렴도'];

  if(radarChart) radarChart.destroy();
  const ctx = document.getElementById('radarChart').getContext('2d');
  radarChart = new Chart(ctx, {{
    type:'radar',
    data:{{
      labels,
      datasets:[
        {{
          label:'나의 가치관',data:userDisplay,
          backgroundColor:'rgba(1,105,111,0.15)',borderColor:'#01696f',borderWidth:2.5,
          pointBackgroundColor:'#01696f',pointRadius:5
        }},
        {{
          label:top1c.country,data:top1Display,
          backgroundColor:'rgba(79,152,163,0.10)',borderColor:'#4f98a3',borderWidth:2,
          borderDash:[5,4],pointBackgroundColor:'#4f98a3',pointRadius:4
        }}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      scales:{{r:{{min:0,max:1,ticks:{{stepSize:0.25,font:{{size:10}}}},pointLabels:{{font:{{size:12,weight:'600'}}}}}}}},
      plugins:{{legend:{{position:'bottom',labels:{{font:{{size:12,family:"'Noto Sans KR',sans-serif"}}}}}}}}
    }}
  }});

  document.getElementById('stats-country-name').textContent = top1c.country;
  document.getElementById('stats-grid').innerHTML = `
    <div class="stat-box"><div class="stat-val">${{top1c.happiness.toFixed(2)}}</div><div class="stat-label">😊 행복지수</div></div>
    <div class="stat-box"><div class="stat-val">${{top1c.gdp.toFixed(2)}}</div><div class="stat-label">💰 GDP(로그)</div></div>
    <div class="stat-box"><div class="stat-val">${{top1c.health.toFixed(1)}}세</div><div class="stat-label">🌿 건강수명</div></div>
    <div class="stat-box"><div class="stat-val">${{top1c.social.toFixed(3)}}</div><div class="stat-label">🤝 사회적 지지</div></div>
    <div class="stat-box"><div class="stat-val">${{top1c.freedom.toFixed(3)}}</div><div class="stat-label">🗽 자유도</div></div>
    <div class="stat-box"><div class="stat-val">${{top1c.corruption.toFixed(3)}}</div><div class="stat-label">🏛️ 부패인식</div></div>
  `;

  document.getElementById('result').scrollIntoView({{behavior:'smooth',block:'start'}});
}}
</script>
</body>
</html>"""

# ── 8. HTML 파일 저장 ─────────────────────────────────────────────
with open("나의_가치관으로_찾는_나라.html", "w", encoding="utf-8") as f:
    f.write(html)

print("완료! 나의_가치관으로_찾는_나라.html 파일을 열어보세요.")
