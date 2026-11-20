import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 페이지 기본 설정
st.set_page_config(
    page_title="V2X & 퍼지 기반 지능형 방어운전 대시보드",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------------
# 1. 퍼지 소속 함수 (삼각형 / 사다리꼴)
# ---------------------------------------------------------
def trimf(x, abc):
    a, b, c = abc
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def trapmf(x, abcd):
    a, b, c, d = abcd
    left = (x - a) / (b - a + 1e-9) if b > a else np.where(x >= a, 1.0, 0.0)
    right = (d - x) / (d - c + 1e-9) if d > c else np.where(x <= d, 1.0, 0.0)
    return np.maximum(0, np.minimum(np.minimum(left, 1.0), right))

# ---------------------------------------------------------
# 2. 맘다니(Mamdani) 퍼지 추론 및 무게중심법 역퍼지화 엔진
# ---------------------------------------------------------
def run_fuzzy_inference(dist, lat_speed, vis_block, weather_noise, v2x_delay):
    # (1) 퍼지화 (Fuzzification)
    # 차간 거리 (5~100m)
    d_close = float(trapmf(dist, [0, 0, 15, 35]))
    d_med   = float(trimf(dist, [20, 45, 70]))
    d_far   = float(trapmf(dist, [55, 75, 100, 100]))

    # V2X 측면 차량 접근 속도 (0~100 km/h)
    v_slow  = float(trapmf(lat_speed, [0, 0, 20, 40]))
    v_med   = float(trimf(lat_speed, [25, 50, 75]))
    v_fast  = float(trapmf(lat_speed, [60, 80, 100, 100]))

    # 전방 시야 차단 정도 (0~100 %)
    b_clear = float(trapmf(vis_block, [0, 0, 20, 40]))
    b_part  = float(trimf(vis_block, [25, 50, 75]))
    b_high  = float(trapmf(vis_block, [60, 80, 100, 100]))

    # (2) IF-THEN 방어운전 규칙 베이스 (최소 연산 MIN 적용)
    r1 = min(b_high, v_fast)          # 시야 차단 높음 AND 측면 접근 빠름 -> 위험 매우 높음
    r2 = min(d_close, v_fast)         # 차간 거리 가까움 AND 측면 접근 빠름 -> 위험 매우 높음
    r3 = d_close                      # 차간 거리 가까움 -> 위험 높음
    r4 = min(b_high, v_med)           # 시야 차단 높음 AND 측면 접근 보통 -> 위험 높음
    r5 = min(b_part, v_fast)          # 시야 일부 차단 AND 측면 접근 빠름 -> 위험 높음
    r6 = min(d_med, v_med)            # 차간 거리 보통 AND 측면 접근 보통 -> 위험 보통
    r7 = min(b_part, v_med)           # 시야 일부 차단 AND 측면 접근 보통 -> 위험 보통
    r8 = min(d_far, v_slow, b_clear)  # 거리 멀고, 느리고, 시야 양호 -> 위험 낮음

    # 후건부 출력 집합 (위험도 0~100)
    x_risk = np.linspace(0, 100, 201)
    risk_low  = trapmf(x_risk, [0, 0, 20, 40])
    risk_med  = trimf(x_risk, [25, 50, 75])
    risk_high = trapmf(x_risk, [60, 80, 100, 100])

    # 규칙별 활성도 결합 (MAX 연산)
    act_high = max(r1, r2, r3, r4, r5)
    act_med  = max(r6, r7)
    act_low  = max(r8, 0.05)  # 기본 안전 기저치

    agg_low  = np.minimum(act_low, risk_low)
    agg_med  = np.minimum(act_med, risk_med)
    agg_high = np.minimum(act_high, risk_high)
    aggregated = np.maximum(agg_low, np.maximum(agg_med, agg_high))

    # (3) 무게중심법(Centroid) 역퍼지화
    if np.sum(aggregated) == 0:
        raw_risk = 0.0
    else:
        raw_risk = float(np.sum(x_risk * aggregated) / np.sum(aggregated))

    # (4) 확신도(Certainty Factor) 산출 및 불확실성 보정
    sensor_conf = 1.0 - (weather_noise / 100.0) * 0.65
    comm_conf   = 1.0 - (v2x_delay / 100.0) * 0.65
    certainty   = float(np.clip(sensor_conf * comm_conf, 0.25, 1.0))

    # 확신도가 낮으면 급제동 대신 완만한 감속(상한 제한) 및 단계별 경고 우선
    if certainty < 0.65:
        decel_control = min(raw_risk * certainty * 0.85, 55.0)
        steer_assist  = raw_risk * certainty * 0.5
        control_mode  = "불확실성 보정 모드 (급감속 억제 · 완만 감속 및 단계별 경고)"
    else:
        decel_control = raw_risk * certainty
        steer_assist  = raw_risk * certainty * 0.75
        control_mode  = "정상 선제적 방어운전 모드 (예비 감속 및 안전거리 확보)"

    rules_info = [
        {"규칙": "R1: 시야 차단 높음 AND 측면 접근 빠름 → 위험 높음", "적합도": round(r1, 2)},
        {"규칙": "R2: 차간 거리 가까움 AND 측면 접근 빠름 → 위험 높음", "적합도": round(r2, 2)},
        {"규칙": "R3: 차간 거리 가까움 → 위험 높음", "적합도": round(r3, 2)},
        {"규칙": "R4: 시야 차단 높음 AND 측면 접근 보통 → 위험 높음", "적합도": round(r4, 2)},
        {"규칙": "R5: 시야 일부 차단 AND 측면 접근 빠름 → 위험 높음", "적합도": round(r5, 2)},
        {"규칙": "R6: 차간 거리 보통 AND 측면 접근 보통 → 위험 보통", "적합도": round(r6, 2)},
        {"규칙": "R7: 시야 일부 차단 AND 측면 접근 보통 → 위험 보통", "적합도": round(r7, 2)},
        {"규칙": "R8: 거리 멀음 AND 측면 느림 AND 시야 양호 → 위험 낮음", "적합도": round(r8, 2)},
    ]

    return {
        "raw_risk": raw_risk,
        "certainty": certainty * 100.0,
        "decel_control": decel_control,
        "steer_assist": steer_assist,
        "control_mode": control_mode,
        "x_risk": x_risk,
        "risk_sets": (risk_low, risk_med, risk_high),
        "aggregated": aggregated,
        "rules_info": rules_info,
    }

# ---------------------------------------------------------
# 3. 세션 상태 초기화 및 원클릭 시나리오 프리셋
# ---------------------------------------------------------
defaults = {
    "dist": 45,
    "lat_speed": 30,
    "vis_block": 20,
    "weather_noise": 10,
    "v2x_delay": 10,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def set_preset(dist, lat_speed, vis_block, weather_noise, v2x_delay):
    st.session_state.dist = dist
    st.session_state.lat_speed = lat_speed
    st.session_state.vis_block = vis_block
    st.session_state.weather_noise = weather_noise
    st.session_state.v2x_delay = v2x_delay

# ---------------------------------------------------------
# 4. 사이드바 UI (입력 제어 & 시나리오 버튼)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 주행 환경 시뮬레이터")
    st.caption("원클릭 시나리오 프리셋")
    st.button("🟢 일반 안전 주행", use_container_width=True,
              on_click=set_preset, args=(75, 15, 10, 5, 5))
    st.button("🚨 사각지대 교차로 측면 접근", use_container_width=True,
              on_click=set_preset, args=(40, 82, 85, 10, 10))
    st.button("🌧️ 악천후 + V2X 통신 지연", use_container_width=True,
              on_click=set_preset, args=(22, 70, 75, 80, 75))

    st.divider()
    st.subheader("온보드 센서 & V2X 입력")
    dist = st.slider("전방 차간 거리 (m)", 5, 100, key="dist")
    lat_speed = st.slider("측면 차량 접근 속도 - V2X (km/h)", 0, 100, key="lat_speed")
    vis_block = st.slider("전방 시야 차단 정도 (%)", 0, 100, key="vis_block")

    st.subheader("불확실성 요인 (확신도 산출)")
    weather_noise = st.slider("악천후 / 센서 노이즈 (%)", 0, 100, key="weather_noise")
    v2x_delay = st.slider("V2X 통신 지연율 (%)", 0, 100, key="v2x_delay")

# 추론 실행
res = run_fuzzy_inference(dist, lat_speed, vis_block, weather_noise, v2x_delay)

# ---------------------------------------------------------
# 5. 메인 화면 구성
# ---------------------------------------------------------
st.title("🚗 V2X & 퍼지 전문가 시스템 기반 지능형 방어운전 대시보드")
st.markdown(
    "온보드 센서와 **V2X 비가시 영역 데이터**를 융합하고, **맘다니(Mamdani) 퍼지 추론 및 확신도 보정**을 통해 "
    "선제적 감속·조향 제어를 시뮬레이션하는 서비스 대시보드입니다."
)

# 핵심 KPI 카드
c1, c2, c3, c4 = st.columns(4)
risk_val = res["raw_risk"]
risk_grade = "높음 (위험)" if risk_val >= 65 else ("보통 (주의)" if risk_val >= 35 else "낮음 (안전)")

c1.metric("방어운전 위험도", f"{risk_val:.1f} / 100", risk_grade)
c2.metric("데이터 확신도", f"{res['certainty']:.1f} %",
          "불확실성 높음" if res["certainty"] < 65 else "신뢰도 양호")
c3.metric("선제적 감속 제어", f"{res['decel_control']:.1f} %")
c4.metric("조향 회피 보조", f"{res['steer_assist']:.1f} %")

# 실시간 경고 및 제어 상태 배너
if res["certainty"] < 65 and risk_val >= 50:
    st.warning(
        f"⚠️ **{res['control_mode']}** — 악천후·통신 지연으로 데이터 불확실성이 높습니다. "
        "오작동 급제동을 방지하기 위해 **완만한 감속과 단계별 운전자 경고**를 우선 수행합니다."
    )
elif risk_val >= 65:
    st.error(
        f"🚨 **위험도 '높음' 감지 ({res['control_mode']})** — 교차로 사각지대 진입 전 "
        "운전자에게 시각·청각 경고를 제공하고 **선제적 서행(예비 감속) 및 안전거리 확보 제어**를 실행합니다."
    )
elif risk_val >= 35:
    st.info(f"ℹ️ **주의 구간 ({res['control_mode']})** — 잠재적 위험 요소를 모니터링하며 감속 준비 상태를 유지합니다.")
else:
    st.success(f"✅ **안전 주행 구간 ({res['control_mode']})** — 현재 주행 환경이 안정적입니다.")

# 하단 시각화 탭
tab1, tab2, tab3 = st.tabs(["📊 퍼지 추론 및 역퍼지화", "⚖️ 기존 AEB vs 제안 모델 비교", "📜 퍼지 규칙 활성도"])

with tab1:
    st.subheader("맘다니 퍼지 추론 결과 및 무게중심법(Centroid) 역퍼지화")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = res["x_risk"]
    r_low, r_med, r_high = res["risk_sets"]

    ax.plot(x, r_low, linestyle="--", alpha=0.5, label="Low Risk")
    ax.plot(x, r_med, linestyle="--", alpha=0.5, label="Medium Risk")
    ax.plot(x, r_high, linestyle="--", alpha=0.5, label="High Risk")
    ax.fill_between(x, 0, res["aggregated"], color="#ff6b6b", alpha=0.45, label="Aggregated Output")
    ax.axvline(risk_val, color="#c92a2a", linewidth=2.5, label=f"Centroid Risk = {risk_val:.1f}")

    ax.set_xlabel("Defensive Driving Risk Score (0 - 100)")
    ax.set_ylabel("Membership Degree")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

with tab2:
    st.subheader("동일 주행 상황에서의 제어 반응 비교")
    # 기존 AEB 로직 (단일 센서 기반 고정 임계값 이진 결정)
    aeb_brake = 100.0 if (dist <= 20 or (weather_noise >= 70 and dist <= 30)) else 0.0
    aeb_status = (
        "최대 급제동 발동 (악천후 노이즈 오작동 위험)" if (aeb_brake == 100.0 and weather_noise >= 70)
        else ("최대 급제동 발동 (충돌 직전 대응)" if aeb_brake == 100.0
              else "미작동 (사각지대 측면 V2X 차량 인식 불가)")
    )

    comp_df = pd.DataFrame({
        "구분": ["제어 방식", "비가시 영역(V2X) 대응", "현재 감속 제어량 (%)", "시스템 판단 결과"],
        "기존 AEB (고정 임계값)": [
            "이진 결정 (0% 또는 100% 급제동)",
            "불가 (직선 시야 내 객체만 인식)",
            f"{aeb_brake:.1f} %",
            aeb_status
        ],
        "제안 모델 (V2X + 퍼지 + 확신도)": [
            "연속적 선제 감속 및 불확실성 보정",
            "가능 (V2X 측면 접근 속도 실시간 융합)",
            f"{res['decel_control']:.1f} %",
            res["control_mode"]
        ]
    })
    st.table(comp_df)

    chart_df = pd.DataFrame({
        "감속 제어량 (%)": [aeb_brake, res["decel_control"]]
    }, index=["기존 AEB", "제안 방어운전 모델"])
    st.bar_chart(chart_df)

with tab3:
    st.subheader("IF-THEN 방어운전 규칙별 전건부 적합도")
    rules_df = pd.DataFrame(res["rules_info"])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)
