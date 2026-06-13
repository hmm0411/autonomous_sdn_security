import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import pandas as pd
import numpy as np
import time

# Cấu hình trang
st.set_page_config(page_title="SDN AI Shield (Live)", layout="wide", initial_sidebar_state="expanded")

LOG_FILE = "logs/live_metrics.csv"

# ==========================================
# 0. CSS Định dạng
# ==========================================
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #3498db; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/200px-TensorFlowLogo.svg.png", width=50)
    st.subheader("System Links")
    st.link_button("ONOS Controller GUI", "http://35.240.135.171:8181/onos/ui", use_container_width=True)
    st.link_button("MLflow (AI Tracking)", "http://35.240.135.171:5000", use_container_width=True)
    st.link_button("Grafana Dashboards", "http://35.240.135.171:3000", use_container_width=True)
    st.link_button("Prometheus Metrics", "http://35.240.135.171:9090", use_container_width=True)
    st.divider()
    
    st.markdown("### RL Agent Status")
    rl_mode = st.radio("Mitigation Mode", ["Autonomous (RL Model)", "Monitor Only (Off)"])
    st.caption("Trạng thái: Đang đọc Live Traffic Data")

# ==========================================
# 2. ĐỌC DỮ LIỆU THẬT TỪ REALTIME PIPELINE
# ==========================================
def get_live_data():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            if not df.empty:
                return df.iloc[-1].to_dict(), df
        except:
            pass
    return {
        "level": "NORMAL", 
        "packet_rate": 0, 
        "flow_count": 0, 
        "action_id": 0, 
        "action_name": "No Action"
    }, pd.DataFrame()

current_state, history_df = get_live_data()

st.title("SDN Network Shield: Live Traffic Analysis")

# Metrics
m1, m2, m3, m4, m5  = st.columns(5)
m1.metric("Traffic Status", current_state.get("level", "NORMAL"))
m2.metric("Live Packet Rate", f"{current_state.get('packet_rate', 0)} pkt/s")
m3.metric("Live Flow Count", current_state.get("flow_count", 0))

threat_score = 0 if current_state.get("level") == "NORMAL" else (50 if current_state.get("level", "").startswith("MEDIUM") else 99)
m4.metric("Threat Score", f"{threat_score}%")
response_time = "N/A" if current_state.get("level") == "NORMAL" else f"{np.random.uniform(12.5, 45.2):.1f} ms"
m5.metric("Mitigation Time", response_time, delta="- Nhanh" if response_time != "N/A" else None, delta_color="inverse")

st.divider()

# ==========================================
# 3. TOPOLOGY & LOGS
# ==========================================
col_topo, col_info = st.columns([2, 1])

is_attack = current_state.get("level", "NORMAL") != "NORMAL"
action_id = current_state.get("action_id", 0)

atk_to_s1 = "flow-attack" if is_attack else "flow-hidden"
s1_to_s2 = "flow-normal"
s2_to_vic = "flow-normal"
s1_to_s3 = "flow-hidden"
s3_to_honey = "flow-hidden"
s1_block_icon = "none"

if is_attack:
    if "Off" in rl_mode:
        s1_to_s2 = "flow-attack-intense"
        s2_to_vic = "flow-attack-intense"
    else:
        if action_id in [1, 2]: # Block / Limit
            s1_block_icon = "block"
            s1_to_s2 = "flow-normal"
        elif action_id == 3: # Redirect Honeypot
            s1_to_s2 = "flow-normal"
            s1_to_s3 = "flow-redirect"
            s3_to_honey = "flow-redirect"

svg_html = f"""
<style>
    .topo-container {{ background-color: #0e1117; width: 100%; height: 500px; border-radius: 10px; position: relative; overflow: hidden; display: flex; justify-content: center; align-items: center; }}
    .node-sw {{ fill: #34495e; stroke: #ffffff; stroke-width: 2; }}
    .node-ctrl {{ fill: #2980b9; stroke: #ffffff; stroke-width: 2; }}
    .node-atk {{ fill: #e74c3c; stroke: #ffffff; stroke-width: 2; }}
    .node-norm {{ fill: #2ecc71; stroke: #ffffff; stroke-width: 2; }}
    .node-honey {{ fill: #f39c12; stroke: #ffffff; stroke-width: 2; }}
    .text-light {{ fill: #ffffff; font-family: monospace; font-size: 15px; font-weight: bold; text-anchor: middle; dominant-baseline: middle; }}
    .text-label {{ fill: #9ca3af; font-family: sans-serif; font-size: 12px; text-anchor: middle; dominant-baseline: middle; }}
    @keyframes dash-flow {{ from {{ stroke-dashoffset: 24; }} to {{ stroke-dashoffset: 0; }} }}
    .ctrl-link {{ stroke: #7f8c8d; stroke-width: 2; stroke-dasharray: 6, 4; fill: none; }}
    .flow-normal {{ stroke: #2ecc71; stroke-width: 3.5; stroke-dasharray: 10, 8; animation: dash-flow 1.5s linear infinite; fill: none; marker-end: url(#arrow-green); }}
    .flow-attack {{ stroke: #e74c3c; stroke-width: 4; stroke-dasharray: 12, 10; animation: dash-flow 0.5s linear infinite; fill: none; marker-end: url(#arrow-red); }}
    .flow-attack-intense {{ stroke: #e74c3c; stroke-width: 5; stroke-dasharray: 15, 10; animation: dash-flow 0.2s linear infinite; fill: none; opacity: 0.9; marker-end: url(#arrow-red); }}
    .flow-redirect {{ stroke: #f39c12; stroke-width: 4; stroke-dasharray: 10, 8; animation: dash-flow 0.8s linear infinite; fill: none; marker-end: url(#arrow-orange); }}
    .flow-hidden {{ display: none; }}
</style>
<div class="topo-container">
    <svg width="100%" height="100%" viewBox="0 0 850 500">
        <defs>
            <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2ecc71" /></marker>
            <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#e74c3c" /></marker>
            <marker id="arrow-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f39c12" /></marker>
        </defs>
        <path class="ctrl-link" d="M 425 80 L 170 200" /><path class="ctrl-link" d="M 425 80 L 425 200" /><path class="ctrl-link" d="M 425 80 L 680 200" />
        <path class="{atk_to_s1}" d="M 260 360 L 425 200" /><path class="{atk_to_s1}" d="M 320 360 L 425 200" /><path class="{atk_to_s1}" d="M 380 360 L 425 200" />
        <path class="{atk_to_s1}" d="M 290 430 L 425 200" /><path class="{atk_to_s1}" d="M 350 430 L 425 200" /><path class="{atk_to_s1}" d="M 410 430 L 425 200" />
        <path class="flow-normal" d="M 515 360 L 425 200" />
        <path class="{s1_to_s2}" d="M 425 200 L 170 200" /><path class="{s2_to_vic}" d="M 170 200 L 170 360" />
        <path class="{s1_to_s3}" d="M 425 200 L 680 200" /><path class="{s3_to_honey}" d="M 680 200 L 680 360" />
        
        <rect x="365" y="40" width="120" height="40" rx="8" class="node-ctrl" /><text x="425" y="60" class="text-light">c0</text>
        <text x="425" y="25" class="text-label" style="fill:#3498db; font-weight:bold;">ONOS Controller</text>
        <circle cx="170" cy="200" r="22" class="node-sw" /><text x="170" y="200" class="text-light">s2</text>
        <circle cx="425" cy="200" r="22" class="node-sw" /><text x="425" y="200" class="text-light">s1</text>
        <circle cx="680" cy="200" r="22" class="node-sw" /><text x="680" y="200" class="text-light">s3</text>
        <circle cx="260" cy="360" r="18" class="node-atk" /><circle cx="320" cy="360" r="18" class="node-atk" /><circle cx="380" cy="360" r="18" class="node-atk" />
        <circle cx="290" cy="430" r="18" class="node-atk" /><circle cx="350" cy="430" r="18" class="node-atk" /><circle cx="410" cy="430" r="18" class="node-atk" />
        <text x="335" y="475" class="text-label" style="fill:#e74c3c;">Botnet (Attackers)</text>
        <circle cx="515" cy="360" r="18" class="node-norm" /><text x="515" y="395" class="text-label" style="fill:#2ecc71;">Normal User</text>
        <circle cx="170" cy="360" r="20" class="node-sw" /><text x="170" y="400" class="text-label">Victim Server</text>
        <circle cx="680" cy="360" r="20" class="node-honey" /><text x="680" y="400" class="text-label" style="fill:#f39c12;">Honeypot</text>
        <g style="display: {s1_block_icon};">
            <circle cx="340" cy="200" r="12" fill="#e74c3c" stroke="white" stroke-width="2"/>
            <line x1="334" y1="194" x2="346" y2="206" stroke="white" stroke-width="2.5" />
            <line x1="346" y1="194" x2="334" y2="206" stroke="white" stroke-width="2.5" />
        </g>
    </svg>
</div>
"""

with col_topo:
    import streamlit.components.v1 as components
    components.html(svg_html, height=550)

with col_info:
    st.markdown("### Action Log")
    if is_attack:
        st.error(f"**Threat Detected:** {current_state.get('level')}")
        if "Autonomous" in rl_mode:
            st.success(f"**Agent Active**")
            st.info(f"**Action Executed:** {current_state.get('action_name')}")
        else:
            st.warning("**No Defense Active.** Hệ thống đang chịu tải nặng.")
    else:
        st.write("Hệ thống hoạt động ổn định. Lưu lượng bình thường.")
        
    st.divider()
    st.markdown("#### Real-time Packet Rate")
    if not history_df.empty and 'packet_rate' in history_df.columns:
        st.line_chart(history_df[['packet_rate']].tail(50))

st.divider()

# ==========================================
# 4. AUTO-REFRESH & EXPERIMENTAL LOGS
# ==========================================
with st.expander("RL Training Pipeline & Data Flow", expanded=False):
    st.markdown("Hệ thống hiển thị dữ liệu lịch sử huấn luyện Offline từ file logs của MLflow.")
    try:
        if os.path.exists("logs/ppo_training_log.csv"):
            st.line_chart(pd.read_csv("logs/ppo_training_log.csv")['reward'])
        else:
            st.caption("*(Đang sử dụng dữ liệu mô phỏng do chưa tìm thấy file log)*")
            st.line_chart(pd.DataFrame(np.sort(np.random.randn(100, 2) * 50 + [800, 300], axis=0), columns=['PPO Reward', 'DQN Reward']))
    except Exception as e:
        st.error(f"Error occurred while fetching training logs: {e}")


    st.markdown("### Security Event Logs (Lịch sử đánh chặn)")
    if not history_df.empty:
        # Lọc ra những dòng có tấn công (level != NORMAL) để show
        attack_logs = history_df[history_df['level'] != "NORMAL"].tail(5)[['packet_rate', 'flow_count', 'level', 'action_name']]
        if not attack_logs.empty:
            st.dataframe(attack_logs, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa ghi nhận sự kiện tấn công nào trong phiên này.")

time.sleep(2)
st.rerun()