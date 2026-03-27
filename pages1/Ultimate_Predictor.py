import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
import os
import gc
import base64
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

st.title("⚡ Ultimate TFLN Modulator Predictor")
st.markdown("""
Instant physics prediction mapping **Raw Electrostatics** to **Full Broadband RF Performance**.  
All predictions are performed at **1330 nm**.
""")

# =====================================================================
# 1. SIDEBAR: GEOMETRY & SYSTEM INPUTS
# =====================================================================
st.sidebar.header("Geometry Parameters")

def user_input_features():
    st.sidebar.subheader("Global Device Parameters")
    length_cm = st.sidebar.number_input("Device Length (L) [cm]", value=1.65, min_value=0.4, max_value=10.0, step=0.1, format="%.2f")
    
    st.sidebar.subheader("Active Region")
    ws = st.sidebar.number_input("WS (Signal Width) [µm]", value=10.69, step=0.01, format="%.2f")
    gap = st.sidebar.number_input("GAP [µm]", value=7.36, step=0.01, format="%.2f")
    mtx = st.sidebar.number_input("MTX (Metal Thickness) [µm]", value=1.50, step=0.01, format="%.2f")
    cap_w = st.sidebar.number_input("CAP_W (Cap Width) [µm]", value=6.25, step=0.01, format="%.2f")
    
    st.sidebar.subheader("T-Structure Dimensions")
    l1 = st.sidebar.number_input("L1 (Inner Length) [µm]", value=4.0, step=0.1, format="%.1f")
    l2 = st.sidebar.number_input("L2 (Outer Length) [µm]", value=57.38, step=0.1, format="%.1f")
    w1 = st.sidebar.number_input("W1 (Inner Width) [µm]", value=10.69, step=0.1, format="%.1f")
    w2 = st.sidebar.number_input("W2 (Outer Width) [µm]", value=35.92, step=0.1, format="%.1f")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Electrical Environment")
    Zs = st.sidebar.number_input("Zs (Lab Source Impedance) [Ω]", value=50.0, step=1.0)
    Rt = st.sidebar.number_input("Rt (Termination Resistor) [Ω]", value=37.16, step=0.1)
    Zs_driver = st.sidebar.number_input("Zs_driver (Telecom Driver) [Ω]", value=65.0, step=1.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Fixed Parameters")
    st.sidebar.info("**Pitch:** 200 µm  \n**WG:** 70 µm  \n**CAP_H:** 1.4 µm  \n**ng:** 2.27")
    
    geom_list = [ws, gap, mtx, cap_w, l1, l2, w1, w2]
    params = {"WS": ws, "GAP": gap, "MTX": mtx, "CAP_W": cap_w, "L1": l1, "L2": l2, "W1": w1, "W2": w2}
    
    return length_cm, geom_list, params, Zs, Rt, Zs_driver

length_cm, geometry_list, params, Zs, Rt, Zs_driver = user_input_features()

# =====================================================================
# 2. SVG DRAWING FUNCTIONS
# =====================================================================
def render_svg(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="100%"/>'

def generate_exact_svg(p):
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W = p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0; BOTTOM_LAYER_H = 0.23; RIDGE_W = 0.8; RIDGE_H = 0.23; CAP_HEIGHT = 1.4
    C_ELEC = '#F5BD02'; C_SUB = '#00BFFF'; C_CAP = '#00BFFF'; C_LINE = 'black'
    
    CV_W = 800; CV_H = 600; CX = CV_W / 2
    
    def svg_arrow(x1, y1, x2, y2, text, text_loc="top", offset=10, font_size=14):
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1.5" marker-start="url(#arrow_start)" marker-end="url(#arrow_end)" />'
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        tx, ty = mx, my
        anchor = "middle"; dominant = "middle"
        if text_loc == "top": ty -= offset; dominant = "auto"
        elif text_loc == "bottom": ty += offset; dominant = "hanging"
        elif text_loc == "left": tx -= offset; anchor = "end"
        elif text_loc == "right": tx += offset; anchor = "start"
        return line + f'<text x="{tx}" y="{ty}" fill="{C_LINE}" font-family="sans-serif" font-size="{font_size}" font-weight="bold" text-anchor="{anchor}" dominant-baseline="{dominant}">{text}</text>'

    # TOP-DOWN
    SCALE_TOP = 3.5; CV_H_TOP = 700; CY_TOP = 380 
    def to_top(x, y): return CX + x*SCALE_TOP, CY_TOP - y*SCALE_TOP
    pts = [(GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
           (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
           (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
           (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)]
    poly_str = " ".join([f"{to_top(x,y)[0]},{to_top(x,y)[1]}" for x, y in pts])
    ws_x1, ws_y1 = to_top(-(GAP/2 + WS), 100)
    
    arrows_top = ""
    arrows_top += svg_arrow(*to_top(-(GAP/2 + WS), 120), *to_top(-GAP/2, 120), "WS", "top", 10)
    arrows_top += svg_arrow(*to_top(GAP/2, 120), *to_top(GAP/2 + WG, 120), "WG", "top", 10)
    arrows_top += svg_arrow(*to_top(-GAP/2, -120), *to_top(GAP/2, -120), "GAP", "bottom", 10)
    l1_x = GAP/2 - 20; arrows_top += svg_arrow(*to_top(l1_x, -L1/2), *to_top(l1_x, L1/2), "L1", "left", 10)
    l2_x = GAP/2 + W1 + W2 + 20; arrows_top += svg_arrow(*to_top(l2_x, -L2/2), *to_top(l2_x, L2/2), "L2", "right", 10)
    arrows_top += svg_arrow(*to_top(GAP/2, -L1/2 - 20), *to_top(GAP/2 + W1, -L1/2 - 20), "W1", "bottom", 10)
    arrows_top += svg_arrow(*to_top(GAP/2 + W1, L2/2 + 20), *to_top(GAP/2 + W1 + W2, L2/2 + 20), "W2", "top", 10)

    svg_top = f"""<svg width="{CV_W}" height="{CV_H_TOP}" viewBox="0 0 {CV_W} {CV_H_TOP}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        <text x="20" y="50" text-anchor="start" font-family="sans-serif" font-size="24" font-weight="bold">Top-Down View</text>
        <rect x="{ws_x1}" y="{ws_y1}" width="{WS*SCALE_TOP}" height="{200*SCALE_TOP}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />
        <polygon points="{poly_str}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />
        {arrows_top}
    </svg>"""

    # CROSS-SECTION
    SCALE_CS = 700.0 / ((WS/2 + GAP) + 1.0) / 2; CY_CS = 350 
    def to_cs(x, y): return CX + x*SCALE_CS, CY_CS - y*SCALE_CS
    
    sub_rect = f'<rect x="0" y="{to_cs(0, 0)[1]}" width="{CV_W}" height="{CV_H}" fill="{C_SUB}" />'
    bl_rect = f'<rect x="0" y="{to_cs(0, BOTTOM_LAYER_H)[1]}" width="{CV_W}" height="{BOTTOM_LAYER_H*SCALE_CS}" fill="black" />'
    ws_rect = f'<rect x="{to_cs(-WS/2, BOTTOM_LAYER_H + MTX)[0]}" y="{to_cs(-WS/2, BOTTOM_LAYER_H + MTX)[1]}" width="{WS*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    rwg_rect = f'<rect x="{to_cs(WS/2 + GAP, BOTTOM_LAYER_H + MTX)[0]}" y="{to_cs(WS/2 + GAP, BOTTOM_LAYER_H + MTX)[1]}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    lwg_rect = f'<rect x="{to_cs(-(WS/2 + GAP + 500), BOTTOM_LAYER_H + MTX)[0]}" y="{to_cs(-(WS/2 + GAP + 500), BOTTOM_LAYER_H + MTX)[1]}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    
    caps_svg = ""
    for cx in [WS/2 + GAP/2, -WS/2 - GAP/2]:
        caps_svg += f'<rect x="{to_cs(cx - CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT)[0]}" y="{to_cs(cx - CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT)[1]}" width="{CAP_W*SCALE_CS}" height="{CAP_HEIGHT*SCALE_CS}" fill="{C_CAP}" stroke="{C_LINE}" stroke-width="1.5" />'
        caps_svg += f'<rect x="{to_cs(cx - RIDGE_W/2, BOTTOM_LAYER_H + RIDGE_H)[0]}" y="{to_cs(cx - RIDGE_W/2, BOTTOM_LAYER_H + RIDGE_H)[1]}" width="{RIDGE_W*SCALE_CS}" height="{RIDGE_H*SCALE_CS}" fill="black" />'

    arr_y = BOTTOM_LAYER_H + max(MTX, CAP_HEIGHT) + (3.0 if MTX < 5 else 0.5 * MTX)
    arrows_cs = ""
    arrows_cs += svg_arrow(*to_cs(-WS/2, arr_y), *to_cs(WS/2, arr_y), "WS", "top", 15)
    arrows_cs += svg_arrow(*to_cs(WS/2, arr_y), *to_cs(WS/2 + GAP, arr_y), "GAP", "top", 15)
    arrows_cs += svg_arrow(*to_cs((-WS/2 - GAP/2) - CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT + 1.0), *to_cs((-WS/2 - GAP/2) + CAP_W/2, BOTTOM_LAYER_H + CAP_HEIGHT + 1.0), "CAP_W", "top", 15)
    arrows_cs += svg_arrow(*to_cs(-WS/2 + 2.0, BOTTOM_LAYER_H), *to_cs(-WS/2 + 2.0, BOTTOM_LAYER_H + MTX), "MTX", "right", 10)

    svg_cross = f"""<svg width="{CV_W}" height="{CV_H}" viewBox="0 0 {CV_W} {CV_H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        <text x="20" y="50" text-anchor="start" font-family="sans-serif" font-size="24" font-weight="bold">Cross-Section View</text>
        {sub_rect} {bl_rect} {ws_rect} {rwg_rect} {lwg_rect} {caps_svg} {arrows_cs}
    </svg>"""
    return svg_top, svg_cross

st.subheader("1. Geometry Visualization")
st.caption("Updated automatically based on sidebar.")
col1, col2 = st.columns([1, 1])
svg_t, svg_c = generate_exact_svg(params)
with col1: st.markdown(render_svg(svg_t), unsafe_allow_html=True)
with col2: st.markdown(render_svg(svg_c), unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 3. LOW-RAM SEQUENTIAL PREDICTOR ENGINE
# =====================================================================
def fit_alpha_scaled(f_GHz, a, b, c): return a * np.sqrt(f_GHz) + b * f_GHz + c

def calculate_eo_response(f_GHz, alpha_dB_cm, nm, Zc, L_device_m, ng, Zs, Rt):
    c0 = 299792458.0
    omega = 2 * np.pi * f_GHz * 1e9
    beta_opt, beta_rf = ng * omega / c0, nm * omega / c0
    alpha_Np_m = alpha_dB_cm * 100.0 / 8.686
    gamma = alpha_Np_m + 1j * beta_rf

    tanh_gL = np.tanh(gamma * L_device_m)
    zin = Zc * ((Rt + Zc * tanh_gL) / (Zc + Rt * tanh_gL))
    M = (Zs + zin) * (Rt + Zc)
    N = (Zs + zin) * (Rt - Zc)
    p1 = zin / (M * np.exp(gamma * L_device_m) + N * np.exp(-gamma * L_device_m))

    up = np.where(L_device_m * (alpha_Np_m + 1j * (beta_rf - beta_opt)) == 0, 1e-12, L_device_m * (alpha_Np_m + 1j * (beta_rf - beta_opt)))
    un = np.where(L_device_m * (-alpha_Np_m + 1j * (-beta_rf - beta_opt)) == 0, 1e-12, L_device_m * (-alpha_Np_m + 1j * (-beta_rf - beta_opt)))
    p2 = (Rt + Zc) * ((1 - np.exp(up)) / up) + (Rt - Zc) * ((1 - np.exp(un)) / un)

    s21_eo_abs = p1 * p2
    transfer_pad_abs = s21_eo_abs / (zin / (Zs + zin))
    
    idx_1GHz = np.argmin(np.abs(f_GHz - 1.0))
    s21_eo_norm_dB = 20 * np.log10(np.abs(s21_eo_abs) / np.abs(s21_eo_abs[idx_1GHz]))
    return s21_eo_norm_dB, transfer_pad_abs

def predict_sequentially(geometry, L_cm, Zs, Zs_driver, Rt):
    results = {}
    model_dir = Path("gp_surrogate_results_ultimate")
    
    if not model_dir.exists():
        st.error(f"❌ Folder '{model_dir}' not found.")
        return None

    progress = st.progress(0)
    geom_um = np.array(geometry)
    geom_m = geom_um / 1000.0
    WS, GAP, MTX, CAP_W, L1, L2, W1, W2 = geom_m
    bragg_proxy = (L1 + W1 + L2 + W2) * (WS / GAP)
    
    X_8dof = np.array([[CAP_W, GAP, L1, L2, MTX, W1, W2, WS]])
    X_9dof = np.array([[CAP_W, GAP, L1, L2, MTX, W1, W2, WS, bragg_proxy]])

    try:
        # --- 1. VPI ---
        progress.progress(0.25, text="Loading VPI Surrogate...")
        with open(model_dir / "gp_vpi_surrogate/scalers_VPI.pkl", 'rb') as f: meta_v = pickle.load(f)
        with open(model_dir / "gp_vpi_surrogate/gp_model_VPI.pkl", 'rb') as f: m_vpi = pickle.load(f)
        v_norm, v_std_norm = m_vpi.predict(meta_v['scaler_X'].transform(X_8dof), return_std=True)
        del m_vpi; gc.collect()
        
        vpi_log = meta_v['scaler_y'].inverse_transform(v_norm.reshape(-1,1)).ravel()[0]
        vpi_base = 10 ** vpi_log
        vpi_std = vpi_base * np.log(10) * v_std_norm[0] * meta_v['scaler_y'].scale_[0]
        results['vpi_L'] = (vpi_base / L_cm, vpi_std / L_cm)

        # --- 2. nm & Zc ---
        progress.progress(0.50, text="Loading Index & Impedance Surrogates...")
        with open(model_dir / "gp_nm_zc_surrogate/scalers_nm_zc.pkl", 'rb') as f: meta_nz = pickle.load(f)
        X_nz = meta_nz['scaler_X'].transform(X_9dof)
        
        with open(model_dir / "gp_nm_zc_surrogate/gp_model_nm_60.pkl", 'rb') as f: m_nm = pickle.load(f)
        nm_n, nm_s = m_nm.predict(X_nz, return_std=True)
        del m_nm; gc.collect()
        
        with open(model_dir / "gp_nm_zc_surrogate/gp_model_Zc_60.pkl", 'rb') as f: m_zc = pickle.load(f)
        zc_n, zc_s = m_zc.predict(X_nz, return_std=True)
        del m_zc; gc.collect()
        
        nm = meta_nz['scalers_y']['nm_60'].inverse_transform(nm_n.reshape(-1,1)).ravel()[0]
        zc = meta_nz['scalers_y']['Zc_60'].inverse_transform(zc_n.reshape(-1,1)).ravel()[0]
        results['nm'] = (nm, nm_s[0] * meta_nz['scalers_y']['nm_60'].scale_[0])
        results['zc'] = (zc, zc_s[0] * meta_nz['scalers_y']['Zc_60'].scale_[0])

        # --- 3. Alpha Anchors (WITH UNCERTAINTY PROPAGATION) ---
        progress.progress(0.75, text="Loading RF Attenuation Surrogates...")
        with open(model_dir / "gp_alpha_anchors/scaler_anchors.pkl", 'rb') as f: s_alpha = pickle.load(f)['scaler_X']
        with open(model_dir / "gp_alpha_anchors/gp_alpha_anchors_suite.pkl", 'rb') as f: m_alpha = pickle.load(f)
        X_a = s_alpha.transform(X_9dof)
        
        # Extract log10 predictions AND standard deviations
        y20, std20 = m_alpha['Alpha_20GHz_dB_cm'].predict(X_a, return_std=True)
        y60, std60 = m_alpha['Alpha_60GHz_dB_cm'].predict(X_a, return_std=True)
        y100, std100 = m_alpha['Alpha_100GHz_dB_cm'].predict(X_a, return_std=True)
        del m_alpha; gc.collect()

        # Nominal Linear Alphas
        a20_nom, a60_nom, a100_nom = 10**y20[0], 10**y60[0], 10**y100[0]
        
        # Worst-Case (Highest Attenuation = Lower Bandwidth Bound)
        a20_wc = 10**(y20[0] + 1.96*std20[0])
        a60_wc = 10**(y60[0] + 1.96*std60[0])
        a100_wc = 10**(y100[0] + 1.96*std100[0])

        # Best-Case (Lowest Attenuation = Upper Bandwidth Bound)
        a20_bc = 10**(y20[0] - 1.96*std20[0])
        a60_bc = 10**(y60[0] - 1.96*std60[0])
        a100_bc = 10**(y100[0] - 1.96*std100[0])

       # --- 4. Cascade Math for all 3 Scenarios ---
        progress.progress(0.95, text="Calculating Broadband Physics...")
        f_axis = np.linspace(1.0, 150.0, 500)
        L_m = L_cm / 100.0

        def get_bw(alphas):
            popt, _ = curve_fit(fit_alpha_scaled, [20.0, 60.0, 100.0], alphas)
            s21_lossy, _ = calculate_eo_response(f_axis, fit_alpha_scaled(f_axis, *popt), nm, zc, L_m, 2.27, Zs, Rt)
            if s21_lossy[-1] > -3.0: return 150.0, s21_lossy, popt
            idx = np.where(s21_lossy <= -3.0)[0][0]
            return f_axis[idx-1] + (f_axis[idx]-f_axis[idx-1])*(-3.0-s21_lossy[idx-1])/(s21_lossy[idx]-s21_lossy[idx-1]), s21_lossy, popt

        bw_nom, s21_nom, popt_nom = get_bw([a20_nom, a60_nom, a100_nom])
        bw_lower, _, popt_wc = get_bw([a20_wc, a60_wc, a100_wc]) # Worst attenuation = lower BW
        bw_upper, _, popt_bc = get_bw([a20_bc, a60_bc, a100_bc]) # Best attenuation = upper BW

        # VPI calculations
        _, t_lossless = calculate_eo_response(f_axis, np.zeros_like(f_axis), nm, zc, L_m, 2.27, Zs, Rt)
        _, t_lossy = calculate_eo_response(f_axis, fit_alpha_scaled(f_axis, *popt_nom), nm, zc, L_m, 2.27, Zs, Rt)
        
        idx_60 = np.argmin(np.abs(f_axis - 60.0))
        v_lossless = results['vpi_L'][0] / np.abs(t_lossless[idx_60])
        v_lossy = results['vpi_L'][0] / np.abs(t_lossy[idx_60])
        
        results['vpi_lossless'] = (v_lossless, results['vpi_L'][1] * (v_lossless / results['vpi_L'][0]))
        results['vpi_lossy'] = (v_lossy, results['vpi_L'][1] * (v_lossy / results['vpi_L'][0]))
        
        results['bw'] = (bw_nom, bw_lower, bw_upper)
        results['f_axis'] = f_axis
        results['s21'] = s21_nom
        
        # --- NEW: EXPORT ATTENUATION DATA ---
        results['alpha_60'] = (a60_nom, a60_bc, a60_wc) # (Nominal, Lowest Loss, Highest Loss)
        results['alpha_curve_nom'] = fit_alpha_scaled(f_axis, *popt_nom)
        results['alpha_curve_bc'] = fit_alpha_scaled(f_axis, *popt_bc)
        results['alpha_curve_wc'] = fit_alpha_scaled(f_axis, *popt_wc)
        
        gamma_limit = 10 ** (-10.0 / 20.0) 
        results['rt_min'] = max(Zs_driver * (1 - gamma_limit) / (1 + gamma_limit), zc * (1 - gamma_limit) / (1 + gamma_limit))

        progress.empty()
        return results

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        progress.empty()
        return None

# =====================================================================
# 4. RESULTS DISPLAY
# =====================================================================
st.subheader("2. Performance Prediction")

if st.button("🚀 Predict Performance", type="primary"):
    res = predict_sequentially(geometry_list, length_cm, Zs, Zs_driver, Rt)
    
    if res:
        # Impedance Warning Block
        st.markdown("### 🔍 Impedance Matching Conditions (S11 ≤ -10 dB)")
        rt_min = res['rt_min']
        st.write(f"Minimum safe **Rt** (accounting for Driver Zs={Zs_driver}Ω and Line Zc={res['zc'][0]:.1f}Ω) : **{rt_min:.2f} Ω**")
        
        if Rt < rt_min:
            st.error(f"⚠️ **WARNING:** Your specified Rt ({Rt} Ω) violates physical reflection limits. Expect heavy signal distortion.")
        else:
            st.success(f"✅ **Status:** Rt ({Rt} Ω) is safely within reflection limits.")
        
        st.markdown("---")
        
       # Top Row Metrics 
        col1, col2, col3 = st.columns(3)
        # Fix: Use res['bw'][0] for the nominal bandwidth
        col1.metric("EO Bandwidth", f"{res['bw'][0]:.1f} GHz")
        col2.metric("Impedance (Zc)", f"{res['zc'][0]:.1f} Ω")
        col3.metric("Index (nm)", f"{res['nm'][0]:.4f}")
        
        # Comprehensive FOM Table
        st.markdown("### 📊 Detailed Figures of Merit")
        
        # Hardcoded Global MAEs from your cross-validation
        mae_nm = 0.0264
        mae_zc = 0.7726     # <-- Put your actual Zc MAE here
        mae_vpi = 0.0130  # <-- Put your actual Vpi MAE here
        mae_alpha60 = 0.15 # <-- Put your actual Alpha 60 MAE here
        
        data = [
            [
                "Microwave Index (nm)", 
                f"{res['nm'][0]:.4f}", 
                f"[{res['nm'][0]-1.96*res['nm'][1]:.4f}, {res['nm'][0]+1.96*res['nm'][1]:.4f}]",
                f"± {mae_nm:.4f}"
            ],
            [
                "Characteristic Impedance (Z0) [Ω]", 
                f"{res['zc'][0]:.1f}", 
                f"[{res['zc'][0]-1.96*res['zc'][1]:.1f}, {res['zc'][0]+1.96*res['zc'][1]:.1f}]",
                f"± {mae_zc:.2f}"
            ],
            [
                "VPI Length Scaled (Electrostatics) [V]", 
                f"{res['vpi_L'][0]:.3f}", 
                f"[{res['vpi_L'][0]-1.96*res['vpi_L'][1]:.3f}, {res['vpi_L'][0]+1.96*res['vpi_L'][1]:.3f}]",
                f"± {mae_vpi:.3f}"
            ],
            [
                "RF Attenuation @ 60 GHz [dB/cm]", 
                f"{res['alpha_60'][0]:.3f}", 
                f"[{res['alpha_60'][1]:.3f}, {res['alpha_60'][2]:.3f}]",
                f"± {mae_alpha60:.3f}"
            ],
            [
                "EO Bandwidth [GHz]", 
                f"{res['bw'][0]:.1f}", 
                f"[{res['bw'][1]:.1f}, {res['bw'][2]:.1f}]",
                "N/A (Derived)"
            ],
            [
                "VPI @ 60 GHz (Walk-off + Term. Mismatch) [V]", 
                f"{res['vpi_lossless'][0]:.3f}", 
                f"[{res['vpi_lossless'][0]-1.96*res['vpi_lossless'][1]:.3f}, {res['vpi_lossless'][0]+1.96*res['vpi_lossless'][1]:.3f}]",
                "N/A (Derived)"
            ],
            [
                "VPI @ 60 GHz (Full RF Attenuation) [V]", 
                f"{res['vpi_lossy'][0]:.3f}", 
                f"[{res['vpi_lossy'][0]-1.96*res['vpi_lossy'][1]:.3f}, {res['vpi_lossy'][0]+1.96*res['vpi_lossy'][1]:.3f}]",
                "N/A (Derived)"
            ],
        ]
        st.table(pd.DataFrame(data, columns=["FOM", "Predicted Value", "95% Confidence Interval", "Global MAE"]))
        
        # --- PLOT 1: S21 Bandwidth ---
        st.markdown("### 📈 Broadband RF Response")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(res['f_axis'], res['s21'], 'b-', lw=2.5, label=f'EO Response (L={length_cm*10:.1f} mm, Rt={Rt} Ω)')
        ax.axhline(-3, color='r', linestyle='--', lw=2)
        
        if res['bw'][0] < 150.0:
            ax.plot(res['bw'][0], -3, 'ko', markersize=8)
            ax.annotate(f"{res['bw'][0]:.1f} GHz", (res['bw'][0] + 3, -1.5), fontsize=12, fontweight='bold')
            
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Normalized S21 (dB)')
        ax.set_ylim(-8, 1)
        ax.set_xlim(0, 150)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # --- PLOT 2: RF Attenuation Curve ---
        st.markdown("### 📉 RF Attenuation Profile")
        fig_alpha, ax_alpha = plt.subplots(figsize=(8, 4))
        
        # Plot the nominal fit
        ax_alpha.plot(res['f_axis'], res['alpha_curve_nom'], 'g-', lw=2.5, label='Predicted Attenuation')
        
        # Fill the 95% Confidence Interval between the best-case (lowest loss) and worst-case (highest loss)
        ax_alpha.fill_between(res['f_axis'], res['alpha_curve_bc'], res['alpha_curve_wc'], color='green', alpha=0.2, label='95% Confidence Interval')
        
        # Mark the 3 GP Anchor points used for the fit
        idx_20 = np.argmin(np.abs(res['f_axis'] - 20.0))
        idx_100 = np.argmin(np.abs(res['f_axis'] - 100.0))
        ax_alpha.scatter([20, 60, 100], [res['alpha_curve_nom'][idx_20], res['alpha_60'][0], res['alpha_curve_nom'][idx_100]], color='black', zorder=5, label='GP Anchors')

        ax_alpha.set_xlabel('Frequency (GHz)')
        ax_alpha.set_ylabel(r'Attenuation $\alpha$ (dB/cm)')
        ax_alpha.set_xlim(0, 150)
        ax_alpha.set_ylim(bottom=0)
        ax_alpha.grid(True, linestyle=':', alpha=0.7)
        ax_alpha.legend()
        plt.tight_layout()
        st.pyplot(fig_alpha)
