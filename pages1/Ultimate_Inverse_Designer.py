import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
import os
import gc
import base64
from torch.quasirandom import SobolEngine
import torch
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE HEADER
# ==========================================
st.title("🔍 Ultimate Inverse Synthesizer")
st.markdown("""
**Goal-Seeking Engine.** Input your exact target broadband FOMs and tolerances. 
The hybrid GP-Physics engine floods the 10-DOF space using a Quasi-Monte Carlo Sobol sequence, 
scores candidates using Weighted Euclidean Distance, and back-calculates the absolute best physical compromises.
""")

# ==========================================
# 2. SIDEBAR: TARGETS & BOUNDS
# ==========================================
st.sidebar.header("1. Performance Targets")

t_vpi = st.sidebar.number_input("Target Vpi (Length Scaled) [V]", value=2.00, step=0.1)
tol_vpi = st.sidebar.number_input("Vpi Tolerance (+/-) [V]", value=0.20, step=0.05)

t_bw = st.sidebar.number_input("Target EO Bandwidth [GHz]", value=80.0, step=5.0)
tol_bw = st.sidebar.number_input("Bandwidth Tol (+/-) [GHz]", value=8.0, step=1.0)

t_zc = st.sidebar.number_input("Target Zc [Ω]", value=70.0, step=1.0)
tol_zc = st.sidebar.number_input("Zc Tolerance (+/-) [Ω]", value=2.0, step=0.5)

t_nm = st.sidebar.number_input("Target Index (nm)", value=2.270, step=0.01)
tol_nm = st.sidebar.number_input("Index Tolerance (+/-)", value=0.03, step=0.005)

st.sidebar.header("2. Search Space (Bounds)")
def range_input(label, min_def, max_def, step=0.1, fmt="%.1f"):
    c1, c2 = st.sidebar.columns(2)
    return (c1.number_input(f"Min {label}", value=float(min_def), step=step, format=fmt),
            c2.number_input(f"Max {label}", value=float(max_def), step=step, format=fmt))

b_WS   = range_input("WS", 10.0, 60.0)
b_GAP  = range_input("GAP", 4.0, 15.0)
b_MTX  = range_input("MTX", 1.5, 12.0)
b_CAPW = range_input("CAP_W", 1.5, 14.0)
b_L1   = range_input("L1", 4.0, 60.0)
b_L2   = range_input("L2", 4.0, 180.0)
b_W1   = range_input("W1", 4.0, 60.0)
b_W2   = range_input("W2", 4.0, 60.0)
b_L    = range_input("L_dev [mm]", 4.0, 20.0)
b_Rt   = range_input("Rt [Ω]", 34.0, 60.0)

VAR_NAMES = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2', 'L_dev', 'Rt']
BOUNDS = [b_WS, b_GAP, b_MTX, b_CAPW, b_L1, b_L2, b_W1, b_W2, b_L, b_Rt]

# ==========================================
# 3. PHYSICS & BATCH ENGINE
# ==========================================
MODEL_DIR = Path("gp_surrogate_results_ultimate")

def fit_alpha(f, a, b, c): return a * np.sqrt(f) + b * f + c

def calc_eo(f_GHz, alpha, nm, Zc, L_m, ng=2.27, Zs=50.0, Rt=42.0):
    c0 = 299792458.0; w = 2 * np.pi * f_GHz * 1e9
    b_opt, b_rf = ng * w / c0, nm * w / c0
    a_Np = alpha * 100.0 / 8.686
    gamma = a_Np + 1j * b_rf

    tanh_gL = np.tanh(gamma * L_m)
    zin = Zc * ((Rt + Zc * tanh_gL) / (Zc + Rt * tanh_gL))
    M = (Zs + zin) * (Rt + Zc)
    N = (Zs + zin) * (Rt - Zc)
    p1 = zin / (M * np.exp(gamma * L_m) + N * np.exp(-gamma * L_m))

    up = np.where(L_m * (a_Np + 1j * (b_rf - b_opt)) == 0, 1e-12, L_m * (a_Np + 1j * (b_rf - b_opt)))
    un = np.where(L_m * (-a_Np + 1j * (-b_rf - b_opt)) == 0, 1e-12, L_m * (-a_Np + 1j * (-b_rf - b_opt)))
    p2 = (Rt + Zc) * ((1 - np.exp(up)) / up) + (Rt - Zc) * ((1 - np.exp(un)) / un)

    s21_abs = p1 * p2
    idx_1G = np.argmin(np.abs(f_GHz - 1.0))
    return 20 * np.log10(np.abs(s21_abs) / np.abs(s21_abs[idx_1G])), s21_abs / (zin / (Zs + zin))

def get_detailed_predictions(geom_um, L_mm, Rt):
    """Heavy inference for the top winners to generate plots and CIs."""
    # BUG FIX: Explicitly scaling um to mm right at the entry point!
    ws, gap, mtx, cap, l1, l2, w1, w2 = np.array(geom_um) / 1000.0
    
    bp = (l1 + w1 + l2 + w2) * (ws / gap)
    x8 = np.array([[cap, gap, l1, l2, mtx, w1, w2, ws]])
    x9 = np.array([[cap, gap, l1, l2, mtx, w1, w2, ws, bp]])

    with open(MODEL_DIR/"gp_vpi_surrogate/scalers_VPI.pkl", 'rb') as f: v_s = pickle.load(f)
    with open(MODEL_DIR/"gp_vpi_surrogate/gp_model_VPI.pkl", 'rb') as f: v_m = pickle.load(f)
    vn, vs = v_m.predict(v_s['scaler_X'].transform(x8), return_std=True)
    v_base = 10 ** v_s['scaler_y'].inverse_transform(vn.reshape(-1,1)).ravel()[0]
    v_std = v_base * np.log(10) * vs[0] * v_s['scaler_y'].scale_[0]

    with open(MODEL_DIR/"gp_nm_zc_surrogate/scalers_nm_zc.pkl", 'rb') as f: nz_s = pickle.load(f)
    with open(MODEL_DIR/"gp_nm_zc_surrogate/gp_model_nm_60.pkl", 'rb') as f: n_m = pickle.load(f)
    with open(MODEL_DIR/"gp_nm_zc_surrogate/gp_model_Zc_60.pkl", 'rb') as f: z_m = pickle.load(f)
    X_nz = nz_s['scaler_X'].transform(x9)
    nn, ns = n_m.predict(X_nz, return_std=True)
    zn, zs = z_m.predict(X_nz, return_std=True)
    nm = nz_s['scalers_y']['nm_60'].inverse_transform(nn.reshape(-1,1)).ravel()[0]
    zc = nz_s['scalers_y']['Zc_60'].inverse_transform(zn.reshape(-1,1)).ravel()[0]
    n_std = ns[0] * nz_s['scalers_y']['nm_60'].scale_[0]
    z_std = zs[0] * nz_s['scalers_y']['Zc_60'].scale_[0]

    with open(MODEL_DIR/"gp_alpha_anchors/scaler_anchors.pkl", 'rb') as f: a_s = pickle.load(f)['scaler_X']
    with open(MODEL_DIR/"gp_alpha_anchors/gp_alpha_anchors_suite.pkl", 'rb') as f: a_m = pickle.load(f)
    Xa = a_s.transform(x9)
    y20, s20 = a_m['Alpha_20GHz_dB_cm'].predict(Xa, return_std=True)
    y60, s60 = a_m['Alpha_60GHz_dB_cm'].predict(Xa, return_std=True)
    y100, s100 = a_m['Alpha_100GHz_dB_cm'].predict(Xa, return_std=True)

    a20, a60, a100 = 10**y20[0], 10**y60[0], 10**y100[0]
    a20_w, a60_w, a100_w = 10**(y20[0]+1.96*s20[0]), 10**(y60[0]+1.96*s60[0]), 10**(y100[0]+1.96*s100[0])
    a20_b, a60_b, a100_b = 10**(y20[0]-1.96*s20[0]), 10**(y60[0]-1.96*s60[0]), 10**(y100[0]-1.96*s100[0])

    f_ax = np.linspace(1.0, 150.0, 500)
    Lm = L_mm * 1e-3

    def get_bw(al):
        p, _ = curve_fit(fit_alpha, [20., 60., 100.], al)
        s21, _ = calc_eo(f_ax, fit_alpha(f_ax, *p), nm, zc, Lm, 2.27, 50.0, Rt)
        if s21[-1] > -3.0: return 150.0, s21, p
        i = np.where(s21 <= -3.0)[0][0]
        return f_ax[i-1] + (f_ax[i]-f_ax[i-1])*(-3.0-s21[i-1])/(s21[i]-s21[i-1]), s21, p

    bw_n, s21_n, p_n = get_bw([a20, a60, a100])
    bw_l, _, p_w = get_bw([a20_w, a60_w, a100_w])
    bw_u, _, p_b = get_bw([a20_b, a60_b, a100_b])

    _, t_lossless = calc_eo(f_ax, np.zeros_like(f_ax), nm, zc, Lm, 2.27, 50.0, Rt)
    _, t_lossy = calc_eo(f_ax, fit_alpha(f_ax, *p_n), nm, zc, Lm, 2.27, 50.0, Rt)
    
    i60 = np.argmin(np.abs(f_ax - 60.0))
    Lcm = L_mm / 10.0
    vl, vfull = (v_base/Lcm)/np.abs(t_lossless[i60]), (v_base/Lcm)/np.abs(t_lossy[i60])

    g_lim = 10**(-10/20)
    rtm = max(65.0 * (1-g_lim)/(1+g_lim), zc * (1-g_lim)/(1+g_lim))

    return {
        'vpi': (v_base/Lcm, v_std/Lcm), 'nm': (nm, n_std), 'zc': (zc, z_std),
        'v_ll': (vl, (v_std/Lcm)*(vl/(v_base/Lcm))), 'v_full': (vfull, (v_std/Lcm)*(vfull/(v_base/Lcm))),
        'bw': (bw_n, bw_l, bw_u), 'rt_min': rtm, 's21': s21_n, 'f_axis': f_ax,
        'a60': (a60, a60_b, a60_w), 'a_nom': fit_alpha(f_ax, *p_n), 
        'a_bc': fit_alpha(f_ax, *p_b), 'a_wc': fit_alpha(f_ax, *p_w)
    }

# ==========================================
# 4. INVERSE SEARCH ALGORITHM
# ==========================================
def run_inverse_search():
    N_CANDS = 250000
    prog = st.progress(0, f"Stage 1: Flooding space with {N_CANDS} Sobol sequence geometries...")
    
    sobol = SobolEngine(10, scramble=True, seed=42)
    X_u = sobol.draw(N_CANDS).numpy()
    mins = np.array([b[0] for b in BOUNDS])
    maxs = np.array([b[1] for b in BOUNDS])
    X_p = mins + X_u * (maxs - mins)
    
    # Fast Geom Filter
    cap_c=3; gap_c=1; w1_c=6; w2_c=7
    mask_geom = (X_p[:, cap_c] < (X_p[:, gap_c] - 1.0)) & ((X_p[:, w1_c] + X_p[:, w2_c]) < 60.0)
    X_v = X_p[mask_geom] # These are in micrometers!
    
    if len(X_v) == 0:
        st.error("No candidates pass geometric bounds."); return
        
    prog.progress(0.2, f"Stage 2: Machine Learning Inference on {len(X_v)} layouts...")
    
    # BUG FIX: Creating a dedicated mm array for the ML models
    X8_mm = np.zeros((len(X_v), 8))
    X8_mm[:, 0] = X_v[:, 3] / 1e3 # CAP_W
    X8_mm[:, 1] = X_v[:, 1] / 1e3 # GAP
    X8_mm[:, 2] = X_v[:, 4] / 1e3 # L1
    X8_mm[:, 3] = X_v[:, 5] / 1e3 # L2
    X8_mm[:, 4] = X_v[:, 2] / 1e3 # MTX
    X8_mm[:, 5] = X_v[:, 6] / 1e3 # W1
    X8_mm[:, 6] = X_v[:, 7] / 1e3 # W2
    X8_mm[:, 7] = X_v[:, 0] / 1e3 # WS
    
    BP_mm = (X8_mm[:,2] + X8_mm[:,5] + X8_mm[:,3] + X8_mm[:,6]) * (X8_mm[:,7] / X8_mm[:,1])
    X9_mm = np.column_stack([X8_mm, BP_mm])
    
    with open(MODEL_DIR/"gp_vpi_surrogate/scalers_VPI.pkl", 'rb') as f: v_s = pickle.load(f)
    with open(MODEL_DIR/"gp_vpi_surrogate/gp_model_VPI.pkl", 'rb') as f: v_m = pickle.load(f)
    v_base = 10 ** v_s['scaler_y'].inverse_transform(v_m.predict(v_s['scaler_X'].transform(X8_mm)).reshape(-1,1)).ravel()
    del v_m; gc.collect()
    
    with open(MODEL_DIR/"gp_nm_zc_surrogate/scalers_nm_zc.pkl", 'rb') as f: nz_s = pickle.load(f)
    X_nz = nz_s['scaler_X'].transform(X9_mm)
    with open(MODEL_DIR/"gp_nm_zc_surrogate/gp_model_nm_60.pkl", 'rb') as f: n_m = pickle.load(f)
    nm = nz_s['scalers_y']['nm_60'].inverse_transform(n_m.predict(X_nz).reshape(-1,1)).ravel()
    del n_m; gc.collect()
    with open(MODEL_DIR/"gp_nm_zc_surrogate/gp_model_Zc_60.pkl", 'rb') as f: z_m = pickle.load(f)
    zc = nz_s['scalers_y']['Zc_60'].inverse_transform(z_m.predict(X_nz).reshape(-1,1)).ravel()
    del z_m; gc.collect()
    
    # Stage 3: Generous Soft-Sieve
    prog.progress(0.6, "Stage 3: Extracting broad performance subsets...")
    vpi_approx = v_base / (X_v[:, 8] / 10.0)
    
    # Using 3x tolerances to generously catch any design that MIGHT polish into the target
    mask_perf = (np.abs(nm - t_nm) <= tol_nm * 3) & \
                (np.abs(zc - t_zc) <= tol_zc * 3) & \
                (vpi_approx <= t_vpi + tol_vpi + 0.5)
                
    X_candidates = X_v[mask_perf]
    X9_surv = X9_mm[mask_perf] # The scaled mm array must be passed!
    v_c = v_base[mask_perf]
    nm_c = nm[mask_perf]
    zc_c = zc[mask_perf]
    
    if len(X_candidates) == 0:
        prog.empty(); st.error("❌ Your combination of Zc, nm, and Vpi targets is physically impossible within these geometric bounds."); return

    prog.progress(0.8, f"Stage 4: Running Broadband Cascade & Scoring top {len(X_candidates)} designs...")
    
    with open(MODEL_DIR/"gp_alpha_anchors/scaler_anchors.pkl", 'rb') as f: a_s = pickle.load(f)['scaler_X']
    with open(MODEL_DIR/"gp_alpha_anchors/gp_alpha_anchors_suite.pkl", 'rb') as f: a_m = pickle.load(f)
    Xa_c = a_s.transform(X9_surv)
    a20 = 10 ** a_m['Alpha_20GHz_dB_cm'].predict(Xa_c)
    a60 = 10 ** a_m['Alpha_60GHz_dB_cm'].predict(Xa_c)
    a100 = 10 ** a_m['Alpha_100GHz_dB_cm'].predict(Xa_c)
    del a_m; gc.collect()
    
    final_results = []
    f_ax = np.linspace(1.0, 150.0, 500)
    
    for i in range(len(X_candidates)):
        Lm = X_candidates[i, 8] / 1000.0  
        Lcm = X_candidates[i, 8] / 10.0   
        Rt = X_candidates[i, 9]
        
        glim = 10**(-10/20)
        rt_min = max(65.0*(1-glim)/(1+glim), zc_c[i]*(1-glim)/(1+glim))
        if Rt < rt_min: continue # Strict Safety violation, drop it
        
        p, _ = curve_fit(fit_alpha, [20., 60., 100.], [a20[i], a60[i], a100[i]])
        s21, ty = calc_eo(f_ax, fit_alpha(f_ax, *p), nm_c[i], zc_c[i], Lm, 2.27, 50.0, Rt)
        
        if s21[-1] > -3.0: bw = 150.0
        else:
            idx = np.where(s21 <= -3.0)[0][0]
            bw = f_ax[idx-1] + (f_ax[idx]-f_ax[idx-1])*(-3.0-s21[idx-1])/(s21[idx]-s21[idx-1])
            
        v_lossy = (v_c[i] / Lcm) / np.abs(ty[np.argmin(np.abs(f_ax - 60.0))])
        
        # Weighted Euclidean Distance Scoring
        err = ((bw-t_bw)/tol_bw)**2 + ((v_lossy-t_vpi)/tol_vpi)**2 + ((zc_c[i]-t_zc)/tol_zc)**2 + ((nm_c[i]-t_nm)/tol_nm)**2
        final_results.append({'x': X_candidates[i], 'err': err})
        
    prog.empty()
    if not final_results:
        st.error("❌ Geometries survived electrostatics, but violated the S11 Reflection limits for Rt. Try increasing the Max Rt bound."); return
        
    final_results.sort(key=lambda item: item['err'])
    st.session_state['inv_res'] = [r['x'] for r in final_results[:5]]

# ==========================================
# 5. VISUALIZATION
# ==========================================
def render_svg(s): return f'<img src="data:image/svg+xml;base64,{base64.b64encode(s.encode("utf-8")).decode("utf-8")}" width="100%"/>'

def generate_exact_svg(p):
    W1, W2, L1, L2, WS, GAP, MTX, CAP_W = p["W1"], p["W2"], p["L1"], p["L2"], p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0; C_ELEC = '#F5BD02'; C_SUB = '#00BFFF'; C_CAP = '#00BFFF'; C_LINE = 'black'; CX = 400
    def arrow(x1, y1, x2, y2, txt, loc="top", off=10):
        m, t, a, d = (x1+x2)/2, (y1+y2)/2, "middle", "middle"
        if loc=="top": t-=off; d="auto"
        elif loc=="bottom": t+=off; d="hanging"
        elif loc=="left": m-=off; a="end"
        elif loc=="right": m+=off; a="start"
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1.5" marker-start="url(#S)" marker-end="url(#E)" /><text x="{m}" y="{t}" fill="{C_LINE}" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="{a}" dominant-baseline="{d}">{txt}</text>'

    def top(x,y): return CX+x*3.5, 380-y*3.5
    pts = [(GAP/2,-100), (GAP/2+WG,-100), (GAP/2+WG,100), (GAP/2,100), (GAP/2,L1/2), (GAP/2+W1,L1/2), (GAP/2+W1,L2/2), (GAP/2+W1+W2,L2/2), (GAP/2+W1+W2,-L2/2), (GAP/2+W1,-L2/2), (GAP/2+W1,-L1/2), (GAP/2,-L1/2)]
    poly = " ".join([f"{top(x,y)[0]},{top(x,y)[1]}" for x,y in pts])
    svg_t = f'<svg width="800" height="700" viewBox="0 0 800 700" xmlns="http://www.w3.org/2000/svg"><defs><marker id="E" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z"/></marker><marker id="S" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto"><path d="M9,0 L9,6 L0,3 z"/></marker></defs><text x="20" y="50" font-family="sans-serif" font-size="24" font-weight="bold">Top-Down View</text><rect x="{top(-(GAP/2+WS),100)[0]}" y="{top(-(GAP/2+WS),100)[1]}" width="{WS*3.5}" height="{200*3.5}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><polygon points="{poly}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/>{arrow(*top(-(GAP/2+WS),120),*top(-GAP/2,120),"WS","top")}{arrow(*top(GAP/2,120),*top(GAP/2+WG,120),"WG","top")}{arrow(*top(-GAP/2,-120),*top(GAP/2,-120),"GAP","bottom")}{arrow(*top(GAP/2-20,-L1/2),*top(GAP/2-20,L1/2),"L1","left")}{arrow(*top(GAP/2+W1+W2+20,-L2/2),*top(GAP/2+W1+W2+20,L2/2),"L2","right")}{arrow(*top(GAP/2,-L1/2-20),*top(GAP/2+W1,-L1/2-20),"W1","bottom")}{arrow(*top(GAP/2+W1,L2/2+20),*top(GAP/2+W1+W2,L2/2+20),"W2","top")}</svg>'

    SC = 700/((WS/2+GAP)+1.0)/2
    def cs(x,y): return CX+x*SC, 350-y*SC
    svg_c = f'<svg width="800" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg"><defs><marker id="E" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z"/></marker><marker id="S" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto"><path d="M9,0 L9,6 L0,3 z"/></marker></defs><text x="20" y="50" font-family="sans-serif" font-size="24" font-weight="bold">Cross-Section</text><rect x="0" y="{cs(0,0)[1]}" width="800" height="600" fill="{C_SUB}"/><rect x="0" y="{cs(0,0.23)[1]}" width="800" height="{0.23*SC}" fill="black"/><rect x="{cs(-WS/2,0.23+MTX)[0]}" y="{cs(-WS/2,0.23+MTX)[1]}" width="{WS*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(WS/2+GAP,0.23+MTX)[0]}" y="{cs(WS/2+GAP,0.23+MTX)[1]}" width="{500*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(-(WS/2+GAP+500),0.23+MTX)[0]}" y="{cs(-(WS/2+GAP+500),0.23+MTX)[1]}" width="{500*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/>'
    for cx in [WS/2+GAP/2, -WS/2-GAP/2]: svg_c += f'<rect x="{cs(cx-CAP_W/2,1.63)[0]}" y="{cs(cx-CAP_W/2,1.63)[1]}" width="{CAP_W*SC}" height="{1.4*SC}" fill="{C_CAP}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(cx-0.4,0.46)[0]}" y="{cs(cx-0.4,0.46)[1]}" width="{0.8*SC}" height="{0.23*SC}" fill="black"/>'
    y = 0.23+max(MTX,1.4)+(3.0 if MTX<5 else 0.5*MTX)
    svg_c += f'{arrow(*cs(-WS/2,y),*cs(WS/2,y),"WS","top",15)}{arrow(*cs(WS/2,y),*cs(WS/2+GAP,y),"GAP","top",15)}{arrow(*cs((-WS/2-GAP/2)-CAP_W/2,2.63),*cs((-WS/2-GAP/2)+CAP_W/2,2.63),"CAP_W","top",15)}{arrow(*cs(-WS/2+2.0,0.23),*cs(-WS/2+2.0,0.23+MTX),"MTX","right")}</svg>'
    return svg_t, svg_c

# ==========================================
# 6. UI EXECUTION
# ==========================================
if st.button("SYNTHESIZE GEOMETRY", type="primary"): run_inverse_search()

if 'inv_res' in st.session_state:
    st.markdown("---")
    st.success(f"✅ Scanning complete! Displaying the top {len(st.session_state['inv_res'])} best mathematical compromises.")
    
    for i, x in enumerate(st.session_state['inv_res']):
        with st.expander(f"🏅 Candidate {i+1} | L={x[8]:.1f} mm | Rt={x[9]:.1f} Ω", expanded=(i==0)):
            res = get_detailed_predictions(x[:8], x[8], x[9])
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("EO Bandwidth", f"{res['bw'][0]:.1f} GHz", delta=f"{res['bw'][0]-t_bw:+.1f} from target")
            c2.metric("VPI (RF)", f"{res['v_full'][0]:.2f} V", delta=f"{res['v_full'][0]-t_vpi:+.2f} from target", delta_color="inverse")
            c3.metric("Zc", f"{res['zc'][0]:.1f} Ω", delta=f"{res['zc'][0]-t_zc:+.1f} from target")
            c4.metric("nm", f"{res['nm'][0]:.4f}", delta=f"{res['nm'][0]-t_nm:+.4f} from target")
            
            # Parameters
            st.markdown("### Geometric Layout & Configuration")
            p_dict = dict(zip(VAR_NAMES, x))
            st.table(pd.DataFrame([p_dict]).style.format("{:.3f}"))
            
            # SVG
            c_svg1, c_svg2 = st.columns(2)
            svg_t, svg_c = generate_exact_svg(p_dict)
            with c_svg1: st.markdown(render_svg(svg_t), unsafe_allow_html=True)
            with c_svg2: st.markdown(render_svg(svg_c), unsafe_allow_html=True)
            
            # FOM Table
            st.markdown("### 📊 Predicted FOMs (with 95% CI & MAE)")
            m_nm, m_zc, m_vpi, m_a = 0.0264, 1.0, 0.045, 0.15 
            data = [
                ["Microwave Index (nm)", f"{res['nm'][0]:.4f}", f"[{res['nm'][0]-1.96*res['nm'][1]:.4f}, {res['nm'][0]+1.96*res['nm'][1]:.4f}]", f"± {m_nm:.4f}"],
                ["Impedance Zc [Ω]", f"{res['zc'][0]:.1f}", f"[{res['zc'][0]-1.96*res['zc'][1]:.1f}, {res['zc'][0]+1.96*res['zc'][1]:.1f}]", f"± {m_zc:.2f}"],
                ["VPI (Electrostatic) [V]", f"{res['vpi'][0]:.3f}", f"[{res['vpi'][0]-1.96*res['vpi'][1]:.3f}, {res['vpi'][0]+1.96*res['vpi'][1]:.3f}]", f"± {m_vpi:.3f}"],
                ["RF Attenuation @ 60 GHz [dB/cm]", f"{res['a60'][0]:.3f}", f"[{res['a60'][1]:.3f}, {res['a60'][2]:.3f}]", f"± {m_a:.3f}"],
                ["EO Bandwidth [GHz]", f"{res['bw'][0]:.1f}", f"[{res['bw'][1]:.1f}, {res['bw'][2]:.1f}]", "N/A"],
                ["VPI @ 60GHz (Walk-off+Mismatch) [V]", f"{res['v_ll'][0]:.3f}", f"[{res['v_ll'][0]-1.96*res['v_ll'][1]:.3f}, {res['v_ll'][0]+1.96*res['v_ll'][1]:.3f}]", "N/A"],
                ["VPI @ 60GHz (Full RF Physics) [V]", f"{res['v_full'][0]:.3f}", f"[{res['v_full'][0]-1.96*res['v_full'][1]:.3f}, {res['v_full'][0]+1.96*res['v_full'][1]:.3f}]", "N/A"]
            ]
            st.table(pd.DataFrame(data, columns=["FOM", "Predicted", "95% CI", "Global MAE"]))
            
            # Plots
            st.markdown("### 📈 Broadband RF Response & Attenuation")
            f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(res['f_axis'], res['s21'], 'b-', lw=2)
            ax1.axhline(-3, color='r', ls='--')
            if res['bw'][0] < 150: ax1.plot(res['bw'][0], -3, 'ko'); ax1.annotate(f"{res['bw'][0]:.1f} GHz", (res['bw'][0]+3, -1.5))
            ax1.set(xlabel='Frequency (GHz)', ylabel='S21 (dB)', title='EO Bandwidth', xlim=(0,150), ylim=(-8,1)); ax1.grid(ls=':')
            
            ax2.plot(res['f_axis'], res['a_nom'], 'g-', lw=2, label='Nominal')
            ax2.fill_between(res['f_axis'], res['a_bc'], res['a_wc'], color='green', alpha=0.2, label='95% CI')
            ax2.set(xlabel='Frequency (GHz)', ylabel='Alpha (dB/cm)', title='RF Attenuation', xlim=(0,150)); ax2.grid(ls=':')
            st.pyplot(f1)
