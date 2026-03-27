import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize as scipy_minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
import pickle
import os
import base64
import io
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE HEADER & STYLING
# ==========================================
st.title("🎯 Ultimate TFLN 10-DOF Optimizer")
st.markdown("""
**Find the absolute physical limit of your design.**
This engine uses a **Memetic Algorithm** (NSGA-II Global Pareto Search + SLSQP Gradient Polishing) 
to map the ultimate trade-offs between Bandwidth and Efficiency, and then mathematically locks your velocity match 
while maximizing characteristic impedance ($Z_c$).
""")

# ==========================================
# 2. SIDEBAR: USER INPUTS & BOUNDS
# ==========================================
st.sidebar.header("1. Global Ranges")
def range_input(label, min_def, max_def, step=0.1, format="%.1f"):
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input(f"Min {label}", value=float(min_def), step=step, format=format)
    max_val = c2.number_input(f"Max {label}", value=float(max_def), step=step, format=format)
    return (min_val, max_val)

b_L  = range_input("Length [mm]", 4.0, 20.0)
b_Rt = range_input("Rt [Ω]", 34.0, 60.0)

st.sidebar.header("2. Geometry Constraints")
st.sidebar.markdown("Search space limits (µm):")
b_WS   = range_input("WS", 10.0, 60.0)
b_GAP  = range_input("GAP", 4.0, 15.0)
b_MTX  = range_input("MTX", 1.5, 15.0)
b_CAPW = range_input("CAP_W", 1.5, 14.0)
b_L1   = range_input("L1", 2.0, 60.0)
b_L2   = range_input("L2", 4.0, 180.0)
b_W1   = range_input("W1", 2.0, 60.0)
b_W2   = range_input("W2", 2.0, 60.0)

# Compile bounds list for optimizers
BOUNDS_LIST = [b_WS, b_GAP, b_MTX, b_CAPW, b_L1, b_L2, b_W1, b_W2, b_L, b_Rt]

st.sidebar.header("3. Performance Targets")
target_vpi = st.sidebar.number_input("Max VPI [V]", value=2.0, step=0.1)
target_bw  = st.sidebar.number_input("Min EO Bandwidth [GHz]", value=65.0, step=1.0)
target_nm  = st.sidebar.number_input("Target Index (nm)", value=2.27, step=0.01)
target_tol = st.sidebar.number_input("Index Tolerance (+/-)", value=0.03, step=0.005)

st.sidebar.markdown("---")
st.sidebar.info("**Fixed:** Zs = 50 Ω | Zs_driver = 65 Ω | ng = 2.27")

# ==========================================
# 3. CACHED PREDICTOR ENGINE
# ==========================================
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

class FastPredictor:
    def __init__(self):
        m_dir = Path("gp_surrogate_results_ultimate")
        with open(m_dir/"gp_vpi_surrogate/scalers_VPI.pkl", 'rb') as f: self.v_s = pickle.load(f)
        with open(m_dir/"gp_vpi_surrogate/gp_model_VPI.pkl", 'rb') as f: self.v_m = pickle.load(f)
        with open(m_dir/"gp_nm_zc_surrogate/scalers_nm_zc.pkl", 'rb') as f: self.nz_s = pickle.load(f)
        with open(m_dir/"gp_nm_zc_surrogate/gp_model_nm_60.pkl", 'rb') as f: self.n_m = pickle.load(f)
        with open(m_dir/"gp_nm_zc_surrogate/gp_model_Zc_60.pkl", 'rb') as f: self.z_m = pickle.load(f)
        with open(m_dir/"gp_alpha_anchors/scaler_anchors.pkl", 'rb') as f: self.a_s = pickle.load(f)['scaler_X']
        with open(m_dir/"gp_alpha_anchors/gp_alpha_anchors_suite.pkl", 'rb') as f: self.a_m = pickle.load(f)

    def predict(self, geom, L_mm, Rt):
        ws, gap, mtx, cap, l1, l2, w1, w2 = geom
        bp = (l1 + w1 + l2 + w2) * (ws / gap)
        x8 = np.array([[cap, gap, l1, l2, mtx, w1, w2, ws]])
        x9 = np.array([[cap, gap, l1, l2, mtx, w1, w2, ws, bp]])

        vn, vs = self.v_m.predict(self.v_s['scaler_X'].transform(x8), return_std=True)
        v_base = 10 ** self.v_s['scaler_y'].inverse_transform(vn.reshape(-1,1)).ravel()[0]
        v_std = v_base * np.log(10) * vs[0] * self.v_s['scaler_y'].scale_[0]

        X_nz = self.nz_s['scaler_X'].transform(x9)
        nn, ns = self.n_m.predict(X_nz, return_std=True)
        zn, zs = self.z_m.predict(X_nz, return_std=True)
        nm = self.nz_s['scalers_y']['nm_60'].inverse_transform(nn.reshape(-1,1)).ravel()[0]
        zc = self.nz_s['scalers_y']['Zc_60'].inverse_transform(zn.reshape(-1,1)).ravel()[0]
        n_std = ns[0] * self.nz_s['scalers_y']['nm_60'].scale_[0]
        z_std = zs[0] * self.nz_s['scalers_y']['Zc_60'].scale_[0]

        Xa = self.a_s.transform(x9)
        y20, s20 = self.a_m['Alpha_20GHz_dB_cm'].predict(Xa, return_std=True)
        y60, s60 = self.a_m['Alpha_60GHz_dB_cm'].predict(Xa, return_std=True)
        y100, s100 = self.a_m['Alpha_100GHz_dB_cm'].predict(Xa, return_std=True)

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

@st.cache_resource
def load_engine(): return FastPredictor()
engine = load_engine()

# ==========================================
# 4. OPTIMIZER CLASSES
# ==========================================
class NSGA2_Problem(ElementwiseProblem):
    def __init__(self, bnds):
        xl = np.array([b[0] for b in bnds])
        xu = np.array([b[1] for b in bnds])
        super().__init__(n_var=10, n_obj=2, n_ieq_constr=8, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        cap_w = x[3]; gap = x[1]; w1 = x[6]; w2 = x[7]
        g1 = cap_w - (gap - 1.0)
        g2 = (w1 + w2) - 60.0
        
        if g1 > 0 or g2 > 0:
            out["F"] = [10.0, 0.0]
            out["G"] = [g1, g2, 10., 10., 10., 10., 10., 10.]
            return

        try:
            res = engine.predict(x[:8]/1000.0, x[8], x[9])
            bw, vpi, nm, zc, rtm = res['bw'][0], res['vpi'][0], res['nm'][0], res['zc'][0], res['rt_min']
        except:
            bw, vpi, nm, zc, rtm = 0.0, 10.0, 0.0, 0.0, 100.0

        g3 = (target_nm - target_tol) - nm
        g4 = nm - (target_nm + target_tol)
        g5 = 45.0 - zc
        g6 = rtm - x[9]
        g7 = target_bw - bw
        g8 = vpi - target_vpi

        out["F"] = [vpi, -bw]
        out["G"] = [g1, g2, g3, g4, g5, g6, g7, g8]

def slsqp_polisher(x0):
    def get_p(x):
        try: return engine.predict(x[:8]/1000.0, x[8], x[9])
        except: return {'zc':(0.,0.), 'nm':(10.,0.), 'bw':(0.,0.,0.), 'vpi':(10.,0.), 'rt_min':100.}
    
    def obj(x): return -get_p(x)['zc'][0] # MAXIMIZE ZC
    def eq_nm(x): return get_p(x)['nm'][0] - target_nm
    
    cons = [
        {'type': 'eq', 'fun': eq_nm},
        {'type': 'ineq', 'fun': lambda x: get_p(x)['bw'][0] - target_bw},
        {'type': 'ineq', 'fun': lambda x: target_vpi - get_p(x)['vpi'][0]},
        {'type': 'ineq', 'fun': lambda x: x[9] - get_p(x)['rt_min']},
        {'type': 'ineq', 'fun': lambda x: (x[1]-1.0) - x[3]},
        {'type': 'ineq', 'fun': lambda x: 60.0 - (x[6]+x[7])}
    ]
    return scipy_minimize(obj, x0, method='SLSQP', bounds=BOUNDS_LIST, constraints=cons, options={'ftol': 1e-4, 'maxiter': 50}).x

# ==========================================
# 5. VISUALIZATION (Exact Copy)
# ==========================================
def render_svg(s):
    return f'<img src="data:image/svg+xml;base64,{base64.b64encode(s.encode("utf-8")).decode("utf-8")}" width="100%"/>'

def generate_exact_svg(p):
    W1, W2, L1, L2, WS, GAP, MTX, CAP_W = p["W1"], p["W2"], p["L1"], p["L2"], p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0; C_ELEC = '#F5BD02'; C_SUB = '#00BFFF'; C_CAP = '#00BFFF'; C_LINE = 'black'
    CX = 400

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
# 6. MAIN EXECUTION
# ==========================================
if st.button("🚀 RUN OPTIMIZATION", type="primary"):
    with st.spinner("Step 1/2: Global Pareto Search (NSGA-II)..."):
        res_ga = pymoo_minimize(NSGA2_Problem(BOUNDS_LIST), NSGA2(pop_size=50, n_offsprings=25), get_termination("n_gen", 40), seed=42)
        
    if res_ga.F is None:
        st.error("No designs met all physics constraints. Relax your bounds or targets.")
    else:
        # Extract Pareto Front
        df_p = pd.DataFrame(res_ga.X, columns=['WS','GAP','MTX','CAP_W','L1','L2','W1','W2','L_dev','Rt'])
        df_p['VPI (V)'] = res_ga.F[:,0]; df_p['BW (GHz)'] = -res_ga.F[:,1]
        df_p = df_p.sort_values(by='BW (GHz)', ascending=False)
        
        # Save to memory buffer for download
        buf = io.BytesIO()
        df_p.to_excel(buf, index=False)
        st.download_button("📥 Download Pareto Front (Excel)", buf.getvalue(), "Pareto_Front.xlsx", "application/vnd.ms-excel")

        st.markdown("---")
        with st.spinner("Step 2/2: Gradient Polishing for Maximum Z0..."):
            best_idx = np.argmax(-res_ga.F[:,1])
            final_x = slsqp_polisher(res_ga.X[best_idx])
            res = engine.predict(final_x[:8]/1000.0, final_x[8], final_x[9])
            
        st.header("🏆 Optimal Geometry Found")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z0 (Impedance)", f"{res['zc'][0]:.2f} Ω", delta="Maximized")
        c2.metric("EO Bandwidth", f"{res['bw'][0]:.1f} GHz")
        c3.metric("VPI", f"{res['vpi'][0]:.2f} V")
        c4.metric("Index (nm)", f"{res['nm'][0]:.4f}", help="Locked to Target")
        
        st.subheader("Optimal Parameters")
        p_dict = dict(zip(['WS','GAP','MTX','CAP_W','L1','L2','W1','W2','L_dev','Rt'], final_x))
        st.table(pd.DataFrame([p_dict]).style.format("{:.3f}"))
        
        col1, col2 = st.columns(2)
        svg_t, svg_c = generate_exact_svg(p_dict)
        with col1: st.markdown(render_svg(svg_t), unsafe_allow_html=True)
        with col2: st.markdown(render_svg(svg_c), unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Detailed Figures of Merit")
        
        m_nm, m_zc, m_vpi, m_a = 0.0264, 0.7726, 0.0130, 0.3559 # Replace with true MAEs
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
        
        # Pareto Plot
        f2, ax3 = plt.subplots(figsize=(8,4))
        ax3.scatter(res_ga.F[:,0], -res_ga.F[:,1], c='blue', alpha=0.7)
        ax3.scatter(res['vpi'][0], res['bw'][0], c='gold', s=200, edgecolors='k', marker='*', label='SLSQP Final Winner')
        ax3.set(xlabel='VPI [V]', ylabel='Bandwidth [GHz]', title='NSGA-II Pareto Front')
        ax3.grid(ls=':'); ax3.legend(); st.pyplot(f2)
