"""
================================================================================
🔍 ULTIMATE TFLN 11-DOF INVERSE SYNTHESIZER (STREAMLIT APP)
Goal-Seeking Engine: Sobol Space Flooding + SLSQP Gradient Polishing
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
import torch
from torch.quasirandom import SobolEngine
import warnings
import base64

# Import the new, highly accurate Ultimate Predictor
from ultimate_predictor import Ultimate_TFLN_Predictor

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE HEADER & STYLING
# ==========================================
st.title("🔍 Ultimate TFLN 11-DOF Inverse Synthesizer")
st.markdown("""
**Goal-Seeking Engine.** Input your exact target broadband FOMs and tolerances.
The hybrid GP-Physics engine floods the 11-DOF space using a Quasi-Monte Carlo Sobol sequence,
scores candidates using Weighted Euclidean Distance, and uses an **SLSQP Gradient Polisher** to mathematically slide the best candidates perfectly into your targets.
""")

# ==========================================
# 2. CACHED ENGINE LOADER
# ==========================================
@st.cache_resource
def load_engine():
    return Ultimate_TFLN_Predictor()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load predictor models. Ensure the folder exists. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: TARGETS & BOUNDS
# ==========================================
st.sidebar.header("1. Performance Targets")
t_vpi = st.sidebar.number_input("Target Vpi (Length Scaled) [V]", value=1.50, step=0.1)
tol_vpi = st.sidebar.number_input("Vpi Tolerance (+/-) [V]", value=0.30, step=0.05)

t_bw = st.sidebar.number_input("Target EO Bandwidth [GHz]", value=80.0, step=5.0)
tol_bw = st.sidebar.number_input("Bandwidth Tol (+/-) [GHz]", value=10.0, step=1.0)

t_zc = st.sidebar.number_input("Target Zc [Ω]", value=50.0, step=1.0)
tol_zc = st.sidebar.number_input("Zc Tolerance (+/-) [Ω]", value=3.0, step=0.5)

t_nm = st.sidebar.number_input("Target Index (nm)", value=2.270, step=0.01)
tol_nm = st.sidebar.number_input("Index Tolerance (+/-)", value=0.03, step=0.005)

st.sidebar.markdown("---")
st.sidebar.header("2. Search Space (Bounds)")
def range_input(label, min_def, max_def, step=0.1, fmt="%.2f"):
    c1, c2 = st.sidebar.columns(2)
    return (c1.number_input(f"Min {label}", value=float(min_def), step=step, format=fmt),
            c2.number_input(f"Max {label}", value=float(max_def), step=step, format=fmt))

# 11 DOFs Exact Mapping
# [0:WS, 1:GAP, 2:MTX, 3:L1, 4:L2, 5:W1, 6:W2, 7:CAP_W, 8:ETCH_DEPTH, 9:L_dev, 10:Rt]
b_WS    = range_input("WS [µm]", 10.0, 60.0)
b_GAP   = range_input("GAP [µm]", 4.0, 12.0)
b_MTX   = range_input("MTX [µm]", 1.5, 12.0)
b_L1    = range_input("L1 [µm]", 4.0, 60.0)
b_L2    = range_input("L2 [µm]", 4.0, 180.0)
b_W1    = range_input("W1 [µm]", 4.0, 60.0)
b_W2    = range_input("W2 [µm]", 4.0, 60.0)
b_CAPW  = range_input("CAP_W [µm]", 1.5, 12.0)
b_ETCH  = range_input("ETCH_DEPTH [µm]", 0.0, 0.40, step=0.01, fmt="%.3f")
b_L     = range_input("L_device [mm]", 4.0, 16.5)
b_Rt    = range_input("Rt [Ω]", 34.0, 60.0)

BOUNDS_LIST = [b_WS, b_GAP, b_MTX, b_L1, b_L2, b_W1, b_W2, b_CAPW, b_ETCH, b_L, b_Rt]
VAR_NAMES = ['WS', 'GAP', 'MTX', 'L1', 'L2', 'W1', 'W2', 'CAP_W', 'ETCH_DEPTH', 'L_dev', 'Rt']

# ==========================================
# 4. INVERSE SEARCH ALGORITHMS
# ==========================================
def slsqp_inverse_polish(best_sobol_x):
    """Gradient-polishes the coarse guess using Euclidean Distance matching."""
    bounds = [(b[0], b[1]) for b in BOUNDS_LIST]
    
    def get_p(x):
        try:
            # FIXED: Flawless list indexing. L_dev = x[9], Rt = x[10]
            res = engine.predict(geom=x[:9].tolist(), L_device_mm=x[9], Rt=x[10], Zs_driver=65.0, ng=2.27, plot=False)
            
            # FIXED: Perfect Tuple Unpacking [0] for physical means
            return {
                'bw': res['bw'][0],
                'vpi_len': res['vpi_len'][0],
                'zc': res['z0'][0],
                'nm': res['nm'][0],
                'rt_min': res['rt_min'] # Float, not a tuple
            }
        except Exception:
            return {'bw': 0.0, 'vpi_len': 10.0, 'zc': 0.0, 'nm': 10.0, 'rt_min': 100.0}

    def objective(x):
        p = get_p(x)
        # Weighted Euclidean Distance from Target
        return ((p['bw'] - t_bw)/tol_bw)**2 + \
               ((p['vpi_len'] - t_vpi)/tol_vpi)**2 + \
               ((p['zc'] - t_zc)/tol_zc)**2 + \
               ((p['nm'] - t_nm)/tol_nm)**2

    # Scipy Constraints (Must be >= 0)
    def ineq_rt(x): return x[10] - get_p(x)['rt_min']
    def ineq_cap(x): return (x[1] - 1.0) - x[7]  # GAP - 1.0 - CAP_W >= 0
    def ineq_wsum(x): return 60.0 - (x[5] + x[6]) # 60 - W1 - W2 >= 0

    cons = [
        {'type': 'ineq', 'fun': ineq_rt},
        {'type': 'ineq', 'fun': ineq_cap},
        {'type': 'ineq', 'fun': ineq_wsum}
    ]

    res = scipy_minimize(
        objective, x0=best_sobol_x, method='SLSQP',
        bounds=bounds, constraints=cons,
        options={'ftol': 1e-4, 'maxiter': 50}
    )
    return res.x

# ==========================================
# 5. SVG RENDERING LOGIC
# ==========================================
def render_svg(s):
    return f'<img src="data:image/svg+xml;base64,{base64.b64encode(s.encode("utf-8")).decode("utf-8")}" width="100%"/>'

def generate_exact_svg(p):
    W1, W2, L1, L2, WS, GAP, MTX, CAP_W, ETCH_DEPTH = p["W1"], p["W2"], p["L1"], p["L2"], p["WS"], p["GAP"], p["MTX"], p["CAP_W"], p["ETCH_DEPTH"]
    WG = 70.0; BOTTOM_LAYER_H = 0.46 - ETCH_DEPTH; RIDGE_H = ETCH_DEPTH; CAP_HEIGHT = 1.4
    C_ELEC = '#F5BD02'; C_SUB = '#00BFFF'; C_CAP = '#00BFFF'; C_LINE = 'black'; CX = 400
    
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
    svg_c = f'<svg width="800" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg"><defs><marker id="E" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z"/></marker><marker id="S" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto"><path d="M9,0 L9,6 L0,3 z"/></marker></defs><text x="20" y="50" font-family="sans-serif" font-size="24" font-weight="bold">Cross-Section</text><rect x="0" y="{cs(0,0)[1]}" width="800" height="600" fill="{C_SUB}"/><rect x="0" y="{cs(0,BOTTOM_LAYER_H)[1]}" width="800" height="{BOTTOM_LAYER_H*SC}" fill="black"/><rect x="{cs(-WS/2,BOTTOM_LAYER_H+MTX)[0]}" y="{cs(-WS/2,BOTTOM_LAYER_H+MTX)[1]}" width="{WS*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(WS/2+GAP,BOTTOM_LAYER_H+MTX)[0]}" y="{cs(WS/2+GAP,BOTTOM_LAYER_H+MTX)[1]}" width="{500*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(-(WS/2+GAP+500),BOTTOM_LAYER_H+MTX)[0]}" y="{cs(-(WS/2+GAP+500),BOTTOM_LAYER_H+MTX)[1]}" width="{500*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/>'
    for cx in [WS/2+GAP/2, -WS/2-GAP/2]: svg_c += f'<rect x="{cs(cx-CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT)[0]}" y="{cs(cx-CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT)[1]}" width="{CAP_W*SC}" height="{CAP_HEIGHT*SC}" fill="{C_CAP}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(cx-0.4,0.46)[0]}" y="{cs(cx-0.4,0.46)[1]}" width="{0.8*SC}" height="{RIDGE_H*SC}" fill="black"/>'
    y = BOTTOM_LAYER_H+max(MTX,CAP_HEIGHT)+(3.0 if MTX<5 else 0.5*MTX)
    svg_c += f'{arrow(*cs(-WS/2,y),*cs(WS/2,y),"WS","top",15)}{arrow(*cs(WS/2,y),*cs(WS/2+GAP,y),"GAP","top",15)}{arrow(*cs((-WS/2-GAP/2)-CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT+1.0),*cs((-WS/2-GAP/2)+CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT+1.0),"CAP_W","top",15)}{arrow(*cs(-WS/2+2.0,BOTTOM_LAYER_H),*cs(-WS/2+2.0,BOTTOM_LAYER_H+MTX),"MTX","right")}</svg>'
    return svg_t, svg_c

# ==========================================
# 6. EXECUTION PIPELINE
# ==========================================
if st.button("🚀 Run Goal-Seeking Synthesizer", use_container_width=True):
    N_CANDS = 250000
    prog = st.progress(0.0, f"Stage 1: Flooding space with {N_CANDS} Sobol sequence geometries...")
    
    # 1. Generate Sobol Matrix
    sobol = SobolEngine(11, scramble=True, seed=42)
    X_u = sobol.draw(N_CANDS).numpy()
    mins = np.array([b[0] for b in BOUNDS_LIST])
    maxs = np.array([b[1] for b in BOUNDS_LIST])
    X_p = mins + X_u * (maxs - mins)
    
    # 2. Fast Geometric Sieve (CAP_W <= GAP - 1.0 & W1 + W2 <= 60.0)
    # [0:WS, 1:GAP, 2:MTX, 3:L1, 4:L2, 5:W1, 6:W2, 7:CAP_W, 8:ETCH_DEPTH, 9:L_dev, 10:Rt]
    mask_geom = (X_p[:, 7] <= X_p[:, 1] - 1.0) & ((X_p[:, 5] + X_p[:, 6]) <= 60.0)
    X_v = X_p[mask_geom]
    
    prog.progress(0.2, f"Stage 2: Instant Vectorized ML Inference on {len(X_v)} valid geometries...")
    
    # Map arrays for Vectorized ML
    WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH, L_dev, Rt = [X_v[:, i] for i in range(11)]
    SLAB_H = 0.460 - ETCH
    Perimeter = L1 + L2 + W1 + W2
    
    # Matching Predictor architecture
    x_c_5 = np.column_stack((WS, GAP, MTX, CAP_W, ETCH))
    x_lc_9 = np.column_stack((WS, GAP, MTX, L1, L2, W1, W2, SLAB_H, Perimeter))
    
    # --- Vectorized Fast Predict ---
    # VPI
    X_vpi_sc = engine.scalers['VPI_X'].transform(x_c_5)
    vpi_norm, _ = engine.models['VPI'].predict(X_vpi_sc, return_std=True)
    vpi_log = engine.scalers['VPI_y'].inverse_transform(vpi_norm.reshape(-1, 1)).ravel()
    vpi_b = (10 ** vpi_log) - 1e-15
    
    # NM and Z0
    X_nm_sc = engine.scalers['NMZ0_C']['X'].transform(x_c_5)
    nm_2d_norm, _ = engine.models['RN_NM'].predict(X_nm_sc, return_std=True)
    nm_2d = engine.scalers['NMZ0_C']['y']['RN NM'].inverse_transform(nm_2d_norm.reshape(-1, 1)).ravel()
    
    z0_2d_norm, _ = engine.models['Z0'].predict(X_nm_sc, return_std=True)
    z0_2d = engine.scalers['NMZ0_C']['y']['Z0 [Ω]'].inverse_transform(z0_2d_norm.reshape(-1, 1)).ravel()
    
    # dL and dC
    X_cst_sc = engine.scalers['NMZ0_CST']['X'].transform(x_lc_9)
    dL_norm, _ = engine.models['dL'].predict(X_cst_sc, return_std=True)
    dL = engine.scalers['NMZ0_CST']['y']['Delta_L_lumped'].inverse_transform(dL_norm.reshape(-1, 1)).ravel() / 1e12
    
    dC_norm, _ = engine.models['dC'].predict(X_cst_sc, return_std=True)
    dC = engine.scalers['NMZ0_CST']['y']['Delta_C_lumped'].inverse_transform(dC_norm.reshape(-1, 1)).ravel() / 1e15
    
    # Physics Merge
    L_dist_3d = np.maximum(1e-15, (z0_2d * nm_2d / engine.c0) + (dL / engine.L_cell))
    C_dist_3d = np.maximum(1e-15, (nm_2d / (z0_2d * engine.c0)) + (dC / engine.L_cell))
    nm_3d = engine.c0 * np.sqrt(L_dist_3d * C_dist_3d)
    z0_3d = np.sqrt(L_dist_3d / C_dist_3d)
    
    duty_cycle = L1 / 200.0
    vpi_duty = vpi_b / (1.0 - duty_cycle)
    vpi_len = vpi_duty / (L_dev / 10.0)
    
    # 3. Soft Sieve Filtering
    prog.progress(0.6, "Stage 3: Extracting broad performance subsets...")
    mask_perf = (np.abs(nm_3d - t_nm) <= tol_nm * 4) & \
                (np.abs(z0_3d - t_zc) <= tol_zc * 4) & \
                (vpi_len <= t_vpi + tol_vpi + 1.0)
                
    X_cands = X_v[mask_perf]
    if len(X_cands) == 0:
        prog.empty()
        st.error("❌ Your combination of Zc, nm, and Vpi targets is physically impossible within these geometric bounds.")
        st.stop()
        
    nm_c, zc_c, vpi_c = nm_3d[mask_perf], z0_3d[mask_perf], vpi_len[mask_perf]
    
    # 4. Pre-rank Top 1500 matches using coarse Euclidean Distance
    pre_err = ((nm_c - t_nm)/tol_nm)**2 + ((zc_c - t_zc)/tol_zc)**2 + ((vpi_c - t_vpi)/tol_vpi)**2
    best_idx = np.argsort(pre_err)[:1500]
    X_cands = X_cands[best_idx]
    
    prog.progress(0.8, f"Stage 4: Running Rigorous Broadband Calculation & Scoring top {len(X_cands)} designs...")
    final_results = []
    
    for i, x in enumerate(X_cands):
        # Precise predictor calls (automatically unpackages values correctly via the engine structure)
        res = engine.predict(geom=x[:9].tolist(), L_device_mm=x[9], Rt=x[10], Zs_driver=65.0, ng=2.27, plot=False)
        
        # EXTRACT NOMINAL VALUES DIRECTLY FROM THE TUPLES!
        nm_v = res['nm'][0]
        zc_v = res['z0'][0]
        vpi_v = res['vpi_len'][0]
        bw_v = res['bw'][0]
        rt_min = res['rt_min']

        if x[10] < rt_min:
            continue # Drops S11 safety violators instantly

        err = ((nm_v - t_nm)/tol_nm)**2 + ((zc_v - t_zc)/tol_zc)**2 + \
              ((vpi_v - t_vpi)/tol_vpi)**2 + ((bw_v - t_bw)/tol_bw)**2

        final_results.append({'x': x, 'err': err})

    if not final_results:
        prog.empty()
        st.error("❌ No designs met the S11 Impedance criteria. Increase Rt or Zc targets.")
        st.stop()
        
    final_results.sort(key=lambda item: item['err'])
    best_coarse_guesses = [r['x'] for r in final_results[:5]]

    # 5. SLSQP Gradient Polish
    prog.progress(0.9, "Stage 5: Gradient-Polishing the best neighborhood guesses to lock into exact targets...")
    st.info("Initiating local gradient descent to find the absolute mathematically exact valleys...")
    
    polished_results = []
    for coarse_x in best_coarse_guesses:
        polished_x = slsqp_inverse_polish(coarse_x)
        polished_results.append(polished_x)
        
    prog.progress(1.0, "Synthesizer Complete!")
    st.session_state['inv_res'] = polished_results
    st.success(f"✅ Search & Optimization complete! Showing the Top {len(st.session_state['inv_res'])} absolute best physical matches.")

# ==========================================
# 7. RESULTS DISPLAY (UI LOOP)
# ==========================================
if 'inv_res' in st.session_state:
    st.markdown("---")
    
    for i, x in enumerate(st.session_state['inv_res']):
        with st.expander(f"🏅 Candidate {i+1} | L={x[9]:.1f} mm | Rt={x[10]:.1f} Ω", expanded=(i==0)):
            res = engine.predict(geom=x[:9].tolist(), L_device_mm=x[9], Rt=x[10], Zs_driver=65.0, ng=2.27, plot=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("EO Bandwidth", f"{res['bw'][0]:.1f} GHz", delta=f"{res['bw'][0]-t_bw:+.1f} from target")
            c2.metric("VPI (Length Scaled)", f"{res['vpi_len'][0]:.2f} V", delta=f"{res['vpi_len'][0]-t_vpi:+.2f} from target", delta_color="inverse")
            c3.metric("Zc", f"{res['z0'][0]:.1f} Ω", delta=f"{res['z0'][0]-t_zc:+.1f} from target")
            c4.metric("nm", f"{res['nm'][0]:.4f}", delta=f"{res['nm'][0]-t_nm:+.4f} from target")
            
            st.markdown("### Geometric Layout & Configuration")
            p_dict = dict(zip(VAR_NAMES, x))
            st.table(pd.DataFrame([p_dict]).style.format("{:.3f}"))
            
            c_svg1, c_svg2 = st.columns(2)
            svg_t, svg_c = generate_exact_svg(p_dict)
            with c_svg1: st.markdown(render_svg(svg_t), unsafe_allow_html=True)
            with c_svg2: st.markdown(render_svg(svg_c), unsafe_allow_html=True)
            
            st.markdown("### 📊 Predicted FOMs (with 95% CI & MAE)")
            def fmt(tup): 
                return f"{tup[0]:.3f}", f"[{max(0, tup[0]-1.96*tup[1]):.3f}, {tup[0]+1.96*tup[1]:.3f}]"

            nm_v, nm_c = fmt(res['nm'])
            zc_v, zc_c = fmt(res['z0'])
            vp_v, vp_c = fmt(res['vpi_len'])
            al_v, al_c = fmt(res['alpha'])
            bw_v, bw_c = fmt(res['bw'])
            vll_v, vll_c = fmt(res['vpi_ll'])
            vf_v, vf_c = fmt(res['vpi_full'])

            data = [
                ["Microwave Index (nm)", nm_v, nm_c, f"± {res['nm'][2]:.4f}"],
                ["Impedance Zc [Ω]", zc_v, zc_c, f"± {res['z0'][2]:.2f}"],
                ["VPI Length Scaled [V]", vp_v, vp_c, f"± {res['vpi_len'][2]:.3f}"],
                ["RF Attenuation @ 60 GHz [dB/cm]", al_v, al_c, f"± {res['alpha'][2]:.3f}"],
                ["EO Bandwidth [GHz]", bw_v, bw_c, f"± {res['bw'][2]:.1f}"],
                ["VPI @ 60GHz (Walk‑off+Mismatch) [V]", vll_v, vll_c, "N/A (Derived)"],
                ["VPI @ 60GHz (Full RF Physics) [V]", vf_v, vf_c, "N/A (Derived)"]
            ]
            st.table(pd.DataFrame(data, columns=["FOM", "Predicted", "95% CI", "Global MAE"]))
            
            st.markdown("### 📈 Broadband RF Response")
            # Natively capture the plot generated by plot=True in the predictor
            fig = plt.gcf()
            st.pyplot(fig)
            plt.clf() # Clean the buffer for the next candidate in the loop
