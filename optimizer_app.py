import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
import base64
from scipy.optimize import minimize
from torch.quasirandom import SobolEngine
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="TFLN Optimizer", layout="wide")
st.title("🚀 AI-Driven TFLN Modulator Optimizer")
st.markdown("""
**Find the perfect geometry.**
This tool uses a **Hybrid Bayesian Optimization** engine (Sobol Sampling + Gradient Polishing) 
to find the optimal TFLN cross-section for your specific constraints.
""")

# ==========================================
# 2. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.header("1. Geometry Constraints")
st.sidebar.markdown("Define the allowable search range.")

# Helper for sidebar ranges
def range_input(label, min_def, max_def, step=0.1, key_prefix=""):
    col1, col2 = st.sidebar.columns(2)
    min_val = col1.number_input(f"Min {label}", value=float(min_def), step=step, key=f"{key_prefix}_min_{label}")
    max_val = col2.number_input(f"Max {label}", value=float(max_def), step=step, key=f"{key_prefix}_max_{label}")
    return (min_val, max_val)

bounds = {}
bounds['WS']    = range_input("Signal Width (WS)", 35.0, 70.0)
bounds['GAP']   = range_input("Gap", 3.0, 15.0)
bounds['MTX']   = range_input("Metal Thickness (MTX)", 1.8, 15.0)
bounds['CAP_W'] = range_input("Cap Width", 1.4, 15.0)
bounds['L1']    = range_input("L1 (T-Rail)", 3.0, 60.0)
bounds['L2']    = range_input("L2 (T-Rail)", 3.0, 190.0)
bounds['W1']    = range_input("W1 (T-Rail)", 3.0, 60.0)
bounds['W2']    = range_input("W2 (T-Rail)", 3.0, 60.0)

st.sidebar.header("2. Performance Targets")
const_vpi = st.sidebar.number_input("Max Vpi·L (V·cm)", value=2.0, step=0.1)
const_s21 = st.sidebar.number_input("Min S21 (dB)", value=-2.0, step=0.1)
const_nm_target = st.sidebar.number_input("Target Index (nm)", value=2.27, step=0.01)
const_nm_tol = st.sidebar.number_input("Index Tolerance (+/-)", value=0.03, step=0.005)
beta = st.sidebar.slider("Optimization Aggressiveness (Beta)", 0.0, 2.0, 0.5, help="0.0 = Safe, 1.0 = Curious")

# Fixed list for ordering
VAR_NAMES = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2']
BOUNDS_LIST = [bounds[name] for name in VAR_NAMES]

# ==========================================
# 3. CORE FUNCTIONS (Memory Optimized)
# ==========================================

# UPDATED: Matches your repo folder name
MODEL_DIR = "gp_surrogate_results_199_8var_fixed"

@st.cache_resource
def load_scalers():
    """Load scalers only once."""
    scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler file not found at {scaler_path}")
        return None
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def load_single_model(name, scalers):
    """Load one model, use it, and return the model object."""
    file_map = {
        'VPI': 'gp_model_VPI.pkl',
        'nm':  'gp_model_nm.pkl',
        'Z0':  'gp_model_Z0.pkl',
        'S21': 'gp_model_S21.pkl'
    }
    scaler_key_map = {
        'VPI': 'VPI (duty cycle)',
        'nm':  'nm',
        'Z0':  'Z0',
        'S21': 'S21'
    }
    
    path = os.path.join(MODEL_DIR, file_map[name])
    if not os.path.exists(path):
        st.error(f"❌ Model {name} not found at {path}")
        return None, None

    with open(path, 'rb') as f:
        model = pickle.load(f)
        
    scaler_y = scalers['y'][scaler_key_map[name]]
    return model, scaler_y

def predict_batch_memory_safe(X_norm, scalers):
    """
    Loads models sequentially to save RAM.
    Returns a dictionary of predictions.
    """
    preds = {}
    model_names = ['VPI', 'nm', 'Z0', 'S21']
    
    prog_bar = st.progress(0, text="Predicting...")
    
    for i, name in enumerate(model_names):
        model, y_scaler = load_single_model(name, scalers)
        if model is None: return {}
        
        # Predict
        mean_norm, std_norm = model.predict(X_norm, return_std=True)
        
        # Inverse Transform
        mean_phys = y_scaler.inverse_transform(mean_norm.reshape(-1, 1)).ravel()
        if hasattr(y_scaler, 'scale_'):
            std_phys = std_norm * y_scaler.scale_[0]
        else:
            std_phys = std_norm
            
        if name == 'VPI':
            mean_phys = 10 ** mean_phys
            std_phys = mean_phys * np.log(10) * std_phys
            
        preds[name] = {'mean': mean_phys, 'std': std_phys}
        
        del model
        gc.collect()
        
        prog_bar.progress((i + 1) / 4, text=f"Predicted {name}")
    
    prog_bar.empty()
    return preds

# ==========================================
# 4. VISUALIZATION ENGINE (FROM APP.PY)
# ==========================================

def render_svg(svg_string):
    """Renders SVG string in Streamlit"""
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="100%"/>'

def generate_exact_svg(p):
    """
    Generates exact SVG strings for Top-Down and Cross-Section views.
    Expects dictionary p with keys: WS, GAP, MTX, CAP_W, L1, L2, W1, W2
    """
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W = p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0
    BOTTOM_LAYER_H = 0.23
    RIDGE_W = 0.8
    RIDGE_H = 0.23
    CAP_HEIGHT = 1.4
    C_ELEC = '#F5BD02' # Gold
    C_SUB = '#00BFFF'  # Deep Sky Blue
    C_CAP = '#00BFFF'
    C_LINE = 'black'
    
    # === GLOBAL CANVAS SETTINGS ===
    CV_W = 800
    CV_H = 600
    CX = CV_W / 2
    
    def svg_arrow(x1, y1, x2, y2, text, text_loc="top", offset=10, font_size=14):
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1.5" marker-start="url(#arrow_start)" marker-end="url(#arrow_end)" />'
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        tx, ty = mx, my
        anchor = "middle"
        dominant = "middle"
        if text_loc == "top": ty -= offset; dominant = "auto"
        elif text_loc == "bottom": ty += offset; dominant = "hanging"
        elif text_loc == "left": tx -= offset; anchor = "end"
        elif text_loc == "right": tx += offset; anchor = "start"
        
        txt = f'<text x="{tx}" y="{ty}" fill="{C_LINE}" font-family="sans-serif" font-size="{font_size}" font-weight="bold" text-anchor="{anchor}" dominant-baseline="{dominant}">{text}</text>'
        return line + txt

    # ==========================================
    # FIGURE 1: TOP-DOWN VIEW
    # ==========================================
    
    SCALE_TOP = 3.5 
    CV_H_TOP = 700 
    CY_TOP = 380 
    
    def to_top(x, y): 
        return CX + x*SCALE_TOP, CY_TOP - y*SCALE_TOP

    # Polygons
    pts = [
        (GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
        (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
        (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
        (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)
    ]
    poly_str = " ".join([f"{to_top(x,y)[0]},{to_top(x,y)[1]}" for x, y in pts])
    ws_x1, ws_y1 = to_top(-(GAP/2 + WS), 100)
    
    # Arrows
    arrows_top = ""
    top_arrow_y = 120 
    bot_arrow_y = -120 
    side_arrow_offset = 20
    
    arrows_top += svg_arrow(*to_top(-(GAP/2 + WS), top_arrow_y), *to_top(-GAP/2, top_arrow_y), "WS", "top", 10)
    arrows_top += svg_arrow(*to_top(GAP/2, top_arrow_y), *to_top(GAP/2 + WG, top_arrow_y), "WG", "top", 10)
    arrows_top += svg_arrow(*to_top(-GAP/2, bot_arrow_y), *to_top(GAP/2, bot_arrow_y), "GAP", "bottom", 10)
    
    l1_x = GAP/2 - side_arrow_offset
    arrows_top += svg_arrow(*to_top(l1_x, -L1/2), *to_top(l1_x, L1/2), "L1", "left", 10)
    l2_x = GAP/2 + W1 + W2 + side_arrow_offset
    arrows_top += svg_arrow(*to_top(l2_x, -L2/2), *to_top(l2_x, L2/2), "L2", "right", 10)
    
    w_arrows_y_top = L2/2 + side_arrow_offset
    w_arrows_y_bot = -L1/2 - side_arrow_offset
    arrows_top += svg_arrow(*to_top(GAP/2, w_arrows_y_bot), *to_top(GAP/2 + W1, w_arrows_y_bot), "W1", "bottom", 10)
    arrows_top += svg_arrow(*to_top(GAP/2 + W1, w_arrows_y_top), *to_top(GAP/2 + W1 + W2, w_arrows_y_top), "W2", "top", 10)

    svg_top = f"""
    <svg width="{CV_W}" height="{CV_H_TOP}" viewBox="0 0 {CV_W} {CV_H_TOP}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        <text x="20" y="50" text-anchor="start" font-family="sans-serif" font-size="24" font-weight="bold">Top-Down View</text>
        
        <rect x="{ws_x1}" y="{ws_y1}" width="{WS*SCALE_TOP}" height="{200*SCALE_TOP}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />
        <polygon points="{poly_str}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />
        {arrows_top}
    </svg>
    """

    # ==========================================
    # FIGURE 2: CROSS SECTION VIEW (ZOOMED)
    # ==========================================
    
    CROP_MARGIN = 1.0
    half_width_um = (WS/2 + GAP) + CROP_MARGIN
    total_width_um = half_width_um * 2
    SCALE_CS = 700.0 / total_width_um
    CY_CS = 350 
    
    def to_cs(x, y): 
        return CX + x*SCALE_CS, CY_CS - y*SCALE_CS

    base_y_math = BOTTOM_LAYER_H
    
    _, sub_y = to_cs(0, 0)
    sub_rect = f'<rect x="0" y="{sub_y}" width="{CV_W}" height="{CV_H}" fill="{C_SUB}" />'
    
    _, bl_y = to_cs(0, BOTTOM_LAYER_H)
    bl_rect = f'<rect x="0" y="{bl_y}" width="{CV_W}" height="{BOTTOM_LAYER_H*SCALE_CS}" fill="black" />'
    
    ws_x, ws_y = to_cs(-WS/2, base_y_math + MTX)
    ws_rect = f'<rect x="{ws_x}" y="{ws_y}" width="{WS*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    
    rwg_x, rwg_y = to_cs(WS/2 + GAP, base_y_math + MTX)
    rwg_rect = f'<rect x="{rwg_x}" y="{rwg_y}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    
    lwg_x, lwg_y = to_cs(-(WS/2 + GAP + 500), base_y_math + MTX)
    lwg_rect = f'<rect x="{lwg_x}" y="{lwg_y}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    
    caps_svg = ""
    for center_x in [WS/2 + GAP/2, -WS/2 - GAP/2]:
        cx_svg, cy_svg = to_cs(center_x - CAP_W/2, base_y_math + CAP_HEIGHT)
        caps_svg += f'<rect x="{cx_svg}" y="{cy_svg}" width="{CAP_W*SCALE_CS}" height="{CAP_HEIGHT*SCALE_CS}" fill="{C_CAP}" stroke="{C_LINE}" stroke-width="1.5" />'
        
        rx_svg, ry_svg = to_cs(center_x - RIDGE_W/2, base_y_math + RIDGE_H)
        caps_svg += f'<rect x="{rx_svg}" y="{ry_svg}" width="{RIDGE_W*SCALE_CS}" height="{RIDGE_H*SCALE_CS}" fill="black" />'

    arrows_cs = ""
    dim_y = base_y_math + max(MTX, CAP_HEIGHT) + (3.0 if MTX < 5 else 0.5 * MTX)
    
    arr_y = dim_y
    arrows_cs += svg_arrow(*to_cs(-WS/2, arr_y), *to_cs(WS/2, arr_y), "WS", "top", 15)
    arrows_cs += svg_arrow(*to_cs(WS/2, arr_y), *to_cs(WS/2 + GAP, arr_y), "GAP", "top", 15)
    
    l_gap_c = -WS/2 - GAP/2
    cap_arr_y = base_y_math + CAP_HEIGHT + 1.0
    arrows_cs += svg_arrow(*to_cs(l_gap_c - CAP_W/2, cap_arr_y), *to_cs(l_gap_c + CAP_W/2, cap_arr_y), "CAP_W", "top", 15)
    
    mtx_x_pos = -WS/2 + 2.0 
    mtx_y_start = base_y_math
    mtx_y_end = base_y_math + MTX
    arrows_cs += svg_arrow(*to_cs(mtx_x_pos, mtx_y_start), *to_cs(mtx_x_pos, mtx_y_end), "MTX", "right", 10)
    
    svg_cross = f"""
    <svg width="{CV_W}" height="{CV_H}" viewBox="0 0 {CV_W} {CV_H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        <text x="20" y="50" text-anchor="start" font-family="sans-serif" font-size="24" font-weight="bold">Cross-Section View</text>
        
        {sub_rect} {bl_rect} {ws_rect} {rwg_rect} {lwg_rect} {caps_svg} {arrows_cs}
    </svg>
    """
    return svg_top, svg_cross

# ==========================================
# 5. OPTIMIZATION LOGIC
# ==========================================

def run_optimization():
    scalers = load_scalers()
    if not scalers: return

    # 1. Generate Candidates
    N_CANDIDATES = 200000 
    
    with st.spinner(f"Generating {N_CANDIDATES} Monte Carlo samples..."):
        sobol = SobolEngine(dimension=8, scramble=True, seed=42)
        X_unit = sobol.draw(N_CANDIDATES).to(dtype=torch.float64)
        
        mins = torch.tensor([b[0] for b in BOUNDS_LIST], dtype=torch.float64)
        maxs = torch.tensor([b[1] for b in BOUNDS_LIST], dtype=torch.float64)
        X_phys = mins + X_unit * (maxs - mins)
        X_phys_np = X_phys.numpy()
        
        # 2. Geometric Filtering
        gap_col = 1; cap_col = 3; w1_col = 6; w2_col = 7
        mask_geom = (X_phys_np[:, cap_col] < (X_phys_np[:, gap_col] - 1.0)) & \
                    ((X_phys_np[:, w1_col] + X_phys_np[:, w2_col]) < 65.0)
        
        X_valid = X_phys_np[mask_geom]
    
    if len(X_valid) == 0:
        st.error("No candidates match geometric constraints!")
        return
    
    st.caption(f"Valid Geometries: {len(X_valid)} / {N_CANDIDATES}")

    # 3. Batch Predict
    scaler_X = scalers['X']['input']
    X_norm = scaler_X.transform(X_valid)
    
    preds = predict_batch_memory_safe(X_norm, scalers)
    if not preds: return
    
    # 4. Apply Performance Constraints
    mask_perf = (preds['VPI']['mean'] < const_vpi) & \
                (np.abs(preds['nm']['mean'] - const_nm_target) < const_nm_tol) & \
                (preds['S21']['mean'] > const_s21)
                
    valid_indices = np.where(mask_perf)[0]
    
    if len(valid_indices) == 0:
        st.warning(f"No designs met constraints. Try relaxing them.")
        return

    # 5. Select Discrete Winner
    z0_mean = preds['Z0']['mean'][valid_indices]
    z0_std = preds['Z0']['std'][valid_indices]
    scores = z0_mean + (beta * z0_std)
    
    best_idx_local = np.argmax(scores)
    best_idx_global = valid_indices[best_idx_local]
    
    winner_x = X_valid[best_idx_global]
    
    # 6. Polishing (SLSQP)
    perform_polishing(winner_x, scalers)

def perform_polishing(x0, scalers):
    st.info("🛠 Polishing geometry for maximum Z0...")
    
    models = {}
    model_names = ['VPI', 'nm', 'Z0', 'S21']
    for name in model_names:
        m, s = load_single_model(name, scalers)
        models[name] = {'model': m, 'scaler': s}
        
    scaler_X = scalers['X']['input']

    def predict_one(x):
        x_n = scaler_X.transform(x.reshape(1, -1))
        res = {}
        for k, v in models.items():
            m_val = v['model'].predict(x_n)[0]
            val = v['scaler'].inverse_transform(m_val.reshape(-1,1)).item()
            if k == 'VPI': val = 10**val
            res[k] = val
        return res

    def obj(x): return -1.0 * predict_one(x)['Z0']
    
    # Constraints for Scipy
    cons = [
        {'type': 'ineq', 'fun': lambda x: const_vpi - predict_one(x)['VPI']},
        {'type': 'ineq', 'fun': lambda x: predict_one(x)['S21'] - const_s21},
        {'type': 'ineq', 'fun': lambda x: (const_nm_target + const_nm_tol) - predict_one(x)['nm']},
        {'type': 'ineq', 'fun': lambda x: predict_one(x)['nm'] - (const_nm_target - const_nm_tol)},
        {'type': 'ineq', 'fun': lambda x: (x[1]-1.0) - x[3]}, # GAP-1 > CAP
        {'type': 'ineq', 'fun': lambda x: 65.0 - (x[6] + x[7])} # W < 65
    ]
    
    res = minimize(obj, x0, method='SLSQP', bounds=BOUNDS_LIST, constraints=cons)
    
    final_x = res.x
    final_preds = predict_one(final_x)
    
    # Save to Session State
    st.session_state['result'] = {
        'x': final_x,
        'preds': final_preds
    }
    
    del models
    gc.collect()

# ==========================================
# 6. MAIN UI EXECUTION
# ==========================================

if st.button("RUN OPTIMIZATION", type="primary"):
    run_optimization()

if 'result' in st.session_state:
    res = st.session_state['result']
    preds = res['preds']
    
    st.markdown("---")
    st.header("🏆 Optimal Geometry Found")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Z0 (Impedance)", f"{preds['Z0']:.2f} Ω", delta="Maximize")
    c2.metric("S21 (Loss)", f"{preds['S21']:.2f} dB", delta_color="normal")
    c3.metric("Vpi·L", f"{preds['VPI']:.2f} V·cm", delta_color="inverse")
    c4.metric("Index (nm)", f"{preds['nm']:.4f}", help="Target: 2.27")
    
    # Params
    st.subheader("Optimal Parameters")
    df_params = pd.DataFrame([res['x']], columns=VAR_NAMES)
    st.table(df_params.style.format("{:.3f}"))
    
    # SVG Plotting
    # Convert array to dictionary for the SVG function
    p_dict = {name: val for name, val in zip(VAR_NAMES, res['x'])}
    
    svg_t, svg_c = generate_exact_svg(p_dict)
    
    st.markdown("### Visualization")
    st.markdown(render_svg(svg_t), unsafe_allow_html=True)
    st.markdown(render_svg(svg_c), unsafe_allow_html=True)
