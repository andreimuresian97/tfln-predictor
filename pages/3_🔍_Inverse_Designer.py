import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
import base64
from torch.quasirandom import SobolEngine
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE HEADER
# ==========================================
st.title("🔍 Inverse Geometry Synthesizer")
st.markdown("""
**Goal-Seeking Engine.** Input your specific target Figures of Merit (FOMs) and their acceptable tolerances. 
The GP surrogate model will scan hundreds of thousands of candidates to back-calculate up to 5 unique geometries that satisfy your exact requirements.
""")

# ==========================================
# 2. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.header("1. Performance Targets")
st.sidebar.markdown("Define the exact physics you want to achieve.")

target_vpi = st.sidebar.number_input("Target Vpi·L (V·cm)", value=1.50, step=0.05)
tol_vpi = st.sidebar.number_input("Vpi·L Tolerance (+/-)", value=0.10, step=0.01)

target_z0 = st.sidebar.number_input("Target Z0 (Ω)", value=55.0, step=1.0)
tol_z0 = st.sidebar.number_input("Z0 Tolerance (+/-)", value=1.0, step=0.5)

target_nm = st.sidebar.number_input("Target Index (nm)", value=2.270, step=0.01)
tol_nm = st.sidebar.number_input("Index Tolerance (+/-)", value=0.002, step=0.001)

min_s21 = st.sidebar.number_input("Min S21 Benchmark (dB) [2mm line]", value=-2.0, step=0.1)

st.sidebar.header("2. Geometry Constraints")
st.sidebar.markdown("Constrain the search space (e.g., for fabrication limits).")

def range_input(label, min_def, max_def, step=0.1, key_prefix="inv"):
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

VAR_NAMES = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2']
BOUNDS_LIST = [bounds[name] for name in VAR_NAMES]

# ==========================================
# 3. CORE FUNCTIONS (Memory Optimized)
# ==========================================
MODEL_DIR = "gp_surrogate_results_199_8var_fixed"
L_CM = 1.0 # Fixed to 1cm because target is VPI*1cm

@st.cache_resource
def load_scalers():
    scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler file not found at {scaler_path}")
        return None
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def load_single_model(name, scalers):
    file_map = {'VPI': 'gp_model_VPI.pkl', 'nm': 'gp_model_nm.pkl', 'Z0': 'gp_model_Z0.pkl', 'S21': 'gp_model_S21.pkl'}
    scaler_key_map = {'VPI': 'VPI (duty cycle)', 'nm': 'nm', 'Z0': 'Z0', 'S21': 'S21'}
    path = os.path.join(MODEL_DIR, file_map[name])
    if not os.path.exists(path):
        st.error(f"❌ Model {name} not found.")
        return None, None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model, scalers['y'][scaler_key_map[name]]

def predict_batch_memory_safe(X_norm, scalers):
    preds = {}
    model_names = ['VPI', 'nm', 'Z0', 'S21']
    prog_bar = st.progress(0, text="Scanning design space...")
    
    for i, name in enumerate(model_names):
        model, y_scaler = load_single_model(name, scalers)
        if model is None: return {}
        
        mean_norm, _ = model.predict(X_norm, return_std=True)
        mean_phys = y_scaler.inverse_transform(mean_norm.reshape(-1, 1)).ravel()
            
        if name == 'VPI':
            mean_phys = (10 ** mean_phys) / L_CM
            
        preds[name] = mean_phys
        
        del model
        gc.collect()
        prog_bar.progress((i + 1) / 4, text=f"Predicted {name}")
    
    prog_bar.empty()
    return preds

# ==========================================
# 4. VISUALIZATION ENGINE
# ==========================================
def render_svg(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="100%"/>'

def generate_exact_svg(p):
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W = p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0
    BOTTOM_LAYER_H = 0.23
    RIDGE_W = 0.8
    RIDGE_H = 0.23
    CAP_HEIGHT = 1.4
    C_ELEC = '#F5BD02'; C_SUB = '#00BFFF'; C_CAP = '#00BFFF'; C_LINE = 'black'
    
    CV_W = 800; CV_H = 600; CX = CV_W / 2
    
    def svg_arrow(x1, y1, x2, y2, text, text_loc="top", offset=10, font_size=14):
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1.5" marker-start="url(#arrow_start)" marker-end="url(#arrow_end)" />'
        tx, ty = (x1 + x2)/2, (y1 + y2)/2
        anchor = "middle"; dominant = "middle"
        if text_loc == "top": ty -= offset; dominant = "auto"
        elif text_loc == "bottom": ty += offset; dominant = "hanging"
        elif text_loc == "left": tx -= offset; anchor = "end"
        elif text_loc == "right": tx += offset; anchor = "start"
        return line + f'<text x="{tx}" y="{ty}" fill="{C_LINE}" font-family="sans-serif" font-size="{font_size}" font-weight="bold" text-anchor="{anchor}" dominant-baseline="{dominant}">{text}</text>'

    # Top-Down View
    SCALE_TOP = 3.5; CV_H_TOP = 700; CY_TOP = 380 
    def to_top(x, y): return CX + x*SCALE_TOP, CY_TOP - y*SCALE_TOP

    pts = [
        (GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
        (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
        (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
        (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)
    ]
    poly_str = " ".join([f"{to_top(x,y)[0]},{to_top(x,y)[1]}" for x, y in pts])
    ws_x1, ws_y1 = to_top(-(GAP/2 + WS), 100)
    
    arrows_top = ""
    top_arrow_y = 120; bot_arrow_y = -120; side_arrow_offset = 20
    arrows_top += svg_arrow(*to_top(-(GAP/2 + WS), top_arrow_y), *to_top(-GAP/2, top_arrow_y), "WS", "top", 10)
    arrows_top += svg_arrow(*to_top(GAP/2, top_arrow_y), *to_top(GAP/2 + WG, top_arrow_y), "WG", "top", 10)
    arrows_top += svg_arrow(*to_top(-GAP/2, bot_arrow_y), *to_top(GAP/2, bot_arrow_y), "GAP", "bottom", 10)
    arrows_top += svg_arrow(*to_top(GAP/2 - side_arrow_offset, -L1/2), *to_top(GAP/2 - side_arrow_offset, L1/2), "L1", "left", 10)
    arrows_top += svg_arrow(*to_top(GAP/2 + W1 + W2 + side_arrow_offset, -L2/2), *to_top(GAP/2 + W1 + W2 + side_arrow_offset, L2/2), "L2", "right", 10)
    arrows_top += svg_arrow(*to_top(GAP/2, -L1/2 - side_arrow_offset), *to_top(GAP/2 + W1, -L1/2 - side_arrow_offset), "W1", "bottom", 10)
    arrows_top += svg_arrow(*to_top(GAP/2 + W1, L2/2 + side_arrow_offset), *to_top(GAP/2 + W1 + W2, L2/2 + side_arrow_offset), "W2", "top", 10)

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

    # Cross Section View
    half_width_um = (WS/2 + GAP) + 1.0; SCALE_CS = 700.0 / (half_width_um * 2); CY_CS = 350 
    def to_cs(x, y): return CX + x*SCALE_CS, CY_CS - y*SCALE_CS

    base_y = BOTTOM_LAYER_H
    _, sub_y = to_cs(0, 0)
    sub_rect = f'<rect x="0" y="{sub_y}" width="{CV_W}" height="{CV_H}" fill="{C_SUB}" />'
    _, bl_y = to_cs(0, BOTTOM_LAYER_H)
    bl_rect = f'<rect x="0" y="{bl_y}" width="{CV_W}" height="{BOTTOM_LAYER_H*SCALE_CS}" fill="black" />'
    
    ws_x, ws_y = to_cs(-WS/2, base_y + MTX)
    ws_rect = f'<rect x="{ws_x}" y="{ws_y}" width="{WS*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    rwg_x, rwg_y = to_cs(WS/2 + GAP, base_y + MTX)
    rwg_rect = f'<rect x="{rwg_x}" y="{rwg_y}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    lwg_x, lwg_y = to_cs(-(WS/2 + GAP + 500), base_y + MTX)
    lwg_rect = f'<rect x="{lwg_x}" y="{lwg_y}" width="{500*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5" />'
    
    caps_svg = ""
    for center_x in [WS/2 + GAP/2, -WS/2 - GAP/2]:
        cx_svg, cy_svg = to_cs(center_x - CAP_W/2, base_y + CAP_HEIGHT)
        caps_svg += f'<rect x="{cx_svg}" y="{cy_svg}" width="{CAP_W*SCALE_CS}" height="{CAP_HEIGHT*SCALE_CS}" fill="{C_CAP}" stroke="{C_LINE}" stroke-width="1.5" />'
        rx_svg, ry_svg = to_cs(center_x - RIDGE_W/2, base_y + RIDGE_H)
        caps_svg += f'<rect x="{rx_svg}" y="{ry_svg}" width="{RIDGE_W*SCALE_CS}" height="{RIDGE_H*SCALE_CS}" fill="black" />'

    arrows_cs = ""
    dim_y = base_y + max(MTX, CAP_HEIGHT) + (3.0 if MTX < 5 else 0.5 * MTX)
    arrows_cs += svg_arrow(*to_cs(-WS/2, dim_y), *to_cs(WS/2, dim_y), "WS", "top", 15)
    arrows_cs += svg_arrow(*to_cs(WS/2, dim_y), *to_cs(WS/2 + GAP, dim_y), "GAP", "top", 15)
    arrows_cs += svg_arrow(*to_cs(-WS/2 - GAP/2 - CAP_W/2, base_y + CAP_HEIGHT + 1.0), *to_cs(-WS/2 - GAP/2 + CAP_W/2, base_y + CAP_HEIGHT + 1.0), "CAP_W", "top", 15)
    arrows_cs += svg_arrow(*to_cs(-WS/2 + 2.0, base_y), *to_cs(-WS/2 + 2.0, base_y + MTX), "MTX", "right", 10)
    
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
# 5. INVERSE SEARCH LOGIC
# ==========================================
def run_inverse_search():
    scalers = load_scalers()
    if not scalers: return

    N_CANDIDATES = 300000 
    
    with st.spinner(f"Flooding search space with {N_CANDIDATES} geometric combinations..."):
        sobol = SobolEngine(dimension=8, scramble=True, seed=42)
        X_unit = sobol.draw(N_CANDIDATES).to(dtype=torch.float64)
        
        mins = torch.tensor([b[0] for b in BOUNDS_LIST], dtype=torch.float64)
        maxs = torch.tensor([b[1] for b in BOUNDS_LIST], dtype=torch.float64)
        X_phys = mins + X_unit * (maxs - mins)
        X_phys_np = X_phys.numpy()
        
        # Hardware validity filters
        gap_col = 1; cap_col = 3; w1_col = 6; w2_col = 7
        mask_geom = (X_phys_np[:, cap_col] < (X_phys_np[:, gap_col] - 1.0)) & \
                    ((X_phys_np[:, w1_col] + X_phys_np[:, w2_col]) < 65.0)
        X_valid = X_phys_np[mask_geom]
    
    if len(X_valid) == 0:
        st.error("No candidates matched the base geometric constraints (e.g., Cap fitting in Gap).")
        return

    scaler_X = scalers['X']['input']
    X_norm = scaler_X.transform(X_valid)
    preds = predict_batch_memory_safe(X_norm, scalers)
    if not preds: return
    
    # Target Filtering
    mask_perf = (np.abs(preds['VPI'] - target_vpi) <= tol_vpi) & \
                (np.abs(preds['Z0'] - target_z0) <= tol_z0) & \
                (np.abs(preds['nm'] - target_nm) <= tol_nm) & \
                (preds['S21'] >= min_s21)
                
    valid_indices = np.where(mask_perf)[0]
    
    if len(valid_indices) == 0:
        st.error("❌ No geometries exist that satisfy all of these conditions simultaneously. Please relax your tolerances or target values.")
        return
        
    # Rank by closeness to exact targets (Normalized Euclidean Error)
    err_vpi = ((preds['VPI'][valid_indices] - target_vpi) / target_vpi) ** 2
    err_z0  = ((preds['Z0'][valid_indices] - target_z0) / target_z0) ** 2
    err_nm  = ((preds['nm'][valid_indices] - target_nm) / target_nm) ** 2
    total_error = err_vpi + err_z0 + err_nm
    
    # Get top 5 indices (or fewer if < 5 exist)
    top_k = min(5, len(valid_indices))
    best_local_indices = np.argsort(total_error)[:top_k]
    best_global_indices = valid_indices[best_local_indices]
    
    results_list = []
    for idx in best_global_indices:
        geom = X_valid[idx]
        res = {
            'x': geom,
            'VPI': preds['VPI'][idx],
            'Z0': preds['Z0'][idx],
            'nm': preds['nm'][idx],
            'S21': preds['S21'][idx]
        }
        results_list.append(res)
        
    st.session_state['inverse_results'] = results_list
    st.session_state['total_found'] = len(valid_indices)

# ==========================================
# 6. MAIN UI EXECUTION
# ==========================================
if st.button("SYNTHESIZE GEOMETRY", type="primary"):
    run_inverse_search()

if 'inverse_results' in st.session_state:
    results_list = st.session_state['inverse_results']
    total_found = st.session_state['total_found']
    
    st.markdown("---")
    st.success(f"✅ Found **{total_found}** valid geometries in the design space. Showing the top {len(results_list)} best matches:")
    
    for i, res in enumerate(results_list):
        # Create an expander for each geometry
        with st.expander(f"Candidate {i+1} | Z0: {res['Z0']:.1f} Ω | VPI: {res['VPI']:.2f} V·cm | nm: {res['nm']:.3f} | S21: {res['S21']:.2f} dB", expanded=(i==0)):
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Z0", f"{res['Z0']:.2f} Ω", delta=f"{res['Z0'] - target_z0:+.2f} from target")
            c2.metric("VPI", f"{res['VPI']:.2f} V·cm", delta=f"{res['VPI'] - target_vpi:+.2f} from target", delta_color="inverse")
            c3.metric("nm", f"{res['nm']:.4f}", delta=f"{res['nm'] - target_nm:+.4f} from target")
            c4.metric("S21", f"{res['S21']:.2f} dB")
            
            st.markdown("**Geometric Parameters (µm)**")
            df_params = pd.DataFrame([res['x']], columns=VAR_NAMES)
            st.table(df_params.style.format("{:.3f}"))
            
            p_dict = {name: val for name, val in zip(VAR_NAMES, res['x'])}
            svg_t, svg_c = generate_exact_svg(p_dict)
            
            st.markdown(render_svg(svg_t), unsafe_allow_html=True)
            st.markdown(render_svg(svg_c), unsafe_allow_html=True)
