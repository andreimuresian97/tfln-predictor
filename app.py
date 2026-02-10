import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gc
import base64
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="TFLN Geometry Predictor", layout="centered")

st.title("⚡ TFLN Performance Predictor")
st.markdown("""
Instant inference for **VPI, nm, Z0, and S21**.  
**Note:** All predictions are performed at **1330 nm**.
""")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Geometry Parameters")

def user_input_features():
    # Global Device Parameters
    st.sidebar.subheader("Global Device Parameters")
    length_cm = st.sidebar.number_input("Device Length (L) [cm]", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.2f")
    
    # Active Region (High precision)
    st.sidebar.subheader("Active Region")
    ws = st.sidebar.number_input("WS (Signal Width) [µm]", value=22.936, step=0.001, format="%.3f")
    gap = st.sidebar.number_input("GAP [µm]", value=10.311, step=0.001, format="%.3f")
    mtx = st.sidebar.number_input("MTX (Metal Thickness) [µm]", value=8.07, step=0.001, format="%.3f")
    cap_w = st.sidebar.number_input("CAP_W (Cap Width) [µm]", value=1.65, step=0.001, format="%.3f")
    
    # T-Structure Dimensions
    st.sidebar.subheader("T-Structure Dimensions")
    l1 = st.sidebar.number_input("L1 (Inner Length) [µm]", value=8.0, step=0.1, format="%.1f")
    l2 = st.sidebar.number_input("L2 (Outer Length) [µm]", value=86.0, step=0.1, format="%.1f")
    w1 = st.sidebar.number_input("W1 (Inner Width) [µm]", value=5.0, step=0.1, format="%.1f")
    w2 = st.sidebar.number_input("W2 (Outer Width) [µm]", value=11.0, step=0.1, format="%.1f")
    
    # Fixed Parameters Report
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fixed Parameters")
    st.sidebar.info(
        """
        **Pitch:** 200 µm  
        **WG:** 70 µm  
        **CAP_H:** 1.4 µm
        """
    )
    
    return length_cm, [ws, gap, mtx, cap_w, l1, l2, w1, w2], {
        "WS": ws, "GAP": gap, "MTX": mtx, "CAP_W": cap_w,
        "L1": l1, "L2": l2, "W1": w1, "W2": w2
    }

length_cm, geometry_list, params = user_input_features()

# --- HELPER: RENDER SVG AS IMAGE ---
def render_svg(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="100%"/>'

# --- LOW MEMORY PREDICTOR ENGINE ---
def predict_sequentially(geometry, L_cm):
    results = {}
    model_dir = Path("gp_surrogate_results_199_8var_fixed")
    
    if not model_dir.exists():
        st.error(f"❌ Folder '{model_dir}' not found.")
        return {}

    try:
        with open(model_dir / "scalers.pkl", 'rb') as f:
            scalers_data = pickle.load(f)
            scaler_X = scalers_data['X']['input']
            scalers_y = scalers_data['y']
    except Exception as e:
        st.error(f"❌ Error loading scalers: {e}")
        return {}

    X_input = np.array(geometry).reshape(1, -1)
    X_norm = scaler_X.transform(X_input)

    targets = {'VPI': 'VPI (duty cycle)', 'nm': 'nm', 'Z0': 'Z0', 'S21': 'S21'}
    
    progress = st.progress(0)
    idx = 0
    
    for safe_name, scaler_key in targets.items():
        try:
            with open(model_dir / f"gp_model_{safe_name}.pkl", 'rb') as f:
                model = pickle.load(f)
            
            y_pred_norm, y_std_norm = model.predict(X_norm, return_std=True)
            
            del model
            gc.collect() 
            
            scaler = scalers_y[scaler_key]
            y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()[0]
            
            if hasattr(scaler, 'scale_'):
                y_std = y_std_norm[0] * scaler.scale_[0]
            else:
                y_std = y_std_norm[0]
            
            if safe_name == "VPI":
                real_val = 10 ** y_pred
                real_std = real_val * np.log(10) * y_std
                y_pred = real_val / L_cm
                y_std = real_std / L_cm
            
            results[safe_name] = {'value': y_pred, 'lower_bound': y_pred - 1.96 * y_std, 'upper_bound': y_pred + 1.96 * y_std}

        except Exception as e:
            st.warning(f"Could not predict {safe_name}: {e}")
        
        idx += 1
        progress.progress(idx / 4)
        
    progress.empty()
    return results

# --- EXACT REPLICA SVG PLOTTING ---
def generate_exact_svg(p):
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

    # [FIX] Title moved to Left (x=20) and anchor=start to completely avoid overlap
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
    
    # [FIX] Title moved to Left (x=20) and anchor=start for consistency
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

# --- LAYOUT ---
st.subheader("1. Geometry Visualization")
st.caption("Updated automatically.")

svg_t, svg_c = generate_exact_svg(params)

st.markdown(render_svg(svg_t), unsafe_allow_html=True)
st.markdown("---")
st.markdown(render_svg(svg_c), unsafe_allow_html=True)

st.markdown("---")
st.subheader("2. Performance Prediction")

if st.button("Predict Performance", type="primary"):
    results = predict_sequentially(geometry_list, length_cm)
    if results:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VPI", f"{results['VPI']['value']:.2f} V")
        col2.metric("Loss (S21)", f"{results['S21']['value']:.2f} dB")
        col3.metric("Impedance (Z0)", f"{results['Z0']['value']:.1f} Ω")
        col4.metric("Index (nm)", f"{results['nm']['value']:.3f}")
        
        data = [[k, f"{v['value']:.4f}", f"[{v['lower_bound']:.4f}, {v['upper_bound']:.4f}]"] for k, v in results.items()]
        st.table(pd.DataFrame(data, columns=["FOM", "Value", "95% CI"]))
