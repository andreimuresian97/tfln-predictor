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
    C_ELEC = '#F5BD02' # Gold-ish Yellow
    C_SUB = '#00BFFF'  # Deep Sky Blue (Matches your image)
    C_CAP = '#00BFFF'  # Same Blue for Caps
    C_LINE = 'black'
    
    # FIXED CANVAS SIZE FOR BOTH IMAGES
    CANVAS_W = 800
    CANVAS_H = 600
    CX = CANVAS_W / 2
    
    def svg_arrow(x1, y1, x2, y2, text, text_loc="top", offset=10, font_size=12):
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1" marker-start="url(#arrow_start)" marker-end="url(#arrow_end)" />'
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        tx, ty = mx, my
        anchor = "middle"
        dominant = "middle"
        if text_loc == "top": ty -= offset; dominant = "auto"
        elif text_loc == "bottom": ty += offset; dominant = "hanging"
        elif text_loc == "left": tx -= offset; anchor = "end"
        elif text_loc == "right": tx += offset; anchor = "start"
        
        txt = f'<text x="{tx}" y="{ty}" fill="{C_LINE}" font-family="sans-serif" font-size="{font_size}" text-anchor="{anchor}" dominant-baseline="{dominant}">{text}</text>'
        return line + txt

    # --- FIGURE 1: TOP-DOWN VIEW ---
    # We use a scale that fits the T-Structure into the 800x600 box
    SCALE_TOP = 4.0 
    CY_TOP = CANVAS_H / 2 + 50 # Centered vertically
    
    def to_svg_top(x, y): return CX + x*SCALE_TOP, CY_TOP - y*SCALE_TOP

    pts = [
        (GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
        (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
        (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
        (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)
    ]
    poly_str = " ".join([f"{to_svg_top(x,y)[0]},{to_svg_top(x,y)[1]}" for x, y in pts])
    ws_x1, ws_y1 = to_svg_top(-(GAP/2 + WS), 100)
    
    arrows_svg = ""
    top_y = 104
    arrows_svg += svg_arrow(*to_svg_top(-(GAP/2 + WS), top_y), *to_svg_top(-GAP/2, top_y), "WS", "top", 10)
    arrows_svg += svg_arrow(*to_svg_top(GAP/2, top_y), *to_svg_top(GAP/2 + WG, top_y), "WG", "top", 10)
    
    bot_y = -104
    arrows_svg += svg_arrow(*to_svg_top(-GAP/2, bot_y), *to_svg_top(GAP/2, bot_y), "GAP", "bottom", 10)
    l1_x = GAP/2 - 4
    arrows_svg += svg_arrow(*to_svg_top(l1_x, -L1/2), *to_svg_top(l1_x, L1/2), "L1", "left", 10)
    l2_x = GAP/2 + W1 + W2 + 4
    arrows_svg += svg_arrow(*to_svg_top(l2_x, -L2/2), *to_svg_top(l2_x, L2/2), "L2", "right", 10)
    w1_y = -L1/2 - 4
    arrows_svg += svg_arrow(*to_svg_top(GAP/2, w1_y), *to_svg_top(GAP/2 + W1, w1_y), "W1", "bottom", 10)
    w2_y = L2/2 + 4
    arrows_svg += svg_arrow(*to_svg_top(GAP/2 + W1, w2_y), *to_svg_top(GAP/2 + W1 + W2, w2_y), "W2", "top", 10)

    svg_top = f"""
    <svg width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        <text x="{CX}" y="40" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">Top-Down View</text>
        <rect x="{ws_x1}" y="{ws_y1}" width="{WS*SCALE_TOP}" height="{200*SCALE_TOP}" fill="{C_ELEC}" stroke="{C_LINE}" />
        <polygon points="{poly_str}" fill="{C_ELEC}" stroke="{C_LINE}" />
        {arrows_svg}
    </svg>
    """

    # --- FIGURE 2: CROSS SECTION (FIXED SIZE, STRICT CROP) ---
    # We use a much larger scale because we are zoomed in
    SCALE_CS = 15.0 
    CY_CS = 400 # Vertically centered in the 600px high canvas
    
    def to_svg_cs(x, y): return CX + x*SCALE_CS, CY_CS - y*SCALE_CS

    # --- VIEWBOX LOGIC ---
    # Center = 0.
    # We want to show from Left Ground Edge + 1um to Right Ground Edge + 1um.
    # Right Ground Edge starts at: (WS/2 + GAP)
    # Visible part ends at: (WS/2 + GAP) + 1.0
    VISIBLE_EXTENSION = 1.0
    x_limit_um = (WS/2 + GAP) + VISIBLE_EXTENSION
    
    # Calculate pixel range centered on CX
    # x_min_px = CX - (x_limit_um * SCALE_CS)
    # x_max_px = CX + (x_limit_um * SCALE_CS)
    # width_px = x_max_px - x_min_px
    
    # Since we want the image to remain 800x600, we simply draw everything relative to CX/CY
    # and use the SVG viewBox to "crop" it to exactly the width we want.
    
    # ViewBox X: The starting X coordinate in the SVG canvas space
    vb_x = CX - (x_limit_um * SCALE_CS)
    vb_w = (x_limit_um * 2) * SCALE_CS
    
    # ViewBox Y: Center around the features (CY_CS)
    # We want to show ~10um above and ~10um below
    y_range_um = 12.0
    vb_y = CY_CS - (y_range_um * SCALE_CS) # Top of box
    vb_h = (y_range_um * 2) * SCALE_CS     # Height of box
    
    base_y_math = BOTTOM_LAYER_H
    
    # Substrate (Draw really wide/deep, crop handles visibility)
    _, sub_y = to_svg_cs(-100, 0)
    sub_rect = f'<rect x="-1000" y="{sub_y}" width="4000" height="2000" fill="{C_SUB}" />'
    
    # Black Layer
    _, bl_y = to_svg_cs(-100, BOTTOM_LAYER_H)
    bl_rect = f'<rect x="-1000" y="{bl_y}" width="4000" height="{BOTTOM_LAYER_H*SCALE_CS}" fill="black" />'
    
    # Electrodes
    ws_x, ws_y = to_svg_cs(-WS/2, base_y_math + MTX)
    ws_rect = f'<rect x="{ws_x}" y="{ws_y}" width="{WS*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" />'
    
    # Right WG (Draw full width, ViewBox cuts it)
    rwg_x, rwg_y = to_svg_cs(WS/2 + GAP, base_y_math + MTX)
    rwg_rect = f'<rect x="{rwg_x}" y="{rwg_y}" width="{WG*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" />'
    
    # Left WG
    lwg_x, lwg_y = to_svg_cs(-(WS/2 + GAP + WG), base_y_math + MTX)
    lwg_rect = f'<rect x="{lwg_x}" y="{lwg_y}" width="{WG*SCALE_CS}" height="{MTX*SCALE_CS}" fill="{C_ELEC}" stroke="{C_LINE}" />'
    
    caps_svg = ""
    for center_x in [WS/2 + GAP/2, -WS/2 - GAP/2]:
        cx_svg, cy_svg = to_svg_cs(center_x - CAP_W/2, base_y_math + CAP_HEIGHT)
        caps_svg += f'<rect x="{cx_svg}" y="{cy_svg}" width="{CAP_W*SCALE_CS}" height="{CAP_HEIGHT*SCALE_CS}" fill="{C_CAP}" stroke="{C_LINE}" />'
        rx_svg, ry_svg = to_svg_cs(center_x - RIDGE_W/2, base_y_math + RIDGE_H)
        caps_svg += f'<rect x="{rx_svg}" y="{ry_svg}" width="{RIDGE_W*SCALE_CS}" height="{RIDGE_H*SCALE_CS}" fill="black" />'

    cs_arrows = ""
    dim_y = base_y_math + max(MTX, CAP_HEIGHT) + 1.0
    
    # Font Size is adjusted relative to scale
    CS_FONT = 12 * (SCALE_TOP / SCALE_CS) * 2 # Heuristic to match visual size
    CS_FONT = 1.5 # In SVG units for this zoom level
    
    cs_arrows += svg_arrow(*to_svg_cs(-WS/2, dim_y), *to_svg_cs(WS/2, dim_y), "WS", "top", 1, font_size=CS_FONT)
    cs_arrows += svg_arrow(*to_svg_cs(WS/2, dim_y), *to_svg_cs(WS/2 + GAP, dim_y), "GAP", "top", 1, font_size=CS_FONT)
    
    l_gap_c = -WS/2 - GAP/2
    cap_y = base_y_math + CAP_HEIGHT + 0.5
    cs_arrows += svg_arrow(*to_svg_cs(l_gap_c - CAP_W/2, cap_y), *to_svg_cs(l_gap_c + CAP_W/2, cap_y), "CAP_W", "top", 1, font_size=CS_FONT)
    
    # We define the SVG with fixed width/height matching TOP view (800x600)
    # But the VIEWBOX crops it to the tiny area we care about.
    svg_cross = f"""
    <svg width="{CANVAS_W}" height="{CANVAS_H}" viewBox="{vb_x} {vb_y} {vb_w} {vb_h}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="black" /></marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M9,0 L9,6 L0,3 z" fill="black" /></marker>
        </defs>
        {sub_rect} {bl_rect} {ws_rect} {rwg_rect} {lwg_rect} {caps_svg} {cs_arrows}
    </svg>
    """
    return svg_top, svg_cross

# --- LAYOUT ---
st.subheader("1. Geometry Visualization")
st.caption("Updated automatically.")

svg_t, svg_c = generate_exact_svg(params)

st.markdown(render_svg(svg_t), unsafe_allow_html=True)
st.markdown("**Cross-Section View** (Zoomed to Center)")
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
