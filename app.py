import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="TFLN Geometry Predictor", layout="centered")

st.title("⚡ TFLN Performance Predictor")
st.markdown("Instant inference for **VPI, nm, Z0, and S21**.")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Geometry Parameters")

def user_input_features():
    st.sidebar.subheader("Active Region")
    ws = st.sidebar.number_input("WS (Signal Width) [µm]", value=22.936, format="%.3f")
    gap = st.sidebar.number_input("GAP [µm]", value=10.311, format="%.3f")
    mtx = st.sidebar.number_input("MTX (Metal Thickness) [µm]", value=8.07, format="%.3f")
    cap_w = st.sidebar.number_input("CAP_W (Cap Width) [µm]", value=1.65, format="%.3f")
    
    st.sidebar.subheader("T-Structure Dimensions")
    l1 = st.sidebar.number_input("L1 (Inner Length) [µm]", value=8.0, format="%.1f")
    l2 = st.sidebar.number_input("L2 (Outer Length) [µm]", value=86.0, format="%.1f")
    w1 = st.sidebar.number_input("W1 (Inner Width) [µm]", value=5.0, format="%.1f")
    w2 = st.sidebar.number_input("W2 (Outer Width) [µm]", value=11.0, format="%.1f")
    
    return [ws, gap, mtx, cap_w, l1, l2, w1, w2], {
        "WS": ws, "GAP": gap, "MTX": mtx, "CAP_W": cap_w,
        "L1": l1, "L2": l2, "W1": w1, "W2": w2
    }

geometry_list, params = user_input_features()

# --- LOW MEMORY PREDICTOR ENGINE ---
def predict_sequentially(geometry):
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
    
    # Progress bar to give feedback during sequential load
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
                y_pred = real_val
                y_std = real_std
            
            results[safe_name] = {'value': y_pred, 'lower_bound': y_pred - 1.96 * y_std, 'upper_bound': y_pred + 1.96 * y_std}

        except Exception as e:
            st.warning(f"Could not predict {safe_name}: {e}")
        
        idx += 1
        progress.progress(idx / 4)
        
    progress.empty()
    return results

# --- EXACT REPLICA SVG PLOTTING ---
def generate_exact_svg(p):
    # Unpack
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W = p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0
    
    # Fixed Constants from your script
    BOTTOM_LAYER_H = 0.23
    RIDGE_W = 0.8
    RIDGE_H = 0.23
    CAP_HEIGHT = 1.4
    
    # Colors
    C_ELEC = '#F5BD02' # Gold
    C_SUB = '#00FFFF'  # Cyan
    C_CAP = '#00FFFF'  # Cyan
    C_LINE = 'black'
    
    # SVG Helpers
    def svg_arrow(x1, y1, x2, y2, text, text_loc="top", offset=10):
        """Draws a double-headed arrow with text label."""
        # Arrow line
        line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1" marker-start="url(#arrow_start)" marker-end="url(#arrow_end)" />'
        
        # Text Position Calculation
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        tx, ty = mx, my
        anchor = "middle"
        dominant = "middle"
        
        if text_loc == "top": ty -= offset; dominant = "auto"
        elif text_loc == "bottom": ty += offset; dominant = "hanging"
        elif text_loc == "left": tx -= offset; anchor = "end"
        elif text_loc == "right": tx += offset; anchor = "start"
            
        txt = f'<text x="{tx}" y="{ty}" fill="{C_LINE}" font-family="sans-serif" font-size="12" text-anchor="{anchor}" dominant-baseline="{dominant}">{text}</text>'
        return line + txt

    # --- FIGURE 1: TOP-DOWN VIEW ---
    scale = 3.0
    cx, cy = 400, 300 # Center
    
    # Coordinates Conversion (Math -> SVG)
    def to_svg(x, y): return cx + x*scale, cy - y*scale

    # Polygon Points
    pts = [
        (GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
        (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
        (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
        (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)
    ]
    poly_str = " ".join([f"{to_svg(x,y)[0]},{to_svg(x,y)[1]}" for x, y in pts])
    
    # WS Rect
    ws_x1, ws_y1 = to_svg(-(GAP/2 + WS), 100)
    ws_w = WS * scale
    ws_h = 200 * scale # Fixed view height
    
    # Arrows Logic
    arrows_svg = ""
    ao = 15 # Arrow Offset in SVG pixels
    
    # Top Arrows
    top_y_math = 100 + 4
    arrows_svg += svg_arrow(*to_svg(-(GAP/2 + WS), top_y_math), *to_svg(-GAP/2, top_y_math), "WS", "top", 10)
    arrows_svg += svg_arrow(*to_svg(GAP/2, top_y_math), *to_svg(GAP/2 + WG, top_y_math), "WG", "top", 10)
    
    # Bottom Arrows
    bot_y_math = -100 - 4
    arrows_svg += svg_arrow(*to_svg(-GAP/2, bot_y_math), *to_svg(GAP/2, bot_y_math), "GAP", "bottom", 10)
    
    # Side Arrows (L1, L2)
    l1_x_math = GAP/2 - 4
    arrows_svg += svg_arrow(*to_svg(l1_x_math, -L1/2), *to_svg(l1_x_math, L1/2), "L1", "left", 10)
    
    l2_x_math = GAP/2 + W1 + W2 + 4
    arrows_svg += svg_arrow(*to_svg(l2_x_math, -L2/2), *to_svg(l2_x_math, L2/2), "L2", "right", 10)

    # W1, W2 Arrows
    w1_y = -L1/2 - 4
    arrows_svg += svg_arrow(*to_svg(GAP/2, w1_y), *to_svg(GAP/2 + W1, w1_y), "W1", "bottom", 10)
    
    w2_y = L2/2 + 4
    arrows_svg += svg_arrow(*to_svg(GAP/2 + W1, w2_y), *to_svg(GAP/2 + W1 + W2, w2_y), "W2", "top", 10)

    svg_top = f"""
    <svg width="100%" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="black" />
            </marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M9,0 L9,6 L0,3 z" fill="black" />
            </marker>
        </defs>
        <text x="400" y="30" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">Top-Down View</text>
        
        <rect x="{ws_x1}" y="{ws_y1}" width="{ws_w}" height="{ws_h}" fill="{C_ELEC}" stroke="{C_LINE}" />
        <polygon points="{poly_str}" fill="{C_ELEC}" stroke="{C_LINE}" />
        
        {arrows_svg}
    </svg>
    """

    # --- FIGURE 2: CROSS SECTION ---
    cs_scale = 8.0 # Zoomed in
    cs_cx, cs_cy = 400, 250
    
    # Base Y (SVG coordinates)
    # Math y=0 is at cs_cy.
    # Math y=BOTTOM_LAYER_H is higher up (smaller svg y).
    
    def cs_to_svg(x, y): return cs_cx + x*cs_scale, cs_cy - y*cs_scale

    base_y_math = BOTTOM_LAYER_H
    
    # 1. Substrate (Cyan, below y=0)
    sub_depth = max(MTX, 5)
    sub_x, sub_y = cs_to_svg(-100, 0) # Top-Left of substrate
    sub_rect = f'<rect x="0" y="{sub_y}" width="800" height="{sub_depth*cs_scale}" fill="{C_SUB}" />'
    
    # 2. Bottom Black Layer (y=0 to y=0.23)
    bl_x, bl_y = cs_to_svg(-100, BOTTOM_LAYER_H)
    bl_rect = f'<rect x="0" y="{bl_y}" width="800" height="{BOTTOM_LAYER_H*cs_scale}" fill="black" />'
    
    # 3. Electrodes (Gold, sit on base_y)
    # WS
    ws_x, ws_y = cs_to_svg(-WS/2, base_y_math + MTX)
    ws_rect = f'<rect x="{ws_x}" y="{ws_y}" width="{WS*cs_scale}" height="{MTX*cs_scale}" fill="{C_ELEC}" stroke="{C_LINE}" />'
    
    # Right WG
    rwg_x, rwg_y = cs_to_svg(WS/2 + GAP, base_y_math + MTX)
    rwg_rect = f'<rect x="{rwg_x}" y="{rwg_y}" width="{WG*cs_scale}" height="{MTX*cs_scale}" fill="{C_ELEC}" stroke="{C_LINE}" />'
    
    # Left WG
    lwg_x, lwg_y = cs_to_svg(-(WS/2 + GAP + WG), base_y_math + MTX)
    lwg_rect = f'<rect x="{lwg_x}" y="{lwg_y}" width="{WG*cs_scale}" height="{MTX*cs_scale}" fill="{C_ELEC}" stroke="{C_LINE}" />'
    
    # 4. Caps (Cyan) & Ridges (Black)
    caps_svg = ""
    for center_x in [WS/2 + GAP/2, -WS/2 - GAP/2]:
        # Cap
        cx_svg, cy_svg = cs_to_svg(center_x - CAP_W/2, base_y_math + CAP_HEIGHT)
        caps_svg += f'<rect x="{cx_svg}" y="{cy_svg}" width="{CAP_W*cs_scale}" height="{CAP_HEIGHT*cs_scale}" fill="{C_CAP}" stroke="{C_LINE}" />'
        
        # Ridge (Inside Cap, at bottom)
        rx_svg, ry_svg = cs_to_svg(center_x - RIDGE_W/2, base_y_math + RIDGE_H)
        caps_svg += f'<rect x="{rx_svg}" y="{ry_svg}" width="{RIDGE_W*cs_scale}" height="{RIDGE_H*cs_scale}" fill="black" />'

    # 5. Arrows
    cs_arrows = ""
    dim_y = base_y_math + max(MTX, CAP_HEIGHT) + 2.0
    
    # WS Arrow
    cs_arrows += svg_arrow(*cs_to_svg(-WS/2, dim_y), *cs_to_svg(WS/2, dim_y), "WS", "top", 10)
    # GAP Arrow
    cs_arrows += svg_arrow(*cs_to_svg(WS/2, dim_y), *cs_to_svg(WS/2 + GAP, dim_y), "GAP", "top", 10)
    # CAP_W Arrow (on Left Gap)
    l_gap_c = -WS/2 - GAP/2
    cap_y = base_y_math + CAP_HEIGHT + 1.0
    cs_arrows += svg_arrow(*cs_to_svg(l_gap_c - CAP_W/2, cap_y), *cs_to_svg(l_gap_c + CAP_W/2, cap_y), "CAP_W", "top", 10)
    
    svg_cross = f"""
    <svg width="100%" height="400" viewBox="200 100 400 300" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrow_end" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="black" />
            </marker>
            <marker id="arrow_start" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M9,0 L9,6 L0,3 z" fill="black" />
            </marker>
        </defs>
        <text x="400" y="120" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">Cross-Section View</text>
        
        {sub_rect}
        {bl_rect}
        {ws_rect} {rwg_rect} {lwg_rect}
        {caps_svg}
        {cs_arrows}
    </svg>
    """
    
    return svg_top, svg_cross

# --- LAYOUT ---
st.subheader("1. Geometry Visualization")
st.caption("Updated automatically.")

svg_t, svg_c = generate_exact_svg(params)

# VERTICAL STACKING (One on top of the other)
st.markdown(svg_t, unsafe_allow_html=True)
st.markdown("---")
st.markdown(svg_c, unsafe_allow_html=True)

st.markdown("---")
st.subheader("2. Performance Prediction")

if st.button("Predict Performance", type="primary"):
    results = predict_sequentially(geometry_list)
    
    if results:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VPI", f"{results['VPI']['value']:.2f} V")
        col2.metric("Loss (S21)", f"{results['S21']['value']:.2f} dB")
        col3.metric("Impedance (Z0)", f"{results['Z0']['value']:.1f} Ω")
        col4.metric("Index (nm)", f"{results['nm']['value']:.3f}")
        
        data = [[k, f"{v['value']:.4f}", f"[{v['lower_bound']:.4f}, {v['upper_bound']:.4f}]"] for k, v in results.items()]
        st.table(pd.DataFrame(data, columns=["FOM", "Value", "95% CI"]))
