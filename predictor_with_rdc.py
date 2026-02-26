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
Instant inference for **VPI, nm, Z0, S21, and Rdc/cm**.  
**Note:** Optical predictions are performed at **1330 nm**.
""")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Geometry Parameters")

def user_input_features():
    # Global Device Parameters
    st.sidebar.subheader("Global Device Parameters")
    length_cm = st.sidebar.number_input("Device Length (L) [cm]", value=1.5, min_value=0.1, max_value=10.0, step=0.1, format="%.2f")
    r_load = st.sidebar.number_input("Load Resistance (RL) [Ω]", value=50.0, min_value=10.0, max_value=100.0, step=1.0, format="%.1f")
    
    # Active Region (High precision)
    st.sidebar.subheader("Active Region")
    ws = st.sidebar.number_input("WS (Signal Width) [µm]", value=10.371, step=0.001, format="%.3f")
    gap = st.sidebar.number_input("GAP [µm]", value=4.992, step=0.001, format="%.3f")
    mtx = st.sidebar.number_input("MTX (Metal Thickness) [µm]", value=2.428, step=0.001, format="%.3f")
    cap_w = st.sidebar.number_input("CAP_W (Cap Width) [µm]", value=3.214, step=0.001, format="%.3f")
    
    # T-Structure Dimensions
    st.sidebar.subheader("T-Structure Dimensions")
    l1 = st.sidebar.number_input("L1 (Inner Length) [µm]", value=10.0, step=0.1, format="%.1f")
    l2 = st.sidebar.number_input("L2 (Outer Length) [µm]", value=60.9, step=0.1, format="%.1f")
    w1 = st.sidebar.number_input("W1 (Inner Width) [µm]", value=50.1, step=0.1, format="%.1f")
    w2 = st.sidebar.number_input("W2 (Outer Width) [µm]", value=13.9, step=0.1, format="%.1f")
    
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
    
    return length_cm, r_load, [ws, gap, mtx, cap_w, l1, l2, w1, w2], {
        "WS": ws, "GAP": gap, "MTX": mtx, "CAP_W": cap_w,
        "L1": l1, "L2": l2, "W1": w1, "W2": w2
    }

length_cm, r_load, geometry_list, params = user_input_features()

# --- HELPER: RENDER SVG AS IMAGE ---
def render_svg(svg_string):
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="100%"/>'

# --- PHYSICS ENGINEERING (For RDC) ---
def engineer_rdc_features(geometry):
    ws, _, mtx, _, l1, l2, w1, w2 = geometry
    
    area_center = ws * mtx
    area_tees = (l1 * w1) + (l2 * w2)
    inv_ws = 1.0 / (ws + 1e-9)
    period = w1 + w2 
    fill_factor = w2 / (period + 1e-9) 
    perimeter = 2 * (w1 + w2) + l2 
    
    return np.array([ws, mtx, l1, l2, w1, w2, area_center, area_tees, inv_ws, fill_factor, perimeter]).reshape(1, -1)

# --- LOW MEMORY PREDICTOR ENGINE ---
def predict_sequentially(geometry, L_cm, R_L):
    results = {}
    model_dir = Path("gp_surrogate_results_final")
    
    if not model_dir.exists():
        st.error(f"❌ Folder '{model_dir}' not found. Please ensure the new .pkl files are uploaded.")
        return {}

    try:
        with open(model_dir / "scalers.pkl", 'rb') as f:
            scalers_data = pickle.load(f)
            scaler_X_base = scalers_data['X']['base_input']
            scaler_X_rdc = scalers_data['X']['rdc_input']
            scalers_y = scalers_data['y']
    except Exception as e:
        st.error(f"❌ Error loading scalers: {e}")
        return {}

    X_input_8D = np.array(geometry).reshape(1, -1)
    X_norm_8D = scaler_X_base.transform(X_input_8D)
    
    X_input_11D = engineer_rdc_features(geometry)
    X_norm_11D = scaler_X_rdc.transform(X_input_11D)

    targets = {
        'VPI': 'VPI (duty cycle)', 
        'nm': 'nm', 
        'Z0': 'Z0', 
        'S21': 'S21', 
        'Rdc_cm': 'Rdc/cm'
    }
    
    progress = st.progress(0)
    
    for idx, (safe_name, scaler_key) in enumerate(targets.items()):
        try:
            with open(model_dir / f"gp_model_{safe_name}.pkl", 'rb') as f:
                model = pickle.load(f)
            
            # Route to correct input space
            if safe_name == 'Rdc_cm':
                y_pred_norm, y_std_norm = model.predict(X_norm_11D, return_std=True)
            else:
                y_pred_norm, y_std_norm = model.predict(X_norm_8D, return_std=True)
            
            del model
            gc.collect() 
            
            scaler = scalers_y[scaler_key]
            y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()[0]
            
            if hasattr(scaler, 'scale_'):
                y_std = y_std_norm[0] * scaler.scale_[0]
            elif hasattr(scaler, 'data_range_'): 
                y_std = y_std_norm[0] * (scaler.data_max_[0] - scaler.data_min_[0])
            else:
                y_std = y_std_norm[0]
            
            if safe_name in ["VPI", "Rdc_cm"]:
                real_val = 10 ** y_pred
                real_std = real_val * np.log(10) * y_std
                y_pred = real_val
                y_std = real_std
            
            results[safe_name] = {
                'value': y_pred, 
                'lower_bound': y_pred - 1.96 * y_std, 
                'upper_bound': y_pred + 1.96 * y_std
            }

        except Exception as e:
            st.warning(f"Could not predict {safe_name}: {e}")
        
        progress.progress((idx + 1) / len(targets))
        
    progress.empty()

    # --- POST-PROCESS LAB VPI ---
    if 'VPI' in results and 'Rdc_cm' in results:
        vpi_ideal_vcm = results['VPI']['value']
        vpi_ideal_vcm_low = results['VPI']['lower_bound']
        vpi_ideal_vcm_high = results['VPI']['upper_bound']
        
        rdc_per_cm = results['Rdc_cm']['value']
        rdc_per_cm_low = results['Rdc_cm']['lower_bound']
        rdc_per_cm_high = results['Rdc_cm']['upper_bound']
        
        # Base Prediction
        vpi_ideal_chip = vpi_ideal_vcm / L_cm
        rdc_total = rdc_per_cm * L_cm
        pf = (2.0 * (rdc_total + R_L)) / (rdc_total + (2.0 * R_L))
        vpi_lab = vpi_ideal_chip * pf
        
        # Lower Bound (Best Case)
        vpi_ideal_chip_low = vpi_ideal_vcm_low / L_cm
        rdc_total_low = max(0, rdc_per_cm_low * L_cm)
        pf_low = (2.0 * (rdc_total_low + R_L)) / (rdc_total_low + (2.0 * R_L))
        vpi_lab_low = vpi_ideal_chip_low * pf_low
        
        # Upper Bound (Worst Case)
        vpi_ideal_chip_high = vpi_ideal_vcm_high / L_cm
        rdc_total_high = rdc_per_cm_high * L_cm
        pf_high = (2.0 * (rdc_total_high + R_L)) / (rdc_total_high + (2.0 * R_L))
        vpi_lab_high = vpi_ideal_chip_high * pf_high

        results['Lab_VPI'] = {
            'value': vpi_lab,
            'lower_bound': vpi_lab_low,
            'upper_bound': vpi_lab_high,
            'ideal_chip_vpi': vpi_ideal_chip,
            'total_chip_ohms': rdc_total,
            'pf': pf,
            'pf_lower': pf_low,   # <-- This fixes the KeyError
            'pf_upper': pf_high   # <-- This fixes the KeyError
        }

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
    results = predict_sequentially(geometry_list, length_cm, r_load)
    if results:
        
        # Determine success color mapping
        def color_ci(low, high):
            return f"<span style='color: gray; font-size: 0.9em'>[{low:.4f}, {high:.4f}]</span>"
            
        st.markdown("### Fundamental Physical Surrogate Predictions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ideal VPI·L", f"{results['VPI']['value']:.3f} V·cm")
        c2.metric("Loss (S21)", f"{results['S21']['value']:.2f} dB")
        c3.metric("Impedance (Z0)", f"{results['Z0']['value']:.1f} Ω")
        c4.metric("Index (nm)", f"{results['nm']['value']:.3f}")

        st.markdown(f"**Specific DC Resistance:** {results['Rdc_cm']['value']:.2f} Ω/cm")

        st.markdown("---")
        st.markdown("### Lab Vπ Simulation (Voltage Divider Correction)")
        
        lab = results['Lab_VPI']
        
        st.info(f"**Chip Length:** {length_cm} cm | **Total Resistance:** {lab['total_chip_ohms']:.1f} Ω | **Voltage Penalty:** {lab['pf']:.3f}x")
        
        # Big metric display for the final Vpi
        st.metric(label="Predicted Oscilloscope Vπ", value=f"{lab['value']:.3f} V")
        
        # Data Table with Confidence Intervals
        st.markdown("#### Detailed 95% Confidence Intervals")
        data = [
            ["Ideal VPI (V)", f"{lab['ideal_chip_vpi']:.4f}", f"[{results['VPI']['lower_bound']/length_cm:.4f}, {results['VPI']['upper_bound']/length_cm:.4f}]"],
            ["Rdc/cm (Ω/cm)", f"{results['Rdc_cm']['value']:.4f}", f"[{results['Rdc_cm']['lower_bound']:.4f}, {results['Rdc_cm']['upper_bound']:.4f}]"],
            ["Penalty Factor", f"{lab['pf']:.4f}", f"[{lab['pf_lower']:.4f}, {lab['pf_upper']:.4f}]"],
            ["Lab VPI (V)", f"{lab['value']:.4f}", f"[{lab['lower_bound']:.4f}, {lab['upper_bound']:.4f}]"]
        ]
        st.table(pd.DataFrame(data, columns=["Parameter", "Predicted Value", "95% CI"]))
