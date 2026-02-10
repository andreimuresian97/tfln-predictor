import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="TFLN Geometry Predictor", layout="wide")

st.title("⚡ TFLN Performance Predictor")
st.markdown("Instant inference for **VPI, nm, Z0, and S21** with real-time geometry visualization.")

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
    
    return {
        "WS": ws, "GAP": gap, "MTX": mtx, "CAP_W": cap_w,
        "L1": l1, "L2": l2, "W1": w1, "W2": w2
    }

params = user_input_features()

# Prepare list for the Predictor
geometry_list = [
    params["WS"], params["GAP"], params["MTX"], params["CAP_W"],
    params["L1"], params["L2"], params["W1"], params["W2"]
]

# --- 100% SAFE SVG PLOTTING LOGIC (NO MATPLOTLIB) ---
def generate_svg_plots(p):
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W = p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0
    
    # Scale factors to fit SVG viewbox
    scale = 3.0 
    cx, cy = 400, 250 # Center of Top View
    
    # Colors
    c_elec = "#F5BD02"
    c_sub = "#00FFFF"
    c_line = "black"

    # --- TOP VIEW SVG ---
    # We build the polygon points string for the Right Electrode
    # Coordinates relative to center
    pts = [
        (GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
        (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
        (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
        (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)
    ]
    poly_pts = " ".join([f"{cx + x*scale},{cy - y*scale}" for x, y in pts])
    
    # WS Rect
    ws_x = cx - (GAP/2 + WS)*scale
    ws_y = cy - 100*scale
    ws_w = WS * scale
    ws_h = 200 * scale

    svg_top = f"""
    <svg width="100%" height="400" xmlns="http://www.w3.org/2000/svg">
        <text x="10" y="20" font-family="sans-serif" font-size="14">Top-Down View</text>
        <polygon points="{poly_pts}" fill="{c_elec}" stroke="{c_line}" stroke-width="1" />
        <rect x="{ws_x}" y="{ws_y}" width="{ws_w}" height="{ws_h}" fill="{c_elec}" stroke="{c_line}" stroke-width="1" />
        <text x="{cx}" y="{cy}" font-family="sans-serif" font-size="10" text-anchor="middle">GAP: {GAP}</text>
        <line x1="{cx - GAP/2*scale}" y1="{cy}" x2="{cx + GAP/2*scale}" y2="{cy}" stroke="black" stroke-width="0.5" marker-end="url(#arrow)" />
    </svg>
    """

    # --- CROSS SECTION SVG ---
    cs_scale = 5.0
    cs_cx, cs_cy = 400, 150
    base_y = cs_cy  # Y position of the "floor"
    
    # Substrate
    sub_h = max(MTX, 5) * cs_scale
    
    # Rectangles (x, y, w, h)
    # Note: SVG y grows downwards. so "Up" is negative Y.
    
    # WS Center
    ws_rect = f'<rect x="{cs_cx - (WS/2)*cs_scale}" y="{base_y - MTX*cs_scale}" width="{WS*cs_scale}" height="{MTX*cs_scale}" fill="{c_elec}" stroke="black" />'
    
    # Right WG
    rwg_rect = f'<rect x="{cs_cx + (WS/2 + GAP)*cs_scale}" y="{base_y - MTX*cs_scale}" width="{WG*cs_scale}" height="{MTX*cs_scale}" fill="{c_elec}" stroke="black" />'
    
    # Left WG
    lwg_rect = f'<rect x="{cs_cx - (WS/2 + GAP + WG)*cs_scale}" y="{base_y - MTX*cs_scale}" width="{WG*cs_scale}" height="{MTX*cs_scale}" fill="{c_elec}" stroke="black" />'
    
    # Substrate
    sub_rect = f'<rect x="0" y="{base_y}" width="800" height="{sub_h}" fill="{c_sub}" />'
    
    svg_cross = f"""
    <svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
        <text x="10" y="20" font-family="sans-serif" font-size="14">Cross-Section</text>
        {sub_rect}
        {ws_rect} {rwg_rect} {lwg_rect}
    </svg>
    """
    
    return svg_top, svg_cross

# --- LAYOUT ---

st.subheader("1. Geometry Visualization")
svg_t, svg_c = generate_svg_plots(params)
c1, c2 = st.columns(2)
with c1:
    st.markdown(svg_t, unsafe_allow_html=True)
with c2:
    st.markdown(svg_c, unsafe_allow_html=True)

st.markdown("---")
st.subheader("2. Performance Prediction")

if st.button("Predict Performance", type="primary"):
    try:
        # Import moved INSIDE the button to prevent startup crashes
        from fast_FOMs_predictor import TFLNPredictor
        
        @st.cache_resource
        def load_predictor():
            return TFLNPredictor(model_dir="gp_surrogate_results_199_8var_fixed")
        
        predictor = load_predictor()
        results = predictor.predict(geometry_list)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VPI", f"{results['VPI']['value']:.2f} V")
        col2.metric("Loss (S21)", f"{results['S21']['value']:.2f} dB")
        col3.metric("Impedance (Z0)", f"{results['Z0']['value']:.1f} Ω")
        col4.metric("Index (nm)", f"{results['nm']['value']:.3f}")
        
        data = []
        for fom, res in results.items():
            data.append([fom, f"{res['value']:.4f}", f"[{res['lower_bound']:.4f}, {res['upper_bound']:.4f}]"])
        st.table(pd.DataFrame(data, columns=["FOM", "Value", "95% CI"]))
            
    except Exception as e:
        st.error(f"Error: {e}")
