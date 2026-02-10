import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.figure import Figure # <--- The safe way to plot
from matplotlib.patches import Rectangle, Polygon

# Import your predictor
# We wrap this in a try-block just in case the file is missing, so the app doesn't crash
try:
    from fast_FOMs_predictor import TFLNPredictor
except ImportError:
    st.error("⚠️ Could not import 'fast_FOMs_predictor.py'. Is the file in the repo?")

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

# --- PLOTTING LOGIC (Object-Oriented / Server Safe) ---
def generate_plots(p):
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W = p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    
    WG_FIXED = 70
    CAP_HEIGHT_FIXED = 1.4
    BOTTOM_LAYER_H = 0.23
    RIDGE_W = 0.8
    RIDGE_H = 0.23
    ELECTRODE_COLOR = '#F5BD02'
    SUBSTRATE_COLOR = '#00FFFF'
    CAP_COLOR = '#00FFFF'
    LINE_COLOR = 'black'
    FONT_SIZE = 8
    
    # Helper to draw arrows on a specific axes 'ax'
    def draw_arrow(ax, x1, y1, x2, y2, text, loc, a_offset, t_offset):
        ax.annotate(
            '', xy=(x1, y1), xytext=(x2, y2),
            arrowprops=dict(arrowstyle='<->', mutation_scale=10, linewidth=0.7, 
                            color=LINE_COLOR, shrinkA=0, shrinkB=0), zorder=20
        )
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ha, va = 'center', 'center'
        off_x, off_y = 0, 0
        
        if loc == 'top': off_y = t_offset; va = 'bottom'
        elif loc == 'bottom': off_y = -t_offset; va = 'top'
        elif loc == 'left': off_x = -t_offset; ha = 'right'
        elif loc == 'right': off_x = t_offset; ha = 'left'
            
        ax.text(mid_x + off_x, mid_y + off_y, text, 
                ha=ha, va=va, fontsize=FONT_SIZE, color=LINE_COLOR, zorder=21)

    # --- FIGURE 1: TOP-DOWN ---
    # We use Figure() directly. This creates a plot in memory, not on screen.
    fig1 = Figure(figsize=(5, 5))
    ax1 = fig1.subplots()
    
    TOP_VIEW_HEIGHT = 200 
    
    # 1. Left Electrode
    ax1.add_patch(Rectangle((-(GAP/2 + WS), -TOP_VIEW_HEIGHT/2), WS, TOP_VIEW_HEIGHT,
                            linewidth=1, edgecolor=LINE_COLOR, facecolor=ELECTRODE_COLOR))

    # 2. Right Electrode
    pts = [
        (GAP/2, -TOP_VIEW_HEIGHT/2), (GAP/2 + WG_FIXED, -TOP_VIEW_HEIGHT/2),
        (GAP/2 + WG_FIXED, TOP_VIEW_HEIGHT/2), (GAP/2, TOP_VIEW_HEIGHT/2),
        (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
        (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
        (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)
    ]
    ax1.add_patch(Polygon(pts, closed=True, linewidth=1, edgecolor=LINE_COLOR, facecolor=ELECTRODE_COLOR))

    # Annotations
    ao, to = 4.0, 2.0
    top_y = TOP_VIEW_HEIGHT/2 + ao
    draw_arrow(ax1, -(GAP/2 + WS), top_y, -GAP/2, top_y, 'WS', 'top', ao, to)
    l2_x = GAP/2 + W1 + W2 + ao
    draw_arrow(ax1, l2_x, -L2/2, l2_x, L2/2, 'L2', 'right', ao, to)

    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Top-Down View", fontsize=10)

    # --- FIGURE 2: CROSS-SECTION ---
    fig2 = Figure(figsize=(5, 2))
    ax2 = fig2.subplots()
    
    base_y = BOTTOM_LAYER_H
    ws_left, ws_right = -WS/2, WS/2
    
    # Substrate
    ax2.add_patch(Rectangle((-1000, -max(MTX, 5)), 2000, max(MTX, 5), fc=SUBSTRATE_COLOR, ec='none'))
    ax2.add_patch(Rectangle((-1000, 0), 2000, BOTTOM_LAYER_H, fc='black', ec='none'))

    # Electrodes
    ax2.add_patch(Rectangle((ws_right + GAP, base_y), WG_FIXED, MTX, fc=ELECTRODE_COLOR, ec=LINE_COLOR))
    ax2.add_patch(Rectangle((ws_left, base_y), WS, MTX, fc=ELECTRODE_COLOR, ec=LINE_COLOR))
    ax2.add_patch(Rectangle((ws_left - GAP - WG_FIXED, base_y), WG_FIXED, MTX, fc=ELECTRODE_COLOR, ec=LINE_COLOR))

    # Caps
    for center_x in [ws_right + GAP/2, ws_left - GAP/2]:
        ax2.add_patch(Rectangle((center_x - CAP_W/2, base_y), CAP_W, CAP_HEIGHT_FIXED, fc=CAP_COLOR, ec=LINE_COLOR))
        ax2.add_patch(Rectangle((center_x - RIDGE_W/2, base_y), RIDGE_W, RIDGE_H, fc='black'))

    # Annotations
    ao, to = 0.5, 0.3
    dim_y = base_y + max(MTX, CAP_HEIGHT_FIXED) + ao
    draw_arrow(ax2, ws_left, dim_y, ws_right, dim_y, 'WS', 'top', ao, to)
    draw_arrow(ax2, ws_right, dim_y, ws_right + GAP, dim_y, 'GAP', 'top', ao, to)
    
    ax2.set_aspect('equal')
    ax2.set_xlim(ws_left - GAP - 10, ws_right + GAP + 10)
    ax2.set_ylim(-2, dim_y + 2)
    ax2.axis('off')
    ax2.set_title("Cross-Section View", fontsize=10)
    
    return fig1, fig2

# --- MAIN LAYOUT ---

st.subheader("1. Geometry Visualization")
st.caption("Updates automatically as you change parameters.")

try:
    fig_top, fig_cross = generate_plots(params)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_top)
    with col2:
        st.pyplot(fig_cross)
except Exception as e:
    st.error(f"Plotting Error: {e}")

st.markdown("---")
st.subheader("2. Performance Prediction")

if st.button("Predict Performance", type="primary"):
    try:
        @st.cache_resource
        def load_predictor():
            # Adjust folder name if necessary!
            return TFLNPredictor(model_dir="gp_surrogate_results_199_8var_fixed")
        
        predictor = load_predictor()
        results = predictor.predict(geometry_list)
        
        # Display Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VPI", f"{results['VPI']['value']:.2f} V")
        c2.metric("Loss (S21)", f"{results['S21']['value']:.2f} dB")
        c3.metric("Impedance (Z0)", f"{results['Z0']['value']:.1f} Ω")
        c4.metric("Index (nm)", f"{results['nm']['value']:.3f}")
        
        # Display Table
        data = []
        for fom, res in results.items():
            val = f"{res['value']:.4f}"
            ci = f"[{res['lower_bound']:.4f}, {res['upper_bound']:.4f}]"
            data.append([fom, val, ci])
            
        st.table(pd.DataFrame(data, columns=["FOM", "Value", "95% CI"]))
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
