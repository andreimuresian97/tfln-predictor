import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gc  # Garbage Collector
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="TFLN Geometry Predictor", layout="wide")

st.title("⚡ TFLN Performance Predictor (Low RAM Mode)")
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
    """
    Loads models one by one, predicts, and clears memory immediately.
    """
    results = {}
    model_dir = Path("gp_surrogate_results_199_8var_fixed")
    
    if not model_dir.exists():
        st.error(f"❌ Folder '{model_dir}' not found. Check GitHub repo.")
        return {}

    # 1. Load Scalers (Small file, safe to keep)
    try:
        with open(model_dir / "scalers.pkl", 'rb') as f:
            scalers_data = pickle.load(f)
            scaler_X = scalers_data['X']['input']
            scalers_y = scalers_data['y']
    except Exception as e:
        st.error(f"❌ Error loading scalers: {e}")
        return {}

    # Normalize Input
    X_input = np.array(geometry).reshape(1, -1)
    X_norm = scaler_X.transform(X_input)

    # 2. Sequential Model Loading
    # Map of "Safe Name" (file) -> "Scaler Key" (original dict key)
    targets = {
        'VPI': 'VPI (duty cycle)',
        'nm': 'nm',
        'Z0': 'Z0',
        'S21': 'S21'
    }

    progress_bar = st.progress(0)
    step = 0

    for safe_name, scaler_key in targets.items():
        model_path = model_dir / f"gp_model_{safe_name}.pkl"
        
        try:
            # A. Load SINGLE Model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # B. Predict
            y_pred_norm, y_std_norm = model.predict(X_norm, return_std=True)
            
            # C. DELETE MODEL FROM RAM IMMEDIATELY
            del model
            gc.collect() # Force memory cleanup
            
            # D. Process Results (Inverse Transform)
            scaler = scalers_y[scaler_key]
            y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()[0]
            
            # Uncertainty Scaling
            if hasattr(scaler, 'scale_'):
                y_std = y_std_norm[0] * scaler.scale_[0]
            else:
                y_std = y_std_norm[0]
            
            # Special Log10 handling for VPI
            if safe_name == "VPI":
                real_val = 10 ** y_pred
                real_std = real_val * np.log(10) * y_std
                y_pred = real_val
                y_std = real_std
            
            # Store Result
            results[safe_name] = {
                'value': y_pred,
                'lower_bound': y_pred - 1.96 * y_std,
                'upper_bound': y_pred + 1.96 * y_std
            }

        except Exception as e:
            st.warning(f"Could not predict {safe_name}: {e}")
        
        step += 1
        progress_bar.progress(step / 4)

    progress_bar.empty()
    return results

# --- SVG PLOTTING (Zero Memory) ---
def generate_svg_plots(p):
    W1, W2, L1, L2 = p["W1"], p["W2"], p["L1"], p["L2"]
    WS, GAP, MTX, CAP_W = p["WS"], p["GAP"], p["MTX"], p["CAP_W"]
    WG = 70.0; scale = 3.0; cx, cy = 400, 250
    c_elec, c_line = "#F5BD02", "black"

    # Points for Top View
    pts = [
        (GAP/2, -100), (GAP/2 + WG, -100), (GAP/2 + WG, 100), (GAP/2, 100),
        (GAP/2, L1/2), (GAP/2 + W1, L1/2), (GAP/2 + W1, L2/2),
        (GAP/2 + W1 + W2, L2/2), (GAP/2 + W1 + W2, -L2/2),
        (GAP/2 + W1, -L2/2), (GAP/2 + W1, -L1/2), (GAP/2, -L1/2)
    ]
    poly_pts = " ".join([f"{cx + x*scale},{cy - y*scale}" for x, y in pts])
    ws_x, ws_y = cx - (GAP/2 + WS)*scale, cy - 100*scale
    
    svg_top = f"""<svg width="100%" height="400" xmlns="http://www.w3.org/2000/svg">
        <text x="10" y="20" font-family="sans-serif" font-size="14">Top-Down View</text>
        <polygon points="{poly_pts}" fill="{c_elec}" stroke="{c_line}" stroke-width="1" />
        <rect x="{ws_x}" y="{ws_y}" width="{WS*scale}" height="{200*scale}" fill="{c_elec}" stroke="{c_line}" />
        <text x="{cx}" y="{cy}" text-anchor="middle" font-family="sans-serif">GAP: {GAP}</text>
    </svg>"""

    cs_cx, base_y, cs_scale = 400, 150, 5.0
    svg_cross = f"""<svg width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
        <text x="10" y="20" font-family="sans-serif" font-size="14">Cross-Section</text>
        <rect x="0" y="{base_y}" width="800" height="{max(MTX, 5)*cs_scale}" fill="#00FFFF" />
        <rect x="{cs_cx - (WS/2)*cs_scale}" y="{base_y - MTX*cs_scale}" width="{WS*cs_scale}" height="{MTX*cs_scale}" fill="{c_elec}" stroke="black" />
        <rect x="{cs_cx + (WS/2 + GAP)*cs_scale}" y="{base_y - MTX*cs_scale}" width="{WG*cs_scale}" height="{MTX*cs_scale}" fill="{c_elec}" stroke="black" />
        <rect x="{cs_cx - (WS/2 + GAP + WG)*cs_scale}" y="{base_y - MTX*cs_scale}" width="{WG*cs_scale}" height="{MTX*cs_scale}" fill="{c_elec}" stroke="black" />
    </svg>"""
    return svg_top, svg_cross

# --- LAYOUT ---
st.subheader("1. Geometry Visualization")
svg_t, svg_c = generate_svg_plots(params)
c1, c2 = st.columns(2)
c1.markdown(svg_t, unsafe_allow_html=True)
c2.markdown(svg_c, unsafe_allow_html=True)

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
