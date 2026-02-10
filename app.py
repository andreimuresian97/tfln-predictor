import streamlit as st
import pandas as pd
from fast_FOMs_predictor import TFLNPredictor

# Page Config
st.set_page_config(page_title="TFLN Geometry Predictor", layout="centered")

st.title("⚡ TFLN Performance Predictor")
st.markdown("Instant inference for **VPI, nm, Z0, and S21** based on device geometry.")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Geometry Parameters")

def user_input_features():
    # Note: I have set default values based on your example geometry
    ws = st.sidebar.number_input("WS (Signal Width) [µm]", value=22.936, format="%.3f")
    gap = st.sidebar.number_input("GAP (Gap Width) [µm]", value=10.311, format="%.3f")
    mtx = st.sidebar.number_input("MTX (Electrodes thickness) [µm]", value=8.07, format="%.3f")
    cap_w = st.sidebar.number_input("CAP_W (SiO2 cap thickness) [µm]", value=1.65, format="%.3f")
    l1 = st.sidebar.number_input("L1 (Slow Wave) [µm]", value=8.0, format="%.1f")
    l2 = st.sidebar.number_input("L2 (Slow Wave) [µm]", value=86.0, format="%.1f")
    w1 = st.sidebar.number_input("W1 (Slow Wave) [µm]", value=5.0, format="%.1f")
    w2 = st.sidebar.number_input("W2 (Slow Wave) [µm]", value=11.0, format="%.1f")
    
    return [ws, gap, mtx, cap_w, l1, l2, w1, w2]

geometry = user_input_features()

# --- MAIN PANEL: PREDICTION ---
st.subheader("Model Output")

# Button to trigger prediction
if st.button("Predict Performance", type="primary"):
    try:
        # Initialize predictor (caches the model loading so it's fast)
        @st.cache_resource
        def load_predictor():
            return TFLNPredictor(model_dir="gp_surrogate_results_199_8var_fixed")
        
        predictor = load_predictor()
        
        # Get predictions
        results = predictor.predict(geometry)
        
        # Format results for display
        data = []
        for fom, res in results.items():
            # Formatting Value +/- Uncertainty
            val_str = f"{res['value']:.4f}"
            range_str = f"[{res['lower_bound']:.4f}, {res['upper_bound']:.4f}]"
            data.append([fom, val_str, range_str])
            
        df = pd.DataFrame(data, columns=["Figure of Merit", "Predicted Value", "95% Confidence Interval"])
        
        # Display as a clean table
        st.table(df)
        
        # Visual Metric Cards (Optional Polish)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VPI", f"{results['VPI']['value']:.2f} V")
        col2.metric("Loss (S21)", f"{results['S21']['value']:.2f} dB")
        col3.metric("Impedance (Z0)", f"{results['Z0']['value']:.1f} Ω")
        col4.metric("Index (nm)", f"{results['nm']['value']:.3f}")

    except FileNotFoundError:
        st.error("⚠️ Model folder not found! Ensure 'gp_surrogate_results_199_8var_fixed' is in this directory.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Powered by GP Surrogate Model • TFLN Predictor v2")