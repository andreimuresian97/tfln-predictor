import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="TFLN Sweep Visualizer", layout="wide")

st.title("📈 TFLN Parametric Sweep Visualizer")
st.markdown("""
Instantly visualize the effect of any geometric parameter on the figures of merit using the GP Surrogate Model.  
**Note:** Optical predictions are performed at **1330 nm**.
""")

# --- SIDEBAR: BASELINE INPUTS ---
st.sidebar.header("Baseline Geometry")
st.sidebar.caption("These values are held constant during the sweep.")

def user_input_features():
    st.sidebar.subheader("Global Device")
    length_cm = st.sidebar.number_input("Device Length (L) [cm]", value=1.5, min_value=0.1, max_value=10.0, step=0.1)
    
    st.sidebar.subheader("Active Region")
    ws = st.sidebar.number_input("WS (Signal Width) [µm]", value=10.371, step=0.1)
    gap = st.sidebar.number_input("GAP [µm]", value=4.992, step=0.1)
    mtx = st.sidebar.number_input("MTX (Metal Thickness) [µm]", value=2.428, step=0.1)
    cap_w = st.sidebar.number_input("CAP_W (Cap Width) [µm]", value=3.214, step=0.1)
    
    st.sidebar.subheader("T-Structure Dimensions")
    l1 = st.sidebar.number_input("L1 (Inner Length) [µm]", value=10.0, step=0.1)
    l2 = st.sidebar.number_input("L2 (Outer Length) [µm]", value=60.9, step=0.1)
    w1 = st.sidebar.number_input("W1 (Inner Width) [µm]", value=50.1, step=0.1)
    w2 = st.sidebar.number_input("W2 (Outer Width) [µm]", value=13.9, step=0.1)
    
    return length_cm, {
        "WS": ws, "GAP": gap, "MTX": mtx, "CAP_W": cap_w,
        "L1": l1, "L2": l2, "W1": w1, "W2": w2
    }

length_cm, base_geometry = user_input_features()

# --- MAIN UI: SWEEP CONFIGURATION ---
col1, col2 = st.columns(2)

with col1:
    sweep_param = st.selectbox(
        "1. Select Parameter to Sweep (X-Axis)",
        options=['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2'],
        index=0 # Defaults to WS to show off the Rdc curve!
    )
    
    # Dynamic bounds based on selected parameter
    default_bounds = {
        'WS': (1.0, 30.0), 'GAP': (4.0, 10.0), 'MTX': (1.5, 12.0), 'CAP_W': (1.0, 5.0),
        'L1': (3.0, 60.0), 'L2': (3.0, 190.0), 'W1': (3.0, 60.0), 'W2': (3.0, 60.0)
    }
    min_val, max_val = st.slider(
        f"Sweep Range for {sweep_param} [µm]",
        min_value=1.0, max_value=200.0, 
        value=default_bounds[sweep_param], step=1.0
    )

with col2:
    target_fom = st.selectbox(
        "2. Select Figure of Merit (Y-Axis)",
        options=['Z0', 'nm', 'VPI', 'S21', 'Rdc_cm'],
        index=4 # Defaults to Rdc_cm
    )

# --- PHYSICS ENGINEERING (For RDC) ---
def engineer_rdc_features_batch(geom_matrix):
    """Vectorized physics engineering for the 100-row sweep array"""
    ws = geom_matrix[:, 0]
    mtx = geom_matrix[:, 2]
    l1 = geom_matrix[:, 4]
    l2 = geom_matrix[:, 5]
    w1 = geom_matrix[:, 6]
    w2 = geom_matrix[:, 7]
    
    area_center = ws * mtx
    area_tees = (l1 * w1) + (l2 * w2)
    inv_ws = 1.0 / (ws + 1e-9)
    period = w1 + w2 
    fill_factor = w2 / (period + 1e-9) 
    perimeter = 2 * (w1 + w2) + l2 
    
    return np.column_stack([geom_matrix[:, [0, 2, 4, 5, 6, 7]], area_center, area_tees, inv_ws, fill_factor, perimeter])

# --- LOW MEMORY BATCH PREDICTOR ENGINE ---
def predict_sweep_ram_safe(base_geom, param_to_sweep, bounds, fom, L_cm):
    model_dir = Path("gp_surrogate_results_final")
    if not model_dir.exists():
        st.error(f"❌ Folder '{model_dir}' not found. Please upload the new .pkl files.")
        return None, None, None, None
        
    # 1. Load Scalers
    try:
        with open(model_dir / "scalers.pkl", 'rb') as f:
            scalers_data = pickle.load(f)
            scaler_X_base = scalers_data['X']['base_input']
            scaler_X_rdc = scalers_data['X']['rdc_input']
            scalers_y = scalers_data['y']
    except Exception as e:
        st.error(f"❌ Error loading scalers: {e}")
        return None, None, None, None

    # 2. Build the Batch Array (100 rows, 8 columns)
    param_order = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2']
    x_values = np.linspace(bounds[0], bounds[1], 100)
    geom_matrix_list = []
    
    for val in x_values:
        current_geom = base_geom.copy()
        current_geom[param_to_sweep] = val
        
        # Physical Constraint: CAP_W must be strictly less than GAP
        if current_geom['CAP_W'] >= current_geom['GAP']:
            current_geom['CAP_W'] = current_geom['GAP'] - 0.1
            
        geom_matrix_list.append([current_geom[p] for p in param_order])
        
    X_input_8D = np.array(geom_matrix_list)

    # Route Input Space Based on Model
    if fom == 'Rdc_cm':
        X_input_11D = engineer_rdc_features_batch(X_input_8D)
        X_norm = scaler_X_rdc.transform(X_input_11D)
    else:
        X_norm = scaler_X_base.transform(X_input_8D)

    # 3. Load ONLY the requested model to save RAM
    scaler_keys = {'VPI': 'VPI (duty cycle)', 'nm': 'nm', 'Z0': 'Z0', 'S21': 'S21', 'Rdc_cm': 'Rdc/cm'}
    scaler_key = scaler_keys[fom]
    
    try:
        with open(model_dir / f"gp_model_{fom}.pkl", 'rb') as f:
            model = pickle.load(f)
            
        # Predict the entire batch at once
        y_pred_norm, y_std_norm = model.predict(X_norm, return_std=True)
        
        # Instantly delete the model from RAM
        del model
        gc.collect()
        
    except Exception as e:
        st.error(f"❌ Could not predict {fom}: {e}")
        return None, None, None, None

    # 4. Inverse Transform Logic
    scaler = scalers_y[scaler_key]
    y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
    
    if hasattr(scaler, 'scale_'):
        y_std = y_std_norm * scaler.scale_[0]
    elif hasattr(scaler, 'data_range_'): 
        y_std = y_std_norm * (scaler.data_max_[0] - scaler.data_min_[0])
    else:
        y_std = y_std_norm

    # Special handling for Log10 Outputs (VPI and Rdc)
    if fom in ["VPI", "Rdc_cm"]:
        real_val = 10 ** y_pred
        real_std = real_val * np.log(10) * y_std
        
        if fom == "VPI":
            y_pred = real_val / L_cm
            y_std = real_std / L_cm
        else:
            y_pred = real_val
            y_std = real_std

    y_lower = y_pred - 1.96 * y_std
    y_upper = y_pred + 1.96 * y_std

    return x_values, y_pred, y_lower, y_upper


# --- PLOT GENERATION ---
st.markdown("---")
if st.button("🚀 Generate Plot", type="primary", use_container_width=True):
    with st.spinner(f"Querying GP Surrogate for {target_fom} vs {sweep_param}..."):
        
        x_vals, y_vals, y_low, y_up = predict_sweep_ram_safe(
            base_geometry, sweep_param, (min_val, max_val), target_fom, length_cm
        )
        
        if x_vals is not None:
            # Setup Plotting
            unit_map = {
                'Z0': r'Characteristic Impedance $Z_0$ ($\Omega$)',
                'nm': r'Microwave Index $n_m$',
                'VPI': r'$V_\pi \cdot L$ (V$\cdot$cm)' if target_fom == "VPI" and length_cm == 1.0 else f'$V_\\pi$ (V) for L={length_cm}cm',
                'S21': r'RF Attenuation $S_{21}$ (dB/cm)',
                'Rdc_cm': r'DC Resistance $R_{DC}$ ($\Omega$/cm)'
            }
            y_label = unit_map.get(target_fom, target_fom)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot line and confidence intervals
            ax.plot(x_vals, y_vals, color='#1f77b4', linewidth=2.5, label=f'Predicted {target_fom}')
            ax.fill_between(x_vals, y_low, y_up, color='#1f77b4', alpha=0.2, label='95% Confidence Interval')
            
            ax.set_title(f"{target_fom} vs {sweep_param}", fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel(f"{sweep_param} [µm]", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')
            
            # Render Plot in Streamlit
            st.pyplot(fig)
            
            # Show a small warning if the CAP_W constraint was triggered during the sweep
            if sweep_param == 'GAP' and min_val <= base_geometry['CAP_W']:
                st.warning("⚠️ **Note:** At narrow GAP dimensions, `CAP_W` was dynamically reduced to fit inside the gap and prevent unphysical predictions.")
