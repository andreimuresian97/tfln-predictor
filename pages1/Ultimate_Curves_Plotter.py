import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

st.title("📈 TFLN Parametric Sweep Visualizer")
st.markdown("""
Instantly visualize the effect of any geometric parameter on the figures of merit using the 10-DOF GP Surrogate + Physics Cascade.  
**Note:** Optical predictions are performed at **1330 nm**.
""")

# --- SIDEBAR: BASELINE INPUTS ---
st.sidebar.header("Baseline Geometry")
st.sidebar.caption("These values are held constant during the sweep.")

def user_input_features():
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
    
    st.sidebar.subheader("Line Configuration")
    rt = st.sidebar.number_input("Rt (Termination) [Ω]", value=50.0, step=1.0)
    
    return {
        "WS": ws, "GAP": gap, "MTX": mtx, "CAP_W": cap_w,
        "L1": l1, "L2": l2, "W1": w1, "W2": w2, "Rt": rt
    }

base_geometry = user_input_features()
# Fixed internal length baseline (10 mm) since it is removed from sidebar
BASE_L_DEV = 10.0 

# --- MAIN UI: SWEEP CONFIGURATION ---
col1, col2 = st.columns(2)

with col1:
    fom_options = [
        'EO Bandwidth [GHz]',
        'Alpha 60GHz [dB/cm]',
        'Microwave Index (nm)',
        'Impedance (Zc) [Ω]',
        'VPI*L [V·cm]'
    ]
    target_fom = st.selectbox("1. Select Figure of Merit (Y-Axis)", options=fom_options, index=0)

with col2:
    if target_fom in ['Microwave Index (nm)', 'Impedance (Zc) [Ω]']:
        sweep_options = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2']
    elif target_fom in ['VPI*L [V·cm]', 'Alpha 60GHz [dB/cm]']:
        sweep_options = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2', 'L_dev']
    else:
        sweep_options = ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2', 'L_dev', 'Rt']

    sweep_param = st.selectbox("2. Select Parameter to Sweep (X-Axis)", options=sweep_options, index=0)

# Dynamic bounds based on selected parameter
default_bounds = {
    'WS': (10.0, 60.0), 'GAP': (4.0, 15.0), 'MTX': (1.5, 12.0), 'CAP_W': (1.5, 14.0),
    'L1': (4.0, 60.0), 'L2': (4.0, 180.0), 'W1': (4.0, 60.0), 'W2': (4.0, 60.0),
    'L_dev': (4.0, 20.0), 'Rt': (34.0, 60.0)
}

unit_param = "mm" if sweep_param == 'L_dev' else "Ω" if sweep_param == 'Rt' else "µm"

min_val, max_val = st.slider(
    f"3. Sweep Range for {sweep_param} [{unit_param}]",
    min_value=0.0, max_value=200.0, 
    value=default_bounds[sweep_param], step=1.0 if unit_param != 'mm' else 0.1
)

# --- PHYSICS ENGINE ---
def fit_alpha(f, a, b, c): return a * np.sqrt(f) + b * f + c

def calc_eo(f_GHz, alpha, nm, Zc, L_m, ng=2.27, Zs=50.0, Rt=42.0):
    c0 = 299792458.0; w = 2 * np.pi * f_GHz * 1e9
    b_opt, b_rf = ng * w / c0, nm * w / c0
    a_Np = alpha * 100.0 / 8.686
    gamma = a_Np + 1j * b_rf

    tanh_gL = np.tanh(gamma * L_m)
    zin = Zc * ((Rt + Zc * tanh_gL) / (Zc + Rt * tanh_gL))
    M = (Zs + zin) * (Rt + Zc)
    N = (Zs + zin) * (Rt - Zc)
    p1 = zin / (M * np.exp(gamma * L_m) + N * np.exp(-gamma * L_m))

    up = np.where(L_m * (a_Np + 1j * (b_rf - b_opt)) == 0, 1e-12, L_m * (a_Np + 1j * (b_rf - b_opt)))
    un = np.where(L_m * (-a_Np + 1j * (-b_rf - b_opt)) == 0, 1e-12, L_m * (-a_Np + 1j * (-b_rf - b_opt)))
    p2 = (Rt + Zc) * ((1 - np.exp(up)) / up) + (Rt - Zc) * ((1 - np.exp(un)) / un)

    s21_abs = p1 * p2
    idx_1G = np.argmin(np.abs(f_GHz - 1.0))
    return 20 * np.log10(np.abs(s21_abs) / np.abs(s21_abs[idx_1G]))

# --- LOW MEMORY BATCH PREDICTOR ENGINE ---
def predict_sweep_ram_safe(base_geom, param_to_sweep, bounds, fom):
    model_dir = Path("gp_surrogate_results_ultimate")
    if not model_dir.exists():
        st.error(f"❌ Folder '{model_dir}' not found.")
        return None, None, None, None
        
    x_values = np.linspace(bounds[0], bounds[1], 100)
    
    # 1. Build Base Matrices
    X8_mm = np.zeros((100, 8))
    X8_mm[:, 0] = base_geom['CAP_W'] / 1e3
    X8_mm[:, 1] = base_geom['GAP'] / 1e3
    X8_mm[:, 2] = base_geom['L1'] / 1e3
    X8_mm[:, 3] = base_geom['L2'] / 1e3
    X8_mm[:, 4] = base_geom['MTX'] / 1e3
    X8_mm[:, 5] = base_geom['W1'] / 1e3
    X8_mm[:, 6] = base_geom['W2'] / 1e3
    X8_mm[:, 7] = base_geom['WS'] / 1e3
    
    L_array = np.full(100, BASE_L_DEV)
    Rt_array = np.full(100, base_geom['Rt'])
    
    # 2. Apply Sweep Vector
    if param_to_sweep == 'CAP_W': X8_mm[:, 0] = x_values / 1e3
    elif param_to_sweep == 'GAP': X8_mm[:, 1] = x_values / 1e3
    elif param_to_sweep == 'L1': X8_mm[:, 2] = x_values / 1e3
    elif param_to_sweep == 'L2': X8_mm[:, 3] = x_values / 1e3
    elif param_to_sweep == 'MTX': X8_mm[:, 4] = x_values / 1e3
    elif param_to_sweep == 'W1': X8_mm[:, 5] = x_values / 1e3
    elif param_to_sweep == 'W2': X8_mm[:, 6] = x_values / 1e3
    elif param_to_sweep == 'WS': X8_mm[:, 7] = x_values / 1e3
    elif param_to_sweep == 'L_dev': L_array = x_values
    elif param_to_sweep == 'Rt': Rt_array = x_values

    # Physical Constraint Enforcement
    mask = X8_mm[:, 0] >= X8_mm[:, 1]
    if np.any(mask): X8_mm[mask, 0] = X8_mm[mask, 1] - 0.0001
        
    BP_mm = (X8_mm[:,2] + X8_mm[:,5] + X8_mm[:,3] + X8_mm[:,6]) * (X8_mm[:,7] / X8_mm[:,1])
    X9_mm = np.column_stack([X8_mm, BP_mm])

    # 3. Model Inference Paths
    try:
        if fom == 'VPI*L [V·cm]':
            with open(model_dir/"gp_vpi_surrogate/scalers_VPI.pkl", 'rb') as f: v_s = pickle.load(f)
            with open(model_dir/"gp_vpi_surrogate/gp_model_VPI.pkl", 'rb') as f: v_m = pickle.load(f)
            vn, vs = v_m.predict(v_s['scaler_X'].transform(X8_mm), return_std=True)
            v_base = 10 ** v_s['scaler_y'].inverse_transform(vn.reshape(-1,1)).ravel()
            v_std = v_base * np.log(10) * vs * v_s['scaler_y'].scale_[0]
            del v_m; gc.collect()
            return x_values, v_base, v_base - 1.96*v_std, v_base + 1.96*v_std

        if fom in ['Microwave Index (nm)', 'Impedance (Zc) [Ω]']:
            with open(model_dir/"gp_nm_zc_surrogate/scalers_nm_zc.pkl", 'rb') as f: nz_s = pickle.load(f)
            X_nz = nz_s['scaler_X'].transform(X9_mm)
            if fom == 'Microwave Index (nm)':
                with open(model_dir/"gp_nm_zc_surrogate/gp_model_nm_60.pkl", 'rb') as f: n_m = pickle.load(f)
                nn, ns = n_m.predict(X_nz, return_std=True)
                y_pred = nz_s['scalers_y']['nm_60'].inverse_transform(nn.reshape(-1,1)).ravel()
                y_std = ns * nz_s['scalers_y']['nm_60'].scale_[0]
                del n_m; gc.collect()
            else:
                with open(model_dir/"gp_nm_zc_surrogate/gp_model_Zc_60.pkl", 'rb') as f: z_m = pickle.load(f)
                zn, zs = z_m.predict(X_nz, return_std=True)
                y_pred = nz_s['scalers_y']['Zc_60'].inverse_transform(zn.reshape(-1,1)).ravel()
                y_std = zs * nz_s['scalers_y']['Zc_60'].scale_[0]
                del z_m; gc.collect()
            return x_values, y_pred, y_pred-1.96*y_std, y_pred+1.96*y_std

        if fom == 'Alpha 60GHz [dB/cm]':
            with open(model_dir/"gp_alpha_anchors/scaler_anchors.pkl", 'rb') as f: a_s = pickle.load(f)['scaler_X']
            with open(model_dir/"gp_alpha_anchors/gp_alpha_anchors_suite.pkl", 'rb') as f: a_m = pickle.load(f)
            Xa = a_s.transform(X9_mm)
            y60, s60 = a_m['Alpha_60GHz_dB_cm'].predict(Xa, return_std=True)
            del a_m; gc.collect()
            a60_nom = 10**y60
            a60_low = 10**(y60 - 1.96*s60)
            a60_up = 10**(y60 + 1.96*s60)
            return x_values, a60_nom, a60_low, a60_up

        if fom == 'EO Bandwidth [GHz]':
            # Predict nm, zc, and alphas
            with open(model_dir/"gp_nm_zc_surrogate/scalers_nm_zc.pkl", 'rb') as f: nz_s = pickle.load(f)
            X_nz = nz_s['scaler_X'].transform(X9_mm)
            with open(model_dir/"gp_nm_zc_surrogate/gp_model_nm_60.pkl", 'rb') as f: n_m = pickle.load(f)
            nm = nz_s['scalers_y']['nm_60'].inverse_transform(n_m.predict(X_nz).reshape(-1,1)).ravel()
            del n_m; gc.collect()

            with open(model_dir/"gp_nm_zc_surrogate/gp_model_Zc_60.pkl", 'rb') as f: z_m = pickle.load(f)
            zc = nz_s['scalers_y']['Zc_60'].inverse_transform(z_m.predict(X_nz).reshape(-1,1)).ravel()
            del z_m; gc.collect()

            with open(model_dir/"gp_alpha_anchors/scaler_anchors.pkl", 'rb') as f: a_s = pickle.load(f)['scaler_X']
            with open(model_dir/"gp_alpha_anchors/gp_alpha_anchors_suite.pkl", 'rb') as f: a_m = pickle.load(f)
            Xa = a_s.transform(X9_mm)
            y20, s20 = a_m['Alpha_20GHz_dB_cm'].predict(Xa, return_std=True)
            y60, s60 = a_m['Alpha_60GHz_dB_cm'].predict(Xa, return_std=True)
            y100, s100 = a_m['Alpha_100GHz_dB_cm'].predict(Xa, return_std=True)
            del a_m; gc.collect()

            bw_n_list, bw_l_list, bw_u_list = [], [], []
            f_ax = np.linspace(1.0, 150.0, 500)
            
            for i in range(100):
                a20 = 10**y20[i]; a60 = 10**y60[i]; a100 = 10**y100[i]
                a20_w = 10**(y20[i]+1.96*s20[i]); a60_w = 10**(y60[i]+1.96*s60[i]); a100_w = 10**(y100[i]+1.96*s100[i])
                a20_b = 10**(y20[i]-1.96*s20[i]); a60_b = 10**(y60[i]-1.96*s60[i]); a100_b = 10**(y100[i]-1.96*s100[i])

                def get_bw(al):
                    p, _ = curve_fit(fit_alpha, [20., 60., 100.], al)
                    s21 = calc_eo(f_ax, fit_alpha(f_ax, *p), nm[i], zc[i], L_array[i]*1e-3, 2.27, 50.0, Rt_array[i])
                    if s21[-1] > -3.0: return 150.0
                    idx = np.where(s21 <= -3.0)[0][0]
                    return f_ax[idx-1] + (f_ax[idx]-f_ax[idx-1])*(-3.0-s21[idx-1])/(s21[idx]-s21[idx-1])

                bw_n_list.append(get_bw([a20, a60, a100]))
                bw_l_list.append(get_bw([a20_w, a60_w, a100_w]))
                bw_u_list.append(get_bw([a20_b, a60_b, a100_b]))

            return x_values, np.array(bw_n_list), np.array(bw_l_list), np.array(bw_u_list)

    except Exception as e:
        st.error(f"❌ Could not predict {fom}: {e}")
        return None, None, None, None

# --- INTERACTIVE PLOT GENERATION ---
st.markdown("---")
if st.button("🚀 Generate Interactive Plot", type="primary", use_container_width=True):
    with st.spinner(f"Querying GP Surrogate & Physics Cascade for {target_fom} vs {sweep_param}..."):
        
        x_vals, y_vals, y_low, y_up = predict_sweep_ram_safe(
            base_geometry, sweep_param, (min_val, max_val), target_fom
        )
        
        if x_vals is not None:
            fig = go.Figure()

            # Add 95% Confidence Interval (Shaded Region)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_vals, x_vals[::-1]]),
                y=np.concatenate([y_up, y_low[::-1]]),
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='95% Confidence Interval'
            ))

            # Add Main Prediction Line
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                name=f'Predicted {target_fom}',
                hovertemplate=f"<b>{sweep_param}</b>: %{{x:.2f}} {unit_param}<br><b>{target_fom}</b>: %{{y:.3f}}<extra></extra>"
            ))

            # Customize Layout
            fig.update_layout(
                title=f"<b>{target_fom} vs {sweep_param}</b>",
                xaxis_title=f"{sweep_param} [{unit_param}]",
                yaxis_title=target_fom,
                hovermode="x unified", 
                template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if sweep_param == 'GAP' and min_val <= base_geometry['CAP_W']:
                st.warning("⚠️ **Note:** At narrow GAP dimensions, `CAP_W` was dynamically reduced to fit inside the gap to maintain physical geometries.")
