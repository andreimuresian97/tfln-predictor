"""
================================================================================
TFLN 11-DOF ULTIMATE OPTIMIZER: NSGA-II PARETO + SLSQP POLISH
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from scipy.optimize import minimize as scipy_minimize
import pickle
import math
import io
import base64
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE HEADER & STYLING
# ==========================================
st.title("🎯 Ultimate TFLN 11-DOF Optimizer")
st.markdown("""
**Find the absolute physical limit of your design.**
This engine uses a **Memetic Algorithm** (NSGA-II Global Pareto Search + SLSQP Gradient Polishing)
to map the ultimate trade-offs between Bandwidth and VPI, mathematically locking the velocity match while maximizing characteristic impedance (Zc).
""")

# =====================================================================
# 2. THE ULTIMATE PREDICTOR ENGINE (Embedded for Streamlit Cache)
# =====================================================================
class Ultimate_TFLN_Predictor:
    def __init__(self, dir_vpi="gp_surrogate_results_ultimate_500LHS/GP_files_Vpi", 
                 dir_nm_z0="gp_surrogate_results_ultimate_500LHS/GP_files_nm_Z0", 
                 dir_alpha="gp_surrogate_results_ultimate_500LHS/GP_files_RF_alpha"):
        
        self.dirs = {'VPI': Path(dir_vpi), 'NM_Z0': Path(dir_nm_z0), 'ALPHA': Path(dir_alpha)}
        self.models, self.scalers = {}, {}
        
        self.maes = {
            'VPI_L': 0.0866, 'RN_NM': 5.77651e-04, 'Z0': 2.51204e-02,
            'Delta_L': 2.59073e-13, 'Delta_C': 6.83945e-17,
            'COMSOL_RF_ATT': 3.49594e-03, 'Net_Ohmic': 0.0563, 'Pure_Radiation': 0.1612
        }
        self.c0 = 299792458.0
        self.L_cell = 200e-6
        self._load_system()

    def _load_system(self):
        with open(self.dirs['VPI'] / "scalers_COMSOL.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['VPI_X'], self.scalers['VPI_y'] = d['X'], d['y']['VPI_L [V*cm]']
        with open(self.dirs['VPI'] / "gp_model_VPI_L_V_cm.pkl", 'rb') as f: self.models['VPI'] = pickle.load(f)

        with open(self.dirs['NM_Z0'] / "scalers_COMSOL.pkl", 'rb') as f: self.scalers['NMZ0_C'] = pickle.load(f)
        with open(self.dirs['NM_Z0'] / "scalers_CST.pkl", 'rb') as f: self.scalers['NMZ0_CST'] = pickle.load(f)
        for m in [('RN_NM', 'gp_model_RN_NM.pkl'), ('Z0', 'gp_model_Z0_Ω.pkl'), ('dL', 'gp_model_Delta_L_lumped.pkl'), ('dC', 'gp_model_Delta_C_lumped.pkl')]:
            with open(self.dirs['NM_Z0'] / m[1], 'rb') as f: self.models[m[0]] = pickle.load(f)

        with open(self.dirs['ALPHA'] / "scalers_COMSOL.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['AL_C_X'], self.scalers['AL_C_y'] = d['X'], d['y']['RF ATT [dB/cm]']
        with open(self.dirs['ALPHA'] / "scalers_Net_Ohmic_Penalty.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['AL_O_X'], self.scalers['AL_O_y'] = d['scaler_X'], d['scaler_y']
        with open(self.dirs['ALPHA'] / "scalers_Pure_Radiation.pkl", 'rb') as f:
            d = pickle.load(f); self.scalers['AL_R_X'], self.scalers['AL_R_y'] = d['scaler_X'], d['scaler_y']
        for m in [('AL_C', 'gp_model_RF_ATT_dB_cm_COMSOL.pkl'), ('AL_O', 'gp_model_Net_Ohmic_Penalty.pkl'), ('AL_R', 'gp_model_Pure_Radiation.pkl')]:
            with open(self.dirs['ALPHA'] / m[1], 'rb') as f: self.models[m[0]] = pickle.load(f)

    def _predict_log10(self, model_key, scaler_X, scaler_y, X_input):
        X_scaled = scaler_X.transform(X_input)
        y_norm, y_std = self.models[model_key].predict(X_scaled, return_std=True)
        y_log = scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()[0]
        std_log = y_std[0] * scaler_y.scale_[0]
        val = (10 ** y_log) - 1e-15
        low = max(0.0, (10 ** (y_log - 1.96 * std_log)) - 1e-15)
        high = (10 ** (y_log + 1.96 * std_log)) - 1e-15
        return val, (high - low) / (2 * 1.96)

    def _predict_lin(self, model_key, scaler_X, scaler_y, X_input, limit_zero=False):
        X_scaled = scaler_X.transform(X_input)
        y_norm, y_std = self.models[model_key].predict(X_scaled, return_std=True)
        if isinstance(scaler_y, dict): pass
        val = scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()[0]
        std = y_std[0] * scaler_y.scale_[0]
        if limit_zero: val = max(0.0, val)
        return val, std

    def calc_eo_response(self, f_GHz, a_cond, a_rad, nm, zc, L_cm, ng, Rt, Zs):
        L_m = L_cm / 100.0
        f_ratio = np.maximum(f_GHz, 1e-9) / 60.0
        alpha_dB_cm = a_cond * np.sqrt(f_ratio) + a_rad * (f_ratio**3)
        alpha_np_m = alpha_dB_cm * (100.0 / 8.686)
        
        beta_rf = 2 * np.pi * f_GHz * 1e9 * nm / self.c0
        gamma_m = alpha_np_m + 1j * beta_rf
        gamma_o = 1j * 2 * np.pi * f_GHz * 1e9 * ng / self.c0

        Gamma_L = (Rt - zc) / (Rt + zc)
        Gamma_S = (Zs - zc) / (Zs + zc)
        denom = 1 - Gamma_S * Gamma_L * np.exp(-2 * gamma_m * L_m)
        delta1 = gamma_o - gamma_m
        delta2 = gamma_o + gamma_m

        int1 = np.where(np.abs(delta1) < 1e-12, L_m, (np.exp(delta1 * L_m) - 1) / delta1)
        int2 = np.where(np.abs(delta2) < 1e-12, L_m * np.exp(-2 * gamma_m * L_m), np.exp(-2 * gamma_m * L_m) * (np.exp(delta2 * L_m) - 1) / delta2)

        S21_eo = (int1 + Gamma_L * int2) / denom
        S21_mag = np.abs(S21_eo)
        idx_1GHz = np.argmin(np.abs(f_GHz - 1.0))
        return 20 * np.log10(S21_mag / S21_mag[idx_1GHz]), S21_mag

    def get_bandwidth(self, f_GHz, s21_db):
        for i in range(len(s21_db)):
            if s21_db[i] <= -3.0:
                if i > 0:
                    f1, f2 = f_GHz[i-1], f_GHz[i]
                    s1, s2 = s21_db[i-1], s21_db[i]
                    return f1 + (f2 - f1) * (-3.0 - s1) / (s2 - s1)
                return f_GHz[i]
        return f_GHz[-1]

    def predict(self, geom, L_device_mm, Rt, Zs_driver=65.0, ng=2.27, plot=False):
        WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH_DEPTH = geom
        L_cm = L_device_mm / 10.0
        SLAB_H = 0.460 - ETCH_DEPTH
        Perimeter = L1 + L2 + W1 + W2
        Bragg_Proxy = Perimeter * (WS / GAP)

        x_c_5  = np.array([[WS, GAP, MTX, CAP_W, ETCH_DEPTH]])
        x_c_8  = np.array([[WS, GAP, MTX, L1, L2, W1, W2, ETCH_DEPTH]])
        x_c_9  = np.array([[WS, GAP, MTX, L1, L2, W1, W2, ETCH_DEPTH, Bragg_Proxy]])
        x_lc_9 = np.array([[WS, GAP, MTX, L1, L2, W1, W2, SLAB_H, Perimeter]])

        vpi_b, vpi_b_std = self._predict_log10('VPI', self.scalers['VPI_X'], self.scalers['VPI_y'], x_c_5)
        nm_2d, nm_2d_std = self._predict_lin('RN_NM', self.scalers['NMZ0_C']['X'], self.scalers['NMZ0_C']['y']['RN NM'], x_c_5)
        z0_2d, z0_2d_std = self._predict_lin('Z0', self.scalers['NMZ0_C']['X'], self.scalers['NMZ0_C']['y']['Z0 [Ω]'], x_c_5)
        dL, dL_std = self._predict_lin('dL', self.scalers['NMZ0_CST']['X'], self.scalers['NMZ0_CST']['y']['Delta_L_lumped'], x_lc_9)
        dC, dC_std = self._predict_lin('dC', self.scalers['NMZ0_CST']['X'], self.scalers['NMZ0_CST']['y']['Delta_C_lumped'], x_lc_9)
        dL, dL_std, dC, dC_std = dL/1e12, dL_std/1e12, dC/1e15, dC_std/1e15

        al_b, al_b_std = self._predict_log10('AL_C', self.scalers['AL_C_X'], self.scalers['AL_C_y'], x_c_5)
        al_o, al_o_std = self._predict_lin('AL_O', self.scalers['AL_O_X'], self.scalers['AL_O_y'], x_c_8)
        al_r, al_r_std = self._predict_lin('AL_R', self.scalers['AL_R_X'], self.scalers['AL_R_y'], x_c_9, limit_zero=True)

        L_dist_3d = max(1e-15, (z0_2d * nm_2d / self.c0) + (dL / self.L_cell))
        C_dist_3d = max(1e-15, (nm_2d / (z0_2d * self.c0)) + (dC / self.L_cell))
        nm_3d = self.c0 * math.sqrt(L_dist_3d * C_dist_3d)
        z0_3d = math.sqrt(L_dist_3d / C_dist_3d)
        
        al_cond = al_b + al_o
        al_total_60 = al_cond + al_r

        duty_cycle = L1 / 200.0
        vpi_duty = vpi_b / (1.0 - duty_cycle)
        vpi_length = vpi_duty / L_cm

        nm_3d_std = math.sqrt(nm_2d_std**2 + (dL_std/L_dist_3d * 0.5 * nm_3d)**2 + (dC_std/C_dist_3d * 0.5 * nm_3d)**2)
        z0_3d_std = math.sqrt(z0_2d_std**2 + (dL_std/L_dist_3d * 0.5 * z0_3d)**2 + (dC_std/C_dist_3d * 0.5 * z0_3d)**2)
        al_cond_std = math.sqrt(al_b_std**2 + al_o_std**2)
        al_total_std = math.sqrt(al_b_std**2 + al_o_std**2 + al_r_std**2)

        nm_3d_mae = math.sqrt(self.maes['RN_NM']**2 + (self.maes['Delta_L']/L_dist_3d * 0.5 * nm_3d)**2 + (self.maes['Delta_C']/C_dist_3d * 0.5 * nm_3d)**2)
        z0_3d_mae = math.sqrt(self.maes['Z0']**2 + (self.maes['Delta_L']/L_dist_3d * 0.5 * z0_3d)**2 + (self.maes['Delta_C']/C_dist_3d * 0.5 * z0_3d)**2)
        al_cond_mae = math.sqrt(self.maes['COMSOL_RF_ATT']**2 + self.maes['Net_Ohmic']**2)
        al_total_mae = math.sqrt(al_cond_mae**2 + self.maes['Pure_Radiation']**2)

        f_axis = np.linspace(0.001, 150.0, 500)
        idx_60 = np.argmin(np.abs(f_axis - 60.0))

        def compute_foms(n_val, z_val, a_c_val, a_r_val):
            s21_dB_ll, s21_mag_ll = self.calc_eo_response(f_axis, 1e-8, 1e-8, n_val, z_val, L_cm, ng, Rt, Zs_driver)
            s21_dB_full, s21_mag_full = self.calc_eo_response(f_axis, a_c_val, a_r_val, n_val, z_val, L_cm, ng, Rt, Zs_driver)
            bw = self.get_bandwidth(f_axis, s21_dB_full)
            vpi_ll = vpi_length / (s21_mag_ll[idx_60] / s21_mag_ll[0])
            vpi_full = vpi_length / (s21_mag_full[idx_60] / s21_mag_full[0])
            return bw, vpi_ll, vpi_full, s21_dB_full

        bw_nom, vll_nom, vfull_nom, s21_nom_curve = compute_foms(nm_3d, z0_3d, al_cond, al_r)

        # FIXED: Protected the Jacobian deltas against ZeroDivisionError
        d_nm = max(0.01 * nm_3d, 1e-6)
        bw_nm, vll_nm, vfull_nm, _ = compute_foms(nm_3d + d_nm, z0_3d, al_cond, al_r)
        
        d_z0 = max(0.01 * z0_3d, 1e-6)
        bw_z0, vll_z0, vfull_z0, _ = compute_foms(nm_3d, z0_3d + d_z0, al_cond, al_r)
        
        d_ac = max(0.01 * abs(al_cond), 1e-6)
        bw_ac, vll_ac, vfull_ac, _ = compute_foms(nm_3d, z0_3d, al_cond + d_ac, al_r)
        
        d_ar = max(0.01 * al_r, 1e-6)
        bw_ar, vll_ar, vfull_ar, _ = compute_foms(nm_3d, z0_3d, al_cond, al_r + d_ar)

        def propagate(f_nom, f_n, f_z, f_c, f_r, uncert_n, uncert_z, uncert_c, uncert_r):
            return math.sqrt( (((f_n - f_nom)/d_nm)*uncert_n)**2 + (((f_z - f_nom)/d_z0)*uncert_z)**2 +
                              (((f_c - f_nom)/d_ac)*uncert_c)**2 + (((f_r - f_nom)/d_ar)*uncert_r)**2 )

        bw_std = propagate(bw_nom, bw_nm, bw_z0, bw_ac, bw_ar, nm_3d_std, z0_3d_std, al_cond_std, al_r_std)
        bw_mae = propagate(bw_nom, bw_nm, bw_z0, bw_ac, bw_ar, nm_3d_mae, z0_3d_mae, al_cond_mae, self.maes['Pure_Radiation'])
        vll_std = propagate(vll_nom, vll_nm, vll_z0, vll_nom, vll_nom, nm_3d_std, z0_3d_std, 0, 0)
        vfull_std = propagate(vfull_nom, vfull_nm, vfull_z0, vfull_ac, vfull_ar, nm_3d_std, z0_3d_std, al_cond_std, al_r_std)

        f_ratio = np.maximum(f_axis, 1e-9) / 60.0
        alpha_curve_nom = al_cond * np.sqrt(f_ratio) + al_r * (f_ratio**3)
        alpha_curve_bc = max(0, al_cond - 1.96*al_cond_std) * np.sqrt(f_ratio) + max(0, al_r - 1.96*al_r_std) * (f_ratio**3)
        alpha_curve_wc = (al_cond + 1.96*al_cond_std) * np.sqrt(f_ratio) + (al_r + 1.96*al_r_std) * (f_ratio**3)

        gamma_limit = 10 ** (-10.0 / 20.0) 
        rt_min_zs = Zs_driver * (1 - gamma_limit) / (1 + gamma_limit) 
        rt_min_zc = z0_3d * (1 - gamma_limit) / (1 + gamma_limit) 

        return {
            'nm': (nm_3d, nm_3d_std, nm_3d_mae), 'z0': (z0_3d, z0_3d_std, z0_3d_mae),
            'alpha': (al_total_60, al_total_std, al_total_mae),
            'vpi_base': (vpi_b, vpi_b_std, self.maes['VPI_L']),
            'vpi_duty': (vpi_duty, vpi_b_std/(1-duty_cycle), self.maes['VPI_L']/(1-duty_cycle)),
            'vpi_len': (vpi_length, (vpi_b_std/(1-duty_cycle))/L_cm, (self.maes['VPI_L']/(1-duty_cycle))/L_cm),
            'vpi_ll': (vll_nom, vll_std, 0.0), 'vpi_full': (vfull_nom, vfull_std, 0.0), 
            'bw': (bw_nom, bw_std, bw_mae), 'rt_min': max(rt_min_zs, rt_min_zc),
            'f_axis': f_axis, 's21': s21_nom_curve,
            'alpha_curve_nom': alpha_curve_nom, 'alpha_curve_bc': alpha_curve_bc, 'alpha_curve_wc': alpha_curve_wc
        }

@st.cache_resource
def load_engine(): return Ultimate_TFLN_Predictor()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load predictor models. Ensure 'gp_surrogate_results_ultimate_500LHS' exists. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: TARGETS & BOUNDS
# ==========================================
st.sidebar.header("1. Performance Targets")
target_vpi = st.sidebar.number_input("Max Target VPI (Length Scaled) [V]", value=2.00, step=0.1)
target_bw = st.sidebar.number_input("Min Target EO Bandwidth [GHz]", value=65.0, step=5.0)
target_zc = st.sidebar.number_input("Min Target Zc [Ω]", value=45.0, step=1.0)
target_nm = st.sidebar.number_input("Target Index (nm)", value=2.270, step=0.01)
target_tol = st.sidebar.number_input("Index Tolerance (+/-)", value=0.03, step=0.005)

st.sidebar.markdown("---")
st.sidebar.header("2. Search Space (Bounds)")
def range_input(label, min_def, max_def, step=0.1, fmt="%.2f"):
    c1, c2 = st.sidebar.columns(2)
    min_val = c1.number_input(f"Min {label}", value=float(min_def), step=step, format=fmt)
    max_val = c2.number_input(f"Max {label}", value=float(max_def), step=step, format=fmt)
    return (min_val, max_val)

# Restored original boundary limits
b_WS    = range_input("WS [µm]", 10.0, 60.0)
b_GAP   = range_input("GAP [µm]", 4.0, 15.0)
b_MTX   = range_input("MTX [µm]", 1.5, 15.0)
b_L1    = range_input("L1 [µm]", 2.0, 60.0)
b_L2    = range_input("L2 [µm]", 4.0, 180.0)
b_W1    = range_input("W1 [µm]", 2.0, 60.0)
b_W2    = range_input("W2 [µm]", 2.0, 60.0)
b_CAPW  = range_input("CAP_W [µm]", 1.5, 14.0)
b_ETCH  = range_input("ETCH_DEPTH [µm]", 0.0, 0.40, step=0.01, fmt="%.3f")
b_L     = range_input("L_device [mm]", 4.0, 20.0)
b_Rt    = range_input("Rt [Ω]", 34.0, 60.0)

BOUNDS_LIST = [b_WS, b_GAP, b_MTX, b_L1, b_L2, b_W1, b_W2, b_CAPW, b_ETCH, b_L, b_Rt]

# ==========================================
# 4. OPTIMIZER CLASSES (NSGA-II & SLSQP)
# ==========================================
class NSGA2_Problem(ElementwiseProblem):
    def __init__(self, bnds):
        xl = np.array([b[0] for b in bnds])
        xu = np.array([b[1] for b in bnds])
        super().__init__(n_var=11, n_obj=2, n_ieq_constr=8, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH_DEPTH, L_dev, Rt = x
        geom_um = [WS, GAP, MTX, L1, L2, W1, W2, CAP_W, ETCH_DEPTH]
        
        g1_cap = CAP_W - (GAP - 1.0)
        g2_w = (W1 + W2) - 60.0
        
        if g1_cap > 0 or g2_w > 0:
            out["F"] = [10.0, 0.0]
            out["G"] = [g1_cap, g2_w, 10., 10., 10., 10., 10., 10.]
            return

        try:
            res = engine.predict(geom=geom_um, L_device_mm=L_dev, Rt=Rt, Zs_driver=65.0, ng=2.27, plot=False)
            bw, vpi, nm, zc, rtm = res['bw'][0], res['vpi_len'][0], res['nm'][0], res['z0'][0], res['rt_min']
        except Exception as e:
            # Explicit printing to catch any remaining physics bugs
            print(f"\n[!] ML Engine Error: {e}")
            bw, vpi, nm, zc, rtm = 0.0, 10.0, 0.0, 0.0, 100.0

        # Mapped directly to target constraints
        g3_nm_low = (target_nm - target_tol) - nm
        g4_nm_high = nm - (target_nm + target_tol)
        g5_zc = target_zc - zc
        g6_rt = rtm - Rt
        g7_bw = target_bw - bw
        g8_vpi = vpi - target_vpi

        out["F"] = [vpi, -bw]
        out["G"] = [g1_cap, g2_w, g3_nm_low, g4_nm_high, g5_zc, g6_rt, g7_bw, g8_vpi]

def slsqp_polisher(x0, bounds_list):
    def get_p(x):
        try: return engine.predict(geom=x[:9].tolist(), L_device_mm=x[9], Rt=x[10], Zs_driver=65.0, ng=2.27, plot=False)
        except: return {'z0':(0.,0.,0.), 'nm':(10.,0.,0.), 'bw':(0.,0.,0.), 'vpi_len':(10.,0.,0.), 'rt_min':100.}
    
    def obj(x): return -get_p(x)['z0'][0] 
    def eq_nm(x): return get_p(x)['nm'][0] - target_nm
    
    cons = [
        {'type': 'eq', 'fun': eq_nm},
        {'type': 'ineq', 'fun': lambda x: get_p(x)['bw'][0] - target_bw},
        {'type': 'ineq', 'fun': lambda x: target_vpi - get_p(x)['vpi_len'][0]},
        {'type': 'ineq', 'fun': lambda x: x[10] - get_p(x)['rt_min']},
        {'type': 'ineq', 'fun': lambda x: (x[1]-1.0) - x[7]},
        {'type': 'ineq', 'fun': lambda x: 60.0 - (x[5]+x[6])}
    ]
    return scipy_minimize(obj, x0, method='SLSQP', bounds=bounds_list, constraints=cons, options={'ftol': 1e-4, 'maxiter': 50}).x

# ==========================================
# 5. VISUALIZATION
# ==========================================
def render_svg(s):
    return f'<img src="data:image/svg+xml;base64,{base64.b64encode(s.encode("utf-8")).decode("utf-8")}" width="100%"/>'

def generate_exact_svg(p):
    W1, W2, L1, L2, WS, GAP, MTX, CAP_W, ETCH_DEPTH = p["W1"], p["W2"], p["L1"], p["L2"], p["WS"], p["GAP"], p["MTX"], p["CAP_W"], p["ETCH_DEPTH"]
    WG = 70.0; BOTTOM_LAYER_H = 0.46 - ETCH_DEPTH; RIDGE_H = ETCH_DEPTH; CAP_HEIGHT = 1.4
    C_ELEC = '#F5BD02'; C_SUB = '#00BFFF'; C_CAP = '#00BFFF'; C_LINE = 'black'; CX = 400

    def arrow(x1, y1, x2, y2, txt, loc="top", off=10):
        m, t, a, d = (x1+x2)/2, (y1+y2)/2, "middle", "middle"
        if loc=="top": t-=off; d="auto"
        elif loc=="bottom": t+=off; d="hanging"
        elif loc=="left": m-=off; a="end"
        elif loc=="right": m+=off; a="start"
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C_LINE}" stroke-width="1.5" marker-start="url(#S)" marker-end="url(#E)" /><text x="{m}" y="{t}" fill="{C_LINE}" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="{a}" dominant-baseline="{d}">{txt}</text>'

    def top(x,y): return CX+x*3.5, 380-y*3.5
    pts = [(GAP/2,-100), (GAP/2+WG,-100), (GAP/2+WG,100), (GAP/2,100), (GAP/2,L1/2), (GAP/2+W1,L1/2), (GAP/2+W1,L2/2), (GAP/2+W1+W2,L2/2), (GAP/2+W1+W2,-L2/2), (GAP/2+W1,-L2/2), (GAP/2+W1,-L1/2), (GAP/2,-L1/2)]
    poly = " ".join([f"{top(x,y)[0]},{top(x,y)[1]}" for x,y in pts])
    svg_t = f'<svg width="800" height="700" viewBox="0 0 800 700" xmlns="http://www.w3.org/2000/svg"><defs><marker id="E" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z"/></marker><marker id="S" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto"><path d="M9,0 L9,6 L0,3 z"/></marker></defs><text x="20" y="50" font-family="sans-serif" font-size="24" font-weight="bold">Top-Down View</text><rect x="{top(-(GAP/2+WS),100)[0]}" y="{top(-(GAP/2+WS),100)[1]}" width="{WS*3.5}" height="{200*3.5}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><polygon points="{poly}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/>{arrow(*top(-(GAP/2+WS),120),*top(-GAP/2,120),"WS","top")}{arrow(*top(GAP/2,120),*top(GAP/2+WG,120),"WG","top")}{arrow(*top(-GAP/2,-120),*top(GAP/2,-120),"GAP","bottom")}{arrow(*top(GAP/2-20,-L1/2),*top(GAP/2-20,L1/2),"L1","left")}{arrow(*top(GAP/2+W1+W2+20,-L2/2),*top(GAP/2+W1+W2+20,L2/2),"L2","right")}{arrow(*top(GAP/2,-L1/2-20),*top(GAP/2+W1,-L1/2-20),"W1","bottom")}{arrow(*top(GAP/2+W1,L2/2+20),*top(GAP/2+W1+W2,L2/2+20),"W2","top")}</svg>'

    SC = 700/((WS/2+GAP)+1.0)/2
    def cs(x,y): return CX+x*SC, 350-y*SC
    svg_c = f'<svg width="800" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg"><defs><marker id="E" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z"/></marker><marker id="S" markerWidth="10" markerHeight="10" refX="1" refY="3" orient="auto"><path d="M9,0 L9,6 L0,3 z"/></marker></defs><text x="20" y="50" font-family="sans-serif" font-size="24" font-weight="bold">Cross-Section</text><rect x="0" y="{cs(0,0)[1]}" width="800" height="600" fill="{C_SUB}"/><rect x="0" y="{cs(0,BOTTOM_LAYER_H)[1]}" width="800" height="{BOTTOM_LAYER_H*SC}" fill="black"/><rect x="{cs(-WS/2,BOTTOM_LAYER_H+MTX)[0]}" y="{cs(-WS/2,BOTTOM_LAYER_H+MTX)[1]}" width="{WS*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(WS/2+GAP,BOTTOM_LAYER_H+MTX)[0]}" y="{cs(WS/2+GAP,BOTTOM_LAYER_H+MTX)[1]}" width="{500*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(-(WS/2+GAP+500),BOTTOM_LAYER_H+MTX)[0]}" y="{cs(-(WS/2+GAP+500),BOTTOM_LAYER_H+MTX)[1]}" width="{500*SC}" height="{MTX*SC}" fill="{C_ELEC}" stroke="{C_LINE}" stroke-width="1.5"/>'
    for cx in [WS/2+GAP/2, -WS/2-GAP/2]: svg_c += f'<rect x="{cs(cx-CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT)[0]}" y="{cs(cx-CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT)[1]}" width="{CAP_W*SC}" height="{CAP_HEIGHT*SC}" fill="{C_CAP}" stroke="{C_LINE}" stroke-width="1.5"/><rect x="{cs(cx-0.4,0.46)[0]}" y="{cs(cx-0.4,0.46)[1]}" width="{0.8*SC}" height="{RIDGE_H*SC}" fill="black"/>'
    y = BOTTOM_LAYER_H+max(MTX,CAP_HEIGHT)+(3.0 if MTX<5 else 0.5*MTX)
    svg_c += f'{arrow(*cs(-WS/2,y),*cs(WS/2,y),"WS","top",15)}{arrow(*cs(WS/2,y),*cs(WS/2+GAP,y),"GAP","top",15)}{arrow(*cs((-WS/2-GAP/2)-CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT+1.0),*cs((-WS/2-GAP/2)+CAP_W/2,BOTTOM_LAYER_H+CAP_HEIGHT+1.0),"CAP_W","top",15)}{arrow(*cs(-WS/2+2.0,BOTTOM_LAYER_H),*cs(-WS/2+2.0,BOTTOM_LAYER_H+MTX),"MTX","right")}</svg>'
    return svg_t, svg_c

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if st.button("🚀 RUN OPTIMIZATION", type="primary"):
    with st.spinner("Step 1/2: Global Pareto Search (NSGA-II)..."):
        res_ga = pymoo_minimize(NSGA2_Problem(BOUNDS_LIST), NSGA2(pop_size=50, n_offsprings=25), get_termination("n_gen", 40), seed=42)
        
    if res_ga.F is None:
        st.error("No designs met all physics constraints. Relax your bounds or targets.")
    else:
        # Extract Pareto Front
        df_p = pd.DataFrame(res_ga.X, columns=['WS','GAP','MTX','L1','L2','W1','W2','CAP_W','ETCH_DEPTH','L_dev','Rt'])
        df_p['VPI (V)'] = res_ga.F[:,0]; df_p['BW (GHz)'] = -res_ga.F[:,1]
        df_p = df_p.sort_values(by='BW (GHz)', ascending=False)
        
        buf = io.BytesIO()
        df_p.to_excel(buf, index=False)
        st.download_button("📥 Download Pareto Front (Excel)", buf.getvalue(), "Pareto_Front.xlsx", "application/vnd.ms-excel")

        st.markdown("---")
        with st.spinner("Step 2/2: Gradient Polishing for Maximum Z0..."):
            best_idx = np.argmax(-res_ga.F[:,1])
            final_x = slsqp_polisher(res_ga.X[best_idx], BOUNDS_LIST)
            res = engine.predict(geom=final_x[:9].tolist(), L_device_mm=final_x[9], Rt=final_x[10], Zs_driver=65.0, ng=2.27, plot=False)
            
        st.header("🏆 Optimal Geometry Found")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z0 (Impedance)", f"{res['z0'][0]:.2f} Ω", delta="Maximized")
        c2.metric("EO Bandwidth", f"{res['bw'][0]:.1f} GHz")
        c3.metric("VPI", f"{res['vpi_len'][0]:.2f} V")
        c4.metric("Index (nm)", f"{res['nm'][0]:.4f}", help="Locked to Target")
        
        st.subheader("Optimal Parameters")
        p_dict = dict(zip(['WS','GAP','MTX','L1','L2','W1','W2','CAP_W','ETCH_DEPTH','L_dev','Rt'], final_x))
        st.table(pd.DataFrame([p_dict]).style.format("{:.3f}"))
        
        col1, col2 = st.columns(2)
        svg_t, svg_c = generate_exact_svg(p_dict)
        with col1: st.markdown(render_svg(svg_t), unsafe_allow_html=True)
        with col2: st.markdown(render_svg(svg_c), unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Detailed Figures of Merit")
        
        m_nm, m_zc, m_vpi, m_a = res['nm'][2], res['z0'][2], res['vpi_len'][2], res['alpha'][2]
        
        def fmt(tup): return f"{tup[0]:.3f}", f"[{max(0, tup[0]-1.96*tup[1]):.3f}, {tup[0]+1.96*tup[1]:.3f}]"

        nm_v, nm_c = fmt(res['nm'])
        zc_v, zc_c = fmt(res['z0'])
        vp_v, vp_c = fmt(res['vpi_len'])
        al_v, al_c = fmt(res['alpha'])
        bw_v, bw_c = fmt(res['bw'])
        vll_v, vll_c = fmt(res['vpi_ll'])
        vf_v, vf_c = fmt(res['vpi_full'])

        data = [
            ["Microwave Index (nm)", nm_v, nm_c, f"± {m_nm:.4f}"],
            ["Impedance Zc [Ω]", zc_v, zc_c, f"± {m_zc:.2f}"],
            ["VPI (Electrostatic) [V]", vp_v, vp_c, f"± {m_vpi:.3f}"],
            ["RF Attenuation @ 60 GHz [dB/cm]", al_v, al_c, f"± {m_a:.3f}"],
            ["EO Bandwidth [GHz]", bw_v, bw_c, f"± {res['bw'][2]:.1f}"],
            ["VPI @ 60GHz (Walk-off+Mismatch) [V]", vll_v, vll_c, "N/A (Derived)"],
            ["VPI @ 60GHz (Full RF Physics) [V]", vf_v, vf_c, "N/A (Derived)"]
        ]
        st.table(pd.DataFrame(data, columns=["FOM", "Predicted", "95% CI", "Global MAE"]))
        
        st.markdown("### 📈 Broadband RF Response & Attenuation")
        f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(res['f_axis'], res['s21'], 'b-', lw=2)
        ax1.axhline(-3, color='r', ls='--')
        if res['bw'][0] < 150: ax1.plot(res['bw'][0], -3, 'ko'); ax1.annotate(f"{res['bw'][0]:.1f} GHz", (res['bw'][0]+3, -1.5))
        ax1.set(xlabel='Frequency (GHz)', ylabel='S21 (dB)', title='EO Bandwidth', xlim=(0,150), ylim=(-8,1)); ax1.grid(ls=':')
        
        ax2.plot(res['f_axis'], res['alpha_curve_nom'], 'g-', lw=2, label='Nominal')
        ax2.fill_between(res['f_axis'], res['alpha_curve_bc'], res['alpha_curve_wc'], color='green', alpha=0.2, label='95% CI')
        ax2.set(xlabel='Frequency (GHz)', ylabel='Alpha (dB/cm)', title='Decoupled RF Attenuation', xlim=(0,150)); ax2.grid(ls=':')
        
        st.pyplot(f1)
        
        # Pareto Plot
        f2, ax3 = plt.subplots(figsize=(8,4))
        ax3.scatter(res_ga.F[:,0], -res_ga.F[:,1], c='blue', alpha=0.7)
        ax3.scatter(res['vpi_len'][0], res['bw'][0], c='gold', s=200, edgecolors='k', marker='*', label='SLSQP Final Winner')
        ax3.set(xlabel='VPI [V]', ylabel='Bandwidth [GHz]', title='NSGA-II Pareto Front')
        ax3.grid(ls=':'); ax3.legend(); st.pyplot(f2)
