"""
================================================================================
TFLN PREDICTOR (8 VARS / 199 SAMPLES)
Instant Inference Engine for VPI, nm, Z0, S21
Adapted for GP Surrogate Version 2 (Fixed S21 Model)
================================================================================
"""

import numpy as np
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class TFLNPredictor:
    # UPDATED: Points to the folder created by 'GP surrogate 2.py'
    def __init__(self, model_dir="gp_surrogate_results_199_8var_fixed"):
        """Load trained models and scalers from disk"""
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        
        # Check if directory exists
        if not self.model_dir.exists():
            # Fallback to the other possible folder name if 'fixed' doesn't exist
            fallback = Path("gp_surrogate_results_199_8var")
            if fallback.exists():
                print(f"Note: '{model_dir}' not found, switching to '{fallback}'")
                self.model_dir = fallback
            else:
                raise FileNotFoundError(f"Model directory '{model_dir}' not found. Did you run the training script?")

        print(f"Loading models from: {self.model_dir} ...")
        
        # 1. Load Scalers
        scaler_path = self.model_dir / "scalers.pkl"
        try:
            with open(scaler_path, 'rb') as f:
                data = pickle.load(f)
                self.scalers_X = data['X']
                self.scalers_y = data['y']
        except FileNotFoundError:
            raise FileNotFoundError("scalers.pkl missing. Re-run training.")
            
        # 2. Load Models
        model_files = list(self.model_dir.glob("gp_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No model files (gp_model_*.pkl) found.")
            
        for p in model_files:
            # Extract name "gp_model_VPI.pkl" -> "VPI"
            # Note: The training script saved them as "gp_model_{safe_name}.pkl"
            # We need to map safe names back to keys if possible, or iterate carefully.
            name_key = p.stem.replace("gp_model_", "") 
            
            with open(p, 'rb') as f:
                self.models[name_key] = pickle.load(f)
                
        print(f"✓ Loaded {len(self.models)} models: {list(self.models.keys())}")

    def predict(self, geometry):
        """
        Predict FOMs for a given geometry [WS, GAP, MTX, CAP_W, L1, L2, W1, W2]
        """
        # 1. Validate Input
        if len(geometry) != 8:
            raise ValueError(f"Expected 8 input variables, got {len(geometry)}")
            
        X_input = np.array(geometry).reshape(1, -1)
        
        # 2. Normalize Input
        # uses the 'input' scaler saved during training
        X_norm = self.scalers_X['input'].transform(X_input)
        
        results = {}
        
        # 3. Predict each objective
        # The keys in self.models are "safe names" (e.g. "VPI", "S21")
        # The keys in self.scalers_y are the original column names (e.g. "VPI (duty cycle)")
        
        # We map safe names back to scaler keys
        scaler_key_map = {
            'VPI': 'VPI (duty cycle)',
            'nm': 'nm',
            'Z0': 'Z0',
            'S21': 'S21'
        }

        for model_name, model in self.models.items():
            # Find corresponding scaler key
            scaler_key = scaler_key_map.get(model_name, model_name)
            
            # Predict (returns normalized value)
            y_pred_norm, y_std_norm = model.predict(X_norm, return_std=True)
            
            # 4. Inverse Transform Logic (Crucial Step!)
            scaler = self.scalers_y[scaler_key]
            
            # Step A: Inverse Scale
            y_pred = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()[0]
            
            # Step B: Handle Uncertainty Scaling
            # std deviation scales linearly with the scaler's scale factor
            if hasattr(scaler, 'scale_'):
                y_std = y_std_norm[0] * scaler.scale_[0]
            elif hasattr(scaler, 'data_range_'): # For MinMaxScaler
                y_std = y_std_norm[0] * (scaler.data_max_[0] - scaler.data_min_[0])
            else:
                y_std = y_std_norm[0] # Fallback
            
            # Step C: Handle Special Transformations (Log10)
            # In 'GP surrogate 2.py', VPI was Log10-transformed BEFORE scaling.
            if "VPI" in model_name:
                # The value we have now is log10(VPI). We need 10^x.
                # Uncertainty propagation for 10^x: sigma_y ≈ y * ln(10) * sigma_x
                log_val = y_pred
                real_val = 10 ** log_val
                
                # Propagate uncertainty roughly
                real_std = real_val * np.log(10) * y_std
                
                y_pred = real_val
                y_std = real_std
            
            # Calculate 95% Confidence Interval
            lower_95 = y_pred - 1.96 * y_std
            upper_95 = y_pred + 1.96 * y_std
            
            results[model_name] = {
                'value': y_pred,
                'std': y_std,
                'lower_bound': lower_95,
                'upper_bound': upper_95
            }
            
        return results

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    # 1. Initialize
    try:
        # Check folder name matches your actual results folder
        predictor = TFLNPredictor(model_dir="gp_surrogate_results_199_8var_fixed")
        
        # 2. Define a test geometry (8 Variables)
        # Order: ['WS', 'GAP', 'MTX', 'CAP_W', 'L1', 'L2', 'W1', 'W2']
        # Using values from the middle of your dataset ranges as a test
        my_geometry = [
            22.936,   # WS (um)
            10.311,    # GAP (um)
            8.07,    # MTX (um)
            1.65,    # CAP_W (um)
            8.0,   # L1 (um)
            86,   # L2 (um)
            5,   # W1 (um)
            11,  # W2 (um)
        ]
        
        print("\n" + "="*60)
        print(f"PREDICTING PERFORMANCE FOR NEW GEOMETRY:")
        print(f"WS={my_geometry[0]}, GAP={my_geometry[1]}, MTX={my_geometry[2]}...")
        print("="*60)
        
        # 3. Get Predictions
        predictions = predictor.predict(my_geometry)
        
        # 4. Display
        print(f"{'FOM':<10} | {'Prediction':<12} | {'Uncertainty (95% CI)':<25}")
        print("-" * 55)
        
        for param, res in predictions.items():
            val = res['value']
            low = res['lower_bound']
            high = res['upper_bound']
            print(f"{param:<10} | {val:<12.4f} | [{low:.4f}, {high:.4f}]")
            
    except Exception as e:
        print(f"\nERROR: {e}")