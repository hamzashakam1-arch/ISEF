"""
=============================================================================
NEPHROSCAN CLINICAL MEDICAL PLATFORM
Point-of-Care Urine Multiplex Analysis & Early CKD Telehealth Portal

MODULE: sample_generator.py
DESCRIPTION:
    Synthesizes photorealistic, scientifically calibrated multiplex urine test strip
    benchmarks for 3 clinical diagnostic profiles:
      1. Healthy Normal (KDIGO Stage A1)
      2. Early Diabetic Nephropathy / Microalbuminuria (KDIGO Stage A2)
      3. Advanced Glomerular Impairment / Severe Macroalbuminuria (KDIGO Stage A3)
=============================================================================
"""

import os
import cv2
import numpy as np
from cv_engine import BIOMARKER_METADATA, PAD_ORDER

OUTPUT_DIR = "sample_strips"


def generate_synthetic_strip(
    case_name: str,
    target_values: dict,
    noise_level: float = 2.5
) -> str:
    """
    Creates a photorealistic vertical urine test strip on a clean medical backdrop.
    
    Parameters:
        case_name: Filename identifier (e.g. 'case1_healthy_a1.png')
        target_values: Dict of biomarker -> target concentration
        noise_level: Sensor Gaussian noise standard deviation
        
    Returns:
        Saved file path.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Canvas dimensions: 400px wide x 900px tall
    img_h, img_w = 900, 400
    
    # Background: clean medical gray-white gradient
    y_coords = np.linspace(245, 235, img_h).astype(np.uint8)
    backdrop = np.tile(y_coords[:, np.newaxis, np.newaxis], (1, img_w, 3))
    
    # Add subtle paper texture
    bg_noise = np.random.normal(0, 1.5, backdrop.shape).astype(np.int16)
    backdrop = np.clip(backdrop.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)
    
    # White plastic strip dimensions
    strip_w = 70
    strip_h = 760
    strip_x1 = (img_w - strip_w) // 2
    strip_x2 = strip_x1 + strip_w
    strip_y1 = (img_h - strip_h) // 2
    strip_y2 = strip_y1 + strip_h
    
    # Strip subtle drop shadow
    shadow = backdrop.copy()
    cv2.rectangle(shadow, (strip_x1 - 4, strip_y1 + 4), (strip_x2 + 6, strip_y2 + 8), (195, 195, 195), -1)
    backdrop = cv2.addWeighted(shadow, 0.4, backdrop, 0.6, 0)
    
    # Draw white plastic strip substrate with slight bevel
    cv2.rectangle(backdrop, (strip_x1, strip_y1), (strip_x2, strip_y2), (250, 250, 250), -1)
    cv2.rectangle(backdrop, (strip_x1, strip_y1), (strip_x2, strip_y2), (210, 210, 210), 1)
    
    # Longitudinal pad positions matching cv_engine.py
    num_pads = len(PAD_ORDER)
    pad_relative_centers = np.linspace(0.12, 0.88, num_pads)
    pad_h = int(strip_h * 0.085)
    pad_w = int(strip_w * 0.72)
    
    for i, pad_name in enumerate(PAD_ORDER):
        val = target_values.get(pad_name, None)
        meta = BIOMARKER_METADATA[pad_name]
        refs = meta["reference_points"]
        
        # Find closest reference color or interpolate
        if val is None:
            chosen_rgb = refs[0]["rgb"]
        else:
            # Interpolate RGB between matching reference points
            ref_vals = [r["val"] for r in refs]
            ref_rgbs = [r["rgb"] for r in refs]
            
            val_clamped = np.clip(val, min(ref_vals), max(ref_vals))
            interp_r = np.interp(val_clamped, ref_vals, [c[0] for c in ref_rgbs])
            interp_g = np.interp(val_clamped, ref_vals, [c[1] for c in ref_rgbs])
            interp_b = np.interp(val_clamped, ref_vals, [c[2] for c in ref_rgbs])
            chosen_rgb = (int(interp_r), int(interp_g), int(interp_b))
        
        center_y_pct = pad_relative_centers[i]
        pad_cy = strip_y1 + int(strip_h * center_y_pct)
        py1 = pad_cy - pad_h // 2
        py2 = pad_cy + pad_h // 2
        px1 = strip_x1 + (strip_w - pad_w) // 2
        px2 = px1 + pad_w
        
        # Color in BGR for OpenCV
        bgr_color = (chosen_rgb[2], chosen_rgb[1], chosen_rgb[0])
        
        # Fill pad
        pad_canvas = np.full((py2 - py1, px2 - px1, 3), bgr_color, dtype=np.uint8)
        
        # Add realistic micro-texture (porous cellulose membrane grain)
        grain = np.random.normal(0, noise_level, pad_canvas.shape).astype(np.int16)
        pad_canvas = np.clip(pad_canvas.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        
        backdrop[py1:py2, px1:px2] = pad_canvas
        
        # Fine pad border shadow (reagent pad depth)
        cv2.rectangle(backdrop, (px1, py1), (px2, py2), (180, 180, 180), 1)

    output_path = os.path.join(OUTPUT_DIR, case_name)
    cv2.imwrite(output_path, backdrop)
    return output_path


def generate_all_sample_cases():
    """Generates the three benchmark medical cases for demonstration and validation."""
    cases = {
        "case1_normal_a1.png": {
            "Microalbumin": 10.0,
            "Creatinine": 100.0,
            "Total Protein": 0.0,
            "Uric Acid": 30.0,
            "pH": 6.5,
            "Specific Gravity": 1.018
        },
        "case2_microalbuminuria_a2.png": {
            "Microalbumin": 65.0,
            "Creatinine": 100.0,
            "Total Protein": 30.0,
            "Uric Acid": 55.0,
            "pH": 6.0,
            "Specific Gravity": 1.020
        },
        "case3_severe_ckd_a3.png": {
            "Microalbumin": 150.0,
            "Creatinine": 30.0,
            "Total Protein": 300.0,
            "Uric Acid": 85.0,
            "pH": 5.0,
            "Specific Gravity": 1.010
        }
    }
    
    paths = {}
    for filename, profile in cases.items():
        p = generate_synthetic_strip(filename, profile)
        paths[filename] = p
        
    print(f"Synthesized {len(paths)} calibrated test strip benchmarks in '{OUTPUT_DIR}/'.")
    return paths


if __name__ == "__main__":
    generate_all_sample_cases()
