"""
=============================================================================
NEPHROSCAN CLINICAL MEDICAL PLATFORM
Point-of-Care Urine Multiplex Analysis & Early CKD Telehealth Portal

MODULE: cv_engine.py
AUTHOR: Senior Biomedical Computer Vision Engineer
DESCRIPTION:
    Computer vision processing, automated 6-pad segmentation, HSV colorimetry,
    intra-pad homogeneity (RSD) quality control, quantitative biomarker mapping,
    uACR/uPCR calculation, and bilingual KDIGO 2024 CKD clinical risk engine.
=============================================================================
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# =============================================================================
# 1. BIOMARKER CLINICAL SPECIFICATIONS & BILINGUAL CALIBRATION DATA
# =============================================================================
# 6 core test pads:
# 1) Microalbumin (mALB) [mg/L]
# 2) Creatinine (CRE) [mg/dL]
# 3) Total Protein (PRO) [mg/dL]
# 4) Uric Acid (UA) [mg/dL]
# 5) pH [pH units]
# 6) Specific Gravity (SG) [ratio]

BIOMARKER_METADATA = {
    "Microalbumin": {
        "key": "ALB",
        "name_en": "Microalbumin",
        "name_ar": "الزلال الدقيق",
        "unit": "mg/L",
        "description_en": "Sensitive early indicator of glomerular barrier injury and diabetic nephropathy.",
        "description_ar": "مؤشر مبكر وحساس لإصابة المرشحات الكبيبية واعتلال الكلى السكري.",
        "normal_range_en": "< 20 mg/L",
        "normal_range_ar": "< 20 ملغ/لتر",
        "reference_points": [
            {
                "val": 10.0,
                "label_en": "Normal (10)",
                "label_ar": "طبيعي (10)",
                "rgb": (228, 220, 115),
                "hsv": (27, 126, 228),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            },
            {
                "val": 30.0,
                "label_en": "Borderline / A2 (30)",
                "label_ar": "حدي / خفيف (30)",
                "rgb": (175, 205, 125),
                "hsv": (41, 99, 205),
                "risk_en": "Moderate",
                "risk_ar": "متوسط"
            },
            {
                "val": 80.0,
                "label_en": "Microalbuminuria (80)",
                "label_ar": "بيلة ألبومينية دقيقة (80)",
                "rgb": (95, 185, 155),
                "hsv": (80, 124, 185),
                "risk_en": "High",
                "risk_ar": "مرتفع"
            },
            {
                "val": 150.0,
                "label_en": "Macroalbuminuria (150)",
                "label_ar": "بيلة ألبومينية شديدة (150)",
                "rgb": (38, 145, 180),
                "hsv": (97, 201, 180),
                "risk_en": "Severe",
                "risk_ar": "شديد"
            }
        ]
    },
    "Creatinine": {
        "key": "CRE",
        "name_en": "Creatinine",
        "name_ar": "الكرياتينين",
        "unit": "mg/dL",
        "description_en": "Muscle metabolite used as an internal reference standard to normalize biomarker dilution.",
        "description_ar": "ناتج أيضي يُستخدم كمعيار داخلي لتعديل تفاوت تركيز أو تخفيف عينة البول.",
        "normal_range_en": "30 - 250 mg/dL",
        "normal_range_ar": "30 - 250 ملغ/ديسيلتر",
        "reference_points": [
            {
                "val": 10.0,
                "label_en": "Very Dilute (10)",
                "label_ar": "مخفف جداً (10)",
                "rgb": (245, 205, 145),
                "hsv": (18, 104, 245),
                "risk_en": "Monitor",
                "risk_ar": "مراقبة"
            },
            {
                "val": 50.0,
                "label_en": "Normal Dilute (50)",
                "label_ar": "طبيعي مخفف (50)",
                "rgb": (220, 160, 95),
                "hsv": (16, 145, 220),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            },
            {
                "val": 100.0,
                "label_en": "Normal Average (100)",
                "label_ar": "طبيعي متوسط (100)",
                "rgb": (185, 115, 60),
                "hsv": (13, 172, 185),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            },
            {
                "val": 200.0,
                "label_en": "Concentrated (200)",
                "label_ar": "مركز (200)",
                "rgb": (150, 75, 45),
                "hsv": (8, 179, 150),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            },
            {
                "val": 300.0,
                "label_en": "Highly Concentrated (300)",
                "label_ar": "شديد التركيز (300)",
                "rgb": (115, 45, 40),
                "hsv": (2, 166, 115),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            }
        ]
    },
    "Total Protein": {
        "key": "PRO",
        "name_en": "Total Protein",
        "name_ar": "البروتين الكلي",
        "unit": "mg/dL",
        "description_en": "Assesses tubular and glomerular protein leakage and gross nephropathy.",
        "description_ar": "يقيس تسرب البروتين العام عبر الكبيبات والأنابيب الكلوية.",
        "normal_range_en": "< 15 mg/dL (Negative)",
        "normal_range_ar": "< 15 ملغ/ديسيلتر (سلبي)",
        "reference_points": [
            {
                "val": 0.0,
                "label_en": "Negative (0)",
                "label_ar": "سلبي (0)",
                "rgb": (245, 235, 130),
                "hsv": (27, 120, 245),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            },
            {
                "val": 15.0,
                "label_en": "Trace (15)",
                "label_ar": "أثر طفيف (15)",
                "rgb": (205, 225, 115),
                "hsv": (36, 125, 225),
                "risk_en": "Monitor",
                "risk_ar": "مراقبة"
            },
            {
                "val": 30.0,
                "label_en": "1+ Positive (30)",
                "label_ar": "+1 إيجابي (30)",
                "rgb": (145, 200, 100),
                "hsv": (46, 128, 200),
                "risk_en": "Moderate",
                "risk_ar": "متوسط"
            },
            {
                "val": 100.0,
                "label_en": "2+ Positive (100)",
                "label_ar": "+2 إيجابي (100)",
                "rgb": (80, 165, 95),
                "hsv": (65, 131, 165),
                "risk_en": "High",
                "risk_ar": "مرتفع"
            },
            {
                "val": 300.0,
                "label_en": "3+ Positive (300)",
                "label_ar": "+3 إيجابي (300)",
                "rgb": (42, 125, 115),
                "hsv": (87, 169, 125),
                "risk_en": "Severe",
                "risk_ar": "شديد"
            },
            {
                "val": 1000.0,
                "label_en": "4+ Severe (1000)",
                "label_ar": "+4 شديد جداً (1000)",
                "rgb": (22, 85, 95),
                "hsv": (94, 196, 95),
                "risk_en": "Severe",
                "risk_ar": "شديد جداً"
            }
        ]
    },
    "Uric Acid": {
        "key": "UA",
        "name_en": "Uric Acid",
        "name_ar": "حمض اليوريك",
        "unit": "mg/dL",
        "description_en": "Purine breakdown product indicating risk of urate nephropathy and renal calculi.",
        "description_ar": "مؤشر لترسب بلورات اليورات، حصى الكلى، واعتلال الكلى النقرسي.",
        "normal_range_en": "15 - 60 mg/dL",
        "normal_range_ar": "15 - 60 ملغ/ديسيلتر",
        "reference_points": [
            {
                "val": 15.0,
                "label_en": "Low Normal (15)",
                "label_ar": "طبيعي منخفض (15)",
                "rgb": (245, 225, 220),
                "hsv": (6, 26, 245),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            },
            {
                "val": 35.0,
                "label_en": "Optimal Normal (35)",
                "label_ar": "طبيعي مثالي (35)",
                "rgb": (230, 160, 175),
                "hsv": (173, 77, 230),
                "risk_en": "Normal",
                "risk_ar": "طبيعي"
            },
            {
                "val": 55.0,
                "label_en": "Borderline High (55)",
                "label_ar": "حدي مرتفع (55)",
                "rgb": (195, 100, 140),
                "hsv": (167, 124, 195),
                "risk_en": "Monitor",
                "risk_ar": "مراقبة"
            },
            {
                "val": 75.0,
                "label_en": "Hyperuricosuria (75)",
                "label_ar": "فرط يوريك البول (75)",
                "rgb": (150, 60, 125),
                "hsv": (158, 153, 150),
                "risk_en": "High",
                "risk_ar": "مرتفع"
            },
            {
                "val": 100.0,
                "label_en": "Severe Hyperuricosuria (100)",
                "label_ar": "فرط يوريك شديد (100)",
                "rgb": (100, 30, 105),
                "hsv": (148, 182, 105),
                "risk_en": "Severe",
                "risk_ar": "شديد"
            }
        ]
    },
    "pH": {
        "key": "pH",
        "name_en": "pH (Acidity)",
        "name_ar": "الرقم الهيدروجيني (الحموضة)",
        "unit": "pH",
        "description_en": "Acid-base homeostasis; influences crystalluria, kidney stone formation, and RTA.",
        "description_ar": "التوازن الحمضي القاعدي؛ يؤثر على تشكل الحصوات والحماض الأنبوبي الكلوي.",
        "normal_range_en": "5.5 - 7.5",
        "normal_range_ar": "5.5 - 7.5",
        "reference_points": [
            {"val": 5.0, "label_en": "Acidic (5.0)", "label_ar": "حمضي (5.0)", "rgb": (240, 115, 45), "hsv": (11, 207, 240), "risk_en": "Monitor", "risk_ar": "مراقبة"},
            {"val": 5.5, "label_en": "Mildly Acidic (5.5)", "label_ar": "حمضي خفيف (5.5)", "rgb": (245, 155, 45), "hsv": (17, 208, 245), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 6.0, "label_en": "Optimal (6.0)", "label_ar": "مثالي (6.0)", "rgb": (245, 195, 45), "hsv": (22, 208, 245), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 6.5, "label_en": "Optimal (6.5)", "label_ar": "مثالي (6.5)", "rgb": (200, 205, 50), "hsv": (31, 193, 205), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 7.0, "label_en": "Neutral (7.0)", "label_ar": "متعادل (7.0)", "rgb": (140, 200, 70), "hsv": (44, 166, 200), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 7.5, "label_en": "Mild Alkaline (7.5)", "label_ar": "قلوي خفيف (7.5)", "rgb": (85, 180, 115), "hsv": (69, 135, 180), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 8.0, "label_en": "Alkaline (8.0)", "label_ar": "قلوي (8.0)", "rgb": (50, 155, 150), "hsv": (89, 173, 155), "risk_en": "Monitor", "risk_ar": "مراقبة"},
            {"val": 8.5, "label_en": "High Alkaline (8.5)", "label_ar": "قلوي مرتفع (8.5)", "rgb": (35, 110, 155), "hsv": (101, 197, 155), "risk_en": "High", "risk_ar": "مرتفع"}
        ]
    },
    "Specific Gravity": {
        "key": "SG",
        "name_en": "Specific Gravity",
        "name_ar": "الكثافة النوعية",
        "unit": "SG",
        "description_en": "Evaluates renal tubular concentrating ability, hydration, and isosthenuria.",
        "description_ar": "يقيم قدرة الأنابيب الكلوية على تركيز البول، حالة الإماهة، وثبوت الكثافة.",
        "normal_range_en": "1.010 - 1.025",
        "normal_range_ar": "1.010 - 1.025",
        "reference_points": [
            {"val": 1.000, "label_en": "Hyposthenuria (1.000)", "label_ar": "تخفيف مفرط (1.000)", "rgb": (30, 100, 95), "hsv": (88, 179, 100), "risk_en": "Monitor", "risk_ar": "مراقبة"},
            {"val": 1.005, "label_en": "Dilute (1.005)", "label_ar": "مخفف (1.005)", "rgb": (50, 125, 90), "hsv": (76, 153, 125), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 1.010, "label_en": "Isosthenuria (1.010)", "label_ar": "ثبوت الكثافة (1.010)", "rgb": (85, 145, 75), "hsv": (55, 123, 145), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 1.015, "label_en": "Optimal (1.015)", "label_ar": "مثالي (1.015)", "rgb": (130, 165, 65), "hsv": (40, 154, 165), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 1.020, "label_en": "Optimal (1.020)", "label_ar": "مثالي (1.020)", "rgb": (175, 180, 60), "hsv": (31, 170, 180), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 1.025, "label_en": "Concentrated (1.025)", "label_ar": "مركز (1.025)", "rgb": (205, 180, 50), "hsv": (25, 193, 205), "risk_en": "Normal", "risk_ar": "طبيعي"},
            {"val": 1.030, "label_en": "Hypersthenuria (1.030)", "label_ar": "تركيز مفرط (1.030)", "rgb": (220, 165, 45), "hsv": (21, 203, 220), "risk_en": "Monitor", "risk_ar": "مراقبة"}
        ]
    }
}

PAD_ORDER = ["Microalbumin", "Creatinine", "Total Protein", "Uric Acid", "pH", "Specific Gravity"]


# =============================================================================
# 2. IMAGE PREPROCESSING & COLOR ILLUMINATION NORMALIZATION
# =============================================================================

def normalize_illumination(image_bgr: np.ndarray) -> np.ndarray:
    """
    Applies Gray-World Color Constancy preprocessing to neutralize ambient
    lighting shifts (incandescent warm casts vs cool daylight).
    """
    img_float = image_bgr.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    
    kb = np.clip(gray_mean / (b_mean + 1e-5), 0.7, 1.4)
    kg = np.clip(gray_mean / (g_mean + 1e-5), 0.7, 1.4)
    kr = np.clip(gray_mean / (r_mean + 1e-5), 0.7, 1.4)
    
    b_balanced = np.clip(b * kb, 0, 255)
    g_balanced = np.clip(g * kg, 0, 255)
    r_balanced = np.clip(r * kr, 0, 255)
    
    return cv2.merge([b_balanced, g_balanced, r_balanced]).astype(np.uint8)


# =============================================================================
# 3. AUTOMATED STRIP DETECTION & ROBUST PAD SEGMENTATION
# =============================================================================

def detect_and_segment_strip(
    image_bgr: np.ndarray,
    manual_crop_pct: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Locates the test strip, corrects skew, and extracts 6 core pad ROIs.
    Utilizes center 60% core sub-sampling to exclude edge meniscus reflections.
    """
    h_orig, w_orig = image_bgr.shape[:2]
    normalized_bgr = normalize_illumination(image_bgr)
    
    gray = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 4
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    strip_rect = None
    best_score = -1.0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h_orig * w_orig * 0.02):
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_box, h_box), angle = rect
        long_dim = max(w_box, h_box)
        short_dim = min(w_box, h_box)
        aspect_ratio = long_dim / (short_dim + 1e-5)
        
        if 2.2 <= aspect_ratio <= 14.0:
            score = area * aspect_ratio
            if score > best_score:
                best_score = score
                strip_rect = rect

    if strip_rect is None or manual_crop_pct is not None:
        box_x = int(w_orig * 0.35)
        box_w = int(w_orig * 0.30)
        box_y = int(h_orig * 0.08)
        box_h = int(h_orig * 0.84)
        strip_crop = normalized_bgr[box_y:box_y + box_h, box_x:box_x + box_w]
        strip_origin = (box_x, box_y, box_w, box_h)
        detection_mode = "Guided Center Grid"
    else:
        (cx, cy), (w_box, h_box), angle = strip_rect
        if w_box > h_box:
            angle += 90.0
            w_box, h_box = h_box, w_box
            
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_full = cv2.warpAffine(normalized_bgr, M, (w_orig, h_orig), flags=cv2.INTER_CUBIC)
        
        x_min = max(0, int(cx - w_box / 2))
        y_min = max(0, int(cy - h_box / 2))
        w_int = min(w_orig - x_min, int(w_box))
        h_int = min(h_orig - y_min, int(h_box))
        
        if w_int > 20 and h_int > 100:
            strip_crop = rotated_full[y_min:y_min + h_int, x_min:x_min + w_int]
            strip_origin = (x_min, y_min, w_int, h_int)
            detection_mode = "Automated Contour & Deskew"
        else:
            box_x = int(w_orig * 0.35)
            box_w = int(w_orig * 0.30)
            box_y = int(h_orig * 0.08)
            box_h = int(h_orig * 0.84)
            strip_crop = normalized_bgr[box_y:box_y + box_h, box_x:box_x + box_w]
            strip_origin = (box_x, box_y, box_w, box_h)
            detection_mode = "Guided Center Grid"

    num_pads = len(PAD_ORDER)
    pad_measurements: List[Dict[str, Any]] = []
    pad_relative_centers = np.linspace(0.12, 0.88, num_pads)
    pad_height_ratio = 0.085
    pad_width_ratio = 0.65
    
    annotated_bgr = normalized_bgr.copy()
    ox, oy, ow, oh = strip_origin
    cv2.rectangle(annotated_bgr, (ox, oy), (ox + ow, oy + oh), (230, 160, 40), 2)
    
    for i, pad_name in enumerate(PAD_ORDER):
        center_y_pct = pad_relative_centers[i]
        pad_cy = int(oh * center_y_pct)
        pad_h = int(oh * pad_height_ratio)
        pad_w = int(ow * pad_width_ratio)
        
        pad_y1 = max(0, pad_cy - pad_h // 2)
        pad_y2 = min(oh, pad_cy + pad_h // 2)
        pad_x1 = max(0, (ow - pad_w) // 2)
        pad_x2 = min(ow, pad_x1 + pad_w)
        
        pad_roi = strip_crop[pad_y1:pad_y2, pad_x1:pad_x2]
        
        # Center-core sub-sampling (inner 60%) to prevent edge glare / substrate bleed
        roi_h, roi_w = pad_roi.shape[:2]
        if roi_h > 4 and roi_w > 4:
            ch_start, ch_end = int(roi_h * 0.20), int(roi_h * 0.80)
            cw_start, cw_end = int(roi_w * 0.20), int(roi_w * 0.80)
            core_roi = pad_roi[ch_start:ch_end, cw_start:cw_end]
        else:
            core_roi = pad_roi

        core_hsv = cv2.cvtColor(core_roi, cv2.COLOR_BGR2HSV)
        core_rgb = cv2.cvtColor(core_roi, cv2.COLOR_BGR2RGB)
        
        mean_h = float(np.mean(core_hsv[:, :, 0]))
        mean_s = float(np.mean(core_hsv[:, :, 1]))
        mean_v = float(np.mean(core_hsv[:, :, 2]))
        
        mean_r = float(np.mean(core_rgb[:, :, 0]))
        mean_g = float(np.mean(core_rgb[:, :, 1]))
        mean_b = float(np.mean(core_rgb[:, :, 2]))
        
        # Intra-pad homogeneity metric (Target RSD < 5%)
        std_v = float(np.std(core_hsv[:, :, 2]))
        rsd_percent = round((std_v / (mean_v + 1e-5)) * 100.0, 2)
        
        global_x1 = ox + pad_x1
        global_y1 = oy + pad_y1
        global_x2 = ox + pad_x2
        global_y2 = oy + pad_y2
        
        # Highlight segmented pad on strip
        cv2.rectangle(annotated_bgr, (global_x1, global_y1), (global_x2, global_y2), (0, 210, 255), 2)
        
        badge_text = f"P{i+1}: {BIOMARKER_METADATA[pad_name]['key']}"
        cv2.putText(
            annotated_bgr, badge_text,
            (global_x2 + 8, (global_y1 + global_y2) // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        chip_x = max(5, global_x1 - 25)
        chip_y = global_y1
        cv2.rectangle(annotated_bgr, (chip_x, chip_y), (chip_x + 18, chip_y + 18), (int(mean_b), int(mean_g), int(mean_r)), -1)
        cv2.rectangle(annotated_bgr, (chip_x, chip_y), (chip_x + 18, chip_y + 18), (255, 255, 255), 1)
        
        quality_flag_en = "OPTIMAL (RSD < 5%)" if rsd_percent < 5.0 else ("ACCEPTABLE (< 10%)" if rsd_percent < 10.0 else "HIGH VARIANCE")
        quality_flag_ar = "مثالي (RSD < 5%)" if rsd_percent < 5.0 else ("مقبول (< 10%)" if rsd_percent < 10.0 else "تباين مرتفع")

        pad_measurements.append({
            "pad_index": i + 1,
            "name": pad_name,
            "name_en": BIOMARKER_METADATA[pad_name]["name_en"],
            "name_ar": BIOMARKER_METADATA[pad_name]["name_ar"],
            "key": BIOMARKER_METADATA[pad_name]["key"],
            "unit": BIOMARKER_METADATA[pad_name]["unit"],
            "mean_hsv": (mean_h, mean_s, mean_v),
            "mean_rgb": (mean_r, mean_g, mean_b),
            "rsd_percent": rsd_percent,
            "quality_flag_en": quality_flag_en,
            "quality_flag_ar": quality_flag_ar,
            "roi_crop": cv2.cvtColor(pad_roi, cv2.COLOR_BGR2RGB)
        })

    strip_telemetry = {
        "detection_mode": detection_mode,
        "strip_bounds": strip_origin,
        "pads_segmented": len(pad_measurements),
        "mean_strip_rsd": round(float(np.mean([p["rsd_percent"] for p in pad_measurements])), 2)
    }

    return annotated_bgr, pad_measurements, strip_telemetry


# =============================================================================
# 4. COLORIMETRIC CALIBRATION & QUANTITATIVE MAPPING
# =============================================================================

def calculate_hsv_color_distance(hsv1: Tuple[float, float, float], hsv2: Tuple[float, float, float]) -> float:
    """Calculates weighted perceptual circular distance in HSV space."""
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    
    dh = abs(h1 - h2)
    dh = min(dh, 180.0 - dh) / 90.0
    ds = abs(s1 - s2) / 255.0
    dv = abs(v1 - v2) / 255.0
    
    return float(np.sqrt(2.8 * (dh ** 2) + 1.4 * (ds ** 2) + 0.4 * (dv ** 2)))


def map_pad_to_concentration(pad_name: str, measured_hsv: Tuple[float, float, float]) -> Dict[str, Any]:
    """Continuous IDW quantitative mapping against calibrated clinical anchors."""
    meta = BIOMARKER_METADATA[pad_name]
    refs = meta["reference_points"]
    
    distances = np.array([calculate_hsv_color_distance(measured_hsv, r["hsv"]) for r in refs])
    best_idx = int(np.argmin(distances))
    best_ref = refs[best_idx]
    
    if len(refs) >= 2:
        sorted_indices = np.argsort(distances)[:2]
        idx1, idx2 = sorted_indices[0], sorted_indices[1]
        d1, d2 = distances[idx1], distances[idx2]
        
        if d1 < 1e-4:
            quant_value = float(refs[idx1]["val"])
        else:
            w1 = 1.0 / (d1 + 1e-4)
            w2 = 1.0 / (d2 + 1e-4)
            quant_value = float((w1 * refs[idx1]["val"] + w2 * refs[idx2]["val"]) / (w1 + w2))
    else:
        quant_value = float(best_ref["val"])
        
    if meta["unit"] == "SG":
        quant_value = round(quant_value, 3)
    elif meta["unit"] == "pH":
        quant_value = round(quant_value, 1)
    else:
        quant_value = round(quant_value, 1)

    confidence_pct = max(10.0, min(99.0, (1.0 - (distances[best_idx] / 1.8)) * 100.0))
    
    return {
        "biomarker": pad_name,
        "name_en": meta["name_en"],
        "name_ar": meta["name_ar"],
        "key": meta["key"],
        "unit": meta["unit"],
        "quantitative_value": quant_value,
        "matched_band": best_ref["label_en"],
        "matched_band_en": best_ref["label_en"],
        "matched_band_ar": best_ref["label_ar"],
        "risk_tier": best_ref["risk_en"],
        "risk_tier_en": best_ref["risk_en"],
        "risk_tier_ar": best_ref["risk_ar"],
        "normal_range": meta["normal_range_en"],
        "normal_range_en": meta["normal_range_en"],
        "normal_range_ar": meta["normal_range_ar"],
        "description": meta["description_en"],
        "description_en": meta["description_en"],
        "description_ar": meta["description_ar"],
        "confidence": round(confidence_pct, 1),
        "calibration_ref_rgb": best_ref["rgb"]
    }


# =============================================================================
# 5. CLINICAL RATIOS & BILINGUAL KDIGO 2024 CKD DECISION RULES
# =============================================================================

KDIGO_STAGE_DATA = {
    "A1": {
        "stage": "A1",
        "name_en": "Normal to Mildly Increased",
        "name_ar": "طبيعي إلى زيادة طفيفة",
        "risk_en": "LOW RISK - NORMAL FINDING",
        "risk_ar": "خطر منخفض - نتيجة طبيعية",
        "color": "#10B981",
        "badge": "success",
        "summary_en": (
            "No significant albuminuria detected (uACR < 30 mg/g). Glomerular filtration barrier is functionally intact. "
            "Re-screen in 12 months for routine preventive monitoring."
        ),
        "summary_ar": (
            "لم يتم رصد أي تسرب غير طبيعي للزلال (uACR أقل من 30 ملغ/غ). المرشحات الكلوية سليمة وتعمل بكفاءة. "
            "يُنصح بإعادة الفحص الوقائي الدوري بعد 12 شهراً."
        ),
        "actions_en": [
            "Maintain optimal cardiovascular lifestyle and normal blood pressure (< 120/80 mmHg).",
            "Maintain adequate hydration (1.5 - 2.0 L water daily unless contraindicated).",
            "Annual follow-up urine strip checkup."
        ],
        "actions_ar": [
            "الحفاظ على نمط حياة صحي للقلب وضغط دم مثالي (أقل من 120/80 مم زئبق).",
            "شرب كميات كافية من الماء بانتظام (1.5 إلى 2 لتر يومياً ما لم يكن هناك مانع طبي).",
            "إجراء فحص وقائي دوري سنوي لوظائف الكلى."
        ]
    },
    "A2": {
        "stage": "A2",
        "name_en": "Moderately Increased (Microalbuminuria)",
        "name_ar": "زيادة متوسطة (بيلة ألبومينية دقيقة)",
        "risk_en": "MODERATE RISK - MONITORING RECOMMENDED",
        "risk_ar": "خطر متوسط - يُنصح بالمتابعة الطبية",
        "color": "#F59E0B",
        "badge": "warning",
        "summary_en": (
            "Early microalbuminuria detected (uACR 30 - 300 mg/g). This is a critical hallmark of early kidney stress, "
            "often reversible with proactive medical management. Represents increased risk of CKD progression "
            "and cardiovascular events."
        ),
        "summary_ar": (
            "تم رصد زلال دقيق في البول (uACR 30 - 300 ملغ/غ). هذه علامة مبكرة وحرجة على إجهاد المرشحات الكلوية، "
            "وهي مرحلة ذهبية قابلة للشفاء والتراجع مع المتابعة الطبية المنتظمة وضبط السكري وضغط الدم."
        ),
        "actions_en": [
            "Consult primary care physician or nephrologist within 2 to 4 weeks.",
            "Confirm finding with a repeat first-morning spot urine test in 1 - 2 weeks.",
            "Screen for hypertension, tight glycemic control (HbA1c < 7.0%), and review ACEi/ARB medication therapy.",
            "Avoid nephrotoxic agents including frequent over-the-counter NSAIDs (ibuprofen, naproxen)."
        ],
        "actions_ar": [
            "مراجعة طبيب الأسرة أو أخصائي الكلى خلال أسبوعين إلى 4 أسابيع.",
            "تأكيد النتيجة بإعادة الفحص في عينة بول صباحية أولى خلال أسبوع إلى أسبوعين.",
            "فحص ضغط الدم، والتحكم الصارم في سكر الدم التراكمي (HbA1c < 7%) ومراجعة الأدوية الوقائية.",
            "تجنب المسكنات ومضادات الالتهاب غير الستيرويدية (مثل البروفين والفولتارين) التي قد ترهق الكلى."
        ]
    },
    "A3": {
        "stage": "A3",
        "name_en": "Severely Increased (Macroalbuminuria / Nephrotic Range)",
        "name_ar": "زيادة شديدة (بيلة ألبومينية صريحة / نطاق كلوي متقدم)",
        "risk_en": "HIGH RISK - ACTION REQUIRED",
        "risk_ar": "خطر مرتفع - يتطلب استشارة طبية عاجلة",
        "color": "#EF4444",
        "badge": "danger",
        "summary_en": (
            "Marked albuminuria / proteinuria detected (uACR > 300 mg/g or uPCR > 500 mg/g). "
            "High probability of progressive glomerular injury. "
            "Urgent clinical nephrology evaluation and comprehensive renal panel recommended."
        ),
        "summary_ar": (
            "تم رصد تسرب شديد للزلال أو البروتين (uACR > 300 ملغ/غ أو uPCR > 500 ملغ/غ). "
            "يشير ذلك إلى احتمالية عالية لإصابة كبيبية متقدمة تستدعي مراجعة عاجلة لطبيب استشاري أمراض الكلى "
            "وإجراء فحص وظائف الكلى الشامل والموجات الصوتية."
        ),
        "actions_en": [
            "Urgent consultation with a board-certified Nephrologist.",
            "Comprehensive laboratory workup: Serum Creatinine, eGFR, 24-hr urine protein, lipid profile.",
            "Renal ultrasound imaging to evaluate kidney parenchyma, size, and cortico-medullary differentiation.",
            "Strict blood pressure management and renal protective therapy (e.g. SGLT2i / ACEi / ARB under physician guidance)."
        ],
        "actions_ar": [
            "استشارة عاجلة لطبيب استشاري أمراض وزراعة الكلى.",
            "إجراء فحص دم شامل: كرياتينين المصل، معدل الترشيح الكبيبي eGFR، وفحص بروتين البول 24 ساعة.",
            "تصوير الكلى بالموجات فوق الصوتية (السونار) لتقييم حجم النسيج الكلوي والتروية الدموية.",
            "ضبط ضغط الدم بدقة وبدء العلاجات الحامية للكلى (مثل مثبطات SGLT2i أو ACEi/ARB بإشراف الطبيب)."
        ]
    }
}


def calculate_clinical_ratios(biomarker_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes spot uACR and uPCR and stratifies CKD stage according to
    KDIGO 2024 Clinical Practice Guidelines with full English and Arabic diagnostics.
    
    Guarantees both 'kdigo_name_en' and 'kdigo_name_ar' are always present in the returned dictionary.
    """
    # Robust multi-key extraction (handles full names, acronyms, or Arabic keys)
    alb_mg_l = float(
        biomarker_dict.get("Microalbumin") or
        biomarker_dict.get("ALB") or
        biomarker_dict.get("mALB") or
        biomarker_dict.get("alb") or
        biomarker_dict.get("الزلال الدقيق") or
        10.0
    )
    cre_mg_dl = max(10.0, float(
        biomarker_dict.get("Creatinine") or
        biomarker_dict.get("CRE") or
        biomarker_dict.get("cre") or
        biomarker_dict.get("الكرياتينين") or
        100.0
    ))
    pro_mg_dl = float(
        biomarker_dict.get("Total Protein") or
        biomarker_dict.get("PRO") or
        biomarker_dict.get("pro") or
        biomarker_dict.get("البروتين الكلي") or
        0.0
    )
    
    # 1. uACR (mg/g) = (Albumin mg/L * 100) / Creatinine mg/dL
    uacr = round((alb_mg_l * 100.0) / cre_mg_dl, 1)
    # 2. uPCR (mg/g) = (Protein mg/dL / Creatinine mg/dL) * 1000
    upcr = round((pro_mg_dl / cre_mg_dl) * 1000.0, 1)
    
    # KDIGO 2024 criteria: A1 (<30 uACR or <150 uPCR), A2 (30-300 uACR or 150-500 uPCR), A3 (>300 uACR or >500 uPCR)
    is_a3 = (uacr > 300.0) or (upcr >= 500.0)
    is_a2 = (30.0 <= uacr <= 300.0) or (150.0 <= upcr < 500.0)
    
    stage_key = "A3" if is_a3 else ("A2" if is_a2 else "A1")
    s_info = KDIGO_STAGE_DATA[stage_key]

    # Secondary physiological alerts
    secondary_flags = []
    ua_val = float(biomarker_dict.get("Uric Acid") or biomarker_dict.get("UA") or biomarker_dict.get("حمض اليوريك") or 35.0)
    if ua_val >= 75.0:
        secondary_flags.append({
            "param_en": "Uric Acid",
            "param_ar": "حمض اليوريك",
            "val": f"{ua_val} mg/dL",
            "note_en": "Elevated uric acid. Risk of urate nephropathy and renal calculi.",
            "note_ar": "ارتفاع حمض اليوريك. خطر ترسب الحصوات واعتلال الكلى النقرسي."
        })
        
    ph_val = float(biomarker_dict.get("pH") or biomarker_dict.get("الرقم الهيدروجيني") or 6.5)
    if ph_val <= 5.0:
        secondary_flags.append({
            "param_en": "pH (Acidity)",
            "param_ar": "درجة الحموضة",
            "val": f"{ph_val}",
            "note_en": "Acidic urine. Increases uric acid and cystine crystallization.",
            "note_ar": "بول شديد الحموضة. يزيد من قابلية تبلور حصوات حمض اليوريك والسيستين."
        })
    elif ph_val >= 8.0:
        secondary_flags.append({
            "param_en": "pH (Acidity)",
            "param_ar": "درجة الحموضة",
            "val": f"{ph_val}",
            "note_en": "Alkaline urine. Rule out urinary tract infection or renal tubular acidosis.",
            "note_ar": "بول قلوي. ينبغي استبعاد التهابات المسالك البولية أو الحماض الأنبوبي."
        })
        
    sg_val = float(biomarker_dict.get("Specific Gravity") or biomarker_dict.get("SG") or biomarker_dict.get("الكثافة النوعية") or 1.015)
    if 1.008 <= sg_val <= 1.012:
        secondary_flags.append({
            "param_en": "Specific Gravity",
            "param_ar": "الكثافة النوعية",
            "val": f"{sg_val}",
            "note_en": "Isosthenuria (fixed SG ~1.010). Consider chronic renal concentrating defect if persistent.",
            "note_ar": "ثبوت الكثافة النوعية (~1.010). قد يشير في حال استمراره لضعف قدرة الأنابيب على تركيز البول."
        })

    return {
        "uACR": uacr,
        "uPCR": upcr,
        "kdigo_stage": s_info["stage"],
        
        # Primary bilingual name keys required by app.py
        "kdigo_name_en": s_info["name_en"],
        "kdigo_name_ar": s_info["name_ar"],
        "kdigo_name": s_info["name_en"],
        
        # Primary bilingual risk keys
        "kdigo_risk_en": s_info["risk_en"],
        "kdigo_risk_ar": s_info["risk_ar"],
        "kdigo_risk": s_info["risk_en"],
        
        # Primary bilingual summary keys
        "kdigo_summary_en": s_info["summary_en"],
        "kdigo_summary_ar": s_info["summary_ar"],
        "kdigo_summary": s_info["summary_en"],
        
        # Primary bilingual action plan keys
        "clinical_actions_en": s_info["actions_en"],
        "clinical_actions_ar": s_info["actions_ar"],
        "clinical_actions": s_info["actions_en"],
        
        # Color & Badge styling
        "kdigo_color": s_info["color"],
        "kdigo_badge": s_info["badge"],
        
        # Secondary physiological alerts
        "secondary_flags": secondary_flags
    }


# Clinical decision aliases
evaluate_clinical_risk = calculate_clinical_ratios
stratify_patient_risk = calculate_clinical_ratios
calculate_kdigo_risk = calculate_clinical_ratios


# =============================================================================
# 6. FULL COMPUTER VISION PIPELINE INTEGRATOR
# =============================================================================

def process_test_strip_image(
    image_bgr: np.ndarray,
    manual_crop_pct: Optional[Tuple[float, float, float, float]] = None
) -> Dict[str, Any]:
    """End-to-end computer vision and bilingual biomedical inference pipeline."""
    annotated_bgr, pad_measurements, telemetry = detect_and_segment_strip(
        image_bgr, manual_crop_pct=manual_crop_pct
    )
    
    quant_results = []
    biomarker_dict = {}
    
    for pad in pad_measurements:
        name = pad["name"]
        hsv_tuple = pad["mean_hsv"]
        mapped = map_pad_to_concentration(name, hsv_tuple)
        pad_full = {**pad, **mapped}
        quant_results.append(pad_full)
        biomarker_dict[name] = mapped["quantitative_value"]
        
    clinical_evaluation = calculate_clinical_ratios(biomarker_dict)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    return {
        "success": True,
        "telemetry": telemetry,
        "biomarkers": quant_results,
        "biomarker_dict": biomarker_dict,
        "clinical_evaluation": clinical_evaluation,
        "annotated_image": annotated_rgb
    }
