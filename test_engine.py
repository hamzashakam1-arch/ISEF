import cv2
import cv_engine
from cv_engine import (
    process_test_strip_image,
    calculate_clinical_ratios,
    evaluate_clinical_risk,
    stratify_patient_risk
)

# 1. Test standalone clinical risk analysis functions
sample_biomarkers = {"Microalbumin": 25.0, "Creatinine": 100.0, "Total Protein": 10.0}
for func, fname in [(calculate_clinical_ratios, "calculate_clinical_ratios"), 
                    (evaluate_clinical_risk, "evaluate_clinical_risk"), 
                    (stratify_patient_risk, "stratify_patient_risk")]:
    clin_dict = func(sample_biomarkers)
    assert "kdigo_name_en" in clin_dict, f"Missing kdigo_name_en in {fname}"
    assert "kdigo_name_ar" in clin_dict, f"Missing kdigo_name_ar in {fname}"
    assert "kdigo_risk_en" in clin_dict, f"Missing kdigo_risk_en in {fname}"
    assert "kdigo_risk_ar" in clin_dict, f"Missing kdigo_risk_ar in {fname}"
    print(f"Verified {fname}: kdigo_name_en='{clin_dict['kdigo_name_en']}', keys present.")

# 2. Test full image pipeline
files = ['case1_normal_a1.png', 'case2_microalbuminuria_a2.png', 'case3_severe_ckd_a3.png']
for filename in files:
    img = cv2.imread(f'sample_strips/{filename}')
    res = process_test_strip_image(img)
    eval_res = res['clinical_evaluation']
    
    assert "kdigo_name_en" in eval_res, "kdigo_name_en missing from image pipeline evaluation!"
    assert "kdigo_name_ar" in eval_res, "kdigo_name_ar missing from image pipeline evaluation!"
    
    print(f"\n==================== {filename} ====================")
    print(f"uACR: {eval_res['uACR']} mg/g | uPCR: {eval_res['uPCR']} mg/g")
    print(f"KDIGO Classification (EN): {eval_res['kdigo_stage']} - {eval_res['kdigo_name_en']} [{eval_res['kdigo_risk_en']}]")
    print(f"Mean Intra-Pad RSD: {res['telemetry']['mean_strip_rsd']}% (Target: < 5%)")
    for b in res['biomarkers']:
        print(f"  [{b['key']}] {b['name_en']}: {b['quantitative_value']} {b['unit']} | Conf: {b['confidence']}% | RSD: {b['rsd_percent']}%")
print("\nALL CLINICAL RISK DICTIONARY KEY TESTS PASSED!")
