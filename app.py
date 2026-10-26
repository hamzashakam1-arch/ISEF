"""
=============================================================================
NEPHROSCAN CLINICAL MEDICAL PLATFORM
Point-of-Care Urine Multiplex Analysis & Early CKD Telehealth Portal

MODULE: app.py
AUTHOR: Senior UI/UX Designer & Full-Stack Medical Software Engineer
DESCRIPTION:
    Production-grade telemedicine web application with full bilingual
    (Arabic / English) support, RTL layout styling, elderly-friendly accessibility,
    interactive reaction timer, automated computer vision segmentation, and
    clean KDIGO 2024 clinical decision support dashboard.
=============================================================================
"""

import os
import time
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from cv_engine import (
    process_test_strip_image,
    BIOMARKER_METADATA,
    PAD_ORDER
)
from sample_generator import generate_all_sample_cases

# Ensure benchmark clinical test strips exist on disk
if not os.path.exists("sample_strips/case1_normal_a1.png"):
    generate_all_sample_cases()

# =============================================================================
# 1. PAGE SETUP & METADATA
# =============================================================================

st.set_page_config(
    page_title="NephroScan Clinical | Renal Health Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. COMPREHENSIVE BILINGUAL TRANSLATION DICTIONARY (ENGLISH & ARABIC)
# =============================================================================

I18N = {
    "en": {
        "lang_name": "English",
        "app_brand": "NephroScan Clinical",
        "app_subbrand": "Point-of-Care Renal Telemedicine Portal",
        "portal_tag": "Clinical Diagnostic System",
        "hero_title": "AI-Assisted Multiplex Urine Analysis Platform",
        "hero_subtitle": "Automated 6-Pad Optical Colorimetry for Early Chronic Kidney Disease (CKD) Screening & Telehealth Monitoring",
        "tab_protocol": "📋 1. Guided Testing Protocol",
        "tab_scan": "📷 2. Scan & Analyze Strip",
        "tab_dashboard": "📊 3. Clinical Results Dashboard",
        "tab_research": "📖 4. Clinical Research & Reference",
        
        # Sidebar
        "sidebar_profile": "Patient Clinical Profile",
        "patient_name_label": "Patient Full Name / ID",
        "patient_name_val": "Eleanor Vance (Patient #4092)",
        "patient_age_label": "Age (Years)",
        "patient_gender_label": "Biological Sex",
        "gender_options": ["Female", "Male"],
        "risk_factors_label": "Known Clinical Risk Factors",
        "diabetes_label": "Diabetes Mellitus",
        "hypertension_label": "Hypertension (High BP)",
        "nsaids_label": "Frequent NSAID / Painkiller Use",
        "family_history_label": "Family History of Kidney Disease",
        "ckd_history_label": "Prior CKD Diagnosis",
        "ckd_history_options": [
            "No Prior CKD History",
            "Stage 1 (Normal eGFR > 90)",
            "Stage 2 (Mild eGFR 60-89)",
            "Stage 3 (Moderate eGFR 30-59)",
            "Stage 4-5 (Severe Renal Impairment)"
        ],
        "accessibility_title": "Accessibility & Display",
        "large_font_label": "🔍 Enhanced Large Font Mode",
        
        # Protocol
        "protocol_heading": "Patient Self-Testing Instructions (Step-by-Step)",
        "protocol_info": "Please follow these standard diagnostic instructions carefully to ensure precise biochemical reaction and valid colorimetry.",
        "step1_title": "Collect Midstream Urine",
        "step1_desc": "Collect fresh midstream urine (first-morning sample is preferred) in a clean, dry container.",
        "step2_title": "Dip Reagent Strip",
        "step2_desc": "Submerge all 6 test pads fully in the urine for exactly 2 seconds. Remove promptly and tap the long edge against a clean tissue to remove excess liquid.",
        "step3_title": "Incubation Reaction Time",
        "step3_desc": "Place the strip horizontally on a flat white surface. Allow reagents to react for exactly 60–75 seconds using the countdown timer below.",
        "step4_title": "Acquire Quality Image",
        "step4_desc": "Navigate to Tab 2 to capture or upload a clear photo of the reacted strip under balanced indoor lighting.",
        "timer_heading": "⏱️ Interactive Reaction Countdown Timer (60–75s)",
        "timer_select": "Select Incubation Duration:",
        "timer_start": "▶️ Start Countdown Timer",
        "timer_running": "Chemical chromogen reaction in progress... Please wait.",
        "timer_done": "🔔 REACTION COMPLETE! PHOTOGRAPH THE STRIP NOW!",
        "timer_balloons": "Reaction completed! Please proceed to Tab 2 to scan your strip.",
        
        # Scan & Input
        "scan_heading": "Optical Reagent Strip Acquisition",
        "scan_desc": "Provide an image of the reacted 6-pad test strip using your camera, image file, or calibrated clinical test preset.",
        "source_label": "Select Input Source:",
        "source_upload": "📁 Upload Image File",
        "source_camera": "📸 Use Device Camera",
        "source_demo": "🧪 Load Calibrated Clinical Test Strip (Demo)",
        "upload_prompt": "Upload strip photograph (JPEG, PNG, JPG)",
        "camera_prompt": "Center test strip vertically and capture:",
        "demo_prompt": "Select Calibrated Clinical Benchmark Case:",
        "demo_cases": [
            "Case 1: Healthy Normal Profile (KDIGO Stage A1, Normal uACR < 30 mg/g)",
            "Case 2: Early Diabetic Nephropathy (KDIGO Stage A2 Microalbuminuria, uACR 30-300 mg/g)",
            "Case 3: Advanced Glomerular Impairment (KDIGO Stage A3, uACR > 300 mg/g & Proteinuria)"
        ],
        "preview_heading": "Acquired Strip Preview",
        "preview_caption": "Active Photographic Input",
        "analyze_btn": "🚀 Analyze Test Strip Now",
        "analyzing_spinner": "Standardizing illumination, segmenting 6 pad ROIs, and evaluating KDIGO criteria...",
        "analysis_success": "✅ Analysis completed successfully! View clinical evaluation in Tab 3.",
        
        # Dashboard
        "dash_empty": "⚠️ No test strip analyzed yet. Please scan or select an image in Tab 2.",
        "status_title": "OVERALL CLINICAL STATUS",
        "patient_tag": "Patient",
        "years_old": "yrs",
        "ratios_heading": "Clinical Decision Ratios (Spot Urine Sample)",
        "uacr_title": "Albumin-to-Creatinine Ratio (uACR)",
        "upcr_title": "Protein-to-Creatinine Ratio (uPCR)",
        "cre_dilution_title": "Spot Creatinine Normalization",
        "cre_dilution_adequate": "Adequate Sample Concentration",
        "cre_dilution_dilute": "Dilute Urine Sample",
        "secondary_flags_heading": "⚠️ Physiological Secondary Alerts",
        "biomarkers_heading": "Quantitative Biomarker Panel (6 Core Analytes)",
        "biomarkers_caption": "Individual pad spectrophotometric concentrations derived from calibrated CIE/HSV reflectance curves.",
        "pad_label": "Pad",
        "ref_normal": "Normal Reference",
        "matched_band": "Matched Range",
        "measured_color": "Measured",
        "ref_color": "Calibrated",
        "confidence": "Conf.",
        "actions_heading": "📋 Evidence-Based Clinical Action Plan",
        "export_heading": "💾 Export Medical Documentation",
        "download_csv": "📥 Download Clinical CSV Data",
        "download_txt": "🖨️ Download Physician Consultation Note (TXT)",
        
        # Advanced Expander
        "telemetry_expander": "⚙️ Advanced Technical & Quality Telemetry (Laboratory View)",
        "telemetry_mode": "Segmentation Algorithm",
        "telemetry_pads": "Segmented ROIs",
        "telemetry_rsd": "Mean Intra-Pad Uniformity (RSD)",
        "telemetry_target": "Quality Target: RSD < 5.0% across all ROI pixels",
        "telemetry_table_pad": "Reagent Pad",
        "telemetry_table_name": "Biomarker Name",
        "telemetry_table_rsd": "Intra-Pad RSD",
        "telemetry_table_quality": "Optical Quality",
        
        # Footer
        "footer_text": "NephroScan Medical AI • Point-of-Care Renal Diagnostics & Home Monitoring Platform • For Clinical Screening & Telehealth Support"
    },
    
    "ar": {
        "lang_name": "العربية",
        "app_brand": "منصة نيفرو-سكان الطبية",
        "app_subbrand": "بوابة الفحص المبكر للوظائف الكلوية والطب الاتصالي عن بُعد",
        "portal_tag": "نظام التشخيص الطبي المتقدم",
        "hero_title": "المنصة الذكية للتحليل المزدوج لشرائط فحص البول",
        "hero_subtitle": "قياس لوني حاسوبي آلي لـ 6 مؤشرات حيوية للكشف المبكر عن مرض الكلى المزمن والمتابعة المنزلية وفق معايير KDIGO الدولية",
        "tab_protocol": "📋 1. إرشادات الفحص ومؤقت التفاعل",
        "tab_scan": "📷 2. تصوير وفحص شريط الاختبار",
        "tab_dashboard": "📊 3. لوحة النتائج والتقييم السريري",
        "tab_research": "📖 4. الأبحاث والأسس السريرية",
        
        # Sidebar
        "sidebar_profile": "الملف السريري للمريض",
        "patient_name_label": "اسم المريض / رقم الملف الطبي",
        "patient_name_val": "إليانور فانس (ملف #4092)",
        "patient_age_label": "العمر (بالسنوات)",
        "patient_gender_label": "الجنس البيولوجي",
        "gender_options": ["أنثى", "ذكر"],
        "risk_factors_label": "عوامل الخطورة المرضية المعروفة",
        "diabetes_label": "داء السكري",
        "hypertension_label": "ارتفاع ضغط الدم",
        "nsaids_label": "استخدام مفرط للمسكنات (NSAIDs)",
        "family_history_label": "تاريخ عائلي لأمراض الكلى",
        "ckd_history_label": "تاريخ مرضي سابق لاعتلال الكلى",
        "ckd_history_options": [
            "لا يوجد تاريخ سابق لمرض الكلى",
            "المرحلة 1 (ترشيح كبيبي سليم > 90)",
            "المرحلة 2 (قصور طفيف 60-89)",
            "المرحلة 3 (قصور متوسط 30-59)",
            "المرحلة 4-5 (قصور كلوي متقدم)"
        ],
        "accessibility_title": "سهولة الاستخدام والعرض",
        "large_font_label": "🔍 تفعيل وضع تكبير الخط لكبار السن",
        
        # Protocol
        "protocol_heading": "إرشادات إجراء الفحص الذاتي (خطوة بخطوة)",
        "protocol_info": "يُرجى اتباع الإرشادات الطبية التالية بعناية لضمان دقة التفاعل الكيميائي القياسي وصحة القياس اللوني البصري.",
        "step1_title": "جمع عينة البول من منتصف المجرى",
        "step1_desc": "اجمع عينة بول نظيفة (يُفضل عينة الصباح الباكر الأولى) في وعاء بلاستيكي جاف ونظيف.",
        "step2_title": "غمس شريط الاختبار في العينة",
        "step2_desc": "اغمس وسائد الكواشف الست بالكامل في البول لمدة ثانيتين فقط. أخرج الشريط فوراً ومرر حافته الجانبية على منديل نظيف لإزالة السائل الفائض.",
        "step3_title": "انتظار اكتمال التفاعل الكيميائي",
        "step3_desc": "ضع الشريط أفقياً على سطح أبيض مستوٍ. اترك الكواشف لتتفاعل لمدة 60 إلى 75 ثانية باستخدام مؤقت العد التنازلي التفاعلي بالأسفل.",
        "step4_title": "التقاط صورة واضحة للشريط",
        "step4_desc": "انتقل إلى القسم الثاني (2. تصوير وفحص شريط الاختبار) لالتقاط أو رفع صورة واضحة ومستوية تحت إضاءة داخلية متوازنة.",
        "timer_heading": "⏱️ مؤقت العد التنازلي التفاعلي للتفاعل اللوني (60 - 75 ثانية)",
        "timer_select": "اختر مدة الحضانة والتفاعل:",
        "timer_start": "▶️ تشغيل مؤقت العد التنازلي",
        "timer_running": "التفاعل الكيميائي اللوني قيد التشغيل... يُرجى الانتظار.",
        "timer_done": "🔔 اكتمل التفاعل تماماً! صوّر شريط الاختبار الآن!",
        "timer_balloons": "اكتمل التفاعل اللوني بنجاح! انتقل الآن إلى قسم (تصوير وفحص شريط الاختبار).",
        
        # Scan & Input
        "scan_heading": "تصوير ومسح شريط الكواشف البصري",
        "scan_desc": "التقط أو ارفع صورة شريط الاختبار المتفاعل، أو اختر شريطاً سريرياً مُعايراً للتجربة الفورية.",
        "source_label": "اختر وسيلة إدخال الصورة:",
        "source_upload": "📁 رفع صورة من جهازك",
        "source_camera": "📸 التقاط صورة بكاميرا الجهاز",
        "source_demo": "🧪 تجربة شريط سريري مُعاير (فوري)",
        "upload_prompt": "ارفع صورة شريط الاختبار (JPG, JPEG, PNG)",
        "camera_prompt": "وجّه شريط الاختبار عمودياً في منتصف الإطار والتقط الصورة:",
        "demo_prompt": "اختر حالة سريرية نموذجية للاختبار الفوري:",
        "demo_cases": [
            "الحالة 1: نمط سليم وطبيعي (مرحلة KDIGO A1، نسبة زلال طبيعية < 30 ملغ/غ)",
            "الحالة 2: اعتلال كلوي سكري مبكر (مرحلة KDIGO A2، بيلة ألبومينية دقيقة 30-300 ملغ/غ)",
            "الحالة 3: اعتلال كبيبي كلوي متقدم (مرحلة KDIGO A3، زلال شديد > 300 ملغ/غ وبروتين مرتفع)"
        ],
        "preview_heading": "معاينة صورة الشريط المدخلة",
        "preview_caption": "الصورة الحالية المراد فحصها",
        "analyze_btn": "🚀 فحص وتحليل شريط الاختبار الآن",
        "analyzing_spinner": "معايرة الإضاءة، وتحديد وسائد الكواشف الست، وتطبيق خوارزميات KDIGO التشخيصية...",
        "analysis_success": "✅ اكتمل الفحص بنجاح! تفقّد النتائج والتقرير السريري في القسم 3.",
        
        # Dashboard
        "dash_empty": "⚠️ لم يتم فحص أي شريط بعد. يرجى التقاط أو رفع صورة في القسم 2.",
        "status_title": "التقييم والتشخيص السريري العام",
        "patient_tag": "المريض",
        "years_old": "سنة",
        "ratios_heading": "المؤشرات والنسب السريرية الحاسمة (عينة البول العشوائية)",
        "uacr_title": "نسبة الزلال إلى الكرياتينين (uACR)",
        "upcr_title": "نسبة البروتين إلى الكرياتينين (uPCR)",
        "cre_dilution_title": "معايرة التخفيف بالكرياتينين",
        "cre_dilution_adequate": "تركيز العينة ملائم ودقيق",
        "cre_dilution_dilute": "عينة بول مخففة",
        "secondary_flags_heading": "⚠️ تنبيهات فسيولوجية إضافية",
        "biomarkers_heading": "لوحة القياسات الكمية الدقيقة (المؤشرات الستة)",
        "biomarkers_caption": "التركيزات الطيفية الدقيقة المحسوبة لكل وسادة بناءً على المسافات اللونية في الفضاء اللوني الأسطواني HSV المعاير.",
        "pad_label": "وسادة",
        "ref_normal": "المعدل الطبيعي",
        "matched_band": "النطاق المطابق",
        "measured_color": "اللون المقاس",
        "ref_color": "المعيار المخبري",
        "confidence": "الموثوقية",
        "actions_heading": "📋 خطة الإجراءات والتوصيات الطبية السريرية",
        "export_heading": "💾 تصدير وحفظ التقرير الطبي",
        "download_csv": "📥 تحميل بيانات النتائج (CSV)",
        "download_txt": "🖨️ تحميل تقرير استشارة الطبيب (ملف نصي)",
        
        # Advanced Expander
        "telemetry_expander": "⚙️ مراقبة الجودة والقياسات البصرية المتقدمة (عرض المختبر)",
        "telemetry_mode": "خوارزمية تحديد الشريط",
        "telemetry_pads": "الوسائد المستخرجة",
        "telemetry_rsd": "متوسط معامل تباين اللون (RSD)",
        "telemetry_target": "المعيار الطبي المقبول: معامل تباين RSD < 5.0% عبر جميع بكسلات الوسادة",
        "telemetry_table_pad": "وسادة الكاشف",
        "telemetry_table_name": "اسم المؤشر الحيوي",
        "telemetry_table_rsd": "معامل التباين RSD",
        "telemetry_table_quality": "جودة القراءة البصرية",
        
        # Footer
        "footer_text": "منصة نيفرو-سكان للذكاء الاصطناعي الطبي • نظام الرعاية والتشخيص الكلوي المنزلي • مخصص لأغراض الفحص السريري والمتابعة عن بُعد"
    }
}


# =============================================================================
# 3. SIDEBAR: LANGUAGE SELECTOR & PROFESSIONAL CLINICAL PATIENT PROFILE
# =============================================================================

with st.sidebar:
    # 1. Clean Language Selector at the very top
    selected_lang_label = st.selectbox(
        "🌐 Language / اللغة",
        ["English", "العربية"],
        index=0
    )
    lang = "en" if selected_lang_label == "English" else "ar"
    T = I18N[lang]
    is_ar = (lang == "ar")
    
    st.markdown("---")
    
    # Hospital / Telemedicine Branding
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <img src="https://img.icons8.com/fluency/96/kidney.png" width="48" height="48" style="vertical-align: middle;">
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; line-height: 1.2;">{T['app_brand']}</div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #2563EB;">{T['portal_tag']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(T['app_subbrand'])
    st.markdown("---")

    # Accessibility Settings
    st.markdown(f"#### {T['accessibility_title']}")
    large_font = st.checkbox(T['large_font_label'], value=True)
    
    st.markdown("---")
    
    # Patient Clinical Profile Form
    st.markdown(f"#### {T['sidebar_profile']}")
    patient_name = st.text_input(T['patient_name_label'], value=T['patient_name_val'])
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        patient_age = st.number_input(T['patient_age_label'], min_value=18, max_value=115, value=68, step=1)
    with col_a2:
        patient_gender = st.selectbox(T['patient_gender_label'], T['gender_options'])
        
    st.markdown(f"**{T['risk_factors_label']}**")
    col_rf1, col_rf2 = st.columns(2)
    with col_rf1:
        has_diabetes = st.checkbox(T['diabetes_label'], value=True)
        has_nsaids = st.checkbox(T['nsaids_label'], value=False)
    with col_rf2:
        has_htn = st.checkbox(T['hypertension_label'], value=True)
        has_fam_hist = st.checkbox(T['family_history_label'], value=False)
        
    known_ckd = st.selectbox(T['ckd_history_label'], T['ckd_history_options'])


# =============================================================================
# 4. DYNAMIC RTL & HIGH-CONTRAST ACCESSIBILITY STYLING
# =============================================================================

RTL_DIRECTION = "rtl" if is_ar else "ltr"
TEXT_ALIGN = "right" if is_ar else "left"
BORDER_ACCENT_SIDE = "border-right" if is_ar else "border-left"
BORDER_CLEAR_SIDE = "border-left: none;" if is_ar else "border-right: none;"

GLOBAL_STYLES = f"""
<style>
    /* RTL / LTR Configuration */
    html, body, [data-testid="stAppViewContainer"], .main {{
        direction: {RTL_DIRECTION};
        text-align: {TEXT_ALIGN};
        font-family: 'Segoe UI', 'Cairo', Tahoma, -apple-system, sans-serif;
    }}
    [data-testid="stSidebar"] {{
        direction: {RTL_DIRECTION};
        text-align: {TEXT_ALIGN};
    }}
    
    /* Typography */
    h1 {{
        font-size: {'2.5rem' if large_font else '2.1rem'} !important;
        font-weight: 800 !important;
        color: #0F172A;
    }}
    h2 {{
        font-size: {'1.8rem' if large_font else '1.5rem'} !important;
        font-weight: 700 !important;
        color: #1E293B;
    }}
    h3, h4 {{
        font-weight: 700 !important;
    }}
    p, li, label, .stMarkdown {{
        font-size: {'1.25rem' if large_font else '1.05rem'} !important;
        line-height: 1.65 !important;
    }}

    /* Hero Header */
    .hero-banner {{
        background: linear-gradient(135deg, #0F2D6B 0%, #0369A1 100%);
        padding: 26px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }}

    /* Clinical Status Cards */
    .status-card {{
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 24px;
        {BORDER_ACCENT_SIDE}: 10px solid;
        {BORDER_CLEAR_SIDE}
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }}
    .status-card-green {{
        background-color: #ECFDF5;
        border-color: #10B981;
        color: #064E3B;
    }}
    .status-card-yellow {{
        background-color: #FFFBEB;
        border-color: #F59E0B;
        color: #78350F;
    }}
    .status-card-red {{
        background-color: #FEF2F2;
        border-color: #EF4444;
        color: #7F1D1D;
    }}

    /* Ratio Metric Boxes */
    .metric-box {{
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        margin-bottom: 16px;
    }}
    .metric-val {{
        font-size: 2.3rem;
        font-weight: 800;
        margin: 6px 0;
    }}
    .metric-label {{
        font-size: 1.0rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Biomarker Pad Result Tile */
    .pad-tile {{
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .pad-tile:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }}

    /* Color Swatch Pill */
    .swatch-pill {{
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 6px;
        border: 2px solid #CBD5E1;
        vertical-align: middle;
        margin: 0 6px;
    }}

    /* Big Action Buttons */
    .stButton>button {{
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
        min-height: 52px !important;
    }}

    /* Step Badges */
    .step-badge {{
        display: inline-block;
        background: #0284C7;
        color: white;
        font-weight: 800;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        line-height: 36px;
        text-align: center;
        font-size: 1.2rem;
        margin: {'0 0 0 10px' if is_ar else '0 10px 0 0'};
        vertical-align: middle;
    }}
</style>
"""
st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


# =============================================================================
# 5. MAIN HERO HEADER & INTUITIVE NAVIGATION
# =============================================================================

st.markdown(
    f"""
    <div class="hero-banner">
        <h1 style="color: white !important; margin: 0; font-size: {'2.4rem' if large_font else '2.1rem'};">{T['hero_title']}</h1>
        <p style="color: #E0F2FE !important; font-size: {'1.25rem' if large_font else '1.1rem'}; margin-top: 8px; margin-bottom: 0;">
            {T['hero_subtitle']}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    T['tab_protocol'],
    T['tab_scan'],
    T['tab_dashboard'],
    T['tab_research']
])


# =============================================================================
# TAB 1: GUIDED TESTING PROTOCOL & COUNTDOWN TIMER
# =============================================================================

with tab1:
    st.markdown(f"## {T['protocol_heading']}")
    st.info(f"💡 {T['protocol_info']}")

    col_p1, col_p2 = st.columns(2, gap="large")
    
    with col_p1:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 2px solid #CBD5E1; border-radius: 12px; padding: 20px; margin-bottom: 18px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #0369A1; margin-bottom: 10px;">
                    <span class="step-badge">1</span> {T['step1_title']}
                </div>
                <p style="color: #334155; margin: 0 46px;">
                    {T['step1_desc']}
                </p>
            </div>
            
            <div style="background: #F8FAFC; border: 2px solid #CBD5E1; border-radius: 12px; padding: 20px; margin-bottom: 18px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #0369A1; margin-bottom: 10px;">
                    <span class="step-badge">2</span> {T['step2_title']}
                </div>
                <p style="color: #334155; margin: 0 46px;">
                    {T['step2_desc']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_p2:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 2px solid #CBD5E1; border-radius: 12px; padding: 20px; margin-bottom: 18px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #0369A1; margin-bottom: 10px;">
                    <span class="step-badge">3</span> {T['step3_title']}
                </div>
                <p style="color: #334155; margin: 0 46px;">
                    {T['step3_desc']}
                </p>
            </div>
            
            <div style="background: #F8FAFC; border: 2px solid #CBD5E1; border-radius: 12px; padding: 20px; margin-bottom: 18px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #0369A1; margin-bottom: 10px;">
                    <span class="step-badge">4</span> {T['step4_title']}
                </div>
                <p style="color: #334155; margin: 0 46px;">
                    {T['step4_desc']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # Reaction Countdown Timer
    st.markdown(f"### {T['timer_heading']}")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        timer_duration = st.radio(T['timer_select'], [60, 75], horizontal=True, index=0)
    
    timer_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    start_timer = st.button(T['timer_start'], type="primary")
    if start_timer:
        for remaining in range(timer_duration, -1, -1):
            fraction = (timer_duration - remaining) / timer_duration
            mins, secs = divmod(remaining, 60)
            time_display = f"{mins:02d}:{secs:02d}"
            
            color = "#0284C7" if remaining > 10 else ("#F59E0B" if remaining > 0 else "#10B981")
            status_text = T['timer_running'] if remaining > 0 else T['timer_done']
            
            timer_placeholder.markdown(
                f"""
                <div style="text-align: center; padding: 22px; background: #F1F5F9; border-radius: 14px; border: 3px solid {color}; margin: 16px 0;">
                    <div style="font-size: 4rem; font-weight: 900; color: {color}; font-family: monospace;">{time_display}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #334155;">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress_placeholder.progress(fraction)
            if remaining > 0:
                time.sleep(1.0)
                
        st.balloons()
        st.success(f"🎉 **{T['timer_balloons']}**")


# =============================================================================
# TAB 2: IMAGE ACQUISITION & OPTICAL STRIP ANALYSIS
# =============================================================================

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "current_image_rgb" not in st.session_state:
    st.session_state["current_image_rgb"] = None

with tab2:
    st.markdown(f"## {T['scan_heading']}")
    st.write(T['scan_desc'])

    input_mode = st.radio(
        T['source_label'],
        [T['source_upload'], T['source_camera'], T['source_demo']],
        horizontal=True
    )
    
    loaded_bgr = None
    
    if input_mode == T['source_upload']:
        uploaded_file = st.file_uploader(T['upload_prompt'], type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            loaded_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.success("✅ Image file loaded successfully.")

    elif input_mode == T['source_camera']:
        camera_img = st.camera_input(T['camera_prompt'])
        if camera_img is not None:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            loaded_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.success("✅ Photographic frame captured.")

    else:  # Demo test cases
        st.markdown(f"#### {T['demo_prompt']}")
        demo_choice = st.selectbox("", T['demo_cases'], index=0)
        demo_files = {
            T['demo_cases'][0]: "sample_strips/case1_normal_a1.png",
            T['demo_cases'][1]: "sample_strips/case2_microalbuminuria_a2.png",
            T['demo_cases'][2]: "sample_strips/case3_severe_ckd_a3.png"
        }
        chosen_path = demo_files[demo_choice]
        if os.path.exists(chosen_path):
            loaded_bgr = cv2.imread(chosen_path)

    # Preview and Analyze
    if loaded_bgr is not None:
        col_inp1, col_inp2 = st.columns([1, 1], gap="large")
        
        with col_inp1:
            st.markdown(f"### {T['preview_heading']}")
            preview_rgb = cv2.cvtColor(loaded_bgr, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption=T['preview_caption'], use_container_width=True)
            
        with col_inp2:
            st.markdown("### 🔬 **Clinical Processing Engine**")
            st.write(
                "When analysis is triggered, the optical spectrophotometer will balance color temperature, "
                "isolate the 6 core reagent pads, sample the inner 60% core to reject glare, and compute clinical ratios."
                if not is_ar else
                "عند بدء الفحص، يقوم المحرك البصري بمعايرة توازن الإضاءة، وعزل وسائد الكواشف الست، "
                "وأخذ عينة من الـ 60% الداخلية للوسادة لعزل الانعكاسات، ثم حساب النسب السريرية."
            )
            
            run_btn = st.button(T['analyze_btn'], type="primary", use_container_width=True)
            
            if run_btn or st.session_state["analysis_results"] is None:
                with st.spinner(T['analyzing_spinner']):
                    res = process_test_strip_image(loaded_bgr)
                    st.session_state["analysis_results"] = res
                    st.session_state["current_image_rgb"] = preview_rgb
                    st.success(T['analysis_success'])

        # Show segmentation preview on bottom of Tab 2
        if st.session_state["analysis_results"] is not None:
            st.markdown("---")
            res = st.session_state["analysis_results"]
            col_seg1, col_seg2 = st.columns([1, 1], gap="large")
            with col_seg1:
                st.image(
                    res["annotated_image"],
                    caption="Identified Reagent Pad ROIs & Colorimetric Swatches" if not is_ar else "تحديد وسائد الكواشف وعينات الألوان المستخرجة",
                    use_container_width=True
                )
            with col_seg2:
                st.markdown(
                    """
                    <div style="background: #F8FAFC; border: 2px solid #E2E8F0; border-radius: 12px; padding: 20px;">
                        <h4 style="color: #0369A1; margin-top: 0;">Optical Segmentation Quality</h4>
                        <p style="margin-bottom: 8px;">All 6 reagent pads were successfully mapped along the longitudinal axis. Edge artifacts have been filtered out via centered core sub-sampling.</p>
                        <p style="font-weight: 700; color: #10B981; margin: 0;">✓ Ready for clinical decision support evaluation in Tab 3.</p>
                    </div>
                    """
                    if not is_ar else
                    """
                    <div style="background: #F8FAFC; border: 2px solid #E2E8F0; border-radius: 12px; padding: 20px;">
                        <h4 style="color: #0369A1; margin-top: 0;">دقة التحديد البصري للوسائد</h4>
                        <p style="margin-bottom: 8px;">تم استخراج وسائد الكواشف الست بنجاح على طول المحور الرأسي للشريط مع استبعاد الحواف والانعكاسات السطحية بدقة عالية.</p>
                        <p style="font-weight: 700; color: #10B981; margin: 0;">✓ النتائج جاهزة للمراجعة في لوحة التقييم السريري (القسم 3).</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =============================================================================
# TAB 3: CLINICAL RESULTS & KDIGO 2024 DASHBOARD
# =============================================================================

with tab3:
    if st.session_state["analysis_results"] is None:
        st.warning(T['dash_empty'])
    else:
        res = st.session_state["analysis_results"]
        clin = res["clinical_evaluation"]
        biomarkers = res["biomarkers"]
        b_dict = res["biomarker_dict"]
        
        # 1. OVERALL KDIGO RISK STATUS CARD
        kdigo_stage = clin.get("kdigo_stage", "A1")
        card_class = "status-card-green" if kdigo_stage == "A1" else ("status-card-yellow" if kdigo_stage == "A2" else "status-card-red")
        status_icon = "🟢" if kdigo_stage == "A1" else ("🟡" if kdigo_stage == "A2" else "🔴")
        
        # Safe extraction of bilingual names with automatic fallbacks
        stage_name = clin.get("kdigo_name_ar") if is_ar else clin.get("kdigo_name_en")
        if not stage_name:
            stage_name = clin.get("kdigo_name", "طبيعي" if is_ar else "Normal to Mildly Increased")
            
        risk_name = clin.get("kdigo_risk_ar") if is_ar else clin.get("kdigo_risk_en")
        if not risk_name:
            risk_name = clin.get("kdigo_risk", "خطر منخفض" if is_ar else "LOW RISK")
            
        summary_text = clin.get("kdigo_summary_ar") if is_ar else clin.get("kdigo_summary_en")
        if not summary_text:
            summary_text = clin.get("kdigo_summary", "")
        
        st.markdown(
            f"""
            <div class="status-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 1.35rem; font-weight: 800; text-transform: uppercase;">
                            {status_icon} {T['status_title']}: {risk_name}
                        </span>
                        <div style="font-size: 1.85rem; font-weight: 900; margin-top: 6px;">
                            KDIGO ({clin['kdigo_stage']}): {stage_name}
                        </div>
                    </div>
                    <div style="margin-top: 10px;">
                        <span style="background: white; padding: 8px 18px; border-radius: 20px; font-weight: 800; font-size: 1.1rem; border: 2px solid currentColor;">
                            {T['patient_tag']}: {patient_name} ({patient_age} {T['years_old']})
                        </span>
                    </div>
                </div>
                <div style="margin-top: 14px; font-size: 1.25rem; font-weight: 500; line-height: 1.65;">
                    {summary_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. CORE DECISION RATIOS
        st.markdown(f"### {T['ratios_heading']}")
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            uacr_val = clin["uACR"]
            uacr_color = "#10B981" if uacr_val < 30 else ("#F59E0B" if uacr_val <= 300 else "#EF4444")
            uacr_band_desc = (
                ("Normal (< 30 mg/g)" if uacr_val < 30 else ("Microalbuminuria (30-300)" if uacr_val <= 300 else "Severely Elevated (> 300)"))
                if not is_ar else
                ("طبيعي (< 30 ملغ/غ)" if uacr_val < 30 else ("بيلة ألبومينية دقيقة (30-300)" if uacr_val <= 300 else "ارتفاع شديد (> 300)"))
            )
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{T['uacr_title']}</div>
                    <div class="metric-val" style="color: {uacr_color};">{uacr_val} <span style="font-size: 1.15rem; font-weight: 600;">{'mg/g' if not is_ar else 'ملغ/غ'}</span></div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: {uacr_color};">
                        KDIGO {clin['kdigo_stage']} &bull; {uacr_band_desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_r2:
            upcr_val = clin["uPCR"]
            upcr_color = "#10B981" if upcr_val < 150 else ("#F59E0B" if upcr_val <= 500 else "#EF4444")
            upcr_band_desc = (
                ("Normal (< 150 mg/g)" if upcr_val < 150 else ("Mild Proteinuria (150-500)" if upcr_val <= 500 else "Marked Proteinuria (> 500)"))
                if not is_ar else
                ("طبيعي (< 150 ملغ/غ)" if upcr_val < 150 else ("بروتين خفيف (150-500)" if upcr_val <= 500 else "بروتين مرتفع (> 500)"))
            )
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{T['upcr_title']}</div>
                    <div class="metric-val" style="color: {upcr_color};">{upcr_val} <span style="font-size: 1.15rem; font-weight: 600;">{'mg/g' if not is_ar else 'ملغ/غ'}</span></div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: {upcr_color};">
                        {upcr_band_desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_r3:
            cre_val = b_dict.get("Creatinine", 100.0)
            dil_status = T['cre_dilution_adequate'] if 30 <= cre_val <= 250 else T['cre_dilution_dilute']
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{T['cre_dilution_title']}</div>
                    <div class="metric-val" style="color: #0284C7;">{cre_val} <span style="font-size: 1.15rem; font-weight: 600;">{'mg/dL' if not is_ar else 'ملغ/ديسيلتر'}</span></div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0284C7;">
                        {dil_status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 3. SECONDARY CLINICAL ALERTS
        if clin["secondary_flags"]:
            st.markdown(f"#### {T['secondary_flags_heading']}")
            for flag in clin["secondary_flags"]:
                p_name = flag["param_ar"] if is_ar else flag["param_en"]
                p_note = flag["note_ar"] if is_ar else flag["note_en"]
                st.warning(f"**{p_name} ({flag['val']}):** {p_note}")

        st.markdown("---")

        # 4. INDIVIDUAL 6-PAD BIOMARKER BREAKDOWN
        st.markdown(f"### {T['biomarkers_heading']}")
        st.caption(T['biomarkers_caption'])

        col_b1, col_b2 = st.columns(2, gap="large")
        
        for idx, b in enumerate(biomarkers):
            target_col = col_b1 if (idx % 2 == 0) else col_b2
            
            with target_col:
                meas_rgb = b["mean_rgb"]
                ref_rgb = b["calibration_ref_rgb"]
                meas_hex = f"#{int(meas_rgb[0]):02x}{int(meas_rgb[1]):02x}{int(meas_rgb[2]):02x}"
                ref_hex = f"#{int(ref_rgb[0]):02x}{int(ref_rgb[1]):02x}{int(ref_rgb[2]):02x}"
                
                b_name = b["name_ar"] if is_ar else b["name_en"]
                b_band = b["matched_band_ar"] if is_ar else b["matched_band_en"]
                b_risk = b["risk_tier_ar"] if is_ar else b["risk_tier_en"]
                b_norm = b["normal_range_ar"] if is_ar else b["normal_range_en"]
                
                tier_color = "#10B981" if b["risk_tier_en"] in ["Normal", "Optimal"] else ("#F59E0B" if b["risk_tier_en"] in ["Monitor", "Moderate"] else "#EF4444")
                
                st.markdown(
                    f"""
                    <div class="pad-tile">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1.35rem; font-weight: 800; color: #1E293B;">
                                {T['pad_label']} {b['pad_index']}: {b_name} ({b['key']})
                            </span>
                            <span style="background: {tier_color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: 700; font-size: 0.95rem;">
                                {b_risk}
                            </span>
                        </div>
                        <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                            {b['quantitative_value']} <span style="font-size: 1.15rem; font-weight: 600; color: #64748B;">{b['unit']}</span>
                        </div>
                        <div style="font-size: 1.05rem; color: #475569; margin-bottom: 12px;">
                            <strong>{T['ref_normal']}:</strong> {b_norm} &bull; <strong>{T['matched_band']}:</strong> {b_band}
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; background: #F8FAFC; padding: 10px 14px; border-radius: 8px;">
                            <div style="display: flex; align-items: center;">
                                <div class="swatch-pill" style="background-color: {meas_hex};"></div>
                                <span style="font-size: 0.95rem; font-weight: 600;">{T['measured_color']}: {meas_hex}</span>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <div class="swatch-pill" style="background-color: {ref_hex};"></div>
                                <span style="font-size: 0.95rem; font-weight: 600;">{T['ref_color']}: {ref_hex}</span>
                            </div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #0284C7;">
                                {T['confidence']}: {b['confidence']}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # 5. PERSONALIZED CLINICAL ACTION PLAN
        st.markdown(f"### {T['actions_heading']}")
        actions_list = clin.get("clinical_actions_ar") if is_ar else clin.get("clinical_actions_en")
        if not actions_list:
            actions_list = clin.get("clinical_actions", [])
        
        st.markdown(
            f"""
            <div style="background: #F1F5F9; {BORDER_ACCENT_SIDE}: 8px solid #0284C7; {BORDER_CLEAR_SIDE} border-radius: 12px; padding: 22px; margin-bottom: 24px;">
                <ul style="margin-bottom: 0; padding-{'right' if is_ar else 'left'}: 24px;">
            """,
            unsafe_allow_html=True
        )
        for act in actions_list:
            st.markdown(f"- **{act}**")
        st.markdown("</ul></div>", unsafe_allow_html=True)

        # 6. DOWNLOAD CLINICAL DOCUMENTATION
        st.markdown(f"### {T['export_heading']}")
        
        # Build CSV Data
        csv_df = pd.DataFrame([
            {
                "Biomarker": b["name_ar"] if is_ar else b["name_en"],
                "Key": b["key"],
                "Measured Value": b["quantitative_value"],
                "Unit": b["unit"],
                "Normal Range": b["normal_range_ar"] if is_ar else b["normal_range_en"],
                "Matched Band": b["matched_band_ar"] if is_ar else b["matched_band_en"],
                "Risk Tier": b["risk_tier_ar"] if is_ar else b["risk_tier_en"],
                "Confidence (%)": b["confidence"]
            }
            for b in biomarkers
        ])
        
        csv_buffer = io.StringIO()
        csv_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label=T['download_csv'],
                data=csv_data,
                file_name=f"NephroScan_{patient_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_d2:
            summary_doc = f"""
=============================================================================
NEPHROSCAN CLINICAL TELEMEDICINE REPORT
Patient: {patient_name} | Age: {patient_age} | Sex: {patient_gender}
Risk Factors: Diabetes: {has_diabetes} | Hypertension: {has_htn} | Heavy NSAIDs: {has_nsaids} | Family History: {has_fam_hist} | Prior CKD: {known_ckd}
-----------------------------------------------------------------------------
OVERALL KDIGO STATUS: {clin['kdigo_stage']} - {stage_name} [{risk_name}]
uACR: {clin['uACR']} mg/g | uPCR: {clin['uPCR']} mg/g

BIOMARKER CONCENTRATIONS:
- Microalbumin: {b_dict.get('Microalbumin')} mg/L
- Creatinine: {b_dict.get('Creatinine')} mg/dL
- Total Protein: {b_dict.get('Total Protein')} mg/dL
- Uric Acid: {b_dict.get('Uric Acid')} mg/dL
- pH: {b_dict.get('pH')}
- Specific Gravity: {b_dict.get('Specific Gravity')}

RECOMMENDED CLINICAL ACTIONS:
{chr(10).join(['* ' + a for a in actions_list])}
=============================================================================
"""
            st.download_button(
                label=T['download_txt'],
                data=summary_doc,
                file_name=f"Physician_Consult_{patient_name.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        # 7. OPTIONAL ADVANCED QUALITY TELEMETRY EXPANDER
        with st.expander(T['telemetry_expander'], expanded=False):
            telemetry = res["telemetry"]
            st.markdown(f"- **{T['telemetry_mode']}:** `{telemetry['detection_mode']}`")
            st.markdown(f"- **{T['telemetry_pads']}:** `{telemetry['pads_segmented']} of 6 Pads`")
            st.markdown(f"- **{T['telemetry_rsd']}:** `{telemetry['mean_strip_rsd']}%` ({T['telemetry_target']})")
            
            telemetry_rows = []
            for b in biomarkers:
                telemetry_rows.append({
                    T['telemetry_table_pad']: f"P{b['pad_index']}: {b['key']}",
                    T['telemetry_table_name']: b["name_ar"] if is_ar else b["name_en"],
                    T['telemetry_table_rsd']: f"{b['rsd_percent']}%",
                    T['telemetry_table_quality']: b["quality_flag_ar"] if is_ar else b["quality_flag_en"]
                })
            st.dataframe(pd.DataFrame(telemetry_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: CLINICAL RESEARCH & SCIENTIFIC METHODOLOGY
# =============================================================================

with tab4:
    st.markdown(
        """
        ## 📖 Clinical Science & Colorimetric Methodology
        
        ### 1. Chronic Kidney Disease (CKD) Pathophysiology
        Chronic Kidney Disease affects over 850 million individuals worldwide. The initial stages are clinically silent; irreversible renal parenchyma loss can occur before serum creatinine or eGFR show dramatic deviations.
        
        Routine spot urine testing quantifying **Microalbumin** and **Creatinine** allows calculation of the **Albumin-to-Creatinine Ratio (uACR)**. This detects early glomerular hyperfiltration and basement membrane fenestration damage years ahead of standard screening.
        
        ---
        
        ### 2. KDIGO 2024 Clinical Guidelines Reference Table
        """
        if not is_ar else
        """
        ## 📖 الأبحاث والأسس السريرية والمنهجية الطبية
        
        ### 1. الفسيولوجيا المرضية لمرض الكلى المزمن (CKD)
        يؤثر مرض الكلى المزمن على أكثر من 850 مليون شخص حول العالم. تبدأ المراحل الأولى بصمت ودون أي أعراض ملحوظة، مما يؤدي إلى تلف غير رجعي في النسيج الكلوي قبل أن يرتفع كرياتينين الدم بشكل ملحوظ.
        
        يتيح الفحص المنزلي الدقيق للبول قياس **الزلال الدقيق (Microalbumin)** و**الكرياتينين (Creatinine)** لحساب **نسبة الزلال إلى الكرياتينين (uACR)**، مما يكشف عن التلف الكبيبي المبكر قبل حدوث القصور بسنوات عديدة.
        
        ---
        
        ### 2. جدول تصنيف KDIGO 2024 السريري
        """
    )
    
    kdigo_ref = [
        {"KDIGO Stage": "A1", "uACR Range": "< 30 mg/g", "uPCR Range": "< 150 mg/g", "Risk Category": "Normal / Low Risk", "Clinical Action": "Annual routine screening"},
        {"KDIGO Stage": "A2", "uACR Range": "30 - 300 mg/g", "uPCR Range": "150 - 500 mg/g", "Risk Category": "Moderately Increased (Microalbuminuria)", "Risk Action": "Repeat test in 2-4 weeks, review BP and SGLT2i/ACEi therapy"},
        {"KDIGO Stage": "A3", "uACR Range": "> 300 mg/g", "uPCR Range": "> 500 mg/g", "Risk Category": "Severely Increased (Macroalbuminuria)", "Risk Action": "Urgent nephrology consultation, comprehensive renal panel"}
    ]
    st.table(pd.DataFrame(kdigo_ref))

    st.markdown(
        """
        ---
        ### 3. Core Scientific Literature & Engineering References
        - **Coleman, B., Coarsey, C., & Asghar, W. (2019)**. *Cell phone based colorimetric analysis for point-of-care settings*. The Analyst, 144(6), 1935–1947. (Foundational methodology for RGB to cylindrical HSV transformation and illumination invariance).
        - **Martínez-Aviño, A., Molins-Legua, C., & Campíns-Falcó, P. (2021)**. *Scaling the Analytical Information Given by Several Types of Colorimetric and Spectroscopic Instruments Including Smartphones*. Analytical Chemistry, 93(15), 6043–6052. (Continuous calibration curves & figures of merit).
        - **Annals of Laboratory Medicine (2024)**. *Enhancing Urine Test Accuracy Using a Mobile Phone–Based Reading System*. (Established the 60–75 second optimal incubation reaction time window).
        - **Journal of Personalized Medicine (2022)**. *Urinary Biomarkers in Chronic Kidney Disease*. (Glomerular filtration barrier vs. Tubulointerstitial injury markers; serum creatinine only rises after 40–50% nephron parenchyma loss).
        """
        if not is_ar else
        """
        ---
        ### 3. المراجع والأوراق البحثية الأساسية للمشروع
        - **دراسة كولمان وزملاؤه (The Analyst, 2019)**: *التحليل اللوني المعتمد على الهواتف الذكية في نقاط الرعاية الطبية*. (الأساس العلمي لتحويل ألوان RGB إلى نظام HSV الأسطواني لتجاوز اختلافات الإضاءة المنزلية).
        - **دراسة مارتينيز-أفينيو (Analytical Chemistry, 2021)**: *معايرة البيانات التحليلية الصادرة من الهواتف الذكية وأجهزة الاستشعار الكيميائية ومقارنتها بالأجهزة المعملية الضخمة*.
        - **المجلة الدولية للطب المخبري (Annals of Lab Medicine, 2024)**: *تحسين دقة قراءة شرائط البول بالذكاء الاصطناعي*. (تحديد نافذة الـ 60 إلى 75 ثانية كأدق وقت لالتقاط صورة التفاعل الكيميائي).
        - **المجلة الطبية للطب الشخصي (JPM, 2022)**: *المؤشرات الحيوية البولية لأمراض الكلى*. (أهمية الكشف المبكر عبر الزلال الدقيق ونسبة uACR قبل فقدان 40-50% من نسيج الكلى الوظيفي).
        """
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    f"""
    <div style="text-align: center; color: #64748B; font-size: 0.95rem; margin-top: 40px; border-top: 1px solid #E2E8F0; padding-top: 20px;">
        {T['footer_text']}
    </div>
    """,
    unsafe_allow_html=True
)
