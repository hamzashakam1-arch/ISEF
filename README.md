# NephroScan Clinical: AI-Assisted Multiplex Urine Analysis Platform for Early CKD Detection and Telehealth Home Monitoring

**NephroScan Clinical Platform**  
**Specialty:** Point-of-Care Renal Telemedicine & Diagnostic Computer Vision  

---

## 📌 1. Platform Overview & Clinical Rationale

Chronic Kidney Disease (CKD) affects more than **850 million individuals** worldwide. Early-stage renal impairment is predominantly asymptomatic, leading to delayed diagnosis and irreversible nephron loss in over 90% of cases.

Clinical screening relies on spot-urine **Microalbumin** and **Creatinine** quantification to calculate the **Albumin-to-Creatinine Ratio (uACR)**. Traditional hospital laboratory tests, however, require formal clinical visits, expensive benchtop spectrophotometers, and days for result delivery.

**NephroScan Clinical** brings clinical-grade urinalysis into the patient's home or outpatient clinic using standard smartphone and webcam cameras:
- **Full Bilingual Operation**: Seamless English and Arabic (العربية) toggle with native right-to-left (RTL) typography.
- **Elderly-Friendly Accessible UI**: High-contrast clinical cards, large typography modes, intuitive guided workflows, and interactive reaction timing.
- **Standardized Optical Colorimetry**: Retinex color balance, core sub-sampling, and continuous multi-point calibration mapping.
- **KDIGO 2024 Clinical Decision Support**: Automated uACR/uPCR calculation and staging aligned with international nephrology standards.

---

## 🌐 2. Bilingual Support (English & Arabic)

The application features full dynamic localization:
- **Language Selector**: Prominent toggle at the top of the clinical sidebar (`English` / `العربية`).
- **Dynamic RTL Layout**: Automatic right-to-left alignment, Arabic typography (`Cairo`, `Tahoma`), flipped step badges, and localized clinical action plans.
- **Bilingual Medical Nomenclature**: Exact medical terms for all 6 biomarkers (e.g. الزلال الدقيق، الكرياتينين، البروتين الكلي، حمض اليوريك، الرقم الهيدروجيني، الكثافة النوعية), KDIGO risk tiers, and exportable physician notes.

---

## 🧪 3. Multiplex 6-Pad Biomarker Panel

| Pad # | Analyte (English) | Analyte (العربية) | Clinical Significance | Normal Range | Unit |
|:---:|:---|:---|:---|:---:|:---:|
| **1** | **Microalbumin (mALB)** | **الزلال الدقيق** | Early marker of glomerular permeability and diabetic nephropathy | < 20 | mg/L |
| **2** | **Creatinine (CRE)** | **الكرياتينين** | Internal metabolic standard for spot-urine dilution normalization | 30 – 250 | mg/dL |
| **3** | **Total Protein (PRO)** | **البروتين الكلي** | Detects gross tubular and glomerular protein leakage | < 15 (Neg) | mg/dL |
| **4** | **Uric Acid (UA)** | **حمض اليوريك** | Marker for hyperuricemic nephropathy, renal calculi, and gout | 15 – 60 | mg/dL |
| **5** | **pH** | **الرقم الهيدروجيني** | Evaluates urinary acidification, tubular acidosis, and stone risk | 5.5 – 7.5 | pH units |
| **6** | **Specific Gravity (SG)** | **الكثافة النوعية** | Assesses renal concentrating capability and isosthenuria | 1.010 – 1.025 | SG ratio |

---

## 🧮 4. Clinical Ratios & KDIGO 2024 Staging Matrix

1. **Spot Urinary Albumin-to-Creatinine Ratio (uACR)**:
   $$\text{uACR (mg/g)} = \frac{\text{Microalbumin (mg/L)} \times 100}{\text{Creatinine (mg/dL)}}$$
2. **Spot Urinary Protein-to-Creatinine Ratio (uPCR)**:
   $$\text{uPCR (mg/g)} = \frac{\text{Total Protein (mg/dL)}}{\text{Creatinine (mg/dL)}} \times 1000$$

### Risk Stratification Matrix
* **🟢 KDIGO Stage A1 (Normal / Low Risk):** $\text{uACR} < 30\text{ mg/g}$  
  *Glomerular barrier functionally intact. Annual routine screening recommended.*
* **🟡 KDIGO Stage A2 (Moderately Increased / Microalbuminuria):** $30 \le \text{uACR} \le 300\text{ mg/g}$  
  *Early renal stress. Highly reversible with proactive glycemic and blood pressure control (ACEi/ARBs, SGLT2i).*
* **🔴 KDIGO Stage A3 (Severely Increased / Macroalbuminuria):** $\text{uACR} > 300\text{ mg/g}$ or $\text{uPCR} > 500\text{ mg/g}$  
  *Marked proteinuria. Significant risk of CKD progression. Urgent nephrology consultation and comprehensive renal panel required.*

---

## 🚀 5. How to Run the Application Locally

### Step 1: Open Terminal in Project Directory
```powershell
cd "d:\ISEF APP"
```

### Step 2: Install Required Libraries
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Automated Verification Suite
```powershell
python test_engine.py
```

### Step 4: Launch the Web Platform
```powershell
streamlit run app.py
```
Open your web browser at:
`http://localhost:8501`

---

## 📁 6. Codebase Architecture

- **`app.py`**: Streamlit frontend featuring full bilingual translation dictionary (`I18N`), clean clinical portal styling, interactive 60–75s reaction timer, and clean KDIGO dashboard.
- **`cv_engine.py`**: Computer vision processing engine (Retinex illumination balance, deskewing, core sub-sampling, HSV color distance, and KDIGO decision rules).
- **`sample_generator.py`**: Synthesizes 3 photorealistic calibrated clinical benchmark strips for demonstration.
- **`test_engine.py`**: Automated verification test suite.
