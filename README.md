# 🏥 Hate Speech Detection & Analysis of its Consequences in Healthcare Industry

> Enhancing Online Health Community (OHC) wellbeing through automated hate speech detection,
> positive sentiment integration, and a progressive NLP modelling pipeline.

---

## 👥 Team Members

| Roll Number  | Name                              |
|-------------|-----------------------------------|
| 2023BCS0237 | Sidhaarth Mohandas                |
| 2023BCS0231 | Kunamalla Tharun                  |
| 2023BCD0066 | Venkat Chiranjeevi Reddy Basireddy|
| 2023BCS0039 | Gangireddy Vishnu Vardhan Reddy   |

**Group No:** 17

---

## 📌 Problem Statement

Online Health Communities (OHCs) — forums, subreddits, and patient networks where people
discuss medical conditions, treatments, and mental health — are increasingly polluted by hate
speech and toxic discourse. This exposure causes measurable harm:

- **Care Avoidance:** Patients delay or cancel seeking medical help due to stigmatizing language
- **Empathy Erosion:** Bystanders and caregivers lose supportive instincts after repeated toxic exposure
- **Narrative Imbalance:** Healthcare discourse is dominated by fear and negativity rather than recovery and wellness

This project builds an automated pipeline to **detect, classify, and mitigate hate speech** in
OHCs, and to **measure the impact of positive content injection** on community health outcomes.

---

## 🎯 Objectives

1. Detect and classify hate speech and toxic language in OHC text data with high precision
2. Engineer clinically meaningful text features (sentiment, toxicity density, aggression signals)
3. Build a **progressive model pipeline**: Logistic Regression baseline → BERT → BioBERT
4. Analyze which post characteristics most strongly predict toxicity using a composite risk score
5. Demonstrate that positive sentiment content can be identified and amplified within OHCs
6. Deploy a privacy-preserving, explainable NLP system suitable for real clinical environments

---

## 📊 Dataset

### Primary: Davidson Hate Speech & Offensive Language Dataset

| Property            | Value                                      |
|--------------------|--------------------------------------------|
| **Source**         | GitHub – t-davidson/hate-speech-and-offensive-language |
| **Observations**   | ~24,783 tweets (after deduplication)       |
| **Variables**      | 10 original + 10 engineered features       |
| **Target Variable**| `label` — 0: Hate Speech, 1: Offensive, 2: Neither |
| **Class Balance**  | Hate ~5.8% · Offensive ~77.4% · Neither ~16.8% |
| **Access**         | Public — no authentication required        |

### Key Attributes

| Attribute               | Description                                              |
|------------------------|----------------------------------------------------------|
| `text`                 | Raw tweet content                                        |
| `label`                | Ground truth class (0=Hate, 1=Offensive, 2=Neither)      |
| `hate_votes`           | Number of annotators who labelled it hate speech         |
| `offensive_votes`      | Number of annotators who labelled it offensive           |
| `count`                | Total number of annotators per tweet                     |
| `SentimentScore`       | VADER compound sentiment score (-1 to +1) [engineered]   |
| `ToxicityDensity`      | Ratio of flagged toxic words to total words [engineered] |
| `CapitalizationRatio`  | Ratio of ALL-CAPS words (aggression proxy) [engineered]  |
| `LexicalDiversity`     | Unique word ratio (repetition = low diversity) [engineered]|
| `RiskScore`            | Composite toxicity risk score 0–7 [engineered]           |

### Secondary (for fine-tuning phase)
- **ETHOS Dataset** — 998 Reddit/YouTube comments labelled for hate speech
- **Synthea Synthetic EHR** — Privacy-safe clinical records for patient journey simulation
- **Reddit OHC Scrape** — Custom scraped data from r/diabetes, r/cancer, r/mentalhealth *(Phase 2)*

---

## 🔬 Methodology

### 1. Data Preprocessing
- Removed URLs, @mentions, non-ASCII characters, and duplicate tweets
- Text length outliers capped via **winsorization** (1st–99th percentile)
- Binary target created: Hate Speech (1) vs. Not Hate (0)
- Class imbalance handled via **stratified upsampling** to 50/50 balance

### 2. Exploratory Analysis (5-Part Pipeline)

| Part | Focus | Key Output |
|------|-------|-----------|
| Part 1 | Exploration & Cleaning | Class distribution, outlier treatment |
| Part 2 | Feature Engineering | 4 derived features, normalization, concept hierarchy |
| Part 3 | PCA & Aggregation | Dimensionality reduction, Sentiment × Toxicity pivot |
| Part 4 | Smoothing & Sampling | Noise reduction, stratified balancing |
| Part 5 | Risk Analysis | Composite toxicity risk score, feature importance |

**4 Engineered Features:**
- `SentimentScore` — VADER compound score per post
- `ToxicityDensity` — Flagged toxic word ratio
- `CapitalizationRatio` — ALL-CAPS aggression marker
- `LexicalDiversity` — Vocabulary richness (inverse repetition)

**Composite Risk Score Logic:**

```
SentimentScore < -0.5       → +2 pts
ToxicityDensity > 0.10      → +2 pts
CapitalizationRatio > 0.30  → +1 pt
Sentiment below median      → +1 pt
LexicalDiversity < 0.5      → +1 pt

Low Risk: 0–2 | Medium Risk: 3–4 | High Risk: 5+
```

**PCA:** Applied to 4 text features — 3 components explain ≥85% variance.
PC1 captures overall negativity/toxicity intensity; PC2 captures expression aggression style.

### 3. Models Used

| Stage | Model | Approach |
|-------|-------|----------|
| Stage 1 | Logistic Regression + TF-IDF | Baseline — fast, interpretable, bigram features |
| Stage 2 | BERT-base Fine-tuned | Context-aware transformer, general English |
| Stage 3 | BioBERT Fine-tuned | Biomedical pre-training, OHC-domain optimized |

All models trained on balanced binary classification (Hate vs. Not Hate).
BERT and BioBERT fine-tuned for 3 epochs on 4,000-sample stratified subset (GPU required).

### 4. Evaluation Methods
- **Precision, Recall, F1-Score** (primary — hate is the minority class)
- **AUC-ROC** — discriminative ability across classification thresholds
- **Confusion Matrix** — false positive/negative analysis
- **Feature Correlation Analysis** — Pearson correlation of all features with hate label

---

## 📈 Results


| Model                    | Precision | Recall | F1-Score | AUC-ROC |
|--------------------------|-----------|--------|----------|---------|
| LR + TF-IDF (Baseline)   | 0.81      | 0.76   | 0.78     | 0.91    |
| BERT-base Fine-tuned     | 0.88      | 0.85   | 0.86     | 0.95    |
| BioBERT Fine-tuned       | 0.91      | 0.88   | 0.89     | 0.97    |

**Risk Score Validation:**

| Risk Category | Post Count | Actual Hate Rate |
|---------------|-----------|-----------------|
| Low (0–2)     | ~18,500   | ~2.1%           |
| Medium (3–4)  | ~4,800    | ~14.7%          |
| High (5+)     | ~1,400    | ~41.3%          |

**Key Finding:** High-risk posts are ~20× more likely to be hate speech than low-risk posts,
confirming the composite risk score is an effective pre-filter for moderation queues.

---

## 📉 Key Visualizations

The notebook generates the following plots (all tagged with student name/roll number):

1. **Class Distribution Bar Chart** — Hate vs Offensive vs Neither counts and percentages
2. **Text Length Boxplot** — Before vs After winsorization outlier treatment
3. **Hate Rate by Sentiment Bin** — Positive / Neutral / Negative → hate rate
4. **SentimentScore Histogram** — Original vs Min-Max normalized
5. **Concept Hierarchy Bar Chart** — Hate rate rolled up by sentiment category
6. **Scree Plot** — PCA variance explained per component with 85% threshold line
7. **Pivot Heatmap** — Sentiment × Toxicity Bin → Hate Rate (%)
8. **Smoothing Histogram** — Original vs smoothed SentimentScore distribution
9. **Class Balance Chart** — Original vs stratified balanced dataset
10. **Risk Score Bar Chart** — Risk category vs actual hate speech rate
11. **Post Length Group Analysis** — Hate rate by text length bucket
12. **Feature Importance Bar Chart** — Top feature correlations with hate label
13. **Confusion Matrices** — One per model (LR / BERT / BioBERT)
14. **Model Comparison Chart** — Grouped bar chart: Precision, Recall, F1, AUC-ROC across all 3 models

---

## 🚀 How to Run the Project

### Requirements
- Google Colab account (free tier works for Parts 1–5; GPU needed for BERT/BioBERT)
- No local installation required

### Steps

```
1. Open Google Colab → https://colab.research.google.com
2. Upload: OHC_HateSpeech_Pipeline_Group17.ipynb
3. Runtime → Run All  (Parts 1–5 run on CPU, ~2–3 minutes)
4. For Model Stages 2 & 3:
      Runtime → Change runtime type → Hardware accelerator → T4 GPU
      Then re-run the BERT and BioBERT cells
5. All plots are auto-saved inline in the notebook output
```

### Folder Organization

```
OHC-HateSpeech-Detection/
│
├── OHC_HateSpeech_Pipeline_Group17.ipynb   ← Main notebook (all 5 parts + 3 models)
├── README.md                                ← This file
│
├── docs/
│   ├── DWM_project_Abstract.pdf            ← Project abstract
│   └── DataCollection_Methodology.docx     ← Full methodology document
│
└── outputs/                                 ← Generated plots & tables
    └── (auto-created when notebook is run)
```

> **Dataset is loaded automatically** inside the notebook from the Davidson GitHub URL —
> no manual download needed.

---

## 🏁 Conclusion

This project demonstrates that hate speech in Online Health Communities can be reliably
detected using a layered NLP pipeline. The key findings are:

- **Sentiment + Toxicity are the dominant signals**: SentimentScore and ToxicityDensity
  together account for the majority of predictive power across all three model stages
- **Context matters**: BERT improved over TF-IDF by understanding that identical words carry
  different threat levels depending on surrounding context
- **BioBERT is the right long-term model**: Its biomedical pre-training gives it a native
  advantage for OHC text, which will compound when fine-tuned on actual Reddit health data
- **The risk score works as a moderation pre-filter**: High-risk posts are flagged with ~20×
  higher precision than random sampling, enabling efficient human-in-the-loop moderation
- **Next phase**: Scrape and annotate Reddit OHC data (r/diabetes, r/mentalhealth, r/cancer)
  and fine-tune BioBERT for the final healthcare-specific production classifier

---

## 🤝 Contributions

| Roll No     | Member                              | Contributions |
|-------------|-------------------------------------|---------------|
| 2023BCS0237 | Sidhaarth Mohandas                  | Part 1 (Data Exploration & Cleaning), Part 2 (Feature Engineering & Transformation), Abstract & Problem Framing |
| 2023BCS0231 | Kunamalla Tharun                    | Part 3 (PCA & Aggregation), Part 4 (Smoothing & Sampling), Dataset research & curation |
| 2023BCD0066 | Venkat Chiranjeevi Reddy Basireddy  | Part 5 (Risk Analysis & Insights), Stage 1 Model (LR + TF-IDF), Feature importance analysis |
| 2023BCS0039 | Gangireddy Vishnu Vardhan Reddy     | Stage 2 & 3 Models (BERT + BioBERT fine-tuning), Model evaluation & comparison, Report writing |

---

## 📚 References

### Datasets
- Davidson, T. et al. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* AAAI. [GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language)
- Mollas, I. et al. (2022). *ETHOS: An Online Hate Speech Detection Dataset.* [arXiv](https://arxiv.org/abs/2006.08328)
- Mody, D. et al. (2022). *A curated dataset for hate speech detection on social media text.* Data in Brief. [PubMed](https://pubmed.ncbi.nlm.nih.gov/36605500/)

### Models & Libraries
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* [arXiv](https://arxiv.org/abs/1810.04805)
- Lee, J. et al. (2020). *BioBERT: a pre-trained biomedical language representation model.* [arXiv](https://arxiv.org/abs/1901.08746)
- Hutto, C. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis.* ICWSM.
- Hugging Face Transformers. [huggingface.co](https://huggingface.co/transformers)

### Related Work
- Nguyen, T. (2024). *Merging public health and automated approaches to address online hate speech.* AI and Ethics.
- Raza, S. & Chatrath, V. (2024). *HarmonyNet: Navigating hate speech detection.* Natural Language Processing Journal.
