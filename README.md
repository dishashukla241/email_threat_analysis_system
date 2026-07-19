# Email Threat Analysis & Spam Detection System

An AI-powered email security system that uses **Machine Learning, Natural Language Processing (NLP), and Cybersecurity Feature Engineering** to detect spam and potentially suspicious emails.

The system analyzes email content and security-related indicators to provide an explainable threat assessment, including **Spam Prediction, Spam Probability, Threat Score, Threat Level, Sender Reputation, Threat Reasons, and Blacklist Recommendations**.

An interactive **Streamlit dashboard** provides live email analysis, threat analytics, and forensic investigation of previously processed emails.

---

## Features

### Live Threat Detector
- Analyze new emails in real time
- Paste email content manually
- Upload supported email files
- Spam/Ham prediction
- Spam probability estimation
- Threat score calculation
- Threat level classification
- Sender reputation analysis
- Blacklist recommendation
- Security indicator detection
- Explainable threat reasons

### Security Feature Analysis
The system analyzes several email characteristics, including:

- Suspicious keywords
- URLs
- IP addresses
- Shortened URLs
- Suspicious Top-Level Domains (TLDs)
- Urgency-related language
- Financial or money-related language
- Uppercase usage
- Special characters
- Attachment indicators
- Email structural characteristics

### Threat Analytics
The dashboard provides visual analysis of the processed email dataset, including:

- Threat Level Distribution
- Spam vs Ham Distribution
- Sender Reputation Distribution
- Threat Score Distribution
- Spam Probability Distribution
- Security Indicator Breakdown
- Urgency and Financial Trigger Analysis

### Email Investigation & Forensics
- Search and investigate processed email records
- View detailed email threat information
- Analyze spam probability and threat scores
- Review detected security indicators
- View threat reasons
- Review blacklist recommendations
- Identify high-risk email records

---

## Machine Learning Pipeline

The system follows the pipeline:

Raw Emails  
↓  
Email Extraction  
↓  
Data Cleaning  
↓  
Basic Feature Engineering  
↓  
Security Feature Engineering  
↓  
NLP Preprocessing  
↓  
Train/Test Split  
↓  
TF-IDF Vectorization  
↓  
TF-IDF + Numerical Security Features  
↓  
Machine Learning Classification  
↓  
Sender Reputation Analysis  
↓  
Threat Scoring Engine  
↓  
Streamlit Dashboard

---

## Dataset

The project uses **4,198 email samples** collected and processed from the SpamAssassin email corpus.

The dataset was divided into:

| Dataset | Samples |
|---|---:|
| Training | 3,345 |
| Testing | 837 |

The machine learning feature representation combines:

- **5,000 TF-IDF textual features**
- **22 numerical security and structural features**
- **5,022 total features**

---

## Machine Learning Models

Three machine learning algorithms were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 96.89% | 96.31% | 94.22% | 95.26% | 99.44% |
| Logistic Regression | 95.94% | 96.55% | 90.97% | 93.68% | 98.79% |
| SVM | 76.11% | 84.68% | 33.94% | 48.45% | 86.20% |

**Random Forest** was selected as the final model based on its strong overall classification performance and highest F1-score.

---

## Threat Analysis Engine

The trained machine learning model is combined with cybersecurity indicators and sender reputation information to generate an explainable threat assessment.

The final analysis can include:

- Spam Prediction
- Spam Probability
- Security Score
- Threat Score
- Threat Level
- Sender Reputation
- Sender Reputation Score
- Suspicious Keyword Count
- URL Count
- IP Address Count
- Shortened URL Count
- Suspicious TLD Count
- Urgency Indicator
- Financial Indicator
- Threat Reason
- Blacklist Recommendation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Random Forest
- Logistic Regression
- Support Vector Machine
- Matplotlib
- Streamlit

---

## Project Structure

```text
email_threat_analysis/
│
├── data/
│   └── processed/
│
├── models/
│
├── outputs/
│
├── src/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
