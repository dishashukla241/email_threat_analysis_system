import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ==========================================================
# LOAD DATA
# ==========================================================

print("\n========================================")
print("EMAIL THREAT ANALYSIS ENGINE")
print("========================================")

df = pd.read_csv("data/processed/emails_reputation.csv")

print(f"\nLoaded {len(df)} emails")

# ==========================================================
# LOAD TRAINED MODELS
# ==========================================================

rf_model = joblib.load(
    "models/random_forest.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("\nModels Loaded Successfully")

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

# ==========================================================
# TEXT FEATURES
# ==========================================================

X_text = tfidf.transform(

    df["Processed_Text"].fillna("")

)

print(
    "\nTF-IDF Shape:",
    X_text.shape
)

# ==========================================================
# SECURITY FEATURES
# ==========================================================

security_features = [

    "URL_Count",
    "Email_Count",
    "Attachment_Count",

    "Character_Count",
    "Word_Count",

    "Uppercase_Count",
    "Uppercase_Ratio",
    "Digit_Count",
    "Exclamation_Count",
    "Question_Count",
    "Special_Character_Count",
    "Line_Count",
    "Average_Word_Length",

    "Has_Subject",
    "Has_Links",
    "Has_Attachment",

    "Suspicious_Keyword_Count",

    "IP_Address_Count",
    "Shortened_URL_Count",
    "Suspicious_TLD_Count",

    "Contains_Urgency",
    "Contains_Money"

]

X_num = df[security_features].copy()

for column in security_features:

    X_num[column] = pd.to_numeric(

        X_num[column],

        errors="coerce"

    )

X_num = X_num.fillna(0)

X_num = X_num.astype(float)

print(
    "\nSecurity Features:",
    len(security_features)
)

# ==========================================================
# FINAL FEATURE MATRIX
# ==========================================================

X = sp.hstack(

    [

        X_text,

        sp.csr_matrix(

            X_num.values

        )

    ]

)

print(
    "Final Shape:",
    X.shape
)

# ==========================================================
# MODEL PREDICTION
# ==========================================================

predictions = rf_model.predict(X)

probabilities = rf_model.predict_proba(X)[:,1]

df["Spam_Prediction"] = predictions

df["Spam_Prediction"] = df["Spam_Prediction"].map({

    0:"Ham",

    1:"Spam"

})

df["Spam_Probability"] = (

    probabilities*100

).round(2)

print("\nPrediction Distribution\n")

print(

df["Spam_Prediction"].value_counts()

)

print(

"\nAverage Spam Probability:",

round(

df["Spam_Probability"].mean(),

2

)

)
# ==========================================================
# SECURITY SCORE
# ==========================================================

# Raw security score from engineered cybersecurity features

# ==========================================================
# SECURITY SCORE
# ==========================================================

# Rule-based Security Score
# Each cybersecurity indicator contributes a weighted score.
# The final score is capped at 100.

security_score = (

    df["Suspicious_Keyword_Count"] * 8 +

    df["URL_Count"] * 5 +

    df["IP_Address_Count"] * 5 +

    df["Shortened_URL_Count"] * 8 +

    df["Suspicious_TLD_Count"] * 8 +

    df["Contains_Urgency"].astype(int) * 15 +

    df["Contains_Money"].astype(int) * 10

)

# Keep score between 0 and 100
security_score = security_score.clip(0, 100)

df["Security_Score"] = security_score.round(2)

print(
    "\nAverage Security Score:",
    round(df["Security_Score"].mean(), 2)
)

print(
    "Maximum Security Score:",
    round(df["Security_Score"].max(), 2)
)

# ==========================================================
# THREAT SCORE
# ==========================================================

# ==========================================================
# THREAT SCORE
# ==========================================================

# Final Threat Score
# 70% Machine Learning
# 20% Security Indicators
# 10% Sender Reputation

df["Threat_Score"] = (

    0.70 * df["Spam_Probability"]

    +

    0.20 * df["Security_Score"]

    +

    0.10 * df["Sender_Reputation_Score"]

)

df["Threat_Score"] = (

    df["Threat_Score"]

    .clip(0, 100)

    .round(2)

)

# ==========================================================
# THREAT LEVEL
# ==========================================================

def get_threat_level(score):

    if score < 30:

        return "Low"

    elif score < 55:

        return "Medium"

    elif score < 80:

        return "High"

    else:

        return "Critical"

df["Threat_Level"] = df["Threat_Score"].apply(

    get_threat_level

)

# ==========================================================
# BLACKLIST RECOMMENDATION
# ==========================================================

blacklist = (

    (df["Threat_Level"] == "Critical")

    |

    (

        (df["Threat_Level"] == "High")

        &

        (df["Sender_Reputation"] == "Low")

    )

    |

    (

        df["Spam_Probability"] >= 95

    )

)

df["Blacklist_Recommendation"] = blacklist.map({

    True: "YES",

    False: "NO"

})

# ==========================================================
# THREAT REASONS
# ==========================================================

def generate_reason(row):

    reasons = []

    if row["Spam_Probability"] >= 90:

        reasons.append(

            f'High Spam Probability ({row["Spam_Probability"]:.2f}%)'

        )

    if row["Suspicious_Keyword_Count"] > 0:

        reasons.append(

            f'{int(row["Suspicious_Keyword_Count"])} Suspicious Keyword(s)'

        )

    if row["URL_Count"] > 0:

        reasons.append(

            f'{int(row["URL_Count"])} URL(s) Found'

        )

    if row["IP_Address_Count"] > 0:

        reasons.append(

            "IP Address Detected"

        )

    if row["Shortened_URL_Count"] > 0:

        reasons.append(

            "Shortened URL"

        )

    if row["Suspicious_TLD_Count"] > 0:

        reasons.append(

            "Suspicious TLD"

        )

    if row["Contains_Urgency"] == 1:

        reasons.append(

            "Urgent Language"

        )

    if row["Contains_Money"] == 1:

        reasons.append(

            "Financial Language"

        )

    if row["Sender_Reputation"] == "Low":

        reasons.append(

            "Low Sender Reputation"

        )

    elif row["Sender_Reputation"] == "Very Low":

        reasons.append(

            "Very Low Sender Reputation"

        )

    if row["Threat_Level"] == "Critical":

        reasons.append(

            "Critical Threat Score"

        )

    elif row["Threat_Level"] == "High":

        reasons.append(

            "High Threat Score"

        )

    if len(reasons) == 0:

        reasons.append(

            "No significant threat indicators"

        )

    return ", ".join(reasons)

df["Threat_Reason"] = df.apply(

    generate_reason,

    axis=1

)

print("\nThreat Analysis Complete")

print(

df[[

"Spam_Prediction",

"Spam_Probability",

"Security_Score",

"Threat_Score",

"Threat_Level"

]].head()

)
# ==========================================================
# SORT EMAILS BY THREAT SCORE
# ==========================================================

df = df.sort_values(

    by="Threat_Score",

    ascending=False

).reset_index(drop=True)

# ==========================================================
# DASHBOARD DATASET
# ==========================================================

dashboard_columns = [

    "Label",

    "From",

    "Sender_Domain",

    "Subject",

    "Spam_Prediction",

    "Spam_Probability",

    "Security_Score",

    "Threat_Score",

    "Threat_Level",

    "Sender_Reputation",

    "Sender_Reputation_Score",

    "Suspicious_Keyword_Count",

    "URL_Count",

    "IP_Address_Count",

    "Shortened_URL_Count",

    "Suspicious_TLD_Count",

    "Contains_Urgency",

    "Contains_Money",

    "Threat_Reason",

    "Blacklist_Recommendation"

]

dashboard_df = df[dashboard_columns]

# ==========================================================
# SAVE DASHBOARD DATASET
# ==========================================================

dashboard_path = "data/processed/dashboard_dataset.csv"

dashboard_df.to_csv(

    dashboard_path,

    index=False

)

print("\n========================================")
print("DASHBOARD DATASET CREATED")
print("========================================")

print(f"\nSaved Successfully:\n{dashboard_path}")

# ==========================================================
# SUMMARY
# ==========================================================

print("\n========================================")
print("SUMMARY")
print("========================================")

print("\nSpam Prediction Distribution:\n")

print(

dashboard_df["Spam_Prediction"].value_counts()

)

print("\nThreat Level Distribution:\n")

print(

dashboard_df["Threat_Level"].value_counts()

)

print("\nSender Reputation Distribution:\n")

print(

dashboard_df["Sender_Reputation"].value_counts()

)

print("\nBlacklist Recommendation:\n")

print(

dashboard_df["Blacklist_Recommendation"].value_counts()

)

print("\nAverage Spam Probability :",

round(

dashboard_df["Spam_Probability"].mean(),

2

),

"%"

)

print("Average Security Score  :",

round(

dashboard_df["Security_Score"].mean(),

2

)

)

print("Average Threat Score    :",

round(

dashboard_df["Threat_Score"].mean(),

2

)

)

print("Highest Threat Score    :",

round(

dashboard_df["Threat_Score"].max(),

2

)

)

print("Lowest Threat Score     :",

round(

dashboard_df["Threat_Score"].min(),

2

)

)

# ==========================================================
# SHOW TOP 10 THREATS
# ==========================================================

print("\n========================================")
print("TOP 10 MOST DANGEROUS EMAILS")
print("========================================")

print(

dashboard_df[[

    "Sender_Domain",

    "Subject",

    "Spam_Prediction",

    "Spam_Probability",

    "Threat_Score",

    "Threat_Level",

    "Sender_Reputation",

    "Blacklist_Recommendation"

]].head(10)

)

# ==========================================================
# FINAL MESSAGE
# ==========================================================

print("\n========================================")
print("EMAIL THREAT ANALYSIS COMPLETED")
print("========================================")

print("\nDashboard is now ready to use.")

print("\nDashboard dataset:")

print("data/processed/dashboard_dataset.csv")

print("\nColumns available:")

for column in dashboard_df.columns:

    print("-", column)

print("\nBackend pipeline completed successfully.\n")