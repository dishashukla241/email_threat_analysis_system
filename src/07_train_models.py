import os
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print("\nDatasets Loaded Successfully\n")

print(f"Training Samples : {len(train_df)}")
print(f"Testing Samples  : {len(test_df)}")

# ============================================================
# TEXT DATA
# ============================================================

X_train_text = train_df["Processed_Text"].fillna("")
X_test_text = test_df["Processed_Text"].fillna("")

y_train = train_df["Label"]
y_test = test_df["Label"]

# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("\nTF-IDF Completed")
print("Training :", X_train_tfidf.shape)
print("Testing  :", X_test_tfidf.shape)

# ============================================================
# SECURITY FEATURES
# ============================================================

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

X_train_num = train_df[security_features].copy()
X_test_num = test_df[security_features].copy()

for col in security_features:

    X_train_num[col] = pd.to_numeric(
        X_train_num[col],
        errors="coerce"
    )

    X_test_num[col] = pd.to_numeric(
        X_test_num[col],
        errors="coerce"
    )

X_train_num = X_train_num.fillna(0).astype(float)
X_test_num = X_test_num.fillna(0).astype(float)

# ============================================================
# FEATURE MATRICES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Logistic Regression (scaled)

X_train_lr = sp.hstack([
    X_train_tfidf,
    sp.csr_matrix(X_train_scaled)
])

X_test_lr = sp.hstack([
    X_test_tfidf,
    sp.csr_matrix(X_test_scaled)
])

# Random Forest & SVM (original numerical values)

X_train_tree = sp.hstack([
    X_train_tfidf,
    sp.csr_matrix(X_train_num.values)
])

X_test_tree = sp.hstack([
    X_test_tfidf,
    sp.csr_matrix(X_test_num.values)
])

print("\nFeature Matrices Ready")

print("Logistic :", X_train_lr.shape)
print("Tree     :", X_train_tree.shape)

results = []

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc
    })

    print("\n===================================")
    print(model_name)
    print("===================================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc:.4f}")

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title(model_name)

    plt.savefig(
        f"outputs/{model_name.replace(' ','_')}_confusion.png"
    )

    plt.close()

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.title(model_name)

    plt.savefig(
        f"outputs/{model_name.replace(' ','_')}_roc.png"
    )

    plt.close()
    # ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...\n")

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_lr,
    y_train
)

joblib.dump(
    logistic_model,
    "models/logistic_regression.pkl"
)

evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_test_lr,
    y_test
)

# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...\n")

rf_model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

rf_model.fit(

    X_train_tree,

    y_train

)

joblib.dump(

    rf_model,

    "models/random_forest.pkl"

)

evaluate_model(

    "Random Forest",

    rf_model,

    X_test_tree,

    y_test

)

# ============================================================
# SVM
# ============================================================

print("\nTraining SVM...\n")

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

base_svm = LinearSVC(
    random_state=42
)

svm_model = CalibratedClassifierCV(
    estimator=base_svm,
    cv=3
)

svm_model.fit(

    X_train_tree,

    y_train

)

joblib.dump(

    svm_model,

    "models/svm.pkl"

)

evaluate_model(

    "SVM",

    svm_model,

    X_test_tree,

    y_test

)
# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================\n")

print(results_df)

# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]["Model"]

best_f1 = results_df.iloc[0]["F1 Score"]

print("\n========================================")
print("BEST MODEL")
print("========================================\n")

print(f"Model    : {best_model}")
print(f"F1 Score : {best_f1:.4f}")

# ============================================================
# SAVE BEST MODEL NAME
# ============================================================

with open(
    "outputs/best_model.txt",
    "w"
) as f:

    f.write(best_model)

# ============================================================
# SAVE SECURITY FEATURE LIST
# ============================================================

feature_df = pd.DataFrame({
    "Security_Feature": security_features
})

feature_df.to_csv(
    "outputs/security_features.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("TRAINING SUMMARY")
print("========================================\n")

print(f"Training Samples : {len(train_df)}")
print(f"Testing Samples  : {len(test_df)}")

print(f"\nTF-IDF Features     : {X_train_tfidf.shape[1]}")
print(f"Security Features  : {len(security_features)}")
print(f"Total Features     : {X_train_tree.shape[1]}")

# ============================================================
# FILES GENERATED
# ============================================================

print("\n========================================")
print("FILES GENERATED")
print("========================================\n")

generated_files = [

    "models/logistic_regression.pkl",
    "models/random_forest.pkl",
    "models/svm.pkl",
    "models/tfidf_vectorizer.pkl",
    "models/scaler.pkl",

    "outputs/model_results.csv",
    "outputs/security_features.csv",
    "outputs/best_model.txt",

    "outputs/Logistic_Regression_confusion.png",
    "outputs/Random_Forest_confusion.png",
    "outputs/SVM_confusion.png",

    "outputs/Logistic_Regression_roc.png",
    "outputs/Random_Forest_roc.png",
    "outputs/SVM_roc.png"

]

for file in generated_files:
    print(file)

print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================\n")