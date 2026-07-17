import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

rf = joblib.load("models/random_forest.pkl")

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

# ==========================================================
# LOAD TF-IDF
# ==========================================================

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

tfidf_features = tfidf.get_feature_names_out()

# ==========================================================
# COMBINE FEATURE NAMES
# ==========================================================

all_features = list(tfidf_features) + security_features

# ==========================================================
# IMPORTANCE
# ==========================================================

importance = rf.feature_importances_

importance_df = pd.DataFrame({

    "Feature": all_features,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

# ==========================================================
# SAVE COMPLETE LIST
# ==========================================================

importance_df.to_csv(

    "outputs/feature_importance.csv",

    index=False

)

# ==========================================================
# TOP 20
# ==========================================================

top20 = importance_df.head(20)

print("\nTop 20 Important Features\n")

print(top20)

# ==========================================================
# GRAPH
# ==========================================================

plt.figure(figsize=(10,8))

plt.barh(

    top20["Feature"],

    top20["Importance"]

)

plt.gca().invert_yaxis()

plt.title(

    "Top 20 Most Important Features"

)

plt.xlabel(

    "Feature Importance"

)

plt.tight_layout()

plt.savefig(

    "outputs/top20_feature_importance.png",

    dpi=300

)

plt.close()

print("\nSaved:")

print("outputs/feature_importance.csv")

print("outputs/top20_feature_importance.png")