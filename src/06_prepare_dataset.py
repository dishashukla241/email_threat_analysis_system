import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------
# Load Dataset
# --------------------------------

df = pd.read_csv("data/processed/emails_final.csv")

# Remove empty processed text
df["Processed_Text"] = df["Processed_Text"].fillna("").astype(str)

df = df[df["Processed_Text"].str.strip() != ""]

# --------------------------------
# Encode Labels
# --------------------------------

label_map = {
    "Ham": 0,
    "Spam": 1
}

df["Label"] = df["Label"].map(label_map)

# --------------------------------
# Train-Test Split
# --------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Label"]
)

# --------------------------------
# Save
# --------------------------------

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)

# --------------------------------
# Summary
# --------------------------------

print("\nDataset Preparation Complete\n")

print(f"Training Samples : {len(train_df)}")
print(f"Testing Samples  : {len(test_df)}")

print("\nTrain Label Distribution")

print(train_df["Label"].value_counts())

print("\nTest Label Distribution")

print(test_df["Label"].value_counts())